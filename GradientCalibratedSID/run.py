"""Run all stages or independently resume one stage of the fixed GCSS protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from LoCoRec.locorec.data import NextItemDataset, collate_eval
from LoCoRec.locorec.io import read_json, write_json
from LoCoRec.locorec.soft_sid import build_semantic_table
from .model import GCSS, load_model
from .candidates import build_candidates
from .calibration import compatibility, sharing_weights, shuffled_compatibility, pad_item
from .profiling import profile, positive_gradients
from .trainer import fit, training_loader, tensor_hash


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_text(directory, n):
    values = np.load(directory / 'item_text_embeddings.npy')
    ids = np.asarray(read_json(directory / 'embedding_item_ids.json'), dtype=int)
    if values.ndim != 2 or len(values) != len(ids) or set(ids.tolist()) != set(range(1,n+1)) or len(ids) != n:
        raise ValueError('Frozen text IDs do not match the item catalog')
    result = np.empty_like(values)
    result[ids-1] = values
    return result


def profile_stats(payload):
    sig, counts = payload['grad_signature'], payload['grad_count']
    active = counts.gt(0)
    norms = sig.norm(dim=-1)
    return {'shape': list(sig.shape), 'examples': payload['examples'], 'valid_examples': payload['valid_examples'],
            'profiled_items': int(active.sum()), 'zero_count_items': int((~active).sum()),
            'active_norm_quantiles': torch.quantile(norms[active], torch.tensor([0.,.25,.5,.75,1.])).tolist() if active.any() else [],
            'active_count_quantiles': torch.quantile(counts[active].float(), torch.tensor([0.,.5,1.])).tolist() if active.any() else [],
            'outer_normalization': False, 'profiling_tf32': False}


def debug_profile(model, data, config, seed):
    selected = sorted({int(x[1]) for x in data.samples})[:32]
    subset = Subset(data, [i for i, x in enumerate(data.samples) if x[1] in selected])
    if not len(subset):
        raise ValueError('No training examples for profile debug')
    state_before = {k: tensor_hash(v) for k,v in model.state_dict().items()}
    model.eval().requires_grad_(False)
    # A small batch is checked against individual-example gradients. A repeated
    # positive token in other rows cannot introduce cross-example gradients.
    h, c = next(iter(training_loader(subset, config, seed + 4001, shuffle=False)))
    h, c = h[:8].to(config['device']), c[:8].to(config['device'])
    batch = positive_gradients(model, h, c[:,0], c[:,1:])
    singles = torch.cat([positive_gradients(model, h[i:i+1], c[i:i+1,0], c[i:i+1,1:]) for i in range(len(h))])
    torch.testing.assert_close(batch, singles, atol=5e-5, rtol=5e-4)
    payload = profile(model, training_loader(subset, config, seed + 4002, shuffle=False), config['device'])
    if state_before != {k: tensor_hash(v) for k,v in model.state_dict().items()}:
        raise RuntimeError('Profiling changed frozen model state')
    result = profile_stats(payload)
    result.update(item_ids=selected, individual_batch_max_error=float((batch-singles).abs().max()),
                  frozen_state_verified=True, parameters_received_gradients=False)
    return result


def summarize(root):
    rows = [read_json(p) for p in sorted(root.glob('seed*/stage2/*/result.json'))]
    variants = ('hard','semantic_soft','gradient_calibrated','random_compatibility')
    summary, comparisons = {}, {}
    for variant in variants:
        group = [r for r in rows if r['variant'] == variant]
        if not group:
            continue
        values = torch.tensor([r['test']['NDCG@10'] for r in group], dtype=torch.double)
        summary[variant] = {'seeds': [r['seed'] for r in group], 'mean': float(values.mean()),
                            'std': float(values.std()) if len(values)>1 else None,
                            'parameters': group[0]['parameters'], 'best_epochs': [r['best_epoch'] for r in group]}
    for baseline in ('hard','semantic_soft','random_compatibility'):
        treated = {r['seed']: r for r in rows if r['variant']=='gradient_calibrated'}
        base = {r['seed']: r for r in rows if r['variant']==baseline}
        seeds = sorted(set(treated)&set(base))
        if seeds:
            delta = [treated[s]['test']['NDCG@10']-base[s]['test']['NDCG@10'] for s in seeds]
            comparisons['gradient_calibrated_minus_'+baseline] = {'seeds': seeds, 'per_seed_delta': delta, 'mean_delta': float(np.mean(delta))}
    write_json(root/'summary.json', {'models': summary, 'paired_comparisons': comparisons})
    lines = ['# Gradient-Calibrated Semantic Sharing', '',
             '四个Stage 2版本从同一个新训练的纯SID checkpoint开始，重置AdamW，仅训练推荐CE。无private、gate、动态attention或辅助损失。Stage 1原始结果在各seed/stage1目录，不混为Stage 2 Hard continuation。', '',
             '| Variant | Seeds | Test NDCG@10 | Sample std | Parameters | Best epochs |',
             '|---|---|---:|---:|---:|---|']
    for v,r in summary.items():
        std = f"{r['std']:.6f}" if r['std'] is not None else 'single seed'
        lines.append(f"| {v} | {r['seeds']} | {r['mean']:.6f} | {std} | {r['parameters']} | {r['best_epochs']} |")
    lines += ['', '核心比较是gradient_calibrated相对于semantic_soft和random_compatibility的同seed差值。部分完成时本表只包含已完成运行。', '', '```json', json.dumps(comparisons,indent=2), '```', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


def main(forced_stage=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all','candidates','hard','profile','calibrate','stage2'], default=forced_stage or 'all')
    parser.add_argument('--dataset-dir', default='runs/office')
    parser.add_argument('--semantic-ids', default='runs/office/semantic_ids_rq.json')
    parser.add_argument('--output-dir', default='runs/office/gcss_20260910')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026])
    parser.add_argument('--device', default='cuda')
    for k,v in [('H',50),('M',4),('dim',128),('max-len',50),('heads',2),('layers',2),('batch-size',256),
                ('negatives',100),('epochs',100),('patience',10),('threads',2)]:
        parser.add_argument('--'+k, type=int, default=v)
    for k,v in [('lambda-grad',1.),('dropout',.2),('lr',.001),('weight-decay',.0001)]:
        parser.add_argument('--'+k, type=float, default=v)
    args = parser.parse_args()
    if forced_stage and args.stage != forced_stage:
        parser.error('This stage wrapper cannot run another stage')
    if args.H<0 or min(args.M,args.dim,args.max_len,args.heads,args.layers,args.batch_size,args.negatives,args.epochs,args.patience)<1 or args.dim%args.heads:
        raise ValueError('Invalid model/training dimensions')
    if args.lambda_grad < 0 or not np.isfinite(args.lambda_grad):
        raise ValueError('lambda_grad must be finite and nonnegative')
    torch.set_num_threads(args.threads)
    root, data_dir = Path(args.output_dir), Path(args.dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows, stats = read_json(data_dir/'sequences.json'), read_json(data_dir/'stats.json')
    sid_obj = read_json(args.semantic_ids)
    sizes = sid_obj['codebook_sizes']
    for item in range(1,stats['num_items']+1):
        codes = sid_obj['semantic_ids'].get(str(item))
        if codes is None or len(codes)!=len(sizes) or any(not 0<=int(c)<s for c,s in zip(codes,sizes)):
            raise ValueError(f'Invalid SID for item {item}')
    hard, _, num_tokens = build_semantic_table(sid_obj, stats['num_items'])
    config = {**vars(args), 'num_items': stats['num_items']}
    stable_config = {k:v for k,v in config.items() if k not in ('stage','device','threads')}
    manifest = {'config': stable_config, 'data_sha256': {str(p):file_hash(p) for p in (
        data_dir/'sequences.json', Path(args.semantic_ids), data_dir/'item_text_embeddings.npy', data_dir/'embedding_item_ids.json')},
        'note': 'New pure SID baseline. No private embeddings or imported trained LoCoRec checkpoints. Profiling is positive-only; g/prototypes are not normalized. Lambda/H/M fixed before this run.'}
    if (root/'protocol.json').exists() and read_json(root/'protocol.json') != manifest:
        raise ValueError('Output directory belongs to a different protocol; use a new directory')
    write_json(root/'protocol.json', manifest)
    candidate_path = root/'semantic_candidates.pt'
    if args.stage in ('all','candidates') and not candidate_path.exists():
        artifact = build_candidates(hard[1:].numpy(), aligned_text(data_dir,stats['num_items']), args.H, args.M)
        artifact['source_manifest'] = manifest['data_sha256']
        torch.save(artifact, candidate_path)
        write_json(root/'candidate_diagnostics.json', {'shape': list(artifact['candidate_ids'].shape),
            'isolated_items': int(artifact['neighbor_ids'].ne(0).sum(-1).eq(0).sum()),
            'singleton_levels': int(artifact['candidate_mask'].sum(-1).eq(1).sum()),
            'support_is_augmented_count_fraction': True, 'dense_N_by_N_matrix': False})
        print(json.dumps({'stage':'candidates','saved':str(candidate_path)}), flush=True)
    if args.stage == 'candidates':
        return
    if not candidate_path.exists():
        raise ValueError('Build semantic candidates first')
    artifact = torch.load(candidate_path, weights_only=True)
    train = NextItemDataset(rows,args.max_len,'train')
    valid = DataLoader(NextItemDataset(rows,args.max_len,'valid'),batch_size=args.batch_size,collate_fn=collate_eval)
    test = DataLoader(NextItemDataset(rows,args.max_len,'test'),batch_size=args.batch_size,collate_fn=collate_eval)
    if not len(train) or not len(valid.dataset) or not len(test.dataset):
        raise ValueError('Empty dataset split')
    for _,target,known in train.samples:
        if stats['num_items']-len({target,*known}) < args.negatives:
            raise ValueError('Insufficient eligible negatives')
    for seed in args.seeds:
        seed_root = root/f'seed{seed}'
        seed_root.mkdir(exist_ok=True)
        hard_path = seed_root/'stage1/hard_sid_checkpoint.pt'
        if args.stage in ('all','hard'):
            random.seed(seed); torch.manual_seed(seed)
            model = GCSS(hard,num_tokens,args.dim,args.max_len,args.heads,args.layers,args.dropout)
            fit(model,train,valid,test,config,seed,seed_root/'stage1',1,'hard')
            del model
        if args.stage == 'hard':
            continue
        if not hard_path.exists():
            raise ValueError('Train the pure Hard SID checkpoint before this stage')
        hard_sha = file_hash(hard_path)
        gradient_path = seed_root/'grad_signature.pt'
        if args.stage in ('all','profile') and not gradient_path.exists():
            model,_ = load_model(hard_path,args.device)
            debug = debug_profile(model,train,config,seed)
            write_json(seed_root/'profile_debug.json',debug)
            print(json.dumps({'seed':seed,'stage':'profile_debug','result':debug}),flush=True)
            result = profile(model,training_loader(train,config,seed+5001,shuffle=False),args.device)
            result['hard_checkpoint_sha256'] = hard_sha
            torch.save(result,gradient_path)
            write_json(seed_root/'profile_diagnostics.json',profile_stats(result))
            del model
        if args.stage == 'profile':
            continue
        if not gradient_path.exists():
            raise ValueError('Profile the fixed Hard checkpoint first')
        gradient = torch.load(gradient_path,weights_only=True)
        if gradient['hard_checkpoint_sha256'] != hard_sha:
            raise ValueError('Gradient signature source differs from Hard checkpoint')
        calibration_path = seed_root/'calibration.pt'
        if args.stage in ('all','calibrate') and not calibration_path.exists():
            comp = compatibility(artifact,gradient['grad_signature'])
            shuffled = shuffled_compatibility(comp,artifact['candidate_mask'],seed+6001)
            weights = {v: sharing_weights(artifact['candidate_support'],artifact['candidate_mask'],c,lam)
                       for v,c,lam in [('semantic_soft',comp,0.),('gradient_calibrated',comp,args.lambda_grad),
                                       ('random_compatibility',shuffled,args.lambda_grad)]}
            torch.save({'compatibility':comp,'random_compatibility':shuffled,'weights':weights,
                        'hard_checkpoint_sha256':hard_sha,'gradient_sha256':file_hash(gradient_path)},calibration_path)
            active = comp[artifact['candidate_mask']]
            write_json(seed_root/'calibration_diagnostics.json',{'compatibility_range':[float(active.min()),float(active.max())],
                'negative_compatibility_fraction':float(active.lt(0).float().mean()),
                'zero_compatibility_fraction':float(active.eq(0).float().mean()),
                'mean_abs_weight_change':float((weights['gradient_calibrated']-weights['semantic_soft']).abs().mean()),
                'lambda_grad':args.lambda_grad,'prototype_normalization':False,'self_in_gradient_group':False,
                'random_control':'within each item/level, permute valid compatibility slots only'})
        if args.stage == 'calibrate':
            continue
        if not calibration_path.exists():
            raise ValueError('Build fixed calibration weights first')
        calibration = torch.load(calibration_path,weights_only=True)
        if calibration['hard_checkpoint_sha256'] != hard_sha or calibration['gradient_sha256'] != file_hash(gradient_path):
            raise ValueError('Calibration provenance mismatch')
        for variant in ('hard','semantic_soft','gradient_calibrated','random_compatibility'):
            model,_ = load_model(hard_path,args.device)
            if variant != 'hard':
                model.item_encoder.set_lookup(pad_item(artifact['candidate_ids']),pad_item(calibration['weights'][variant]))
            fit(model,train,valid,test,config,seed,seed_root/'stage2'/variant,2,variant,
                source={'checkpoint':str(hard_path),'sha256':hard_sha})
            del model
            summarize(root)
    if args.stage in ('all','stage2'):
        expected = len(args.seeds)*4
        completed = len(list(root.glob('seed*/stage2/*/result.json')))
        write_json(root/'completion.json',{'expected':expected,'completed':completed,'complete':completed==expected})


if __name__ == '__main__':
    main()
