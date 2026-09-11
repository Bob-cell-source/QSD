import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from CCSR.trainer import NegativeSampler
from LoCoRec.locorec.data import NextItemDataset, collate_train, collate_eval
from LoCoRec.locorec.io import read_json, write_json
from LoCoRec.locorec.soft_sid import build_semantic_table, build_soft_sid_table, build_train_item_frequency, SoftSIDConfig
from LoCoRec.locorec.model import LoCoRec
from .model import SimpleSharing, initialize_matched, configure_fusion_control
from .transfer import SIDGradientTransfer


@torch.no_grad()
def evaluate(model, loader, device, num_items, chunk=256, mask_seen_items=True):
    model.eval()
    vectors = torch.cat([model.item_encoder(torch.arange(s, min(s+chunk, num_items+1), device=device))['vectors']
                         for s in range(1, num_items+1, chunk)])
    totals = {f'{metric}@{k}': 0. for k in (5,10,20) for metric in ('NDCG','HR','MRR')}
    users, total = [], 0
    for sequences, targets, histories in loader:
        h, _ = model.encode_sequence(sequences.to(device))
        scores = h @ vectors.T
        for row, (history, target) in enumerate(zip(histories, targets.tolist())):
            seen = list(set(history) - {target, 0})
            if mask_seen_items and seen:
                scores[row, torch.tensor(seen, device=device)-1] = -torch.inf
        order = scores.topk(min(20, num_items), dim=-1).indices + 1
        for k in (5,10,20):
            matches = order[:, :k].eq(targets.to(device)[:,None])
            hit = matches.any(-1).float()
            rank = matches.float().argmax(-1) + 1
            ndcg = hit / torch.log2(rank.float()+1)
            mrr = hit / rank.float()
            totals[f'HR@{k}'] += hit.sum().item()
            totals[f'NDCG@{k}'] += ndcg.sum().item()
            totals[f'MRR@{k}'] += mrr.sum().item()
            if k == 10:
                users.append(ndcg.cpu())
        total += len(sequences)
    if not total:
        raise ValueError('Empty evaluation split')
    return {k:v/total for k,v in totals.items()}, torch.cat(users)


def train_one(args, variant, seed, datasets, hard, num_tokens, soft, prior, reliability, frequency):
    output = Path(args.output_dir)/variant/f'seed{seed}'
    if (output/'result.json').exists():
        existing = read_json(output/'result.json')
        if existing['args'] != vars(args):
            raise ValueError(f'Existing run has different arguments: {output}')
        return existing
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    architecture = variant.removesuffix('_scaled')
    kw = dict(dim=args.dim, max_len=args.max_len, num_heads=2, num_layers=2, dropout=args.dropout)
    is_full = architecture.startswith('full')
    is_consolidated = architecture.startswith('consolidated')
    if is_full or is_consolidated:
        model = LoCoRec(num_items=len(hard)-1, num_semantic_tokens=num_tokens, soft_sid_table=soft,
                       candidate_prior=prior, local_consistency=reliability, item_frequency=frequency, **kw)
        # Apply SRA-CL-style embedding dropout to history inputs only; target
        # and catalog item representations remain deterministic during scoring.
        model.item_encoder.dropout = torch.nn.Identity()
        model.embedding_dropout = torch.nn.Dropout(args.embedding_dropout)
    else:
        if architecture in ('hard_shuffled', 'id_transfer_shuffled'):
            # Fixed catalog permutation independent of training seeds and labels.
            permutation = torch.randperm(len(hard)-1, generator=torch.Generator().manual_seed(1701))+1
            control_hard = torch.cat([hard[:1], hard[permutation]], dim=0)
            model = SimpleSharing('hard' if architecture == 'hard_shuffled' else 'id', control_hard, num_tokens, soft, prior, embedding_dropout=args.embedding_dropout, **kw)
        elif architecture == 'id_transfer':
            model = SimpleSharing('id', hard, num_tokens, soft, prior, embedding_dropout=args.embedding_dropout, **kw)
        else:
            model = SimpleSharing(architecture, hard, num_tokens, soft, prior, embedding_dropout=args.embedding_dropout, **kw)
    initialize_matched(model, seed, args.dim, args.max_len, 2, 2, args.dropout, len(hard)-1, num_tokens)
    if is_consolidated:
        from .compile_fixed import compile_model, configure_training_ablation
        # Transform RANDOM parameters before any training; no checkpoint is loaded.
        model = configure_training_ablation(compile_model(model, vars(args)), architecture)
    if is_full:
        configure_fusion_control(model, architecture)
    params = sum(p.numel() for p in model.parameters())
    model.to(args.device)
    if architecture in ('id_transfer', 'id_transfer_shuffled'):
        model.item_encoder.private_embedding.weight.register_hook(
            SIDGradientTransfer(model.item_encoder.hard_sid_table, strength=.5))
    gate_params = list(model.item_encoder.residual_gate.parameters()) if is_full else []
    gate_ids = {id(p) for p in gate_params}
    optimizer_cls = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.AdamW
    optimizer = optimizer_cls([{'params':[p for p in model.parameters() if id(p) not in gate_ids], 'lr':args.lr},
                               {'params':gate_params, 'lr':args.lr*.1}], weight_decay=args.weight_decay) if args.optimizer == 'adamw' else optimizer_cls([{'params':[p for p in model.parameters() if id(p) not in gate_ids], 'lr':args.lr},
                               {'params':gate_params, 'lr':args.lr*.1}])
    sampler = NegativeSampler(len(hard)-1, args.negatives)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, generator=generator,
                        collate_fn=partial(collate_train, sampler=sampler))
    valid_loader = DataLoader(datasets['valid'], batch_size=args.eval_batch_size, collate_fn=collate_eval)
    test_loader = DataLoader(datasets['test'], batch_size=args.eval_batch_size, collate_fn=collate_eval)
    best, best_epoch, stale, history = -1., 0, 0, []
    began = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        # Candidate draws and training dropout streams are reset per epoch for matched controls.
        random.seed(seed*1000+epoch)
        torch.manual_seed(seed*1000+epoch)
        model.train()
        for p in gate_params:
            p.requires_grad_(architecture not in ('full_fixed', 'full_equal') and epoch > args.gate_warmup)
        total, count = 0., 0
        for sequence, candidates in loader:
            sequence = sequence.to(args.device)
            if args.train_objective == 'full_softmax':
                targets = candidates[:, 0].to(args.device)
                h, _ = model.encode_sequence(sequence)
                target_output = model.item_encoder(targets)
                target_vectors = target_output['vectors']
                target_logits = (h * target_vectors).sum(-1)
                log_partition = None
                all_items = torch.arange(1, len(hard), device=args.device)
                for start in range(0, len(all_items), args.full_softmax_chunk_size):
                    item_chunk = all_items[start:start + args.full_softmax_chunk_size]
                    chunk_vectors = model.item_encoder(item_chunk)['vectors']
                    chunk_logits = h @ chunk_vectors.T
                    if variant.endswith('_scaled'):
                        chunk_logits = chunk_logits / math.sqrt(args.dim)
                    chunk_lse = torch.logsumexp(chunk_logits, dim=-1)
                    log_partition = chunk_lse if log_partition is None else torch.logaddexp(log_partition, chunk_lse)
                if variant.endswith('_scaled'):
                    target_logits = target_logits / math.sqrt(args.dim)
                ce = (log_partition - target_logits).mean()
                # Preserve LoCoRec's auxiliary losses while evaluating the
                # full catalog in chunks.
                if architecture == 'full':
                    sequence_output = model.item_encoder(sequence)
                    result = {
                        'gate_kl': 0.5 * (sequence_output['gate_kl'] + target_output['gate_kl']),
                        'private_penalty': 0.5 * (sequence_output['private_penalty'] + target_output['private_penalty']),
                    }
                else:
                    result = {'gate_kl': ce.new_zeros(()), 'private_penalty': ce.new_zeros(())}
            else:
                result = model(sequence, candidates.to(args.device))
                logits = result['score'] / math.sqrt(args.dim) if variant.endswith('_scaled') else result['score']
                ce = torch.nn.functional.cross_entropy(logits, torch.zeros(len(sequence), device=args.device, dtype=torch.long))
            loss = ce
            if architecture == 'full':
                loss = loss + args.gate_kl_weight*result['gate_kl'] + args.private_weight*result['private_penalty']
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(sequence)
            count += len(sequence)
        metrics,_ = evaluate(model,valid_loader,args.device,len(hard)-1, mask_seen_items=not args.keep_seen_items)
        row={'epoch':epoch,'loss':total/count,'valid':metrics,'elapsed_seconds':time.perf_counter()-began}
        history.append(row)
        write_json(output/'history.json',history)
        print(json.dumps({'variant':variant,'seed':seed,**row}),flush=True)
        monitor = metrics[args.early_stop_metric]
        if monitor > best:
            best, best_epoch, stale = monitor,epoch,0
            torch.save({'model':model.state_dict(),'args':vars(args),'variant':variant,'seed':seed,'epoch':epoch},output/'best.pt')
        else:
            stale += 1
            # Equal minimum training duration includes original full-gate warm-up.
            if epoch >= args.gate_warmup+args.patience and stale >= args.patience:
                break
    model.load_state_dict(torch.load(output/'best.pt',map_location=args.device,weights_only=True)['model'])
    metrics, per_user = evaluate(model,test_loader,args.device,len(hard)-1, mask_seen_items=not args.keep_seen_items)
    torch.save(per_user,output/'test_per_user.pt')
    result={'variant':variant,'seed':seed,'best_epoch':best_epoch,'best_valid_metric':args.early_stop_metric,'best_valid_value':best,
            'test':metrics,'parameters':params,'elapsed_seconds':time.perf_counter()-began,'args':vars(args)}
    write_json(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-dir',default='runs/office')
    p.add_argument('--semantic-ids',default='runs/office/semantic_ids_rq.json')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--protocol',choices=['local','sracl'],default='local')
    p.add_argument('--variants',nargs='+',choices=['id','hard','soft','hard_shuffled','id_transfer','id_transfer_shuffled','full','full_fixed','full_global','full_equal','id_scaled','hard_scaled','soft_scaled','full_scaled','consolidated_scaled','consolidated_hard_scaled','consolidated_no_bias_scaled','consolidated_uniform_scaled'],default=['id','hard','soft','full'])
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    for name,default in [('epochs',100),('patience',10),('gate-warmup',10),('batch-size',256),('eval-batch-size',256),('dim',128),('max-len',50),('negatives',100),('threads',2)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('dropout',.2),('lr',.001),('weight-decay',.0001),('gate-kl-weight',.05),('private-weight',.1)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--optimizer',choices=['adam','adamw'],default='adamw')
    p.add_argument('--train-objective',choices=['sampled','full_softmax'],default='sampled')
    p.add_argument('--early-stop-metric',choices=['NDCG@10','MRR@10'],default='NDCG@10')
    p.add_argument('--keep-seen-items',action='store_true')
    p.add_argument('--embedding-dropout',type=float,default=0.0)
    p.add_argument('--full-softmax-chunk-size',type=int,default=512)
    args=p.parse_args()
    if args.protocol == 'sracl':
        args.max_len = 20
        args.dim, args.dropout = 64, 0.5
        args.weight_decay = 0.0
        args.optimizer = 'adam'
        args.train_objective = 'full_softmax'
        args.early_stop_metric = 'MRR@10'
        args.keep_seen_items = True
        args.embedding_dropout = 0.5
        args.negatives = 0
    torch.set_num_threads(args.threads)
    if any(getattr(args,k)<1 for k in ['epochs','patience','batch_size','eval_batch_size','dim','max_len']):
        raise ValueError('Positive sizes required')
    if args.negatives < 0:
        raise ValueError('Number of negatives cannot be negative')
    rows=read_json(Path(args.dataset_dir)/'sequences.json')
    n=int(read_json(Path(args.dataset_dir)/'stats.json')['num_items'])
    hard, codes, tokens=build_semantic_table(read_json(args.semantic_ids),n)
    # Same original local scope for simple Soft and Full; fixed preprocessing seed.
    soft,prior,reliability=build_soft_sid_table(hard,codes,SoftSIDConfig(top_m=4,min_overlap_slots=3,leave_one_level_out=False,tie_break_seed=2026))
    frequency=build_train_item_frequency(rows,n)
    data={s:NextItemDataset(rows,args.max_len,s) for s in ['train','valid','test']}
    for ds in data.values():
        if not len(ds): raise ValueError('Empty split')
        for _,target,known in ds.samples:
            if n-len({target,*known})<args.negatives: raise ValueError('Insufficient negatives')
    write_json(Path(args.output_dir)/'protocol.json',{'args':vars(args),'sizes':{s:len(ds) for s,ds in data.items()},
      'notes':['all from scratch; same encoder/private initialization','same sampled CE, unscaled dot-product, full catalog masking',
               'no item-side dropout for any variant; common encoder dropout','full retains original hierarchical gate and auxiliary losses',
               'full_fixed freezes original prior gate; full_global uses two global bounded corrections and mean alpha; full_equal uses weights [1,1,1]; these controls use CE only',
               'hard_shuffled permutes nonpadding SID rows using fixed preprocessing seed 1701; preserves tuple multiset and all occupancy counts',
               'id_transfer uses fixed eta=.5, diagonal-preserving SID group-mean gradient hook before global clipping and AdamW; shuffled version uses permutation seed 1701',
               '*_scaled variants divide training logits by sqrt(dim); evaluation rank is invariant to this positive constant',
               'consolidated_* starts from algebraically transformed RANDOM Full parameters, before any training; no checkpoint loading; CE only, fixed private coefficients, one shared table and common bias',
               'consolidated ablations retain surviving initial tensors: hard uses singleton candidates; no_bias zeros/freezes common bias; uniform replaces positive private coefficients by their mean, retaining the cold-item guard',
               'Soft and Full use same original all-level-overlap scope; fixed preprocessing seed 2026']})
    results=[]
    for seed in args.seeds:
        for variant in args.variants:
            results.append(train_one(args,variant,seed,data,hard,tokens,soft,prior,reliability,frequency))
            write_json(Path(args.output_dir)/'results.json',results)
    summary={}
    for variant in args.variants:
        runs=[r for r in results if r['variant']==variant]
        vals=torch.tensor([r['test']['NDCG@10'] for r in runs],dtype=torch.double)
        summary[variant]={'NDCG@10_mean':vals.mean().item(),'NDCG@10_std':vals.std().item() if len(vals)>1 else None,
                          'seeds':[r['seed'] for r in runs],'parameters':runs[0]['parameters']}
    write_json(Path(args.output_dir)/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    main()
