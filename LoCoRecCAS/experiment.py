"""Static warm-up followed by controlled frozen-backbone or joint CAS training."""
import argparse
import copy
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from CCSR.trainer import NegativeSampler
from LoCoRec.locorec.data import NextItemDataset, collate_eval, collate_train
from LoCoRec.locorec.io import read_json, write_json
from LoCoRec.locorec.soft_sid import build_semantic_table
from .model import LoCoRecCAS, FrozenCatalog, build_scope, transfer_locorec


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_of(model, args):
    return {key: getattr(args, key) for key in ('dim', 'max_len', 'num_heads', 'num_layers', 'dropout', 'gate_init')}


def save_model(path, model, args, epoch, metric, gate_mode='static'):
    torch.save({'model': model.state_dict(), 'args': vars(args), 'config': config_of(model, args),
                'epoch': epoch, 'valid_NDCG@10': metric, 'gate_mode': gate_mode}, path)


def build_data(args):
    sequences = read_json(Path(args.dataset_dir) / 'sequences.json')
    num_items = int(read_json(Path(args.dataset_dir) / 'stats.json')['num_items'])
    if any(not 1 <= int(i) <= num_items for row in sequences for i in row['items']):
        raise ValueError('Sequence item IDs outside the catalog.')
    semantic = read_json(args.semantic_ids)
    for codes in semantic['semantic_ids'].values():
        if len(codes) != len(semantic['codebook_sizes']) or any(not 0 <= int(c) < int(s) for c, s in zip(codes, semantic['codebook_sizes'])):
            raise ValueError('Invalid SID codebook values.')
    table, _, tokens = build_semantic_table(semantic, num_items)
    datasets = {split: NextItemDataset(sequences, args.max_len, split) for split in ('train', 'valid', 'test')}
    if any(len(data) == 0 for data in datasets.values()):
        raise ValueError('All data splits must be nonempty.')
    for split, data in datasets.items():
        for _, target, known in data.samples:
            if num_items - len({target, *known}) < args.negatives:
                raise ValueError(f'Too few unique negatives in {split}.')
    return datasets, table, tokens


@torch.no_grad()
def contexts(model, dataset, args):
    model.eval()
    encoder = model.item_encoder
    # Scope and item representations are evaluated once for this checkpoint.
    all_vectors = []
    for start in range(0, len(encoder.hard_sid_table), args.item_chunk):
        items = torch.arange(start, min(start + args.item_chunk, len(encoder.hard_sid_table)), device=args.device)
        all_vectors.append(encoder.static(items))
    all_vectors = torch.cat(all_vectors)
    rows = []
    for sequence, _, _ in DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_eval):
        sequence = sequence.to(args.device)
        rows.append(model.sequence_encoder(sequence, all_vectors[sequence]).detach())
    return torch.cat(rows)


def rank_metrics(scores, target, cutoffs=(5, 10, 20)):
    order = scores.topk(min(max(cutoffs), scores.size(-1)), -1).indices + 1
    result = {}
    per_user = {}
    for k in cutoffs:
        matches = order[:, :k].eq(target[:, None])
        hits = matches.any(-1).float()
        ranks = matches.float().argmax(-1) + 1
        ndcg = hits / torch.log2(ranks.float() + 1)
        result[f'HR@{k}'] = hits.sum().item()
        result[f'NDCG@{k}'] = ndcg.sum().item()
        per_user[f'NDCG@{k}'] = ndcg.cpu()
    return result, per_user


@torch.no_grad()
def evaluate_gates(catalog, h, dataset, gates_by_name, args):
    """All methods share the same cached contexts and catalog. Full seen masking."""
    n = len(catalog.centered) - 1
    totals = {name: {} for name in gates_by_name}
    per_user = {name: [] for name in gates_by_name}
    for start in range(0, len(h), args.eval_batch_size):
        user = h[start:start + args.eval_batch_size]
        rows = dataset.samples[start:start + len(user)]
        targets = torch.tensor([row[1] for row in rows], device=args.device)
        scores = {name: [] for name in gates_by_name}
        for cstart in range(1, n + 1, args.candidate_chunk):
            items = torch.arange(cstart, min(cstart + args.candidate_chunk, n + 1), device=args.device)
            dot = catalog.features(user, items)
            for name, gates in gates_by_name.items():
                scores[name].append(catalog.score(dot, items, gates[start:start + len(user)]))
        columns, batch_rows = [], []
        for index, (_, target, history) in enumerate(rows):
            seen = set(history) - {target, 0}
            columns.extend(i - 1 for i in seen)
            batch_rows.extend([index] * len(seen))
        for name in gates_by_name:
            full = torch.cat(scores[name], -1)
            full[batch_rows, columns] = -torch.inf
            metrics, user_metrics = rank_metrics(full, targets)
            for key, value in metrics.items():
                totals[name][key] = totals[name].get(key, 0.) + value
            per_user[name].append(user_metrics['NDCG@10'])
    for name in totals:
        totals[name] = {k: v / len(h) for k, v in totals[name].items()}
        totals[name].update({f'Recall@{k}': totals[name][f'HR@{k}'] for k in (5, 10, 20)})
        gates = gates_by_name[name]
        totals[name]['gate_mean'] = gates.mean(0).cpu().tolist()
        totals[name]['gate_std'] = gates.std(0, unbiased=False).cpu().tolist()
    return totals, {name: torch.cat(values) for name, values in per_user.items()}


@torch.no_grad()
def make_frozen_training(catalog, h, dataset, args, seed):
    random.seed(seed)
    sampler = NegativeSampler(len(catalog.centered) - 1, args.negatives)
    candidates = torch.tensor([sampler.sample(target, known) for _, target, known in dataset.samples], device=args.device)
    features, deltas = [], []
    for start in range(0, len(h), args.batch_size):
        user, items = h[start:start + args.batch_size], candidates[start:start + args.batch_size]
        dot = catalog.features(user, items)
        all_gates = user.new_ones(len(user), catalog.centered.size(1) - 1)
        all_loss = -catalog.score(dot, items, all_gates).log_softmax(-1)[:, 0]
        delta = []
        for level in range(all_gates.size(-1)):
            gate = all_gates.clone()
            gate[:, level] = 0
            loss = -catalog.score(dot, items, gate).log_softmax(-1)[:, 0]
            delta.append(loss - all_loss)
        features.append(dot)
        deltas.append(torch.stack(delta, -1))
    return {'h': h, 'items': candidates, 'dot': torch.cat(features), 'delta': torch.cat(deltas)}


def utility_metrics(gates, delta, temperature, train_target_mean):
    target = (delta / temperature).sigmoid()
    centered_target = target - target.mean(0)
    centered_gates = gates - gates.mean(0)
    denominator = (centered_gates.square().sum(0) * centered_target.square().sum(0)).sqrt()
    correlations = (centered_gates * centered_target).sum(0) / denominator.clamp_min(1e-15)
    return {'target_mean': target.mean(0).cpu().tolist(), 'target_std': target.std(0, unbiased=False).cpu().tolist(),
            'delta_mean': delta.mean(0).cpu().tolist(), 'delta_std': delta.std(0, unbiased=False).cpu().tolist(),
            'positive_delta_fraction': delta.gt(0).float().mean(0).cpu().tolist(),
            'target_MSE': float((gates - target).square().mean()),
            'constant_train_mean_MSE': float((train_target_mean - target).square().mean()),
            'correlation_per_level': [float(correlations[i]) if denominator[i] > 1e-8 else None for i in range(len(correlations))]}


def run_frozen(model, datasets, args, output):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    catalog = FrozenCatalog(model.item_encoder, args.item_chunk)
    cache = {}
    for index, split in enumerate(('train', 'valid', 'test')):
        print(f'Caching {split} contexts and utility targets ({len(datasets[split])} instances)', flush=True)
        h = contexts(model, datasets[split], args)
        cache[split] = make_frozen_training(catalog, h, datasets[split], args, args.seed + index)
    train = cache['train']
    target = (train['delta'] / args.temperature).sigmoid()
    train_mean = target.mean(0)
    write_json(output / 'utility_targets.json', {split: utility_metrics((values['delta'] / args.temperature).sigmoid(),
               values['delta'], args.temperature, train_mean) for split, values in cache.items()})
    states, records = {}, []
    # Paired initialization/order/candidates for each control.
    for name, mode, weight in [('global', 'global', 0.), ('context_rec', 'context', 0.),
                                ('context_utility', 'context', args.lambda_util)]:
        seed_all(args.seed)
        head = copy.deepcopy(model.context_head if mode == 'context' else torch.nn.ParameterList([model.global_logits]))
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(head.parameters(), lr=args.gate_lr)
        best, best_state, best_epoch, stale = -math.inf, None, 0, 0
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(train['h']), device=args.device)
            loss_sum, rec_sum, util_sum = 0., 0., 0.
            for start in range(0, len(order), args.batch_size):
                idx = order[start:start + args.batch_size]
                logits = head(train['h'][idx]) if mode == 'context' else head[0].expand(len(idx), -1)
                gates = logits.sigmoid()
                rec = -catalog.score(train['dot'][idx], train['items'][idx], gates).log_softmax(-1)[:, 0].mean()
                util = F.binary_cross_entropy_with_logits(logits, target[idx])
                loss = rec + weight * util
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.item() * len(idx)
                rec_sum += rec.item() * len(idx)
                util_sum += util.item() * len(idx)
            with torch.no_grad():
                vh = cache['valid']['h']
                vg = (head(vh) if mode == 'context' else head[0].expand(len(vh), -1)).sigmoid()
                valid, _ = evaluate_gates(catalog, vh, datasets['valid'], {name: vg}, args)
            metric = valid[name]['NDCG@10']
            record = {'variant': name, 'epoch': epoch, 'loss': loss_sum / len(order),
                      'loss_rec': rec_sum / len(order), 'loss_util': util_sum / len(order),
                      'valid': valid[name]}
            records.append(record)
            write_json(output / 'gate_history.json', records)
            print(record, flush=True)
            if metric > best:
                best, best_epoch, stale = metric, epoch, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
                if stale >= args.patience:
                    break
        head.load_state_dict(best_state)
        states[name] = {'head': head, 'mode': mode, 'epoch': best_epoch, 'valid_NDCG@10': best}
        torch.save({'head': best_state, 'mode': mode, 'epoch': best_epoch, 'args': vars(args),
                    'backbone': str(output / 'static_best.pt')}, output / f'{name}_head.pt')
    results = {'protocol': 'full-data frozen-backbone probe; fixed sampled candidates per instance',
               'args': vars(args), 'best_epochs': {name: values['epoch'] for name, values in states.items()}}
    for split in ('valid', 'test'):
        h = cache[split]['h']
        gates = {name: model.gates(h, name) for name in ('static', 'none', 'half')}
        with torch.no_grad():
            for name, values in states.items():
                head = values['head']
                gates[name] = (head(h) if values['mode'] == 'context' else head[0].expand(len(h), -1)).sigmoid()
            learned = gates['context_utility']
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            gates['utility_shuffled'] = learned[torch.randperm(len(h), generator=generator, device=args.device)]
            gates['utility_mean'] = learned.mean(0).expand(len(h), -1)
            metrics, per_user = evaluate_gates(catalog, h, datasets[split], gates, args)
            utility = {name: utility_metrics(gate, cache[split]['delta'], args.temperature, train_mean)
                       for name, gate in gates.items()}
        comparisons = {}
        generator = torch.Generator().manual_seed(args.seed)
        for baseline in ('static', 'half', 'global', 'context_rec', 'utility_shuffled'):
            difference = per_user['context_utility'] - per_user[baseline]
            samples = torch.stack([difference[torch.randint(len(difference), (len(difference),), generator=generator)].mean()
                                   for _ in range(1000)])
            comparisons[baseline] = {'delta_NDCG@10': float(difference.mean()),
                                     'paired_bootstrap_95ci': torch.quantile(samples, torch.tensor([.025, .975])).tolist()}
        results[split] = {'ranking': metrics, 'utility_prediction': utility, 'paired_comparisons': comparisons}
        torch.save(per_user, output / f'{split}_per_user_ndcg10.pt')
    write_json(output / 'results.json', results)
    print({'results': str(output / 'results.json'), 'test': results['test']['ranking']}, flush=True)
    return results


def run_joint(model, datasets, args, output):
    mode = args.joint_mode
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not (name.startswith('context_head.') and mode != 'context')
                                 and not (name == 'global_logits' and mode != 'global'))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sampler = NegativeSampler(len(model.item_encoder.hard_sid_table) - 1, args.negatives)
    loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                        collate_fn=partial(collate_train, sampler=sampler))
    best, stale, history = -math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for sequence, candidates in loader:
            result = model(sequence.to(args.device), candidates.to(args.device), mode, args.lambda_util, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += result['loss'].item() * len(sequence)
            count += len(sequence)
        h = contexts(model, datasets['valid'], args)
        catalog = FrozenCatalog(model.item_encoder, args.item_chunk)
        with torch.no_grad():
            valid, _ = evaluate_gates(catalog, h, datasets['valid'], {mode: model.gates(h, mode)}, args)
        metric = valid[mode]['NDCG@10']
        history.append({'epoch': epoch, 'loss': total / count, 'valid': valid[mode]})
        write_json(output / 'joint_history.json', history)
        print(history[-1], flush=True)
        if metric > best:
            best, stale = metric, 0
            save_model(output / 'joint_best.pt', model, args, epoch, metric, mode)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(output / 'joint_best.pt', map_location=args.device, weights_only=True)['model'])
    h = contexts(model, datasets['test'], args)
    with torch.no_grad():
        metrics, _ = evaluate_gates(FrozenCatalog(model.item_encoder, args.item_chunk), h, datasets['test'],
                                   {mode: model.gates(h, mode)}, args)
    result = {'protocol': 'joint training', 'args': vars(args), 'best_valid_NDCG@10': best, 'test': metrics}
    write_json(output / 'results.json', result)
    return result


def run(args):
    for name in ('epochs', 'batch_size', 'eval_batch_size', 'item_chunk', 'candidate_chunk', 'negatives', 'patience', 'threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if args.warmup_epochs < 1 and not args.static_checkpoint:
        raise ValueError('Need static warm-up or --static-checkpoint.')
    if args.temperature <= 0 or args.lambda_util < 0 or args.lr <= 0 or args.gate_lr <= 0 or args.grad_clip <= 0:
        raise ValueError('Invalid learning rate, utility settings or clipping threshold.')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise ValueError('Output already contains results; choose a new directory.')
    datasets, table, num_tokens = build_data(args)
    tokens, prior, info = build_scope(table, args.max_neighbors, args.min_overlap, args.seed)
    write_json(output / 'scope.json', info)
    model = LoCoRecCAS(table, num_tokens, tokens, prior, args.dim, args.max_len,
                      args.num_heads, args.num_layers, args.dropout, args.gate_init).to(args.device)
    if args.static_checkpoint:
        checkpoint = torch.load(args.static_checkpoint, map_location=args.device, weights_only=True)
        if checkpoint['config'] != config_of(model, args):
            raise ValueError('Static checkpoint model configuration differs.')
        if not torch.equal(checkpoint['model']['item_encoder.hard_sid_table'].cpu(), table):
            raise ValueError('Static checkpoint SID mapping differs.')
        if not torch.equal(checkpoint['model']['item_encoder.scope_tokens'].cpu(), tokens) or not torch.equal(checkpoint['model']['item_encoder.candidate_prior'].cpu(), prior):
            raise ValueError('Static checkpoint scope differs.')
        model.load_state_dict(checkpoint['model'])
        save_model(output / 'static_best.pt', model, args, checkpoint['epoch'], checkpoint['valid_NDCG@10'])
    else:
        if args.init_checkpoint:
            checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
            source = checkpoint['args']
            for key in ('dim', 'max_len', 'num_heads', 'num_layers'):
                if source[key] != getattr(args, key):
                    raise ValueError(f'Source checkpoint {key} differs.')
            source_table, _, _ = build_semantic_table(read_json(source['semantic_ids']), len(table) - 1)
            report = transfer_locorec(model, checkpoint, source_table)
            write_json(output / 'initialization.json', report)
        elif not args.from_scratch:
            raise ValueError('Specify --init-checkpoint, --static-checkpoint or --from-scratch.')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        sampler = NegativeSampler(len(table) - 1, args.negatives)
        loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                            collate_fn=partial(collate_train, sampler=sampler))
        best, history = -math.inf, []
        for epoch in range(1, args.warmup_epochs + 1):
            model.train()
            start_time, total, count = time.perf_counter(), 0., 0
            for sequence, candidates in loader:
                result = model(sequence.to(args.device), candidates.to(args.device), 'static', 0, args.temperature)
                optimizer.zero_grad(set_to_none=True)
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                total += result['loss'].item() * len(sequence)
                count += len(sequence)
            h = contexts(model, datasets['valid'], args)
            valid, _ = evaluate_gates(FrozenCatalog(model.item_encoder, args.item_chunk), h, datasets['valid'],
                                      {'static': model.gates(h, 'static')}, args)
            metric = valid['static']['NDCG@10']
            history.append({'epoch': epoch, 'loss': total / count, 'seconds': time.perf_counter() - start_time,
                            'valid': valid['static']})
            write_json(output / 'warmup_history.json', history)
            print({'stage': 'static_warmup', **history[-1]}, flush=True)
            if metric > best:
                best = metric
                save_model(output / 'static_best.pt', model, args, epoch, metric)
        model.load_state_dict(torch.load(output / 'static_best.pt', map_location=args.device, weights_only=True)['model'])
    write_json(output / 'protocol.json', {'args': vars(args), 'split_sizes': {k: len(v) for k, v in datasets.items()},
               'selection': 'validation NDCG@10 only; test never used for checkpoint selection',
               'utility_reference': 'same context, same candidates, remove one shared residual; keep 1/L',
               'normalization': 'exact candidate-dependent LayerNorm; no additive approximation'})
    return run_frozen(model, datasets, args, output) if args.protocol == 'frozen' else run_joint(model, datasets, args, output)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', default='runs/office')
    p.add_argument('--semantic-ids', default='runs/office/semantic_ids_rq.json')
    p.add_argument('--output-dir', required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint')
    source.add_argument('--static-checkpoint')
    source.add_argument('--from-scratch', action='store_true')
    p.add_argument('--protocol', choices=('frozen', 'joint'), default='frozen')
    p.add_argument('--joint-mode', choices=('static', 'global', 'context'), default='context')
    p.add_argument('--device', default='cuda')
    for name, default in [('warmup-epochs', 3), ('epochs', 10), ('patience', 4), ('batch-size', 256),
                          ('eval-batch-size', 128), ('item-chunk', 128), ('candidate-chunk', 512),
                          ('negatives', 100), ('dim', 128), ('max-len', 50), ('num-heads', 2),
                          ('num-layers', 2), ('max-neighbors', 50), ('min-overlap', 3), ('seed', 2026), ('threads', 2)]:
        p.add_argument(f'--{name}', type=int, default=default)
    for name, default in [('lr', 1e-4), ('gate-lr', 1e-3), ('dropout', .2), ('lambda-util', 1.),
                          ('temperature', .1), ('gate-init', .9), ('grad-clip', 5.)]:
        p.add_argument(f'--{name}', type=float, default=default)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
