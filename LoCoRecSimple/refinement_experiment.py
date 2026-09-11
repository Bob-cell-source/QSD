"""Matched continuations from a shared random-start Hard SID warm-up."""
import argparse
from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from CCSR.trainer import NegativeSampler
from LoCoRec.locorec.data import NextItemDataset, collate_eval, collate_train
from LoCoRec.locorec.io import read_json, write_json
from LoCoRec.locorec.soft_sid import build_semantic_table, build_train_item_frequency
from .experiment import evaluate
from .model import SimpleSharing, initialize_matched
from .refinement import RefinementItemEncoder, apply_partitions, partition_score, propose_partitions


def train_loader(data, args, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    return DataLoader(data, batch_size=args.batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=partial(collate_train, sampler=NegativeSampler(args.num_items, args.negatives)))


def train_epoch(model, optimizer, data, args, seed):
    model.train()
    total, count = 0., 0
    for sequence, candidates in train_loader(data, args, seed):
        logits = model(sequence.to(args.device), candidates.to(args.device))['score'] / math.sqrt(args.dim)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(sequence), dtype=torch.long, device=args.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(sequence)
        count += len(sequence)
    return total / count


def collect_gradients(model, data, args, seed):
    model.eval()
    enc = model.item_encoder
    enc.captured = torch.zeros_like(enc.private_embedding.weight)
    count = 0
    for step, (sequence, candidates) in enumerate(train_loader(data, args, seed)):
        if step >= args.probe_batches:
            break
        model.zero_grad(set_to_none=True)
        logits = model(sequence.to(args.device), candidates.to(args.device))['score'] / math.sqrt(args.dim)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(sequence), dtype=torch.long, device=args.device), reduction='sum')
        loss.backward()
        count += len(sequence)
    if count == 0:
        raise ValueError('Empty gradient probe')
    result = enc.captured.detach().cpu() / count
    result[0].zero_()
    enc.captured = None
    model.zero_grad(set_to_none=True)
    return result, count


def run_seed(args, seed, rows, hard, tokens):
    root = Path(args.output_dir) / f'seed{seed}'
    if (root / 'results.json').exists():
        payload = read_json(root / 'results.json')
        if payload['args'] != vars(args):
            raise ValueError('Existing run has different settings')
        return payload
    root.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    model = SimpleSharing('hard', hard, tokens, None, None, args.dim, args.max_len, 2, 2, args.dropout)
    model.item_encoder = RefinementItemEncoder(hard, tokens, args.dim)
    initialize_matched(model, seed, args.dim, args.max_len, 2, 2, args.dropout, args.num_items, tokens)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train = NextItemDataset(rows, args.max_len, 'train')
    valid = DataLoader(NextItemDataset(rows, args.max_len, 'valid'), batch_size=args.batch_size, collate_fn=collate_eval)
    test = DataLoader(NextItemDataset(rows, args.max_len, 'test'), batch_size=args.batch_size, collate_fn=collate_eval)
    start = time.perf_counter()
    for epoch in range(1, args.warmup_epochs + 1):
        loss = train_epoch(model, optimizer, train, args, seed * 1000 + epoch)
        print(json.dumps({'seed': seed, 'stage': 'warmup', 'epoch': epoch, 'loss': loss}), flush=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'args': vars(args), 'seed': seed}, root / 'warmup.pt')
    # Split by complete user rows, not overlapping sequence prefixes. Both views
    # contain TRAIN prefixes only. The model has trained on both; view B is an
    # independent signal-estimation audit, not unseen-user generalization.
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(1707)).tolist()
    first = NextItemDataset([rows[i] for i in order[::2]], args.max_len, 'train')
    second = NextItemDataset([rows[i] for i in order[1::2]], args.max_len, 'train')
    grad_a, na = collect_gradients(model, first, args, seed * 1000 + 801)
    grad_b, nb = collect_gradients(model, second, args, seed * 1000 + 802)
    frequency = build_train_item_frequency(rows, args.num_items)
    plans = propose_partitions(hard, grad_a, frequency, args.split_budget, args.min_group, seed + 1707)
    audit = {}
    for variant, partitions in plans.items():
        audit[variant] = []
        for row in partitions:
            audit[variant].append({**row,
                'fit_score': partition_score(grad_a, grad_a, row['left'], row['right']),
                'audit_score': partition_score(grad_a, grad_b, row['left'], row['right'])})
    write_json(root / 'partitions.json', audit)
    torch.save({'a': grad_a, 'b': grad_b, 'examples_a': na, 'examples_b': nb}, root / 'gradient_audit.pt')
    print(json.dumps({'seed': seed, 'stage': 'audit', 'examples': [na, nb],
                      'splits': len(plans['gradient_split']),
                      'audit_mean': {v: sum(r['audit_score'] for r in rs) / max(len(rs), 1) for v, rs in audit.items()}}), flush=True)
    model.eval()
    with torch.no_grad():
        ids = torch.arange(args.num_items + 1, device=args.device)
        reference = model.item_encoder(ids)['vectors']
    results = []
    for variant, partitions in plans.items():
        dest = root / variant
        dest.mkdir(exist_ok=True)
        branch = deepcopy(model)
        opt = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt.load_state_dict(deepcopy(optimizer.state_dict()))
        apply_partitions(branch, opt, partitions)
        with torch.no_grad():
            error = float((branch.item_encoder(ids)['vectors'] - reference).abs().max())
        if error > 1e-6:
            raise ValueError(f'Splitting changed initial function: {error}')
        initial, _ = evaluate(branch, valid, args.device, args.num_items)
        best, best_epoch, stale = initial['NDCG@10'], args.warmup_epochs, 0
        history = [{'epoch': args.warmup_epochs, 'valid': initial}]
        def save_best():
            torch.save({'model': branch.state_dict(), 'args': vars(args), 'seed': seed,
                        'variant': variant, 'epoch': best_epoch}, dest / 'best.pt')
        save_best()
        for epoch in range(args.warmup_epochs + 1, args.epochs + 1):
            loss = train_epoch(branch, opt, train, args, seed * 1000 + epoch)
            metrics, _ = evaluate(branch, valid, args.device, args.num_items)
            row = {'epoch': epoch, 'loss': loss, 'valid': metrics}
            history.append(row)
            write_json(dest / 'history.json', history)
            print(json.dumps({'seed': seed, 'variant': variant, **row}), flush=True)
            if metrics['NDCG@10'] > best:
                best, best_epoch, stale = metrics['NDCG@10'], epoch, 0
                save_best()
            else:
                stale += 1
                if epoch >= args.warmup_epochs + args.min_post_epochs and stale >= args.patience:
                    break
        branch.load_state_dict(torch.load(dest / 'best.pt', map_location=args.device, weights_only=True)['model'])
        metrics, per_user = evaluate(branch, test, args.device, args.num_items)
        torch.save(per_user, dest / 'test_per_user.pt')
        result = {'variant': variant, 'seed': seed, 'test': metrics, 'best_epoch': best_epoch,
                  'best_valid_NDCG@10': best, 'splits': len(partitions), 'initial_vector_error': error,
                  'parameters': sum(p.numel() for p in branch.parameters()), 'args': vars(args)}
        write_json(dest / 'result.json', result)
        results.append(result)
        print(json.dumps({'completed': result}), flush=True)
    payload = {'args': vars(args), 'results': results, 'elapsed_seconds': time.perf_counter() - start}
    write_json(root / 'results.json', payload)
    return payload


def summarize(root):
    paths = sorted(Path(root).glob('seed*/*/result.json'))
    rows = [read_json(p) for p in paths]
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row['variant'], []).append(row)
    summary = {}
    for variant, group in by_variant.items():
        values = torch.tensor([r['test']['NDCG@10'] for r in group], dtype=torch.double)
        summary[variant] = {'seeds': [r['seed'] for r in group], 'mean': float(values.mean()),
                            'std': float(values.std()) if len(values) > 1 else None,
                            'parameters': group[0]['parameters']}
    write_json(Path(root) / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default='runs/office')
    p.add_argument('--semantic-ids', default='runs/office/semantic_ids_rq.json')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=[2026])
    for key, value in [('dim', 128), ('max-len', 50), ('batch-size', 256), ('negatives', 100),
                       ('epochs', 30), ('warmup-epochs', 3), ('min-post-epochs', 10), ('patience', 8),
                       ('probe-batches', 64), ('split-budget', 32), ('min-group', 8), ('threads', 2)]:
        p.add_argument('--' + key, type=int, default=value)
    for key, value in [('lr', .001), ('weight-decay', .0001), ('dropout', .2)]:
        p.add_argument('--' + key, type=float, default=value)
    args = p.parse_args()
    if args.epochs <= args.warmup_epochs or args.min_group < 2 or args.split_budget < 1 or args.probe_batches < 1:
        raise ValueError('Invalid refinement settings')
    torch.set_num_threads(args.threads)
    args.num_items = int(read_json(Path(args.dataset_dir) / 'stats.json')['num_items'])
    rows = read_json(Path(args.dataset_dir) / 'sequences.json')
    hard, _, tokens = build_semantic_table(read_json(args.semantic_ids), args.num_items)
    data = NextItemDataset(rows, args.max_len, 'train')
    for _, target, known in data.samples:
        if args.num_items - len({target, *known}) < args.negatives:
            raise ValueError('Insufficient eligible negatives')
    write_json(Path(args.output_dir) / 'protocol.json', {'args': vars(args),
        'note': 'All arms branch from the same RANDOM-START Hard SID warm-up and optimizer. No previous experiment checkpoint. No Soft SID, gate, frequency prior, or auxiliary loss. Frequency is used only by the frequency-selection control. View B never selects splits. Each split adds exactly one shared row; same-parent random control matches group sizes.'})
    for seed in args.seeds:
        run_seed(args, seed, rows, hard, tokens)
        print(summarize(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
