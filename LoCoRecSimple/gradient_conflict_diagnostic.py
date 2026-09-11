"""Read-only trained-checkpoint audit of target-conditioned SID gradients.

L_i is mean sampled recommendation CE over TRAIN examples with next item i.
The full gradient includes history and negative-candidate paths. A separate
positive-path derivative isolates the direct positive-item lookup contribution.
Neither cosine sign nor a cross-sectional correlation proves negative transfer.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
import torch

from CCSR.trainer import NegativeSampler
from LoCoRec.locorec.data import NextItemDataset
from .compile_fixed import empty_compiled
from .model import SimpleSharing


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_hard(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state, args = payload['model'], payload['args']
    hard = state['item_encoder.hard_sid_table']
    tokens = len(state['item_encoder.shared_embedding.weight']) - 1
    if payload['variant'] == 'consolidated_hard_scaled':
        model = empty_compiled(hard, state['item_encoder.soft_sid_table'],
                               state['item_encoder.candidate_prior'], tokens, args)
        if model.item_encoder.soft_sid_table.shape[-1] != 1:
            raise ValueError('Expected singleton Hard SID assignments')
    elif payload['variant'] in ('hard', 'hard_scaled'):
        model = SimpleSharing('hard', hard, tokens, None, None, args['dim'], args['max_len'], 2, 2, args['dropout'])
    else:
        raise ValueError(f"Unsupported Hard checkpoint: {payload['variant']}")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, payload


def split_examples(rows, max_len, seed=1707):
    """Disjoint users, TRAIN prefixes only; never include validation/test labels."""
    order = np.random.default_rng(seed).permutation(len(rows))
    views = []
    for selected in (order[::2], order[1::2]):
        data = NextItemDataset([rows[int(i)] for i in selected], max_len, 'train')
        grouped = defaultdict(list)
        for index, (_, target, _) in enumerate(data.samples):
            grouped[target].append(index)
        views.append((data, grouped))
    return views


def target_gradients(model, sequences, candidates, temperature):
    """True d L_i / d E plus a direct positive-path gradient, all at fixed weights."""
    h, _ = model.encode_sequence(sequences)
    unique, inverse = candidates.unique(return_inverse=True)
    vectors = model.item_encoder(unique)['vectors'][inverse]
    logits = torch.einsum('bd,bcd->bc', h, vectors) / temperature
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(sequences), dtype=torch.long, device=sequences.device))
    # Derivative of the same CE w.r.t. its positive candidate vector. Detach
    # upstream terms to exclude history/negative lookup paths in the second audit.
    upstream = ((logits.detach().softmax(-1)[:, :1] - 1) * h.detach() / (temperature * len(sequences))).sum(0, keepdim=True)
    weight = model.item_encoder.shared_embedding.weight
    full = torch.autograd.grad(loss, weight)[0]
    target = candidates[0, 0:1]
    if not candidates[:, 0].eq(target).all():
        raise ValueError('Each gradient batch must have a single next-item target')
    positive_vector = model.item_encoder(target)['vectors']
    positive = torch.autograd.grad(positive_vector, weight, grad_outputs=upstream)[0]
    return full.detach(), positive.detach(), float(loss.detach())


def collect(model, settings, rows, args, destination):
    hard = model.item_encoder.hard_sid_table.detach().cpu()
    n, levels, dim = len(hard) - 1, hard.shape[1], settings['args']['dim']
    cache = destination / 'gradients.npz'
    if cache.exists():
        return dict(np.load(cache))
    views = split_examples(rows, settings['args']['max_len'])
    available = np.zeros((2, n + 1), dtype=np.int64)
    for view, (_, groups) in enumerate(views):
        for item, examples in groups.items():
            available[view, item] = len(examples)
    eligible = np.where((available >= args.min_events).all(0))[0]
    full = np.zeros((2, n + 1, levels, dim), dtype=np.float32)
    positive = np.zeros_like(full)
    counts = np.zeros_like(available)
    losses = np.zeros((2, n + 1), dtype=np.float32)
    temperature = math.sqrt(dim) if settings['variant'].endswith('_scaled') else 1.
    for view, (data, groups) in enumerate(views):
        for position, item in enumerate(eligible):
            # Same contexts and negative draws across checkpoints/training seeds.
            sample_seed = 9107 + view * 100000 + int(item)
            local = random.Random(sample_seed)
            chosen = local.sample(groups[int(item)], min(args.max_events, len(groups[int(item)])))
            random.seed(sample_seed)
            sampler = NegativeSampler(n, settings['args']['negatives'])
            batch = [data[index] for index in chosen]
            sequences = torch.stack([x[0] for x in batch]).to(args.device)
            candidates = torch.tensor([sampler.sample(x[1], x[2]) for x in batch], device=args.device)
            gf, gp, loss = target_gradients(model, sequences, candidates, temperature)
            full[view, item] = gf[hard[item].to(args.device)].cpu().numpy()
            positive[view, item] = gp[hard[item].to(args.device)].cpu().numpy()
            counts[view, item], losses[view, item] = len(batch), loss
            if (position + 1) % 100 == 0 or position + 1 == len(eligible):
                print(json.dumps({'checkpoint': str(destination), 'view': view, 'items': position + 1,
                                  'total_items': len(eligible)}), flush=True)
    values = dict(full=full, positive=positive, counts=counts, available=available, losses=losses)
    np.savez_compressed(cache, **values)
    return values


def cosines(a):
    unit = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-30)
    return np.clip(unit @ unit.T, -1., 1.)


def behavior_matrices(rows, n):
    """Text-independent training behavior: shared users and preceding-item contexts."""
    item_index, user_index = [], []
    dest, context, weights = [], [], []
    for user, row in enumerate(rows):
        items = list(map(int, row['items'][:-2]))
        for item in set(items):
            item_index.append(item); user_index.append(user)
        for t in range(1, len(items)):
            for lag in range(1, min(t, 5) + 1):
                dest.append(items[t]); context.append(items[t-lag]); weights.append(1. / lag)
    matrices = []
    for matrix in (sp.csr_matrix((np.ones(len(item_index)), (item_index, user_index)), shape=(n+1, len(rows))),
                   sp.csr_matrix((weights, (dest, context)), shape=(n+1, n+1))):
        norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel())
        normalized = sp.diags(1. / np.maximum(norm, 1e-30)) @ matrix
        matrices.append((normalized @ normalized.T).toarray().astype(np.float32))
    frequency = np.bincount([i for r in rows for i in r['items'][:-2]], minlength=n+1)
    return *matrices, frequency


def text_matrix(dataset_dir, n):
    raw = np.load(dataset_dir / 'item_text_embeddings.npy')
    ids = list(map(int, json.loads((dataset_dir / 'embedding_item_ids.json').read_text())))
    if len(ids) != len(raw) or len(set(ids)) != len(ids) or set(ids) != set(range(1, n+1)):
        raise ValueError('Text embedding IDs must exactly match the catalog')
    vectors = np.zeros((n+1, raw.shape[1]), dtype=np.float32)
    vectors[ids] = raw
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-30)
    return np.clip(vectors @ vectors.T, -1., 1.)


def safe_corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def partial_rank_corr(x, y, controls):
    if len(x) < 10:
        return None
    design = np.column_stack([np.ones(len(x)), *[rankdata(c) for c in controls]])
    rx, ry = rankdata(x), rankdata(y)
    rx -= design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry -= design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(hard, values, rows, hard_score, soft_score, semantic, usersim, contextsim, frequency, args, dest):
    levels, n = hard.shape[1], len(hard) - 1
    targets = np.array([r['items'][-1] for r in rows if len(r['items']) >= 3])
    if len(targets) != len(hard_score) or hard_score.shape != soft_score.shape:
        raise ValueError('Paired scores must match evaluation user order')
    gains = soft_score - hard_score
    all_token_rows, summaries = [], {}
    visualization = {}
    for mode in ('full', 'positive'):
        gradient = values[mode]
        mode_rows, pair_chunks = [], []
        for level in range(levels):
            for token in np.unique(hard[1:, level]):
                members = np.where(hard[:, level] == token)[0]
                ok = (values['counts'][:, members] >= args.min_events).all(0)
                ok &= (np.linalg.norm(gradient[:, members, level], axis=-1) > 1e-12).all(0)
                items = members[ok]
                if len(items) < 4:
                    continue
                a, b = gradient[0, items, level], gradient[1, items, level]
                ca, cb = cosines(a), cosines(b)
                average = (a * values['counts'][0, items, None] + b * values['counts'][1, items, None]) / values['counts'][:, items].sum(0)[:, None]
                cosine = cosines(average)
                ix, jx = np.triu_indices(len(items), 1)
                i, j = items[ix], items[jx]
                negative = cosine[ix, jx] < 0
                stable = (ca[ix, jx] < -.01) & (cb[ix, jx] < -.01)
                label_mask = np.isin(targets, members)
                item_stability = np.sum(a*b, -1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-30)
                row = {'mode': mode, 'level': int(level + 1), 'token': int(token), 'occupancy': len(members),
                       'eligible_items': len(items), 'coverage': len(items)/len(members), 'pairs': len(ix),
                       'conflict_rate': float(negative.mean()), 'conflict_margin_rate': float((cosine[ix, jx] < -.01).mean()),
                       'stable_conflict_rate': float(stable.mean()), 'mean_cosine': float(cosine[ix, jx].mean()),
                       'view_a_conflict_rate': float((ca[ix, jx] < 0).mean()), 'view_b_conflict_rate': float((cb[ix, jx] < 0).mean()),
                       'mean_item_gradient_reproducibility': float(item_stability.mean()),
                       'mean_gradient_norm': float(np.linalg.norm(average, axis=-1).mean()),
                       'mean_frequency': float(frequency[members].mean()), 'test_count': int(label_mask.sum()),
                       'hard_ndcg10': float(hard_score[label_mask].mean()) if label_mask.any() else None,
                       'soft_ndcg10': float(soft_score[label_mask].mean()) if label_mask.any() else None,
                       'soft_gain': float(gains[label_mask].mean()) if label_mask.any() else None}
                mode_rows.append(row)
                pair_chunks.append(np.column_stack([np.full(len(ix), level+1), np.full(len(ix), token), i, j,
                    cosine[ix,jx], stable, semantic[i,j], contextsim[i,j], usersim[i,j],
                    np.log1p(frequency[i]), np.log1p(frequency[j])]).astype(np.float32))
                if len(items) > len(visualization.get(mode, {}).get('items', [])):
                    # Choose largest eligible occupancy, not largest conflict.
                    order = np.argsort(-frequency[items], kind='stable')[:60]
                    visualization[mode] = {'level': int(level+1), 'token': int(token), 'items': items[order],
                                           'cosine': cosine[np.ix_(order, order)]}
        all_token_rows.extend(mode_rows)
        if not pair_chunks:
            summaries[mode] = {'tokens': 0}
            continue
        pairs = np.concatenate(pair_chunks)
        np.savez_compressed(dest / f'pairs_{mode}.npz', pairs=pairs,
                            columns=np.array(['level','token','item_i','item_j','cosine','stable_conflict','semantic_cosine','context_cosine','user_cosine','log_frequency_i','log_frequency_j']))
        valid = [r for r in mode_rows if r['test_count'] >= args.min_test]
        by_level = {}
        for level in range(1, levels+1):
            rr = [r for r in mode_rows if r['level'] == level]
            vv = [r for r in valid if r['level'] == level]
            by_level[str(level)] = {'tokens': len(rr), 'gain_tokens': len(vv),
                'occupancy_conflict_rho': safe_corr([r['occupancy'] for r in rr], [r['conflict_rate'] for r in rr]),
                'conflict_gain_rho': safe_corr([r['conflict_rate'] for r in vv], [r['soft_gain'] for r in vv]),
                'conflict_hard_ndcg_rho': safe_corr([r['conflict_rate'] for r in vv], [r['hard_ndcg10'] for r in vv]),
                'stable_conflict_gain_rho': safe_corr([r['stable_conflict_rate'] for r in vv], [r['soft_gain'] for r in vv]),
                'adjusted_conflict_gain_r': partial_rank_corr([r['conflict_rate'] for r in vv], [r['soft_gain'] for r in vv],
                     [[r[k] for r in vv] for k in ('occupancy','mean_frequency','test_count','mean_gradient_norm')])}
        # Pooled pairs repeat across tokens/levels; these are descriptive rates,
        # not independent observations for a significance test.
        high_sem = pairs[:,6] >= np.quantile(pairs[:,6], .75)
        behavior_rows = {}
        for col, name in ((7, 'preceding_item_context'), (8, 'shared_training_users')):
            subset = pairs[high_sem]
            zero = subset[:,col] <= 1e-12
            nonzero = subset[:,col] > 1e-12
            threshold = float(np.median(subset[nonzero,col])) if nonzero.any() else None
            high = subset[:,col] >= threshold if threshold is not None else np.zeros(len(subset), dtype=bool)
            behavior_rows[name] = {'high_semantic_pair_count': len(subset), 'zero_overlap_pairs': int(zero.sum()),
                'high_overlap_pairs': int(high.sum()), 'high_overlap_threshold': threshold,
                'zero_overlap_conflict_rate': float((subset[zero,4] < 0).mean()) if zero.any() else None,
                'high_overlap_conflict_rate': float((subset[high,4] < 0).mean()) if high.any() else None,
                'behavior_vs_conflict_rho': safe_corr(subset[:,col], (subset[:,4] < 0).astype(float))}
        rng = np.random.default_rng(1707)
        all_items = np.where((values['counts'] >= args.min_events).all(0))[0]
        cross = []
        for level in range(levels):
            ok = (np.linalg.norm(gradient[:, all_items, level], axis=-1) > 1e-12).all(0)
            ii = all_items[ok]
            if len(ii) < 2:
                continue
            aa, bb = rng.choice(ii, (2, 20000), replace=True)
            take = (aa != bb) & (hard[aa, level] != hard[bb, level])
            aa, bb = aa[take], bb[take]
            means = (gradient[:,:,level] * values['counts'][:,:,None]).sum(0) / np.maximum(values['counts'].sum(0)[:,None], 1)
            co = np.sum(means[aa]*means[bb], -1) / np.maximum(np.linalg.norm(means[aa],axis=-1)*np.linalg.norm(means[bb],axis=-1),1e-30)
            cross.extend(co.tolist())
        summaries[mode] = {'tokens': len(mode_rows), 'gain_tokens': len(valid), 'token_pair_entries': len(pairs),
            'within_token_pair_conflict_rate': float((pairs[:,4] < 0).mean()),
            'within_token_stable_pair_conflict_rate': float(pairs[:,5].mean()),
            'random_cross_token_conflict_rate': float((np.array(cross) < 0).mean()) if cross else None,
            'median_token_conflict_rate': float(np.median([r['conflict_rate'] for r in mode_rows])),
            'mean_item_reproducibility_over_tokens': float(np.mean([r['mean_item_gradient_reproducibility'] for r in mode_rows])),
            'occupancy_conflict_rho': safe_corr([r['occupancy'] for r in mode_rows], [r['conflict_rate'] for r in mode_rows]),
            'conflict_gain_rho': safe_corr([r['conflict_rate'] for r in valid], [r['soft_gain'] for r in valid]),
            'by_level': by_level, 'semantic_behavior': behavior_rows}
    if all_token_rows:
        with (dest / 'tokens.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_token_rows[0]))
            writer.writeheader(); writer.writerows(all_token_rows)
    np.savez_compressed(dest / 'heatmaps.npz', **{f'{m}_{k}': v for m,x in visualization.items() for k,v in x.items()})
    result = {'gradient_eligible_items': int(((values['counts'] >= args.min_events).all(0)).sum()),
              'catalog_items': n, 'test_users': len(targets), 'hard_ndcg10': float(hard_score.mean()),
              'soft_ndcg10': float(soft_score.mean()), 'soft_gain': float(gains.mean()), 'modes': summaries}
    write_json(dest / 'summary.json', result)
    plot(all_token_rows, visualization, dest, args.min_test)
    return result


def plot(rows, heatmaps, dest, min_test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for line, mode in enumerate(('full', 'positive')):
        rr = [r for r in rows if r['mode'] == mode]
        if not rr:
            continue
        for level in range(1,5):
            sub = [r for r in rr if r['level'] == level]
            axes[line,0].scatter([r['occupancy'] for r in sub], [r['conflict_rate'] for r in sub], s=15, alpha=.65, label=f'L{level}')
            sub = [r for r in sub if r['test_count'] >= min_test]
            axes[line,1].scatter([r['conflict_rate'] for r in sub], [r['soft_gain'] for r in sub],
                                 s=[min(100, 8+math.sqrt(r['test_count'])*2) for r in sub], alpha=.65, label=f'L{level}')
        axes[line,0].set(xscale='log', xlabel='Token occupancy (all catalog items)', ylabel='Negative pair cosine rate', title=f'{mode}: occupancy vs conflict')
        axes[line,1].axhline(0, c='gray', lw=.8)
        axes[line,1].set(xlabel='Token gradient conflict rate', ylabel='Soft minus Hard test NDCG@10', title=f'{mode}: conflict vs observed Soft gain')
        axes[line,0].legend(fontsize=8)
        h = heatmaps[mode]
        im = axes[line,2].imshow(h['cosine'], vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
        axes[line,2].set(title=f"{mode}: L{h['level']} token {h['token']}\nLargest eligible group, up to 60 items", xlabel='Item (ordered by training frequency)', ylabel='Item')
        fig.colorbar(im, ax=axes[line,2], shrink=.8, label='Gradient cosine')
    fig.suptitle(f'{dest.name}: trained Hard SID gradient diagnostic\nDescriptive associations; token groups overlap; no causal claim', fontsize=13)
    fig.savefig(dest / 'diagnostic.png', dpi=180)
    fig.savefig(dest / 'diagnostic.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='runs/office/gradient_conflict_diagnostic_20260910')
    p.add_argument('--dataset-dir', default='runs/office')
    p.add_argument('--pair-root', default='runs/office/consolidated_ablation_20260910')
    p.add_argument('--hard-variant', default='consolidated_hard_scaled')
    p.add_argument('--soft-variant', default='consolidated_scaled')
    p.add_argument('--seeds', nargs='+', type=int, default=[2026,2027,2028])
    p.add_argument('--min-events', type=int, default=4)
    p.add_argument('--max-events', type=int, default=16)
    p.add_argument('--min-test', type=int, default=20)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not 1 <= args.min_events <= args.max_events:
        raise ValueError('Invalid event limits')
    torch.set_num_threads(2)
    root, data_dir = Path(args.output_dir), Path(args.dataset_dir)
    rows = json.loads((data_dir / 'sequences.json').read_text())
    n = json.loads((data_dir / 'stats.json').read_text())['num_items']
    semantic = text_matrix(data_dir, n)
    usersim, contextsim, frequency = behavior_matrices(rows, n)
    results = {}
    for seed in args.seeds:
        pair = Path(args.pair_root)
        hard_dir, soft_dir = pair / args.hard_variant / f'seed{seed}', pair / args.soft_variant / f'seed{seed}'
        model, settings = load_hard(hard_dir / 'best.pt', args.device)
        hard_result = json.loads((hard_dir / 'result.json').read_text())
        soft_result = json.loads((soft_dir / 'result.json').read_text())
        if settings['seed'] != seed or soft_result['seed'] != seed:
            raise ValueError('Seed mismatch')
        for key in ('dataset_dir','semantic_ids','dim','max_len','negatives','lr','weight_decay','dropout'):
            if hard_result['args'][key] != soft_result['args'][key]:
                raise ValueError(f'Unmatched Hard/Soft protocol: {key}')
        if Path(settings['args']['dataset_dir']).resolve() != data_dir.resolve():
            raise ValueError('Checkpoint dataset mismatch')
        dest = root / f'seed{seed}'
        manifest = {'args': vars(args), 'hard_checkpoint': str(hard_dir / 'best.pt'),
                    'hard_sha256': digest(hard_dir / 'best.pt'), 'soft_checkpoint': str(soft_dir / 'best.pt'),
                    'soft_sha256': digest(soft_dir / 'best.pt'), 'sequences_sha256': digest(data_dir / 'sequences.json')}
        if (dest / 'manifest.json').exists() and json.loads((dest / 'manifest.json').read_text()) != manifest:
            raise ValueError('Existing diagnostic has different inputs')
        write_json(dest / 'manifest.json', manifest)
        values = collect(model, settings, rows, args, dest)
        hard_score = torch.load(hard_dir / 'test_per_user.pt', weights_only=True).numpy()
        soft_score = torch.load(soft_dir / 'test_per_user.pt', weights_only=True).numpy()
        results[str(seed)] = analyze(model.item_encoder.hard_sid_table.cpu().numpy(), values, rows,
                                     hard_score, soft_score, semantic, usersim, contextsim, frequency, args, dest)
        write_json(root / 'summary.json', results)
        print(json.dumps({'seed': seed, 'summary': results[str(seed)]}), flush=True)
        del model


if __name__ == '__main__':
    main()
