"""Diagnostic only: can a linear controller predict held-out LOO utility?

Uses a frozen static backbone, train-only targets, validation BCE selection,
and a constant train-mean target baseline. Does not alter recommendation models.
"""
import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from LoCoRec.locorec.io import write_json
from .experiment import build_data, contexts, make_frozen_training, utility_metrics, seed_all
from .model import FrozenCatalog, LoCoRecCAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = argparse.Namespace(**checkpoint['args'])
    cfg.device = args.device
    torch.set_num_threads(cfg.threads)
    seed_all(cfg.seed)
    datasets, table, tokens = build_data(cfg)
    state = checkpoint['model']
    model = LoCoRecCAS(table, tokens, state['item_encoder.scope_tokens'], state['item_encoder.candidate_prior'], **checkpoint['config'])
    model.load_state_dict(state)
    model.to(cfg.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    catalog = FrozenCatalog(model.item_encoder, cfg.item_chunk)
    data = {}
    for i, split in enumerate(('train', 'valid', 'test')):
        print(f'Utility-only probe: caching {split}', flush=True)
        h = contexts(model, datasets[split], cfg)
        values = make_frozen_training(catalog, h, datasets[split], cfg, cfg.seed + i)
        data[split] = {'h': h, 'delta': values['delta'], 'target': (values['delta'] / cfg.temperature).sigmoid()}
        del values
    mean = data['train']['target'].mean(0)
    head = nn.Linear(cfg.dim, model.depth).to(cfg.device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.logit(mean.clamp(1e-6, 1 - 1e-6)))
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(args.epochs + 1):
        if epoch > 0:
            order = torch.randperm(len(data['train']['h']), device=cfg.device)
            for start in range(0, len(order), cfg.batch_size):
                ids = order[start:start + cfg.batch_size]
                loss = F.binary_cross_entropy_with_logits(head(data['train']['h'][ids]), data['train']['target'][ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            value = float(F.binary_cross_entropy_with_logits(head(data['valid']['h']), data['valid']['target']))
        history.append({'epoch': epoch, 'valid_BCE': value})
        if value < best:
            best, best_epoch, best_state = value, epoch, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    result = {'protocol': 'diagnostic utility-only linear probe; train labels only; validation BCE selects epoch including constant initialization',
              'best_epoch': best_epoch, 'temperature': cfg.temperature, 'history': history}
    with torch.no_grad():
        for split in ('train', 'valid', 'test'):
            values = data[split]
            logits = head(values['h'])
            metrics = utility_metrics(logits.sigmoid(), values['delta'], cfg.temperature, mean)
            metrics['BCE'] = float(F.binary_cross_entropy_with_logits(logits, values['target']))
            metrics['constant_BCE'] = float(F.binary_cross_entropy(mean.expand_as(values['target']), values['target']))
            result[split] = metrics
        # Repeat held-out negatives to estimate label noise from candidate sampling.
        repeated = make_frozen_training(catalog, data['test']['h'], datasets['test'], cfg, cfg.seed + 999)
        original = data['test']['delta']
        alternate = repeated['delta']
        x, y = original - original.mean(0), alternate - alternate.mean(0)
        result['negative_resample_stability'] = {
            'delta_correlation_per_level': ((x * y).sum(0) / (x.square().sum(0) * y.square().sum(0)).sqrt().clamp_min(1e-12)).cpu().tolist(),
            'sign_agreement_per_level': original.gt(0).eq(alternate.gt(0)).float().mean(0).cpu().tolist()}
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k != 'history'}, flush=True)


if __name__ == '__main__':
    main()
