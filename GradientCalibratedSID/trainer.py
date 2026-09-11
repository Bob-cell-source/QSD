import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader
from functools import partial

from CCSR.trainer import NegativeSampler
from LoCoRec.locorec.data import collate_train
from LoCoRec.locorec.io import write_json
from LoCoRecSimple.experiment import evaluate


def training_loader(data, config, seed, shuffle=True):
    random.seed(seed)
    torch.manual_seed(seed)
    return DataLoader(data, batch_size=config['batch_size'], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=partial(collate_train, sampler=NegativeSampler(config['num_items'], config['negatives'])))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def fit(model, train, valid, test, config, seed, output, stage, variant, source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / ('hard_sid_checkpoint.pt' if stage == 1 else 'best.pt')
    if (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text())
        if not checkpoint_path.exists():
            raise ValueError('Completed result has no checkpoint')
        return result, checkpoint_path
    device = config['device']
    model.to(device).requires_grad_(True)
    # All stage-2 variants restart AdamW consistently at the same source model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    start_weights = tensor_hash(model.item_encoder.sharing_weight)
    start_ids = tensor_hash(model.item_encoder.candidate_ids)
    best, best_epoch, stale, history = -float('inf'), 0, 0, []
    started = time.perf_counter()

    def save():
        torch.save({'format': 'gcss-v1', 'config': model.config, 'model': model.state_dict(),
                    'stage': stage, 'variant': variant, 'seed': seed, 'epoch': best_epoch,
                    'source': source, 'training_config': config}, checkpoint_path)

    if stage == 2:
        metrics, _ = evaluate(model, valid, device, config['num_items'])
        best = metrics['NDCG@10']
        history.append({'epoch': 0, 'valid': metrics})
        save()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total, count = 0., 0
        # Arms have identical epoch-level sample order, negatives and dropout RNG.
        loader = training_loader(train, config, seed * 1000 + stage * 100000 + epoch)
        for history_items, candidates in loader:
            logits = model(history_items.to(device), candidates.to(device))['score'] / math.sqrt(model.config['dim'])
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(history_items), dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(history_items)
            count += len(history_items)
        metrics, _ = evaluate(model, valid, device, config['num_items'])
        record = {'epoch': epoch, 'loss': total/count, 'valid': metrics, 'elapsed_seconds': time.perf_counter()-started}
        history.append(record)
        write_json(output / 'history.json', history)
        print(json.dumps({'seed': seed, 'stage': stage, 'variant': variant, **record}), flush=True)
        if metrics['NDCG@10'] > best:
            best, best_epoch, stale = metrics['NDCG@10'], epoch, 0
            save()
        else:
            stale += 1
            if stale >= config['patience']:
                break
    if start_weights != tensor_hash(model.item_encoder.sharing_weight) or start_ids != tensor_hash(model.item_encoder.candidate_ids):
        raise RuntimeError('Fixed sharing lookup changed during training')
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    metrics, per_user = evaluate(model, test, device, config['num_items'])
    torch.save(per_user, output / 'test_per_user.pt')
    result = {'stage': stage, 'variant': variant, 'seed': seed, 'best_epoch': best_epoch,
              'best_valid_NDCG@10': best, 'test': metrics, 'source': source,
              'parameters': sum(p.numel() for p in model.parameters()),
              'fixed_lookup_verified': True, 'sharing_weight_sha256': start_weights,
              'elapsed_seconds': time.perf_counter()-started, 'training_config': config}
    write_json(output / 'result.json', result)
    print(json.dumps({'completed': result}), flush=True)
    return result, checkpoint_path
