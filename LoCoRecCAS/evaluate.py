"""Read-only full-catalog evaluation of a frozen CAS controller or joint checkpoint."""
import argparse
from pathlib import Path

import torch

from LoCoRec.locorec.io import write_json
from .experiment import build_data, contexts, evaluate_gates
from .model import FrozenCatalog, LoCoRecCAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='static_best.pt or joint_best.pt')
    parser.add_argument('--head', help='Optional frozen *_head.pt')
    parser.add_argument('--split', choices=('valid', 'test'), default='test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    torch.set_num_threads(cli.threads)
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    cfg = argparse.Namespace(**checkpoint['args'])
    cfg.device = cli.device
    datasets, table, _ = build_data(cfg)
    state = checkpoint['model']
    if not torch.equal(table, state['item_encoder.hard_sid_table']):
        raise ValueError('Checkpoint SID mapping differs from the dataset.')
    tokens = state['item_encoder.semantic_basis_embedding.weight'].size(0) - 1
    model = LoCoRecCAS(table, tokens, state['item_encoder.scope_tokens'],
                      state['item_encoder.candidate_prior'], **checkpoint['config'])
    model.load_state_dict(state)
    model.to(cli.device).eval()
    mode = checkpoint.get('gate_mode', 'static')
    if cli.head:
        head = torch.load(cli.head, map_location=cli.device, weights_only=True)
        # Heads are meaningful only with the exact frozen backbone used for fitting.
        original = torch.load(head['backbone'], map_location='cpu', weights_only=True)['model']
        if set(original) != set(state) or any(not torch.equal(original[k].cpu(), state[k].cpu()) for k in state):
            raise ValueError('Controller head was fitted to a different static backbone.')
        mode = head['mode']
        if mode == 'context':
            model.context_head.load_state_dict(head['head'])
        else:
            with torch.no_grad():
                model.global_logits.copy_(head['head']['0'])
    with torch.no_grad():
        h = contexts(model, datasets[cli.split], cfg)
        gates = model.gates(h, mode)
        values, _ = evaluate_gates(FrozenCatalog(model.item_encoder, cfg.item_chunk), h, datasets[cli.split],
                                   {mode: gates}, cfg)
    result = {'checkpoint': cli.checkpoint, 'head': cli.head, 'split': cli.split, 'ranking': values}
    write_json(cli.output, result)
    print(result)


if __name__ == '__main__':
    main()
