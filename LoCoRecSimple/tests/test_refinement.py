import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from LoCoRecSimple.model import SimpleSharing
from LoCoRecSimple.refinement import RefinementItemEncoder, apply_partitions, partition_score, propose_partitions


def make():
    hard = torch.tensor([[0, 0], [1, 3], [1, 4], [1, 3], [1, 4]])
    model = SimpleSharing('hard', hard, 4, None, None, 8, 4, 2, 2, 0)
    model.item_encoder = RefinementItemEncoder(hard, 4, 8)
    return model


def test_gradient_capture_matches_shared_table_gradient_and_split_preserves_function():
    model = make()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    seq = torch.tensor([[0, 0, 1, 2], [0, 0, 3, 4]])
    cand = torch.tensor([[3, 4], [1, 2]])
    enc = model.item_encoder
    enc.captured = torch.zeros_like(enc.private_embedding.weight)
    torch.nn.functional.cross_entropy(model(seq, cand)['score'], torch.tensor([0, 0])).backward()
    expected = torch.zeros_like(enc.shared_embedding.weight)
    for level in range(2):
        expected.index_add_(0, enc.hard_sid_table[:, level], enc.captured)
    torch.testing.assert_close(expected, enc.shared_embedding.weight.grad)
    enc.captured = None
    optimizer.step()
    reference = model(seq, cand)['score'].detach().clone()
    moment = optimizer.state[enc.shared_embedding.weight]['exp_avg'].clone()
    hard_original = enc.hard_sid_table.clone()
    apply_partitions(model, optimizer, [{'level': 0, 'parent': 1, 'left': [1, 2], 'right': [3, 4]}])
    torch.testing.assert_close(model(seq, cand)['score'], reference, atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state[enc.shared_embedding.weight]['exp_avg'][-1], moment[1])
    torch.testing.assert_close(enc.hard_sid_table, hard_original)
    assert enc.lookup_table[1, 0] != enc.lookup_table[3, 0]
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(seq, cand)['score'], torch.tensor([0, 0])).backward()
    assert enc.shared_embedding.weight.grad[-1].abs().sum() > 0
    optimizer.step()


def test_split_surrogate_detects_conflict_and_audit_can_reject_it():
    gradient = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
    assert partition_score(gradient, gradient, [1, 2], [3, 4]) == 4.
    assert partition_score(gradient, -gradient, [1, 2], [3, 4]) == -4.
    aligned = torch.ones_like(gradient)
    assert partition_score(aligned, aligned, [1, 2], [3, 4]) == 0.
    plans = propose_partitions(make().item_encoder.hard_sid_table, gradient, torch.ones(5), budget=1, min_group=4)
    assert plans['gradient_split'][0]['fit_score'] == 4.
    assert all(len(rows) == 1 for key, rows in plans.items() if key != 'no_split')
    learned, random = plans['gradient_split'][0], plans['same_parent_random'][0]
    assert learned['parent'] == random['parent']
    assert len(learned['left']) == len(random['left'])


def test_refinement_end_to_end(tmp_path):
    rows = [{'items': [1, 2, 3, 4, 5]}, {'items': [2, 3, 4, 5, 6]},
            {'items': [3, 1, 4, 6, 2]}, {'items': [4, 2, 5, 3, 1]}]
    (tmp_path / 'sequences.json').write_text(json.dumps(rows))
    (tmp_path / 'stats.json').write_text(json.dumps({'num_items': 8}))
    (tmp_path / 'sid.json').write_text(json.dumps({'codebook_sizes': [2] * 4,
        'semantic_ids': {str(i): [i % 2, (i // 2) % 2, (i // 4) % 2, 0] for i in range(1, 9)}}))
    subprocess.run([sys.executable, '-m', 'LoCoRecSimple.refinement_experiment',
        '--dataset-dir', str(tmp_path), '--semantic-ids', str(tmp_path / 'sid.json'),
        '--output-dir', str(tmp_path / 'out'), '--device', 'cpu', '--epochs', '2',
        '--warmup-epochs', '1', '--min-post-epochs', '1', '--probe-batches', '2',
        '--split-budget', '1', '--min-group', '4', '--dim', '8', '--max-len', '4',
        '--batch-size', '2', '--negatives', '2', '--threads', '1'], check=True, capture_output=True)
    results = json.loads((tmp_path / 'out' / 'seed2026' / 'results.json').read_text())['results']
    assert len(results) == 5
    assert all(r['initial_vector_error'] == 0 for r in results)
    counts = [r['parameters'] for r in results if r['variant'] != 'no_split']
    assert len(set(counts)) == 1
