import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LoCoRecCAS.model import LoCoRecCAS, FrozenCatalog, build_scope
from LoCoRecCAS.experiment import parser, run


@pytest.fixture
def model():
    torch.manual_seed(22)
    table = torch.tensor([[0, 0], [1, 3], [1, 4], [2, 4], [2, 3]])
    tokens, prior, _ = build_scope(table, 50, 1)
    return LoCoRecCAS(table, 4, tokens, prior, dim=8, max_len=4, num_heads=2, num_layers=1, dropout=0)


def test_scope_exact_local_support_and_empty_neighborhood():
    table = torch.tensor([[0, 0], [1, 3], [1, 4], [2, 4]])
    tokens, prior, _ = build_scope(table, 50, 1)
    # Item 1's sole neighbor is item 2: hard support 0 + anchor 1, alt support 1.
    assert tokens[1, 1].tolist() == [3, 4]
    torch.testing.assert_close(prior[1, 1], torch.tensor([.5, .5]))
    # Level 1 is unchanged in that neighbor: hard support 1 + anchor 1.
    assert prior[1, 0, 0] == 1
    isolated, probabilities, info = build_scope(table, 50, 2)
    assert info['empty_neighborhoods'] == 3
    torch.testing.assert_close(isolated[..., 0], table)
    assert probabilities[1:].eq(1).all() and probabilities[0].eq(0).all()


def test_static_and_gated_formula_and_padding(model):
    enc = model.item_encoder
    items = torch.tensor([0, 1, 2, 3])
    weights = enc.assignment(items)
    assert weights[0].eq(0).all()
    torch.testing.assert_close(weights[1:].sum(-1), torch.ones(3, 2))
    base, shared = enc.components(items)
    manual_basis = []
    manual_shared = []
    for level in range(2):
        tokens = enc.scope_tokens[items, level]
        manual_basis.append((enc.semantic_basis_embedding(tokens) * weights[:, level, :, None]).sum(1))
        manual_shared.append((enc.shared_residual_embedding(tokens) * weights[:, level, :, None]).sum(1))
    b = enc.basis_projection(torch.stack(manual_basis, 1).mean(1))
    torch.testing.assert_close(base[1:], (b + enc.private_residual_embedding(items))[1:])
    torch.testing.assert_close(shared, torch.stack(manual_shared, 1))
    assert enc.static(items)[0].eq(0).all()
    h = torch.randn(2, 8)
    g = torch.tensor([[.2, .7], [.8, .1]])
    v = enc.output_norm(base[None] + torch.einsum('bl,cld->bcd', g, shared) / 2)
    expected = torch.einsum('bd,bcd->bc', h, v) / math.sqrt(8)
    torch.testing.assert_close(model.score_components(h, base, shared, g), expected)
    assert not any('residual_gate' in name or 'frequency' in name for name, _ in model.named_parameters())


def test_frozen_gram_scoring_preserves_full_layernorm_and_gradients(model):
    enc = model.item_encoder
    with torch.no_grad():
        enc.output_norm.weight.copy_(torch.randn(8))
        enc.output_norm.bias.copy_(torch.randn(8))
    h = torch.randn(3, 8)
    items = torch.tensor([[1, 2, 3], [3, 1, 4], [2, 4, 1]])
    base, shared = enc.components(items)
    gate = torch.rand(3, 2, requires_grad=True)
    direct = model.score_components(h, base, shared, gate)
    direct_grad = torch.autograd.grad(direct.sum(), gate)[0]
    catalog = FrozenCatalog(enc, 2)
    fast = catalog.score(catalog.features(h, items), items, gate, h)
    fast_grad = torch.autograd.grad(fast.sum(), gate)[0]
    torch.testing.assert_close(fast, direct, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(fast_grad, direct_grad, atol=2e-6, rtol=2e-5)
    items = torch.arange(1, 5)
    base, shared = enc.components(items)
    torch.testing.assert_close(catalog.score(catalog.features(h, items), items, gate, h),
                               model.score_components(h, base, shared, gate), atol=2e-6, rtol=2e-5)


def test_utility_targets_are_detached_exact_loo_losses(model):
    h = torch.randn(2, 8, requires_grad=True)
    base, shared = model.item_encoder.components(torch.tensor([[1, 2, 3], [4, 2, 1]]))
    target, delta = model.utility_targets(h, base, shared, .1)
    assert not target.requires_grad and not delta.requires_grad
    all_score = model.score_components(h, base, shared, torch.ones(2, 2))
    ce_all = F.cross_entropy(all_score, torch.zeros(2, dtype=torch.long), reduction='none')
    for level in range(2):
        gate = torch.ones(2, 2)
        gate[:, level] = 0
        score = model.score_components(h, base, shared, gate)
        expected = F.cross_entropy(score, torch.zeros(2, dtype=torch.long), reduction='none') - ce_all
        torch.testing.assert_close(delta[:, level], expected)
    torch.testing.assert_close(target, (delta / .1).sigmoid())
    with pytest.raises(ValueError):
        model.utility_targets(h, base, shared, 0)


def test_joint_gradients_and_controller_independence(model):
    seq = torch.tensor([[0, 0, 1, 2], [0, 0, 0, 3]])
    candidates = torch.tensor([[3, 4], [2, 1]])
    output = model(seq, candidates, lambda_util=.4)
    torch.testing.assert_close(output['loss'], output['loss_rec'] + .4 * output['loss_util'])
    output['loss'].backward()
    for name, parameter in model.named_parameters():
        if name == 'global_logits':
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    torch.testing.assert_close(output['gates'], model(seq, candidates.flip(-1))['gates'])
    model.zero_grad(set_to_none=True)
    model(seq, candidates, 'static', 0)['loss'].backward()
    assert model.context_head.weight.grad is None


@pytest.mark.parametrize('protocol,mode', [('frozen', 'context'), ('joint', 'context'), ('joint', 'global')])
def test_small_end_to_end(tmp_path, protocol, mode):
    rows = [{'items': [1, 2, 3, 4, 5]}, {'items': [2, 3, 4, 5, 6]}]
    (tmp_path / 'sequences.json').write_text(json.dumps(rows))
    (tmp_path / 'stats.json').write_text(json.dumps({'num_items': 8}))
    (tmp_path / 'sid.json').write_text(json.dumps({'codebook_sizes': [2, 2],
        'semantic_ids': {str(i): [i % 2, (i // 2) % 2] for i in range(1, 9)}}))
    args = parser().parse_args(['--dataset-dir', str(tmp_path), '--semantic-ids', str(tmp_path / 'sid.json'),
        '--output-dir', str(tmp_path / 'out'), '--from-scratch', '--device', 'cpu',
        '--dim', '8', '--max-len', '4', '--num-layers', '1', '--min-overlap', '1',
        '--batch-size', '2', '--eval-batch-size', '2', '--candidate-chunk', '3', '--negatives', '2',
        '--warmup-epochs', '1', '--epochs', '2', '--protocol', protocol, '--joint-mode', mode,
        '--threads', '1', '--dropout', '0'])
    result = run(args)
    assert (tmp_path / 'out/static_best.pt').is_file()
    assert (tmp_path / 'out/results.json').is_file()
    if protocol == 'frozen':
        assert set(result['test']['ranking']) == {'static', 'none', 'half', 'global', 'context_rec',
                                                 'context_utility', 'utility_shuffled', 'utility_mean'}
        assert len(result['valid']['utility_prediction']['context_utility']['target_mean']) == 2
        assert all(p >= 1 for p in result['best_epochs'].values())
        import subprocess
        subprocess.run([sys.executable, '-m', 'LoCoRecCAS.evaluate',
                        '--checkpoint', str(tmp_path / 'out/static_best.pt'),
                        '--head', str(tmp_path / 'out/context_utility_head.pt'),
                        '--device', 'cpu', '--threads', '1', '--output', str(tmp_path / 'eval.json')],
                       cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True)
        reevaluation = json.loads((tmp_path / 'eval.json').read_text())
        assert reevaluation['ranking']['context']['NDCG@10'] == result['test']['ranking']['context_utility']['NDCG@10']
    else:
        assert (tmp_path / 'out/joint_best.pt').is_file()


def test_assignment_reduces_to_prior_when_attention_is_neutral(model):
    with torch.no_grad():
        model.item_encoder.selector_query.weight.zero_()
    items = torch.arange(1, 5)
    torch.testing.assert_close(model.item_encoder.assignment(items), model.item_encoder.candidate_prior[items])
