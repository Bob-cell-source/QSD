import math
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CCSR.model import CCSR, load_hard_checkpoint, prefix_statistics
from CCSR.trainer import NegativeSampler, evaluate, training_stage
from LoCoRec.locorec.data import NextItemDataset, collate_eval


@pytest.fixture
def model():
    torch.manual_seed(7)
    table = torch.tensor([[0, 0], [1, 3], [1, 3], [1, 4], [2, 4]])
    return CCSR(4, 4, table, dim=8, max_len=4, num_heads=2, num_layers=1, dropout=0)


def test_prefix_entropy_excludes_padding_and_counts_prefixes(model):
    groups, entropy, cost, counts = prefix_statistics(model.item_encoder.hard_sid_table)
    assert counts == [2, 3]
    assert groups[0].eq(0).all()
    assert groups[1].equal(groups[2])
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert entropy[0].item() == pytest.approx(expected)
    assert cost[1].item() == pytest.approx(0.75)
    assert cost[0] <= cost[1] <= cost[2] == 1
    _, _, constant_cost, _ = prefix_statistics(torch.tensor([[0], [1], [1]]))
    assert constant_cost.tolist() == [0, 1]


def test_representation_matches_formula_and_history_is_exact(model):
    enc = model.item_encoder
    items = torch.tensor([0, 1, 2, 3])
    result = enc(items)
    levels = enc.basis_projection(enc.semantic_basis_embedding(enc.hard_sid_table[items]))
    levels = levels + enc.shared_residual_embedding(enc.hard_sid_table[items])
    for r in range(3):
        expected = enc.output_norm(levels[:, :min(r + 1, 2)].mean(1)
                                   + enc.resolution_cost[r] * enc.private_residual_embedding(items))
        torch.testing.assert_close(result[1:, r], expected[1:])
    assert result[0].eq(0).all()
    torch.testing.assert_close(result[:, -1], enc.exact(items))
    assert not torch.allclose(result[1, 0], result[2, 0])
    sequence = torch.tensor([[0, 0, 1, 2]])
    torch.testing.assert_close(model.encode_sequence(sequence), model.sequence_encoder(sequence, enc(sequence)[..., -1, :]))
    assert not any(any(name in key for name in ("selector", "gate", "prior", "soft")) for key, _ in model.named_parameters())


def test_hazard_initialization_and_nonuniform_distribution(model):
    h = torch.randn(5, 8)
    torch.testing.assert_close(model.stopping_distribution(h), torch.full((5, 3), 1 / 3))
    with torch.no_grad():
        model.stop_head.bias.copy_(torch.logit(torch.tensor([0.2, 0.6])))
    expected = torch.tensor([0.2, 0.48, 0.32]).expand(5, -1)
    torch.testing.assert_close(model.stopping_distribution(h), expected)
    with torch.no_grad():
        model.stop_head.bias.copy_(torch.tensor([-1000., 1000.]))
    torch.testing.assert_close(model.stopping_distribution(h), torch.tensor([0., 1., 0.]).expand(5, -1))


def test_expected_ce_not_ce_of_mixed_scores_and_gradient(model):
    scores = torch.tensor([[[4., 0.], [0., 4.], [1., 0.]]], requires_grad=True)
    pi = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
    losses = model.objective({"scores": scores, "pi": pi}, 0.1)
    expected_ce = sum(pi[0, r] * torch.nn.functional.cross_entropy(scores[:, r], torch.tensor([0])) for r in range(3))
    torch.testing.assert_close(losses["loss_pred"], expected_ce)
    mixed_ce = torch.nn.functional.cross_entropy((scores * pi[..., None]).sum(1), torch.tensor([0]))
    assert not torch.isclose(mixed_ce, expected_ce)
    losses["loss"].backward()
    torch.testing.assert_close(pi.grad, losses["loss_per_resolution"].detach() + 0.1 * model.item_encoder.resolution_cost)
    sequence = torch.tensor([[0, 0, 1, 2], [0, 0, 0, 3]])
    candidates = torch.tensor([[3, 4, 1], [2, 1, 4]])
    for uniform in (True, False):
        model.zero_grad(set_to_none=True)
        model.objective(model(sequence, candidates, uniform), 0.03)["loss"].backward()
        for name, parameter in model.named_parameters():
            if uniform and name.startswith("stop_head"):
                assert parameter.grad is None
            else:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_grouped_hard_and_soft_scores_match_direct_scoring(model):
    h = torch.randn(3, 8)
    vectors = model.item_encoder(torch.arange(1, 5))
    pi = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    scores = torch.einsum("bd,crd->brc", h, vectors) / math.sqrt(8)
    torch.testing.assert_close(model.score_catalog(h, vectors, pi), scores[torch.arange(3), pi.argmax(-1)])
    torch.testing.assert_close(model.score_catalog(h, vectors, pi, "soft"), (scores * pi[..., None]).sum(1))


def test_full_ranking_chunking_masking_and_groups(model):
    rows = [{"items": [1, 2, 1, 3, 4]}, {"items": [3, 2, 4, 1, 2]}]
    loader = DataLoader(NextItemDataset(rows, 4, "valid"), batch_size=2, collate_fn=collate_eval)
    result = evaluate(model, loader, torch.device("cpu"), 4, 1, ["Mem", "Gen"])
    other = evaluate(model, loader, torch.device("cpu"), 4, 4, ["Mem", "Gen"])
    assert result == other
    assert result["groups"]["Mem"]["count"] == 1
    # Direct full-catalog reference with seen history masking.
    sequences, targets, histories = next(iter(loader))
    h = model.encode_sequence(sequences)
    pi = model.stopping_distribution(h)
    for mode in ("hard", "soft", "1", "2", "3"):
        scores = model.score_catalog(h, model.item_encoder(torch.arange(1, 5)), pi, mode)
        for row, history in enumerate(histories):
            for item in set(history) - {int(targets[row])}:
                scores[row, item - 1] = -torch.inf
        order = scores.argsort(-1, descending=True) + 1
        rank = order.eq(targets[:, None]).float().argmax(-1) + 1
        expected = float((1 / torch.log2(rank.float() + 1)).mean())
        assert result["ranking"][mode]["NDCG@5"] == pytest.approx(expected)


def test_transfer_hard_only_and_mapping_validation(model):
    state = {key: value.clone() for key, value in model.state_dict().items()}
    table = state.pop("item_encoder.hard_sid_table")
    state["item_encoder.soft_sid_table"] = table.unsqueeze(-1)
    state["item_encoder.candidate_prior"] = table.ne(0).float().unsqueeze(-1)
    report = load_hard_checkpoint(model, {"model": state})
    assert "item_encoder.shared_residual_embedding.weight" in report["loaded_parameters"]
    state["item_encoder.soft_sid_table"] = torch.cat([table.unsqueeze(-1)] * 2, -1)
    state["item_encoder.candidate_prior"] = torch.cat([table.ne(0).float().unsqueeze(-1)] * 2, -1) / 2
    with pytest.raises(ValueError, match="Soft SID checkpoint rejected"):
        load_hard_checkpoint(model, {"model": state})
    state = dict(model.state_dict())
    state["item_encoder.hard_sid_table"] = table.clone()
    state["item_encoder.hard_sid_table"][1, 0] = 2
    with pytest.raises(ValueError, match="mapping differs"):
        load_hard_checkpoint(model, {"model": state})
    state = dict(model.state_dict())
    del state["item_encoder.shared_residual_embedding.weight"]
    with pytest.raises(ValueError, match="missing/mismatched"):
        load_hard_checkpoint(model, {"model": state})


def test_warmup_ramp_fixed_baseline_and_negative_guard():
    assert training_stage(3, 3, 0.1) == (True, 0)
    assert [training_stage(e, 3, 0.1)[1] for e in range(4, 9)] == pytest.approx([0, .025, .05, .075, .1])
    assert training_stage(1, 3, .1, fixed_resolution=2) == (False, 0)
    with pytest.raises(ValueError, match="eligible negatives"):
        NegativeSampler(4, 2).sample(1, [1, 2, 3])
    negatives = NegativeSampler(6, 3).sample(1, [1, 2, 3])
    assert negatives[0] == 1 and set(negatives[1:]) == {4, 5, 6}


def test_end_to_end_training_and_checkpoint(tmp_path):
    import json
    from CCSR.trainer import build_parser, train
    rows = [{"items": [1, 2, 3, 4, 5]}, {"items": [2, 3, 4, 5, 6]}]
    (tmp_path / "sequences.json").write_text(json.dumps(rows))
    (tmp_path / "stats.json").write_text(json.dumps({"num_items": 8}))
    (tmp_path / "sid.json").write_text(json.dumps({"codebook_sizes": [2, 2], "semantic_ids":
        {str(i): [(i // 2) % 2, i % 2] for i in range(1, 9)}}))
    output = tmp_path / "output"
    args = build_parser().parse_args([
        "--dataset-dir", str(tmp_path), "--semantic-ids", str(tmp_path / "sid.json"),
        "--output-dir", str(output), "--from-scratch", "--device", "cpu",
        "--epochs", "3", "--warmup-epochs", "1", "--lambda-warmup-epochs", "2",
        "--batch-size", "2", "--max-len", "4", "--dim", "8", "--num-layers", "1",
        "--num-random-negatives", "2", "--eval-candidate-chunk-size", "3", "--dropout", "0",
    ])
    train(args)
    history = json.loads((output / "history.json").read_text())
    assert [row["stage"] for row in history] == ["warmup", "router", "router"]
    assert [row["lambda_res"] for row in history] == [0, 0, .03]
    assert history[0]["train_router"]["probability_distribution"] == pytest.approx([1/3] * 3)
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert checkpoint["epoch"] > 1
    assert not checkpoint["model"]["stop_head.weight"].eq(0).all()
    result = json.loads((output / "test_metrics.json").read_text())
    assert set(result["test"]["ranking"]) == {"hard", "soft", "1", "2", "3"}
    assert result["best_valid_NDCG@10"] == max(row["valid"]["ranking"]["hard"]["NDCG@10"] for row in history[1:])
    # Exercise the standalone inference entry point against the saved checkpoint.
    import subprocess
    subprocess.run([sys.executable, "-m", "CCSR.evaluate", "--checkpoint", str(output / "best.pt"),
                    "--device", "cpu", "--candidate-chunk-size", "2", "--output", str(tmp_path / "eval.json")],
                   check=True, capture_output=True, cwd=Path(__file__).resolve().parents[2])
    reeval = json.loads((tmp_path / "eval.json").read_text())
    assert reeval["ranking"] == result["test"]["ranking"]
