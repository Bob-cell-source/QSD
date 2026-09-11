#!/usr/bin/env python3
"""Diagnostics for whether LoCoRec Soft SID really changes Hard SID sharing.

The script is intentionally read-only with respect to checkpoints. It rebuilds
the LoCoRec model from a saved run, loads best.pt, and reports:

1. hard-token mass after learned candidate attention/correction;
2. argmax flip rate away from the original hard token;
3. active candidate count;
4. per-item topology-change score D_i;
5. optional D_i vs per-target delta NDCG@10 if a matching HardSID checkpoint is
   provided.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "LoCoRecLOOCorrection"
sys.path.insert(0, str(CODE_ROOT))
LEGACY_CODE_ROOT = ROOT / "LCSoftCRSID"
sys.path.insert(0, str(LEGACY_CODE_ROOT))

from locorec.data import NextItemDataset, collate_eval  # noqa: E402
from locorec.io import read_json  # noqa: E402
from locorec.model import HardSIDFusion, LoCoRec  # noqa: E402
from locorec.soft_sid import (  # noqa: E402
    SoftSIDConfig,
    build_semantic_table,
    build_soft_sid_table,
    build_train_item_frequency,
)
from lcsoftcrsid.model import LCSoftCRSID  # noqa: E402
from lcsoftcrsid.soft_sid import (  # noqa: E402
    SoftSIDConfig as LegacySoftSIDConfig,
    build_semantic_table as legacy_build_semantic_table,
    build_soft_sid_table as legacy_build_soft_sid_table,
    build_train_item_frequency as legacy_build_train_item_frequency,
)


def _load_checkpoint(run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def _get_arg(args: dict[str, Any], key: str, default: Any) -> Any:
    return args[key] if key in args and args[key] is not None else default


def _build_common_tables(args: dict[str, Any]):
    dataset_dir = Path(args["dataset_dir"])
    sequences = read_json(dataset_dir / "sequences.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(args["semantic_ids"])
    num_items = int(stats["num_items"])
    hard_table, item_codes, num_semantic_tokens = build_semantic_table(
        semantic_obj, num_items
    )
    item_frequency = build_train_item_frequency(sequences, num_items)
    return sequences, num_items, hard_table, item_codes, num_semantic_tokens, item_frequency


def _build_locorec(run_dir: Path, device: torch.device) -> tuple[LoCoRec, dict[str, Any], int]:
    checkpoint = _load_checkpoint(run_dir)
    args = checkpoint["args"]
    sequences, num_items, hard_table, item_codes, num_semantic_tokens, item_frequency = (
        _build_common_tables(args)
    )
    soft_ids, priors, local_consistency, _hard_consistency, correction_features = (
        build_soft_sid_table(
            hard_table,
            item_codes,
            SoftSIDConfig(
                top_m=int(_get_arg(args, "soft_top_m", 4)),
                loo_min_overlap_slots=int(_get_arg(args, "loo_min_overlap_slots", 2)),
                min_support=float(_get_arg(args, "soft_min_support", 0.05)),
                min_conditional_lift=float(
                    _get_arg(args, "soft_min_conditional_lift", 0.0)
                ),
                max_neighbors=int(_get_arg(args, "soft_max_neighbors", 50)),
                tie_break_seed=int(_get_arg(args, "seed", 2026)),
            ),
        )
    )
    model = LoCoRec(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        soft_sid_table=soft_ids,
        candidate_prior=priors,
        correction_features=correction_features,
        local_consistency=local_consistency,
        item_frequency=item_frequency,
        dim=int(_get_arg(args, "dim", 128)),
        max_len=int(_get_arg(args, "max_len", 50)),
        num_heads=int(_get_arg(args, "num_heads", 2)),
        num_layers=int(_get_arg(args, "num_layers", 2)),
        dropout=float(_get_arg(args, "dropout", 0.2)),
        tail_tau=float(_get_arg(args, "tail_tau", 20.0)),
        residual_scale=float(_get_arg(args, "residual_scale", 1.0)),
        gate_correction_scale=float(_get_arg(args, "gate_correction_scale", 0.3)),
        gate_private_margin=float(_get_arg(args, "gate_private_margin", 0.05)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, args, num_items


def _build_legacy_locorec(
    run_dir: Path, device: torch.device
) -> tuple[LCSoftCRSID, dict[str, Any], int]:
    checkpoint = _load_checkpoint(run_dir)
    args = checkpoint["args"]
    dataset_dir = Path(args["dataset_dir"])
    sequences = read_json(dataset_dir / "sequences.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(args["semantic_ids"])
    num_items = int(stats["num_items"])
    hard_table, item_codes, num_semantic_tokens = legacy_build_semantic_table(
        semantic_obj, num_items
    )
    soft_ids, soft_weights, reliability = legacy_build_soft_sid_table(
        hard_table,
        item_codes,
        LegacySoftSIDConfig(
            top_m=int(_get_arg(args, "soft_top_m", _get_arg(args, "top_m", 4))),
            min_overlap_slots=int(
                _get_arg(args, "soft_min_overlap_slots", _get_arg(args, "min_overlap_slots", 3))
            ),
            min_support=float(_get_arg(args, "soft_min_support", _get_arg(args, "min_support", 0.05))),
            reliability_floor=float(_get_arg(args, "reliability_floor", 0.10)),
            max_neighbors=int(_get_arg(args, "soft_max_neighbors", _get_arg(args, "max_neighbors", 50))),
        ),
    )
    item_frequency = legacy_build_train_item_frequency(sequences, num_items)
    model = LCSoftCRSID(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        soft_sid_table=soft_ids,
        soft_sid_weights=soft_weights,
        semantic_reliability=reliability,
        item_frequency=item_frequency,
        dim=int(_get_arg(args, "dim", 128)),
        max_len=int(_get_arg(args, "max_len", 50)),
        num_heads=int(_get_arg(args, "num_heads", 2)),
        num_layers=int(_get_arg(args, "num_layers", 2)),
        dropout=float(_get_arg(args, "dropout", 0.2)),
        tail_tau=float(_get_arg(args, "tail_tau", 20.0)),
        alpha_mode=_get_arg(args, "alpha_mode", "fixed"),
        fusion_mode=_get_arg(args, "fusion_mode", "fixed"),
        residual_scale=float(_get_arg(args, "residual_scale", 1.0)),
        gate_correction_scale=float(_get_arg(args, "gate_correction_scale", 1.0)),
        gate_private_margin=float(_get_arg(args, "gate_private_margin", 0.0)),
        candidate_weight_mode=_get_arg(args, "candidate_weight_mode", "prior_guided"),
        disable_semantic_basis=bool(_get_arg(args, "disable_semantic_basis", False)),
        disable_shared_residual=bool(_get_arg(args, "disable_shared_residual", False)),
        disable_private_residual=bool(_get_arg(args, "disable_private_residual", False)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, args, num_items


def _build_soft_model(run_dir: Path, device: torch.device):
    try:
        model, args, num_items = _build_locorec(run_dir, device)
        return model, args, num_items, "hard_centered"
    except RuntimeError as exc:
        text = str(exc)
        if "soft_correction_gate" not in text and "correction_features" not in text:
            raise
    model, args, num_items = _build_legacy_locorec(run_dir, device)
    return model, args, num_items, "legacy_attention"


def _build_hard(run_dir: Path, device: torch.device) -> tuple[HardSIDFusion, dict[str, Any], int]:
    checkpoint = _load_checkpoint(run_dir)
    args = checkpoint["args"]
    _sequences, num_items, hard_table, _item_codes, num_semantic_tokens, _freq = (
        _build_common_tables(args)
    )
    model = HardSIDFusion(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        hard_sid_table=hard_table,
        dim=int(_get_arg(args, "dim", 128)),
        max_len=int(_get_arg(args, "max_len", 50)),
        num_heads=int(_get_arg(args, "num_heads", 2)),
        num_layers=int(_get_arg(args, "num_layers", 2)),
        dropout=float(_get_arg(args, "dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, args, num_items


@torch.no_grad()
def soft_distribution(model: LoCoRec, batch_size: int, device: torch.device):
    encoder = model.item_encoder
    num_items = encoder.soft_sid_table.size(0) - 1
    depth = encoder.soft_sid_table.size(1)
    top_m = encoder.soft_sid_table.size(2)

    all_items = torch.arange(1, num_items + 1, device=device)
    hard_mass_chunks = []
    flip_chunks = []
    active_counts_chunks = []
    dist_chunks = []
    token_chunks = []

    for start in range(0, num_items, batch_size):
        items = all_items[start : start + batch_size]
        tokens = encoder.soft_sid_table[items]
        attention, _entropy = encoder.candidate_weights(items)
        has_alt = tokens[..., 1:].ne(0).any(dim=-1)
        correction = torch.sigmoid(
            encoder.soft_correction_gate(encoder.correction_features[items])
        ).squeeze(-1)
        correction = correction * has_alt

        dist = torch.zeros_like(attention)
        dist[..., 0] = 1.0 - correction
        alt_attention = attention[..., 1:] * tokens[..., 1:].ne(0)
        alt_attention = alt_attention / alt_attention.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        dist[..., 1:] = correction.unsqueeze(-1) * alt_attention
        no_alt = ~has_alt
        dist[..., 0] = torch.where(no_alt, torch.ones_like(dist[..., 0]), dist[..., 0])
        dist[..., 1:] = torch.where(
            no_alt.unsqueeze(-1), torch.zeros_like(dist[..., 1:]), dist[..., 1:]
        )

        hard_mass_chunks.append(dist[..., 0].detach().cpu())
        flip_chunks.append(dist.argmax(dim=-1).ne(0).detach().cpu())
        active_counts_chunks.append((tokens.ne(0) & encoder.candidate_prior[items].gt(0)).sum(dim=-1).cpu())
        dist_chunks.append(dist.detach().cpu())
        token_chunks.append(tokens.detach().cpu())

    return {
        "prior_hard_mass": encoder.candidate_prior[1:, :, 0].detach().cpu(),
        "prior_dist": encoder.candidate_prior[1:].detach().cpu(),
        "hard_mass": torch.cat(hard_mass_chunks, dim=0),
        "flip": torch.cat(flip_chunks, dim=0),
        "active_counts": torch.cat(active_counts_chunks, dim=0),
        "dist": torch.cat(dist_chunks, dim=0),
        "tokens": torch.cat(token_chunks, dim=0),
        "depth": depth,
        "top_m": top_m,
    }


@torch.no_grad()
def legacy_soft_distribution(model: LCSoftCRSID, batch_size: int, device: torch.device):
    encoder = model.item_encoder
    num_items = encoder.soft_sid_table.size(0) - 1
    depth = encoder.soft_sid_table.size(1)
    top_m = encoder.soft_sid_table.size(2)

    all_items = torch.arange(1, num_items + 1, device=device)
    hard_mass_chunks = []
    flip_chunks = []
    active_counts_chunks = []
    dist_chunks = []
    token_chunks = []

    for start in range(0, num_items, batch_size):
        items = all_items[start : start + batch_size]
        tokens = encoder.soft_sid_table[items]
        weights, *_ = encoder.candidate_weights(items)
        mask = tokens.ne(0) & encoder.soft_sid_weights[items].gt(0)
        weights = weights * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hard_mass_chunks.append(weights[..., 0].detach().cpu())
        flip_chunks.append(weights.argmax(dim=-1).ne(0).detach().cpu())
        active_counts_chunks.append(mask.sum(dim=-1).cpu())
        dist_chunks.append(weights.detach().cpu())
        token_chunks.append(tokens.detach().cpu())

    return {
        "prior_hard_mass": encoder.soft_sid_weights[1:, :, 0].detach().cpu(),
        "prior_dist": encoder.soft_sid_weights[1:].detach().cpu(),
        "hard_mass": torch.cat(hard_mass_chunks, dim=0),
        "flip": torch.cat(flip_chunks, dim=0),
        "active_counts": torch.cat(active_counts_chunks, dim=0),
        "dist": torch.cat(dist_chunks, dim=0),
        "tokens": torch.cat(token_chunks, dim=0),
        "depth": depth,
        "top_m": top_m,
    }


def _summarize(values: torch.Tensor) -> dict[str, float]:
    flat = values.float().flatten()
    return {
        "mean": float(flat.mean()),
        "p50": float(flat.quantile(0.50)),
        "p90": float(flat.quantile(0.90)),
        "p95": float(flat.quantile(0.95)),
        "p99": float(flat.quantile(0.99)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def topology_change(diag: dict[str, Any], batch_size: int) -> torch.Tensor:
    tokens = diag["tokens"]
    dist = diag["dist"]
    num_items, depth, _top_m = tokens.shape
    hard_tokens = tokens[:, :, 0]
    d_values = torch.zeros(num_items)

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        soft_sim = torch.zeros(end - start, num_items)
        hard_sim = torch.zeros(end - start, num_items)
        for slot in range(depth):
            eq = tokens[start:end, slot, :, None].eq(hard_tokens[:, slot])
            soft_vs_hard_j = (dist[start:end, slot, :, None] * eq).sum(dim=1)
            hard_vs_hard_j = hard_tokens[start:end, slot, None].eq(
                hard_tokens[:, slot]
            ).float()
            soft_sim += soft_vs_hard_j
            hard_sim += hard_vs_hard_j
        soft_sim /= depth
        hard_sim /= depth
        d_values[start:end] = (soft_sim - hard_sim).abs().sum(dim=1)
    return d_values


@torch.no_grad()
def per_sample_ndcg10(
    model,
    args: dict[str, Any],
    num_items: int,
    device: torch.device,
    candidate_chunk_size: int,
) -> list[tuple[int, float]]:
    sequences = read_json(Path(args["dataset_dir"]) / "sequences.json")
    dataset = NextItemDataset(sequences, int(_get_arg(args, "max_len", 50)), "test")
    loader = DataLoader(
        dataset,
        batch_size=int(_get_arg(args, "batch_size", 1024)),
        shuffle=False,
        collate_fn=collate_eval,
    )
    all_items = torch.arange(1, num_items + 1, device=device)
    out: list[tuple[int, float]] = []

    for sequences_batch, targets, full_histories in loader:
        sequences_batch = sequences_batch.to(device)
        targets = targets.to(device)
        chunks = []
        if hasattr(model, "encode_sequence"):
            user_vectors, _ = model.encode_sequence(sequences_batch)
            for start in range(0, num_items, candidate_chunk_size):
                candidates = all_items[start : start + candidate_chunk_size]
                candidate_vectors = model.item_encoder(candidates)["vectors"]
                chunks.append(user_vectors @ candidate_vectors.transpose(0, 1))
            scores = torch.cat(chunks, dim=1)
        else:
            for start in range(0, num_items, candidate_chunk_size):
                candidates = all_items[start : start + candidate_chunk_size]
                chunks.append(model.full_catalog_forward(sequences_batch, candidates)["score"])
            scores = torch.cat(chunks, dim=1)
        for row, (history, target) in enumerate(zip(full_histories, targets.tolist())):
            seen_items = {int(item) for item in history if int(item) != target}
            if seen_items:
                columns = torch.tensor(
                    [item - 1 for item in seen_items], dtype=torch.long, device=device
                )
                scores[row, columns] = float("-inf")
        top_items = scores.topk(k=10, dim=1).indices + 1
        matches = top_items.eq(targets.unsqueeze(1))
        hit = matches.any(dim=1)
        rank = matches.float().argmax(dim=1) + 1
        ndcg = hit.float() / torch.log2(rank.float() + 1.0)
        out.extend((int(t), float(v)) for t, v in zip(targets.cpu(), ndcg.cpu()))
    return out


def write_item_csv(path: Path, d_values: torch.Tensor, delta_by_item: dict[int, float] | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["item_id", "topology_D"]
        if delta_by_item is not None:
            header.append("delta_ndcg10_full_minus_hard")
        writer.writerow(header)
        for idx, value in enumerate(d_values.tolist(), start=1):
            row: list[Any] = [idx, value]
            if delta_by_item is not None:
                row.append(delta_by_item.get(idx, ""))
            writer.writerow(row)


def write_scatter_plot(
    path: Path, d_values: torch.Tensor, delta_by_item: dict[int, float]
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = []
    ys = []
    for item, delta in delta_by_item.items():
        xs.append(float(d_values[item - 1]))
        ys.append(float(delta))
    if not xs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 3.8))
    plt.scatter(xs, ys, s=8, alpha=0.35, edgecolors="none")
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel(r"Topology change $D_i$")
    plt.ylabel(r"$\Delta$NDCG@10 (Full - HardSID)")
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _group_delta_by_item_property(
    delta_by_item: dict[int, float],
    property_by_item: dict[int, Any],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for item, delta in delta_by_item.items():
        key = str(property_by_item.get(item, "missing"))
        grouped.setdefault(key, []).append(float(delta))
    return {
        key: {
            "count": len(values),
            "mean_delta_ndcg10": float(sum(values) / len(values)),
        }
        for key, values in sorted(grouped.items(), key=lambda row: row[0])
        if values
    }


def _group_sample_deltas_by_item_property(
    sample_deltas: list[tuple[int, float]],
    property_by_item: dict[int, Any],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for item, delta in sample_deltas:
        key = str(property_by_item.get(item, "missing"))
        grouped.setdefault(key, []).append(float(delta))
    return {
        key: {
            "sample_count": len(values),
            "mean_delta_ndcg10_micro": float(sum(values) / len(values)),
        }
        for key, values in sorted(grouped.items(), key=lambda row: row[0])
        if values
    }


def _auc_score(scores: list[float], labels: list[int]) -> float | None:
    positives = [(score, label) for score, label in zip(scores, labels) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    ranked = sorted(enumerate(scores), key=lambda row: row[1])
    ranks = [0.0] * len(scores)
    idx = 0
    while idx < len(ranked):
        j = idx + 1
        while j < len(ranked) and ranked[j][1] == ranked[idx][1]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[ranked[k][0]] = avg_rank
        idx = j
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    n_pos = len(positives)
    n_neg = len(negatives)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def advanced_group_analyses(
    diag: dict[str, Any],
    delta_by_item: dict[int, float],
    sample_deltas: list[tuple[int, float]],
    embedding_path: Path,
    embedding_item_ids_path: Path,
) -> dict[str, Any]:
    tokens = diag["tokens"]
    dist = diag["dist"].float()
    active_counts = diag["active_counts"]
    hard_mass = diag["hard_mass"].float()
    num_items, depth, top_m = tokens.shape
    has_alt = active_counts.gt(1)
    flip = dist.argmax(dim=-1).ne(0) & has_alt

    active_level_count = has_alt.sum(dim=1)
    flip_level_count = flip.sum(dim=1)
    active_group = {}
    flip_group = {}
    for item in range(1, num_items + 1):
        active = int(active_level_count[item - 1])
        active_key = str(active) if active < 4 else "4"
        active_group[item] = active_key
        flips = int(flip_level_count[item - 1])
        if active == 0:
            flip_key = "0_no_active"
        elif flips == 0:
            flip_key = "1_active_no_flip"
        elif flips == 1:
            flip_key = "2_single_level_flip"
        else:
            flip_key = "3_multi_level_flip"
        flip_group[item] = flip_key

    result: dict[str, Any] = {
        "delta_by_active_level_count_macro_item": _group_delta_by_item_property(
            delta_by_item, active_group
        ),
        "delta_by_active_level_count_micro_sample": _group_sample_deltas_by_item_property(
            sample_deltas, active_group
        ),
        "delta_by_flip_group_macro_item": _group_delta_by_item_property(delta_by_item, flip_group),
        "delta_by_flip_group_micro_sample": _group_sample_deltas_by_item_property(
            sample_deltas, flip_group
        ),
        "active_slot_hard_mass_by_level": [
            _summarize(hard_mass[:, slot][has_alt[:, slot]])
            if has_alt[:, slot].any()
            else None
            for slot in range(depth)
        ],
    }

    if not embedding_path.exists() or not embedding_item_ids_path.exists():
        result["candidate_quality"] = {
            "skipped": True,
            "reason": "missing item text embeddings or item-id mapping",
        }
        return result

    import numpy as np

    embeddings = np.load(embedding_path).astype("float32")
    with embedding_item_ids_path.open("r", encoding="utf-8") as handle:
        raw_ids = [int(value) for value in json.load(handle)]
    if len(raw_ids) != embeddings.shape[0]:
        result["candidate_quality"] = {
            "skipped": True,
            "reason": "embedding rows do not match item IDs",
        }
        return result
    row_for_item = {item: idx for idx, item in enumerate(raw_ids)}
    if any(item not in row_for_item for item in range(1, num_items + 1)):
        result["candidate_quality"] = {
            "skipped": True,
            "reason": "embedding item IDs do not cover all internal item IDs",
        }
        return result
    ordered = embeddings[[row_for_item[item] for item in range(1, num_items + 1)]]
    norm = np.linalg.norm(ordered, axis=1, keepdims=True)
    ordered = ordered / np.maximum(norm, 1e-12)

    peers_by_level_token: list[dict[int, list[int]]] = []
    for slot in range(depth):
        slot_map: dict[int, list[int]] = {}
        for item in range(1, num_items + 1):
            token = int(tokens[item - 1, slot, 0])
            slot_map.setdefault(token, []).append(item)
        peers_by_level_token.append(slot_map)

    rng = random.Random(2026)
    item_universe = list(range(1, num_items + 1))
    quality: dict[str, dict[str, list[float]]] = {
        "successful_flip": {"alternative": [], "hard_peer": [], "random": []},
        "harmful_flip": {"alternative": [], "hard_peer": [], "random": []},
        "neutral_flip": {"alternative": [], "hard_peer": [], "random": []},
        "no_flip": {"alternative": [], "hard_peer": [], "random": []},
    }
    delta_sim_by_item: dict[int, list[float]] = {}
    margin_by_item: dict[int, list[float]] = {}

    def peer_mean_similarity(item: int, peers: list[int]) -> float | None:
        valid = [peer for peer in peers if peer != item and 1 <= peer <= num_items]
        if not valid:
            return None
        target = ordered[item - 1]
        peer_matrix = ordered[[peer - 1 for peer in valid]]
        return float((peer_matrix @ target).mean())

    def random_mean_similarity(item: int, count: int) -> float | None:
        count = max(1, min(count, 50))
        candidates = [peer for peer in item_universe if peer != item]
        sample = rng.sample(candidates, min(count, len(candidates)))
        return peer_mean_similarity(item, sample)

    for item, delta in delta_by_item.items():
        idx = item - 1
        item_flips = int(flip_level_count[idx])
        if item_flips > 0 and delta > 0:
            bucket = "successful_flip"
        elif item_flips > 0 and delta < 0:
            bucket = "harmful_flip"
        elif item_flips > 0:
            bucket = "neutral_flip"
        elif int(active_level_count[idx]) > 0:
            bucket = "no_flip"
        else:
            continue

        for slot in range(depth):
            if not bool(has_alt[idx, slot]):
                continue
            if bool(flip[idx, slot]):
                alt_rank = int(dist[idx, slot].argmax())
            else:
                alt_scores = dist[idx, slot, 1:].clone()
                alt_scores[tokens[idx, slot, 1:].eq(0)] = -1
                if float(alt_scores.max()) < 0:
                    continue
                alt_rank = 1 + int(alt_scores.argmax())
            if alt_rank <= 0 or alt_rank >= top_m:
                continue
            alt_token = int(tokens[idx, slot, alt_rank])
            hard_token = int(tokens[idx, slot, 0])
            alt_peers = peers_by_level_token[slot].get(alt_token, [])
            hard_peers = peers_by_level_token[slot].get(hard_token, [])
            alt_sim = peer_mean_similarity(item, alt_peers)
            hard_sim = peer_mean_similarity(item, hard_peers)
            random_sim = random_mean_similarity(item, len(alt_peers))
            if alt_sim is not None:
                quality[bucket]["alternative"].append(alt_sim)
            if hard_sim is not None:
                quality[bucket]["hard_peer"].append(hard_sim)
            if random_sim is not None:
                quality[bucket]["random"].append(random_sim)
            if alt_sim is not None and hard_sim is not None:
                delta_sim_by_item.setdefault(item, []).append(alt_sim - hard_sim)
            prior = diag["prior_hard_mass"]
            alt_prior = dist.new_tensor(0.0)
            if top_m > 1:
                valid_alt = tokens[idx, slot, 1:].ne(0)
                if bool(valid_alt.any()):
                    alt_prior = prior.new_tensor(0.0)
                    if "prior_dist" in diag:
                        alt_prior = diag["prior_dist"][idx, slot, 1:][valid_alt].max()
            margin_by_item.setdefault(item, []).append(
                float(prior[idx, slot] - alt_prior)
            )

    result["candidate_quality"] = {
        bucket: {
            "slot_count": len(values["alternative"]),
            "alternative_mean_bge_cosine": _mean(values["alternative"]),
            "hard_peer_mean_bge_cosine": _mean(values["hard_peer"]),
            "random_mean_bge_cosine": _mean(values["random"]),
        }
        for bucket, values in quality.items()
    }
    item_delta_sim = {
        item: float(sum(values) / len(values))
        for item, values in delta_sim_by_item.items()
        if values
    }
    item_margin = {
        item: float(sum(values) / len(values))
        for item, values in margin_by_item.items()
        if values
    }
    sample_scores = []
    sample_labels = []
    sample_delta_pairs = []
    for item, delta in sample_deltas:
        if item not in item_delta_sim:
            continue
        score = item_delta_sim[item]
        sample_scores.append(score)
        sample_labels.append(1 if delta > 0 else 0)
        sample_delta_pairs.append((score, delta, item))
    sorted_pairs = sorted(sample_delta_pairs, key=lambda row: row[0])
    quartiles: dict[str, dict[str, float]] = {}
    if sorted_pairs:
        n = len(sorted_pairs)
        for q in range(4):
            begin = q * n // 4
            end = (q + 1) * n // 4
            values = sorted_pairs[begin:end]
            if values:
                quartiles[f"Q{q + 1}"] = {
                    "sample_count": len(values),
                    "delta_sim_min": float(values[0][0]),
                    "delta_sim_max": float(values[-1][0]),
                    "mean_delta_sim": float(sum(row[0] for row in values) / len(values)),
                    "mean_delta_ndcg10": float(sum(row[1] for row in values) / len(values)),
                    "positive_delta_rate": float(
                        sum(1 for row in values if row[1] > 0) / len(values)
                    ),
                }
    result["delta_sim_predictiveness"] = {
        "sample_count": len(sample_scores),
        "auc_for_positive_delta_ndcg10": _auc_score(sample_scores, sample_labels),
        "quartile_gain_by_delta_sim": quartiles,
    }

    two_d: dict[str, list[float]] = {}
    scored = [
        (item_delta_sim[item], item_margin.get(item), delta)
        for item, delta in sample_deltas
        if item in item_delta_sim and item in item_margin
    ]
    if scored:
        sim_values = sorted(row[0] for row in scored)
        margin_values = sorted(row[1] for row in scored if row[1] is not None)
        sim_mid = sim_values[len(sim_values) // 2]
        margin_mid = margin_values[len(margin_values) // 2]
        for sim, margin, delta in scored:
            sim_key = "high_dsim" if sim >= sim_mid else "low_dsim"
            margin_key = "high_margin" if margin >= margin_mid else "low_margin"
            two_d.setdefault(f"{margin_key}__{sim_key}", []).append(delta)
        result["quantization_margin_x_delta_sim"] = {
            key: {
                "sample_count": len(values),
                "mean_delta_ndcg10": float(sum(values) / len(values)),
            }
            for key, values in sorted(two_d.items())
        }
        result["quantization_margin_x_delta_sim"]["split"] = {
            "margin_median": float(margin_mid),
            "delta_sim_median": float(sim_mid),
        }
    return result


@torch.no_grad()
def evaluate_metrics_ndcg10(
    model,
    args: dict[str, Any],
    num_items: int,
    device: torch.device,
    candidate_chunk_size: int,
) -> dict[str, float]:
    scores = per_sample_ndcg10(model, args, num_items, device, candidate_chunk_size)
    total = len(scores)
    return {
        "NDCG@10": float(sum(value for _item, value in scores) / max(total, 1)),
        "num_test_samples": total,
    }


def make_hard_routing_intervention(model):
    intervened = copy.deepcopy(model)
    encoder = intervened.item_encoder
    with torch.no_grad():
        if hasattr(encoder, "candidate_prior"):
            encoder.candidate_prior[..., 1:] = 0.0
            encoder.candidate_prior[..., 0] = encoder.soft_sid_table[..., 0].ne(0).float()
            encoder.soft_sid_table[..., 1:] = 0
        elif hasattr(encoder, "soft_sid_weights"):
            encoder.soft_sid_weights[..., 1:] = 0.0
            encoder.soft_sid_weights[..., 0] = encoder.soft_sid_table[..., 0].ne(0).float()
            encoder.soft_sid_table[..., 1:] = 0
    intervened.eval()
    return intervened


def task_aware_evidence_vs_route_delta(
    diag: dict[str, Any],
    route_sample_deltas: list[tuple[int, float]],
    dataset_dir: Path,
) -> dict[str, Any]:
    tokens = diag["tokens"]
    dist = diag["dist"].float()
    active_counts = diag["active_counts"]
    prior = diag["prior_dist"].float()
    num_items, depth, top_m = tokens.shape
    has_alt = active_counts.gt(1)

    sequences = read_json(dataset_dir / "sequences.json")

    import numpy as np

    embedding_path = dataset_dir / "item_text_embeddings.npy"
    embedding_item_ids_path = dataset_dir / "embedding_item_ids.json"
    if not embedding_path.exists() or not embedding_item_ids_path.exists():
        return {
            "skipped": True,
            "reason": "missing item_text_embeddings.npy or embedding_item_ids.json",
        }

    embeddings = np.load(embedding_path).astype("float32")
    with embedding_item_ids_path.open("r", encoding="utf-8") as handle:
        raw_ids = [int(value) for value in json.load(handle)]
    row_for_item = {item: idx for idx, item in enumerate(raw_ids)}
    if len(raw_ids) != embeddings.shape[0] or any(
        item not in row_for_item for item in range(1, num_items + 1)
    ):
        return {
            "skipped": True,
            "reason": "embedding item IDs are not aligned with internal item IDs",
        }
    sem_matrix = embeddings[[row_for_item[item] for item in range(1, num_items + 1)]]
    sem_matrix = sem_matrix / np.maximum(
        np.linalg.norm(sem_matrix, axis=1, keepdims=True), 1e-12
    )

    user_matrix = np.zeros((num_items, len(sequences)), dtype="float32")
    trans_matrix = np.zeros((num_items, num_items * 2), dtype="float32")
    for user_idx, row in enumerate(sequences):
        train_items = [int(item) for item in row["items"][:-2]]
        for item in train_items:
            if 1 <= item <= num_items:
                user_matrix[item - 1, user_idx] += 1.0
        for prev, nxt in zip(train_items[:-1], train_items[1:]):
            if 1 <= prev <= num_items and 1 <= nxt <= num_items:
                trans_matrix[nxt - 1, prev - 1] += 1.0
                trans_matrix[prev - 1, num_items + nxt - 1] += 1.0
    user_matrix = user_matrix / np.maximum(
        np.linalg.norm(user_matrix, axis=1, keepdims=True), 1e-12
    )
    trans_matrix = trans_matrix / np.maximum(
        np.linalg.norm(trans_matrix, axis=1, keepdims=True), 1e-12
    )

    peers_by_level_token: list[dict[int, list[int]]] = []
    for slot in range(depth):
        slot_map: dict[int, list[int]] = {}
        for item in range(1, num_items + 1):
            token = int(tokens[item - 1, slot, 0])
            slot_map.setdefault(token, []).append(item)
        peers_by_level_token.append(slot_map)

    def peer_mean_similarity(
        matrix: np.ndarray, item: int, peers: list[int]
    ) -> float | None:
        valid = [peer for peer in peers if peer != item and 1 <= peer <= num_items]
        if not valid:
            return None
        return float((matrix[[peer - 1 for peer in valid]] @ matrix[item - 1]).mean())

    item_features: dict[int, dict[str, float]] = {}
    for item in range(1, num_items + 1):
        idx = item - 1
        values = {
            "delta_sem": [],
            "delta_coll": [],
            "delta_trans": [],
            "quantization_margin": [],
            "learned_alt_mass": [],
        }
        for slot in range(depth):
            if not bool(has_alt[idx, slot]):
                continue
            alt_scores = dist[idx, slot, 1:].clone()
            alt_scores[tokens[idx, slot, 1:].eq(0)] = -1
            if float(alt_scores.max()) < 0:
                continue
            alt_rank = 1 + int(alt_scores.argmax())
            alt_token = int(tokens[idx, slot, alt_rank])
            hard_token = int(tokens[idx, slot, 0])
            alt_peers = peers_by_level_token[slot].get(alt_token, [])
            hard_peers = peers_by_level_token[slot].get(hard_token, [])

            alt_sem = peer_mean_similarity(sem_matrix, item, alt_peers)
            hard_sem = peer_mean_similarity(sem_matrix, item, hard_peers)
            alt_coll = peer_mean_similarity(user_matrix, item, alt_peers)
            hard_coll = peer_mean_similarity(user_matrix, item, hard_peers)
            alt_trans = peer_mean_similarity(trans_matrix, item, alt_peers)
            hard_trans = peer_mean_similarity(trans_matrix, item, hard_peers)
            if alt_sem is not None and hard_sem is not None:
                values["delta_sem"].append(alt_sem - hard_sem)
            if alt_coll is not None and hard_coll is not None:
                values["delta_coll"].append(alt_coll - hard_coll)
            if alt_trans is not None and hard_trans is not None:
                values["delta_trans"].append(alt_trans - hard_trans)
            valid_alt = tokens[idx, slot, 1:].ne(0)
            if bool(valid_alt.any()):
                values["quantization_margin"].append(
                    float(prior[idx, slot, 0] - prior[idx, slot, 1:][valid_alt].max())
                )
            values["learned_alt_mass"].append(float(1.0 - dist[idx, slot, 0]))

        reduced = {
            key: float(sum(vals) / len(vals))
            for key, vals in values.items()
            if vals
        }
        if reduced:
            item_features[item] = reduced

    def metric_report(metric: str) -> dict[str, Any]:
        pairs = [
            (item_features[item][metric], delta)
            for item, delta in route_sample_deltas
            if item in item_features and metric in item_features[item]
        ]
        labels = [1 if delta > 0 else 0 for _score, delta in pairs]
        scores = [score for score, _delta in pairs]
        sorted_pairs = sorted(pairs, key=lambda row: row[0])
        quartiles: dict[str, dict[str, float]] = {}
        if sorted_pairs:
            n = len(sorted_pairs)
            for q in range(4):
                begin = q * n // 4
                end = (q + 1) * n // 4
                rows = sorted_pairs[begin:end]
                if rows:
                    quartiles[f"Q{q + 1}"] = {
                        "sample_count": len(rows),
                        f"{metric}_min": float(rows[0][0]),
                        f"{metric}_max": float(rows[-1][0]),
                        f"mean_{metric}": float(
                            sum(score for score, _delta in rows) / len(rows)
                        ),
                        "mean_route_delta_ndcg10": float(
                            sum(delta for _score, delta in rows) / len(rows)
                        ),
                        "positive_route_delta_rate": float(
                            sum(1 for _score, delta in rows if delta > 0) / len(rows)
                        ),
                    }
        return {
            "sample_count": len(pairs),
            "auc_for_positive_route_delta": _auc_score(scores, labels),
            "pearson_with_route_delta": float(
                torch.nan_to_num(
                    torch.corrcoef(
                        torch.stack(
                            [
                                torch.tensor(scores, dtype=torch.float),
                                torch.tensor(
                                    [delta for _score, delta in pairs],
                                    dtype=torch.float,
                                ),
                            ]
                        )
                    )[0, 1]
                )
            )
            if len(pairs) >= 2
            else None,
            "quartile_gain": quartiles,
        }

    reports = {
        metric: metric_report(metric)
        for metric in [
            "delta_sem",
            "delta_coll",
            "delta_trans",
            "quantization_margin",
            "learned_alt_mass",
        ]
    }

    metrics_for_2d = ["delta_sem", "delta_coll", "delta_trans"]
    two_d: dict[str, Any] = {}
    for first in metrics_for_2d:
        for second in metrics_for_2d:
            if first >= second:
                continue
            rows = [
                (item_features[item][first], item_features[item][second], delta)
                for item, delta in route_sample_deltas
                if item in item_features
                and first in item_features[item]
                and second in item_features[item]
            ]
            if not rows:
                continue
            first_mid = sorted(row[0] for row in rows)[len(rows) // 2]
            second_mid = sorted(row[1] for row in rows)[len(rows) // 2]
            grouped: dict[str, list[float]] = {}
            for a, b, delta in rows:
                key = (
                    ("high_" if a >= first_mid else "low_")
                    + first
                    + "__"
                    + ("high_" if b >= second_mid else "low_")
                    + second
                )
                grouped.setdefault(key, []).append(delta)
            two_d[f"{first}_x_{second}"] = {
                "split": {f"{first}_median": float(first_mid), f"{second}_median": float(second_mid)},
                "groups": {
                    key: {
                        "sample_count": len(values),
                        "mean_route_delta_ndcg10": float(sum(values) / len(values)),
                    }
                    for key, values in sorted(grouped.items())
                },
            }

    return {
        "target": "same-checkpoint route delta: NDCG_soft - NDCG_hard-routing-intervention",
        "num_items_with_any_evidence": len(item_features),
        "metrics": reports,
        "two_dimensional_groups": two_d,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locorec-run", required=True, type=Path)
    parser.add_argument("--hard-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topology-batch-size", type=int, default=128)
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    locorec, locorec_args, num_items, distribution_mode = _build_soft_model(
        args.locorec_run, device
    )
    if distribution_mode == "hard_centered":
        diag = soft_distribution(locorec, args.batch_size, device)
    else:
        diag = legacy_soft_distribution(locorec, args.batch_size, device)
    d_values = topology_change(diag, args.topology_batch_size)

    hard_mass = diag["hard_mass"]
    prior_hard_mass = diag["prior_hard_mass"]
    active_counts = diag["active_counts"]
    flip = diag["flip"]
    has_alt = active_counts.gt(1)
    candidate_distribution = {}
    total_slots = active_counts.numel()
    for count in range(1, int(diag["top_m"]) + 1):
        candidate_distribution[str(count)] = float(
            active_counts.eq(count).sum().item() / max(total_slots, 1)
        )

    summary: dict[str, Any] = {
        "locorec_run": str(args.locorec_run),
        "distribution_mode": distribution_mode,
        "num_items": num_items,
        "hard_token_average_mass": {
            "overall": _summarize(hard_mass),
            "by_level": [
                _summarize(hard_mass[:, slot]) for slot in range(hard_mass.size(1))
            ],
            "alternative_slots_only": _summarize(hard_mass[has_alt])
            if has_alt.any()
            else None,
        },
        "requested_active_slot_metrics": {
            "hard_mass_active_slot": _summarize(hard_mass[has_alt])
            if has_alt.any()
            else None,
            "prior_hard_mass_active_slot": _summarize(prior_hard_mass[has_alt])
            if has_alt.any()
            else None,
            "learned_hard_mass_active_slot": _summarize(hard_mass[has_alt])
            if has_alt.any()
            else None,
            "flip_rate_active_slot": float(flip[has_alt].float().mean())
            if has_alt.any()
            else None,
            "candidate_count_distribution": candidate_distribution,
        },
        "argmax_flip_rate": {
            "overall": float(flip.float().mean()),
            "by_level": [
                float(flip[:, slot].float().mean()) for slot in range(flip.size(1))
            ],
            "alternative_slots_only": float(flip[has_alt].float().mean())
            if has_alt.any()
            else None,
        },
        "active_candidate_count": {
            "overall": _summarize(active_counts),
            "item_level_mean": _summarize(active_counts.float().mean(dim=1)),
            "slot_fraction_gt1": float(has_alt.float().mean()),
            "item_fraction_any_gt1": float(has_alt.any(dim=1).float().mean()),
            "item_fraction_all_singleton": float((~has_alt).all(dim=1).float().mean()),
        },
        "topology_change_D": _summarize(d_values),
    }

    delta_by_item = None
    if args.hard_run is not None:
        hard, hard_args, hard_num_items = _build_hard(args.hard_run, device)
        if hard_num_items != num_items:
            raise ValueError("LoCoRec and HardSID runs have different num_items")
        full_scores = per_sample_ndcg10(
            locorec, locorec_args, num_items, device, args.candidate_chunk_size
        )
        hard_scores = per_sample_ndcg10(
            hard, hard_args, num_items, device, args.candidate_chunk_size
        )
        if len(full_scores) != len(hard_scores):
            raise ValueError("LoCoRec and HardSID test sets have different sizes")
        sample_deltas: list[tuple[int, float]] = []
        accum: dict[int, list[float]] = {}
        for (target_full, ndcg_full), (target_hard, ndcg_hard) in zip(
            full_scores, hard_scores
        ):
            if target_full != target_hard:
                raise ValueError("test target order mismatch")
            delta = ndcg_full - ndcg_hard
            sample_deltas.append((target_full, delta))
            accum.setdefault(target_full, []).append(delta)
        delta_by_item = {
            item: sum(values) / len(values) for item, values in accum.items()
        }
        micro_delta = sum(delta for _item, delta in sample_deltas) / max(
            len(sample_deltas), 1
        )
        macro_delta = sum(delta_by_item.values()) / max(len(delta_by_item), 1)
        weighted_item_delta = sum(sum(values) for values in accum.values()) / max(
            sum(len(values) for values in accum.values()), 1
        )
        summary["delta_ndcg10_aggregation_check"] = {
            "num_test_samples": len(sample_deltas),
            "num_unique_target_items": len(delta_by_item),
            "sample_micro_delta": float(micro_delta),
            "unique_item_macro_delta": float(macro_delta),
            "test_frequency_weighted_item_delta": float(weighted_item_delta),
            "full_sample_micro_ndcg10": float(
                sum(value for _item, value in full_scores) / max(len(full_scores), 1)
            ),
            "hard_sample_micro_ndcg10": float(
                sum(value for _item, value in hard_scores) / max(len(hard_scores), 1)
            ),
        }
        hard_intervention = make_hard_routing_intervention(locorec).to(device)
        intervention_scores = per_sample_ndcg10(
            hard_intervention,
            locorec_args,
            num_items,
            device,
            args.candidate_chunk_size,
        )
        if len(intervention_scores) != len(full_scores):
            raise ValueError("intervention test set has different size")
        intervention_deltas = []
        route_sample_deltas: list[tuple[int, float]] = []
        for (target_full, ndcg_full), (target_intervention, ndcg_intervention) in zip(
            full_scores, intervention_scores
        ):
            if target_full != target_intervention:
                raise ValueError("intervention target order mismatch")
            intervention_deltas.append(ndcg_full - ndcg_intervention)
            route_sample_deltas.append((target_full, ndcg_full - ndcg_intervention))
        summary["same_checkpoint_soft_to_hard_intervention"] = {
            "full_ndcg10": float(
                sum(value for _item, value in full_scores) / max(len(full_scores), 1)
            ),
            "hard_routing_intervention_ndcg10": float(
                sum(value for _item, value in intervention_scores)
                / max(len(intervention_scores), 1)
            ),
            "routing_contribution_delta_ndcg10": float(
                sum(intervention_deltas) / max(len(intervention_deltas), 1)
            ),
        }
        paired_d = []
        paired_delta = []
        for item, delta in delta_by_item.items():
            paired_d.append(float(d_values[item - 1]))
            paired_delta.append(delta)
        if len(paired_d) >= 2:
            d_tensor = torch.tensor(paired_d)
            delta_tensor = torch.tensor(paired_delta)
            corr = torch.corrcoef(torch.stack([d_tensor, delta_tensor]))[0, 1]
            summary["topology_D_vs_delta_ndcg10"] = {
            "num_test_target_items": len(paired_d),
                "pearson": float(torch.nan_to_num(corr)),
                "mean_delta_ndcg10": float(delta_tensor.mean()),
                "note": "unique-item macro; use delta_ndcg10_aggregation_check for sample-micro main-result alignment",
            }
        summary["advanced_group_analyses"] = advanced_group_analyses(
            diag,
            delta_by_item,
            sample_deltas,
            Path(locorec_args["dataset_dir"]) / "item_text_embeddings.npy",
            Path(locorec_args["dataset_dir"]) / "embedding_item_ids.json",
        )
        summary["task_aware_evidence_vs_route_delta"] = (
            task_aware_evidence_vs_route_delta(
                diag,
                route_sample_deltas,
                Path(locorec_args["dataset_dir"]),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "softsid_diagnostics_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_item_csv(args.output_dir / "softsid_item_diagnostics.csv", d_values, delta_by_item)
    if delta_by_item is not None:
        write_scatter_plot(
            args.output_dir / "topology_D_vs_delta_ndcg10.png",
            d_values,
            delta_by_item,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
