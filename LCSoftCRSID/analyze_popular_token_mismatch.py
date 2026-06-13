#!/usr/bin/env python3
"""Evaluate models on items mismatched with high-occupancy SID tokens.

The script does not retrain a model. It reconstructs each model from the
configuration stored in its checkpoint, evaluates full-item ranking on the
selected test targets, and compares learned item-representation similarities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lcsoftcrsid.data import NextItemDataset, collate_eval
from lcsoftcrsid.model import LCSoftCRSID
from lcsoftcrsid.soft_sid import (
    SoftSIDConfig,
    build_semantic_table,
    build_soft_sid_table,
    build_text_knn_neighbors,
    build_train_item_frequency,
)
from scripts.evaluate_lcsoft_group_benchmarks import instantiate_model as instantiate_legacy_model
from qsdrec.train import build_semantic_table as build_legacy_semantic_table


TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with", "new", "pack", "set", "size",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must be label=path, got {value!r}.")
        label, raw_path = value.split("=", 1)
        checkpoints[label.strip()] = Path(raw_path)
    return checkpoints


def leaf_category(meta: dict[str, Any]) -> str:
    categories = meta.get("categories") or []
    return str(categories[-1]).strip().lower() if categories else ""


def title_tokens(meta: dict[str, Any]) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(str(meta.get("title", "")).lower())
        if len(token) > 1 and token not in TITLE_STOPWORDS
    }


def related_items(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if leaf_category(left) and leaf_category(left) == leaf_category(right):
        return True
    left_brand = str(left.get("brand", "") or "").strip().lower()
    right_brand = str(right.get("brand", "") or "").strip().lower()
    if not left_brand or left_brand != right_brand:
        return False
    left_title = title_tokens(left)
    right_title = title_tokens(right)
    overlap = len(left_title & right_title)
    jaccard = overlap / max(len(left_title | right_title), 1)
    return overlap >= 2 and jaccard >= 0.30


def quantile_threshold(values: list[int], quantile: float) -> int:
    if not values:
        return 1
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return int(ordered[index])


def load_text_embeddings(
    embeddings_path: Path,
    item_ids_path: Path,
) -> tuple[np.ndarray, dict[int, int]]:
    embeddings = np.load(embeddings_path).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    item_ids = [int(item) for item in read_json(item_ids_path)]
    if len(item_ids) != len(embeddings):
        raise ValueError("Embedding rows and embedding item IDs have different lengths.")
    return embeddings, {item: row for row, item in enumerate(item_ids)}


def build_mismatch_cases(
    item_codes: dict[int, list[int]],
    item_meta: dict[str, Any],
    embeddings: np.ndarray,
    embedding_rows: dict[int, int],
    occupancy_quantile: float,
    min_token_occupancy: int,
    max_category_share: float,
    min_text_similarity: float,
    min_alternative_gain: float,
    max_peers: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    depth = len(next(iter(item_codes.values())))
    slot_items: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(depth)]
    categories = {
        item: leaf_category(item_meta.get(str(item), {}))
        for item in item_codes
    }
    for item, sid in item_codes.items():
        for slot, code in enumerate(sid):
            slot_items[slot][int(code)].append(item)

    thresholds = [
        max(
            min_token_occupancy,
            quantile_threshold([len(items) for items in groups.values()], occupancy_quantile),
        )
        for groups in slot_items
    ]
    category_counts: list[dict[int, Counter[str]]] = []
    for groups in slot_items:
        category_counts.append(
            {
                code: Counter(categories[item] for item in items if categories[item])
                for code, items in groups.items()
            }
        )

    related: dict[int, list[int]] = defaultdict(list)
    embedded_items = [item for item in item_codes if item in embedding_rows]
    for left_index, left in enumerate(embedded_items):
        left_meta = item_meta.get(str(left), {})
        for right in embedded_items[left_index + 1 :]:
            if related_items(left_meta, item_meta.get(str(right), {})):
                related[left].append(right)
                related[right].append(left)

    cases: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item, sid in item_codes.items():
        category = categories[item]
        row = embedding_rows.get(item)
        if not category or row is None:
            continue
        target_vector = embeddings[row]
        semantic_peers = related[item]
        for slot, code in enumerate(sid):
            occupancy = len(slot_items[slot][int(code)])
            if occupancy < thresholds[slot]:
                continue
            target_share = category_counts[slot][int(code)][category] / max(occupancy, 1)
            if target_share > max_category_share:
                continue

            alternatives = []
            for peer in semantic_peers:
                if item_codes[peer][slot] == code:
                    continue
                similarity = float(target_vector @ embeddings[embedding_rows[peer]])
                if similarity < min_text_similarity:
                    continue
                peer_code = int(item_codes[peer][slot])
                peer_occupancy = len(slot_items[slot][peer_code])
                peer_share = category_counts[slot][peer_code][category] / max(peer_occupancy, 1)
                if peer_share < target_share + min_alternative_gain:
                    continue
                alternatives.append((similarity, peer_share, peer))
            if not alternatives:
                continue

            alternatives.sort(reverse=True)
            wrong_peers = [
                peer
                for peer in slot_items[slot][int(code)]
                if peer != item
                and categories[peer] != category
                and peer not in semantic_peers
                and peer in embedding_rows
            ]
            wrong_peers.sort(
                key=lambda peer: float(target_vector @ embeddings[embedding_rows[peer]]),
                reverse=True,
            )
            correct_peers = [peer for _, _, peer in alternatives[:max_peers]]
            wrong_peers = wrong_peers[:max_peers]
            if not wrong_peers:
                continue

            best_similarity, best_share, best_peer = alternatives[0]
            cases[item].append(
                {
                    "slot": slot,
                    "token": int(code),
                    "token_occupancy": occupancy,
                    "slot_threshold": thresholds[slot],
                    "target_category": category,
                    "target_category_share": target_share,
                    "alternative_item": best_peer,
                    "alternative_token": int(item_codes[best_peer][slot]),
                    "alternative_category_share": best_share,
                    "alternative_text_similarity": best_similarity,
                    "correct_peers": correct_peers,
                    "wrong_peers": wrong_peers,
                }
            )

    return dict(cases), {
        "occupancy_quantile": occupancy_quantile,
        "min_token_occupancy": min_token_occupancy,
        "slot_occupancy_thresholds": thresholds,
        "max_category_share": max_category_share,
        "min_text_similarity": min_text_similarity,
        "min_alternative_gain": min_alternative_gain,
        "num_flagged_items": len(cases),
        "num_flagged_item_slots": sum(len(item_cases) for item_cases in cases.values()),
    }


def is_current_checkpoint(state: dict[str, Any]) -> bool:
    return any(key.startswith("sequence_encoder.") for key in state["model"])


def instantiate_current_model(
    state: dict[str, Any],
    sequences: list[dict[str, Any]],
    semantic_obj: dict[str, Any],
    num_items: int,
) -> LCSoftCRSID:
    cfg = state.get("args", {})
    hard_table, item_codes, num_tokens = build_semantic_table(semantic_obj, num_items)
    soft_config = SoftSIDConfig(
        top_m=int(cfg.get("soft_top_m", 4)),
        min_overlap_slots=int(cfg.get("soft_min_overlap_slots", 3)),
        min_support=float(cfg.get("soft_min_support", 0.05)),
        reliability_floor=float(cfg.get("soft_reliability_floor", 0.1)),
        max_neighbors=int(cfg.get("soft_max_neighbors", 50)),
        candidate_construction=(
            "uniform_topk"
            if cfg.get("candidate_weight_mode", "prior_guided") == "neighborhood_learned"
            else "local_prior"
        ),
    )
    base_neighbors = None
    if cfg.get("soft_neighbor_source", "sid_overlap") == "text_knn":
        base_neighbors, _ = build_text_knn_neighbors(
            embeddings_path=cfg["soft_text_embeddings"],
            item_ids_path=cfg["soft_text_item_ids"],
            num_items=num_items,
            max_neighbors=int(cfg.get("soft_max_neighbors", 50)),
            chunk_size=int(cfg.get("soft_text_knn_chunk_size", 256)),
        )
    soft_table, soft_weights, reliability = build_soft_sid_table(
        semantic_table=hard_table,
        item_codes=item_codes,
        config=soft_config,
        base_neighbors=base_neighbors,
    )
    model = LCSoftCRSID(
        num_items=num_items,
        num_semantic_tokens=num_tokens,
        soft_sid_table=soft_table,
        soft_sid_weights=soft_weights,
        semantic_reliability=reliability,
        item_frequency=build_train_item_frequency(sequences, num_items),
        dim=int(cfg.get("dim", 128)),
        max_len=int(cfg.get("max_len", 50)),
        num_heads=int(cfg.get("num_heads", 2)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        tail_tau=float(cfg.get("tail_tau", 20.0)),
        alpha_mode=str(cfg.get("alpha_mode", "fixed")),
        candidate_weight_mode=str(cfg.get("candidate_weight_mode", "prior_guided")),
        disable_semantic_basis=bool(cfg.get("disable_semantic_basis", False)),
        disable_shared_residual=bool(cfg.get("disable_shared_residual", False)),
        disable_private_residual=bool(cfg.get("disable_private_residual", False)),
    )
    model.load_state_dict(state["model"], strict=True)
    return model


def instantiate_checkpoint(
    state: dict[str, Any],
    sequences: list[dict[str, Any]],
    semantic_obj: dict[str, Any],
    num_items: int,
):
    if is_current_checkpoint(state):
        return instantiate_current_model(state, sequences, semantic_obj, num_items)

    semantic_table, item_codes, num_tokens = build_legacy_semantic_table(semantic_obj, num_items)
    cfg = dict(state.get("args", {}))
    return instantiate_legacy_model(
        state=state,
        cfg=cfg,
        sequences=sequences,
        semantic_table=semantic_table,
        item_semantic_ids=item_codes,
        num_semantic_tokens=num_tokens,
        num_items=num_items,
    )


def model_scores(model, sequences: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    try:
        return model(sequences, candidates)["score"]
    except TypeError:
        return model(sequences, candidates, sem_weight=1.0)["score"]


def item_representations(model, items: torch.Tensor) -> torch.Tensor | None:
    if hasattr(model, "item_encoder"):
        return model.item_encoder(items)
    if hasattr(model, "item_representation"):
        return model.item_representation(items)
    return None


@torch.no_grad()
def evaluate_ranking(
    model,
    loader: DataLoader,
    flagged_items: set[int],
    device: torch.device,
    num_items: int,
    candidate_chunk_size: int,
) -> dict[str, float | int]:
    model.eval()
    cutoffs = (5, 10, 20)
    hits = Counter()
    ndcgs = Counter()
    total = 0
    all_items = torch.arange(1, num_items + 1, device=device)
    for sequences, targets in loader:
        selected = torch.tensor(
            [int(target) in flagged_items for target in targets],
            dtype=torch.bool,
        )
        if not selected.any():
            continue
        sequences = sequences[selected].to(device)
        targets = targets[selected].to(device)
        score_chunks = []
        for start in range(0, num_items, candidate_chunk_size):
            candidates = all_items[start : start + candidate_chunk_size]
            candidates = candidates.unsqueeze(0).expand(len(targets), -1)
            score_chunks.append(model_scores(model, sequences, candidates))
        scores = torch.cat(score_chunks, dim=1)
        seen = sequences.gt(0)
        rows, columns = seen.nonzero(as_tuple=True)
        scores[rows, sequences[rows, columns] - 1] = float("-inf")
        top_items = scores.topk(max(cutoffs), dim=1).indices + 1
        for cutoff in cutoffs:
            matches = top_items[:, :cutoff].eq(targets.unsqueeze(1))
            hit = matches.any(dim=1)
            rank = matches.float().argmax(dim=1) + 1
            hits[cutoff] += float(hit.float().sum())
            ndcgs[cutoff] += float((hit.float() / torch.log2(rank.float() + 1.0)).sum())
        total += len(targets)
    result: dict[str, float | int] = {"count": total}
    for cutoff in cutoffs:
        result[f"HR@{cutoff}"] = hits[cutoff] / max(total, 1)
        result[f"NDCG@{cutoff}"] = ndcgs[cutoff] / max(total, 1)
    return result


@torch.no_grad()
def evaluate_representations(
    model,
    cases: dict[int, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, float | int | None]:
    model.eval()
    margins = []
    correct_similarities = []
    wrong_similarities = []
    for item, item_cases in cases.items():
        for case in item_cases:
            correct = case["correct_peers"]
            wrong = case["wrong_peers"]
            ids = [item, *correct, *wrong]
            vectors = item_representations(model, torch.tensor(ids, device=device))
            if vectors is None:
                continue
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
            correct_similarity = float((vectors[1 : 1 + len(correct)] @ vectors[0]).mean())
            wrong_similarity = float((vectors[1 + len(correct) :] @ vectors[0]).mean())
            correct_similarities.append(correct_similarity)
            wrong_similarities.append(wrong_similarity)
            margins.append(correct_similarity - wrong_similarity)
    return {
        "count": len(margins),
        "mean_correct_peer_cosine": sum(correct_similarities) / len(correct_similarities) if margins else None,
        "mean_wrong_token_peer_cosine": sum(wrong_similarities) / len(wrong_similarities) if margins else None,
        "mean_similarity_margin": sum(margins) / len(margins) if margins else None,
        "positive_margin_rate": sum(value > 0 for value in margins) / len(margins) if margins else None,
    }


def write_case_csv(
    path: Path,
    cases: dict[int, list[dict[str, Any]]],
    item_codes: dict[int, list[int]],
    item_meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "item_id", "title", "sid", "slot", "token", "token_occupancy",
        "slot_threshold", "target_category", "target_category_share",
        "alternative_item", "alternative_title", "alternative_sid",
        "alternative_token", "alternative_category_share", "alternative_text_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, item_cases in sorted(cases.items()):
            for case in item_cases:
                alternative = case["alternative_item"]
                writer.writerow(
                    {
                        "item_id": item,
                        "title": item_meta.get(str(item), {}).get("title", ""),
                        "sid": item_codes[item],
                        **{key: case[key] for key in fields if key in case},
                        "slot": case["slot"] + 1,
                        "alternative_title": item_meta.get(str(alternative), {}).get("title", ""),
                        "alternative_sid": item_codes[alternative],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="label=checkpoint.pt")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", default=None)
    parser.add_argument("--text-embeddings", default=None)
    parser.add_argument("--text-item-ids", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-chunk-size", type=int, default=512)
    parser.add_argument("--occupancy-quantile", type=float, default=0.80)
    parser.add_argument("--min-token-occupancy", type=int, default=30)
    parser.add_argument("--max-category-share", type=float, default=0.05)
    parser.add_argument("--min-text-similarity", type=float, default=0.85)
    parser.add_argument("--min-alternative-gain", type=float, default=0.20)
    parser.add_argument("--max-peers", type=int, default=10)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    semantic_path = Path(args.semantic_ids or dataset_dir / "semantic_ids_rq.json")
    embeddings_path = Path(args.text_embeddings or dataset_dir / "item_text_embeddings.npy")
    item_ids_path = Path(args.text_item_ids or dataset_dir / "embedding_item_ids.json")
    output_path = Path(args.output)
    checkpoints = parse_checkpoints(args.checkpoint)

    sequences = read_json(dataset_dir / "sequences.json")
    item_meta = read_json(dataset_dir / "item_meta.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(semantic_path)
    item_codes = {int(item): list(map(int, sid)) for item, sid in semantic_obj["semantic_ids"].items()}
    embeddings, embedding_rows = load_text_embeddings(embeddings_path, item_ids_path)
    cases, definition = build_mismatch_cases(
        item_codes=item_codes,
        item_meta=item_meta,
        embeddings=embeddings,
        embedding_rows=embedding_rows,
        occupancy_quantile=args.occupancy_quantile,
        min_token_occupancy=args.min_token_occupancy,
        max_category_share=args.max_category_share,
        min_text_similarity=args.min_text_similarity,
        min_alternative_gain=args.min_alternative_gain,
        max_peers=args.max_peers,
    )

    first_state = torch.load(next(iter(checkpoints.values())), map_location="cpu")
    max_len = int(first_state.get("args", {}).get("max_len", 50))
    test_dataset = NextItemDataset(sequences, max_len=max_len, split="test")
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval)
    test_targets = Counter(int(target) for _, target in test_dataset)
    flagged_items = set(cases)
    definition["num_flagged_test_samples"] = sum(test_targets[item] for item in flagged_items)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_results = {}
    for label, checkpoint in checkpoints.items():
        state = torch.load(checkpoint, map_location="cpu")
        model = instantiate_checkpoint(state, sequences, semantic_obj, int(stats["num_items"]))
        model.to(device)
        model_results[label] = {
            "checkpoint": str(checkpoint),
            "checkpoint_args": state.get("args", {}),
            "ranking": evaluate_ranking(
                model=model,
                loader=loader,
                flagged_items=flagged_items,
                device=device,
                num_items=int(stats["num_items"]),
                candidate_chunk_size=args.candidate_chunk_size,
            ),
            "representation": evaluate_representations(model, cases, device),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_dir": str(dataset_dir),
                "definition": definition,
                "models": model_results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    write_case_csv(output_path.with_suffix(".cases.csv"), cases, item_codes, item_meta)
    print(json.dumps({"definition": definition, "models": model_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
