#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "LCSoftCRSID"))

from lcsoftcrsid.data import NextItemDataset, collate_eval
from lcsoftcrsid.io import read_json, write_json
from lcsoftcrsid.model import LCSoftCRSID
from lcsoftcrsid.soft_sid import (
    SoftSIDConfig,
    build_semantic_table,
    build_soft_sid_table,
    build_text_knn_neighbors,
    build_train_item_frequency,
)
from qsdrec.model import SASRecEncoder


DEFAULT_BUCKETS = "0,1-2,3-5,6-10,>10"
METRICS = ("NDCG@5", "HR@5", "NDCG@10", "HR@10", "NDCG@20", "HR@20")


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Checkpoint must be NAME=PATH or PATH.")
    return name.strip(), Path(path)


def parse_buckets(spec: str) -> list[tuple[str, int, int | None]]:
    buckets = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        if name.startswith(">"):
            buckets.append((name, int(name[1:]) + 1, None))
        elif "-" in name:
            lower, upper = name.split("-", 1)
            buckets.append((name, int(lower), int(upper)))
        else:
            value = int(name)
            buckets.append((name, value, value))
    if not buckets:
        raise ValueError("At least one frequency bucket is required.")
    return buckets


def bucket_for(frequency: int, buckets: Iterable[tuple[str, int, int | None]]) -> str:
    for name, lower, upper in buckets:
        if frequency >= lower and (upper is None or frequency <= upper):
            return name
    return "other"


def empty_stats() -> dict[str, float | int]:
    return {"count": 0, **{metric: 0.0 for metric in METRICS}}


def add_rank(stats: dict[str, float | int], rank: int) -> None:
    stats["count"] += 1
    for cutoff in (5, 10, 20):
        if rank <= cutoff:
            stats[f"HR@{cutoff}"] += 1.0
            stats[f"NDCG@{cutoff}"] += 1.0 / torch.log2(
                torch.tensor(float(rank + 1))
            ).item()


def finalize(stats: dict[str, float | int]) -> dict[str, float | int]:
    count = int(stats["count"])
    return {
        "count": count,
        **{metric: float(stats[metric]) / max(count, 1) for metric in METRICS},
    }


def resolve_path(path: str | Path, checkpoint: Path) -> Path:
    candidate = Path(str(path).replace("\\", "/"))
    if candidate.exists():
        return candidate
    relative_to_root = PROJECT_ROOT / candidate
    if relative_to_root.exists():
        return relative_to_root
    relative_to_checkpoint = checkpoint.parent / candidate
    if relative_to_checkpoint.exists():
        return relative_to_checkpoint
    return candidate


def build_model(
    checkpoint_path: Path,
    state: dict[str, Any],
    device: torch.device,
) -> tuple[
    LCSoftCRSID | SASRecEncoder,
    list[dict[str, Any]],
    torch.Tensor,
    int,
    dict[str, Any],
]:
    cfg = dict(state["args"])
    dataset_dir = resolve_path(cfg["dataset_dir"], checkpoint_path)
    semantic_ids = resolve_path(cfg["semantic_ids"], checkpoint_path)
    sequences = read_json(dataset_dir / "sequences.json")
    stats = read_json(dataset_dir / "stats.json")
    num_items = int(stats["num_items"])
    frequency = build_train_item_frequency(sequences, num_items)

    # Legacy ID-only SASRec checkpoints were trained through QSDRec with
    # sem_weight=0. Load only their encoder so no unused semantic branch can
    # influence the evaluation.
    if "encoder.item_emb.weight" in state["model"]:
        if float(cfg.get("sem_weight", 0.0)) != 0.0:
            raise ValueError(
                f"{checkpoint_path} is not an ID-only SASRec checkpoint."
            )
        model = SASRecEncoder(
            num_items=num_items,
            dim=int(cfg.get("dim", 128)),
            max_len=int(cfg.get("max_len", 50)),
            num_heads=int(cfg.get("num_heads", 2)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state["model"].items()
            if key.startswith("encoder.")
        }
        model.load_state_dict(encoder_state, strict=True)
        model.to(device).eval()
        cfg["dataset_dir"] = str(dataset_dir)
        cfg["semantic_ids"] = str(semantic_ids)
        cfg["evaluation_family"] = "sasrec_id_only"
        return model, sequences, frequency, num_items, cfg

    semantic_obj = read_json(semantic_ids)
    hard_table, item_codes, num_semantic_tokens = build_semantic_table(
        semantic_obj, num_items
    )
    candidate_mode = str(cfg.get("candidate_weight_mode", "prior_guided"))
    soft_config = SoftSIDConfig(
        top_m=int(cfg.get("soft_top_m", 4)),
        min_overlap_slots=int(cfg.get("soft_min_overlap_slots", 3)),
        min_support=float(cfg.get("soft_min_support", 0.05)),
        reliability_floor=float(cfg.get("soft_reliability_floor", 0.10)),
        max_neighbors=int(cfg.get("soft_max_neighbors", 50)),
        candidate_construction=(
            "uniform_topk" if candidate_mode == "neighborhood_learned" else "local_prior"
        ),
    )
    base_neighbors = None
    if cfg.get("soft_neighbor_source", "sid_overlap") == "text_knn":
        base_neighbors, _ = build_text_knn_neighbors(
            embeddings_path=resolve_path(cfg["soft_text_embeddings"], checkpoint_path),
            item_ids_path=resolve_path(cfg["soft_text_item_ids"], checkpoint_path),
            num_items=num_items,
            max_neighbors=soft_config.max_neighbors,
            chunk_size=int(cfg.get("soft_text_knn_chunk_size", 256)),
        )
    soft_ids, soft_weights, reliability = build_soft_sid_table(
        hard_table,
        item_codes,
        soft_config,
        base_neighbors=base_neighbors,
    )
    model = LCSoftCRSID(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        soft_sid_table=soft_ids,
        soft_sid_weights=soft_weights,
        semantic_reliability=reliability,
        item_frequency=frequency,
        dim=int(cfg.get("dim", 128)),
        max_len=int(cfg.get("max_len", 50)),
        num_heads=int(cfg.get("num_heads", 2)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        tail_tau=float(cfg.get("tail_tau", 20.0)),
        alpha_mode=str(cfg.get("alpha_mode", "fixed")),
        fusion_mode=str(cfg.get("fusion_mode", "fixed")),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
        gate_correction_scale=float(cfg.get("gate_correction_scale", 1.0)),
        gate_private_margin=float(cfg.get("gate_private_margin", 0.0)),
        candidate_weight_mode=candidate_mode,
        disable_semantic_basis=bool(cfg.get("disable_semantic_basis", False)),
        disable_shared_residual=bool(cfg.get("disable_shared_residual", False)),
        disable_private_residual=bool(cfg.get("disable_private_residual", False)),
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    cfg["dataset_dir"] = str(dataset_dir)
    cfg["semantic_ids"] = str(semantic_ids)
    return model, sequences, frequency, num_items, cfg


@torch.no_grad()
def evaluate(
    model: LCSoftCRSID | SASRecEncoder,
    sequences: list[dict[str, Any]],
    frequency: torch.Tensor,
    num_items: int,
    max_len: int,
    device: torch.device,
    batch_size: int,
    candidate_chunk_size: int,
    buckets: list[tuple[str, int, int | None]],
    cold_threshold: int,
) -> OrderedDict[str, dict[str, float | int]]:
    dataset = NextItemDataset(sequences, max_len, "test")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )
    names = [
        "overall",
        f"cold_0-{cold_threshold}",
        f"warm_gt{cold_threshold}",
        *(name for name, _, _ in buckets),
        "other",
    ]
    stats = OrderedDict((name, empty_stats()) for name in names)
    all_items = torch.arange(1, num_items + 1, device=device)

    # Item representations are user-independent. Cache them once instead of
    # rebuilding the complete candidate catalogue for every evaluation batch.
    if isinstance(model, LCSoftCRSID):
        item_vector_chunks = []
        for start in range(0, num_items, candidate_chunk_size):
            item_vector_chunks.append(
                model.item_encoder(all_items[start : start + candidate_chunk_size])
            )
        all_item_vectors = torch.cat(item_vector_chunks, dim=0)
    else:
        all_item_vectors = model.item_emb(all_items)

    for history, targets in loader:
        history = history.to(device)
        targets = targets.to(device)
        if isinstance(model, LCSoftCRSID):
            history_vectors = model.item_encoder(history)
            user_vectors, _ = model.sequence_encoder(history, history_vectors)
        else:
            user_vectors, _ = model(history)
        scores = user_vectors @ all_item_vectors.transpose(0, 1)

        seen = history.gt(0)
        if seen.any():
            rows, cols = seen.nonzero(as_tuple=True)
            scores[rows, history[rows, cols] - 1] = float("-inf")

        top_items = scores.topk(k=20, dim=1).indices + 1
        matches = top_items.eq(targets.unsqueeze(1))
        for row in range(history.size(0)):
            positions = matches[row].nonzero(as_tuple=False)
            rank = int(positions[0].item()) + 1 if positions.numel() else 21
            target = int(targets[row].item())
            target_frequency = int(frequency[target].item())
            group = bucket_for(target_frequency, buckets)
            add_rank(stats["overall"], rank)
            if target_frequency <= cold_threshold:
                add_rank(stats[f"cold_0-{cold_threshold}"], rank)
            else:
                add_rank(stats[f"warm_gt{cold_threshold}"], rank)
            add_rank(stats[group], rank)

    return OrderedDict(
        (name, finalize(value))
        for name, value in stats.items()
        if int(value["count"]) > 0
    )


def write_csv(results: list[dict[str, Any]], output: Path) -> None:
    rows = []
    reference: dict[str, dict[str, float | int]] = {}
    if results:
        reference = results[0]["groups"]
    for result in results:
        for group, values in result["groups"].items():
            row = {
                "model": result["name"],
                "group": group,
                **values,
                "delta_vs_reference_NDCG@10": "",
                "relative_gain_vs_reference_NDCG@10": "",
                "checkpoint": result["checkpoint"],
            }
            if group in reference:
                baseline = float(reference[group]["NDCG@10"])
                delta = float(values["NDCG@10"]) - baseline
                row["delta_vs_reference_NDCG@10"] = delta
                row["relative_gain_vs_reference_NDCG@10"] = (
                    delta / baseline if baseline > 0 else ""
                )
            rows.append(row)
    fieldnames = [
        "model",
        "group",
        "count",
        *METRICS,
        "delta_vs_reference_NDCG@10",
        "relative_gain_vs_reference_NDCG@10",
        "checkpoint",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate clean LC-SoftCRSID checkpoints by training-frequency bucket."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="Repeatable NAME=PATH. The first checkpoint is the comparison reference.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", default=DEFAULT_BUCKETS)
    parser.add_argument("--cold-threshold", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-chunk-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    buckets = parse_buckets(args.buckets)
    results = []
    dataset_dir = None

    for name, checkpoint in args.checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model, sequences, frequency, num_items, cfg = build_model(
            checkpoint, state, device
        )
        current_dataset = cfg["dataset_dir"]
        if dataset_dir is None:
            dataset_dir = current_dataset
        elif current_dataset != dataset_dir:
            raise ValueError("All checkpoints must use the same dataset.")
        groups = evaluate(
            model=model,
            sequences=sequences,
            frequency=frequency,
            num_items=num_items,
            max_len=int(cfg.get("max_len", 50)),
            device=device,
            batch_size=args.batch_size,
            candidate_chunk_size=args.candidate_chunk_size,
            buckets=buckets,
            cold_threshold=args.cold_threshold,
        )
        results.append(
            {
                "name": name,
                "checkpoint": str(checkpoint),
                "groups": groups,
            }
        )
        print(json.dumps({name: groups}, ensure_ascii=False, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_dir": dataset_dir,
        "frequency_definition": (
            "Count of item occurrences in items[:-2], identical to the frequency "
            "used by the final LC-SoftCRSID item encoder."
        ),
        "buckets": args.buckets,
        "cold_threshold": args.cold_threshold,
        "reference_model": results[0]["name"],
        "results": results,
    }
    write_json(args.output_dir / "cold_start_metrics.json", payload)
    write_csv(results, args.output_dir / "cold_start_metrics.csv")
    print(args.output_dir / "cold_start_metrics.csv")


if __name__ == "__main__":
    main()
