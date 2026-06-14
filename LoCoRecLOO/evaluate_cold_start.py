#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from locorec.data import NextItemDataset, collate_eval
from locorec.io import read_json, write_json
from locorec.model import LoCoRec
from locorec.soft_sid import (
    SoftSIDConfig,
    build_semantic_table,
    build_soft_sid_table,
    build_train_item_frequency,
)


METRICS = ("NDCG@5", "HR@5", "NDCG@10", "HR@10", "NDCG@20", "HR@20")


def build_popular_sid_flags(item_codes, quantile):
    depth = len(next(iter(item_codes.values()))) if item_codes else 0
    slot_counts = [Counter() for _ in range(depth)]
    for sid in item_codes.values():
        for slot, code in enumerate(sid):
            slot_counts[slot][int(code)] += 1

    slot_max = [max(counts.values(), default=1) for counts in slot_counts]
    scores = {}
    for item, sid in item_codes.items():
        normalized = [
            math.log1p(slot_counts[slot][int(code)])
            / math.log1p(slot_max[slot])
            for slot, code in enumerate(sid)
        ]
        scores[item] = sum(normalized) / max(len(normalized), 1)

    values = sorted(scores.values())
    if values:
        index = min(
            len(values) - 1,
            max(0, int(math.ceil(quantile * len(values))) - 1),
        )
        threshold = values[index]
    else:
        threshold = float("inf")
    flags = {item: score >= threshold for item, score in scores.items()}
    metadata = {
        "quantile": quantile,
        "threshold": threshold,
        "num_flagged_items": sum(flags.values()),
        "num_items": len(flags),
        "definition": "mean slot-wise log-normalized hard SID token frequency",
    }
    return flags, metadata


def parse_buckets(spec):
    buckets = []
    for raw in spec.split(","):
        name = raw.strip()
        if name.startswith(">"):
            buckets.append((name, int(name[1:]) + 1, None))
        elif "-" in name:
            lower, upper = name.split("-", 1)
            buckets.append((name, int(lower), int(upper)))
        else:
            value = int(name)
            buckets.append((name, value, value))
    return buckets


def bucket_for(frequency, buckets):
    for name, lower, upper in buckets:
        if frequency >= lower and (upper is None or frequency <= upper):
            return name
    return "other"


def new_stats():
    return {"count": 0, **{metric: 0.0 for metric in METRICS}}


def add_rank(stats, rank):
    stats["count"] += 1
    for cutoff in (5, 10, 20):
        if rank <= cutoff:
            stats[f"HR@{cutoff}"] += 1.0
            stats[f"NDCG@{cutoff}"] += 1.0 / torch.log2(
                torch.tensor(float(rank + 1))
            ).item()


def finalize(stats):
    count = stats["count"]
    return {
        "count": count,
        **{metric: stats[metric] / max(count, 1) for metric in METRICS},
    }


def build_model(checkpoint, device, popular_sid_quantile):
    state = torch.load(checkpoint, map_location="cpu")
    cfg = dict(state["args"])
    dataset_dir = Path(cfg["dataset_dir"])
    semantic_ids = Path(cfg["semantic_ids"])
    sequences = read_json(dataset_dir / "sequences.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(semantic_ids)
    num_items = int(stats["num_items"])

    hard_table, item_codes, num_semantic_tokens = build_semantic_table(
        semantic_obj, num_items
    )
    soft_ids, candidate_prior, local_consistency, _ = build_soft_sid_table(
        hard_table,
        item_codes,
        SoftSIDConfig(
            top_m=int(cfg.get("soft_top_m", 4)),
            loo_min_overlap_slots=int(cfg.get("loo_min_overlap_slots", 2)),
            min_support=float(cfg.get("soft_min_support", 0.05)),
            max_neighbors=int(cfg.get("soft_max_neighbors", 50)),
            tie_break_seed=int(cfg.get("seed", 2026)),
        ),
    )
    frequency = build_train_item_frequency(sequences, num_items)
    model = LoCoRec(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        soft_sid_table=soft_ids,
        candidate_prior=candidate_prior,
        local_consistency=local_consistency,
        item_frequency=frequency,
        dim=int(cfg.get("dim", 128)),
        max_len=int(cfg.get("max_len", 50)),
        num_heads=int(cfg.get("num_heads", 2)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        tail_tau=float(cfg.get("tail_tau", 20.0)),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
        gate_correction_scale=float(cfg.get("gate_correction_scale", 0.3)),
        gate_private_margin=float(cfg.get("gate_private_margin", 0.05)),
    )
    model.load_state_dict(state["model"], strict=True)
    popular_flags, popular_metadata = build_popular_sid_flags(
        item_codes, popular_sid_quantile
    )
    return (
        model.to(device).eval(),
        sequences,
        frequency,
        num_items,
        cfg,
        popular_flags,
        popular_metadata,
    )


@torch.no_grad()
def evaluate(model, sequences, frequency, num_items, cfg, popular_flags, args):
    dataset = NextItemDataset(sequences, int(cfg.get("max_len", 50)), "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )
    buckets = parse_buckets(args.buckets)
    names = [
        "overall",
        f"cold_0-{args.cold_threshold}",
        f"warm_gt{args.cold_threshold}",
        "popular_sid",
        "non_popular_sid",
        *(name for name, _, _ in buckets),
        "other",
    ]
    grouped = OrderedDict((name, new_stats()) for name in names)
    all_items = torch.arange(1, num_items + 1, device=args.device)
    item_chunks = []
    for start in range(0, num_items, args.candidate_chunk_size):
        items = all_items[start : start + args.candidate_chunk_size]
        item_chunks.append(model.item_encoder(items)["vectors"])
    all_item_vectors = torch.cat(item_chunks, dim=0)

    for sequence, targets, full_histories in loader:
        sequence = sequence.to(args.device)
        targets = targets.to(args.device)
        user_vectors, _ = model.encode_sequence(sequence)
        scores = user_vectors @ all_item_vectors.transpose(0, 1)
        for row, (history, target) in enumerate(zip(full_histories, targets.tolist())):
            seen = {int(item) for item in history if int(item) != target}
            if seen:
                columns = torch.tensor(
                    [item - 1 for item in seen], device=args.device, dtype=torch.long
                )
                scores[row, columns] = float("-inf")

        top_items = scores.topk(k=20, dim=1).indices + 1
        matches = top_items.eq(targets.unsqueeze(1))
        for row, target in enumerate(targets.tolist()):
            positions = matches[row].nonzero(as_tuple=False)
            rank = int(positions[0].item()) + 1 if positions.numel() else 21
            target_frequency = int(frequency[target].item())
            group = bucket_for(target_frequency, buckets)
            add_rank(grouped["overall"], rank)
            if target_frequency <= args.cold_threshold:
                add_rank(grouped[f"cold_0-{args.cold_threshold}"], rank)
            else:
                add_rank(grouped[f"warm_gt{args.cold_threshold}"], rank)
            if popular_flags.get(target, False):
                add_rank(grouped["popular_sid"], rank)
            else:
                add_rank(grouped["non_popular_sid"], rank)
            add_rank(grouped[group], rank)
    return OrderedDict(
        (name, finalize(stats))
        for name, stats in grouped.items()
        if stats["count"] > 0
    )


def write_csv(groups, output):
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "count", *METRICS])
        writer.writeheader()
        for group, values in groups.items():
            writer.writerow({"group": group, **values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-chunk-size", type=int, default=512)
    parser.add_argument("--cold-threshold", type=int, default=5)
    parser.add_argument("--buckets", default="0,1-2,3-5,6-10,>10")
    parser.add_argument("--popular-sid-quantile", type=float, default=0.90)
    args = parser.parse_args()
    args.device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    (
        model,
        sequences,
        frequency,
        num_items,
        cfg,
        popular_flags,
        popular_metadata,
    ) = build_model(
        args.checkpoint, args.device, args.popular_sid_quantile
    )
    groups = evaluate(
        model, sequences, frequency, num_items, cfg, popular_flags, args
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "frequency_definition": "Occurrences in the training split items[:-2].",
        "cold_threshold": args.cold_threshold,
        "buckets": args.buckets,
        "popular_sid": popular_metadata,
        "groups": groups,
    }
    write_json(args.output_dir / "cold_start_metrics.json", payload)
    write_csv(groups, args.output_dir / "cold_start_metrics.csv")
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
