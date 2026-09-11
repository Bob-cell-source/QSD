"""Grouped full-catalog evaluation for the current LoCoRecSimple checkpoints."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LoCoRec.locorec.data import NextItemDataset, collate_eval
from LoCoRec.locorec.io import read_json, write_json
from LoCoRec.locorec.model import LoCoRec
from LoCoRec.locorec.soft_sid import (
    SoftSIDConfig,
    build_semantic_table,
    build_soft_sid_table,
    build_train_item_frequency,
)
from qsdrec.train import build_semantic_table as build_group_semantic_table
from scripts.evaluate_lcsoft_group_benchmarks import (
    build_overlap_sizes,
    build_popular_token_flags,
    build_prefix_sizes,
    build_train_target_counts,
    item_features,
    sample_groups,
)
from LoCoRecSimple.model import SimpleSharing, configure_fusion_control


def parse_checkpoints(values):
    result = {}
    for value in values:
        label, path = value.split("=", 1)
        result[label] = Path(path)
    return result


def make_model(state, hard, soft, prior, reliability, frequency, num_tokens, device):
    args = state["args"]
    variant = state["variant"]
    kw = dict(
        dim=args["dim"],
        max_len=args["max_len"],
        num_heads=2,
        num_layers=2,
        dropout=args["dropout"],
    )
    if variant == "full":
        model = LoCoRec(
            num_items=len(hard) - 1,
            num_semantic_tokens=num_tokens,
            soft_sid_table=soft,
            candidate_prior=prior,
            local_consistency=reliability,
            item_frequency=frequency,
            **kw,
        )
        model.item_encoder.dropout = torch.nn.Identity()
        model.embedding_dropout = torch.nn.Dropout(args.get("embedding_dropout", 0.0))
        configure_fusion_control(model, variant)
    elif variant == "hard":
        model = SimpleSharing(
            "hard", hard, num_tokens, soft, prior,
            embedding_dropout=args.get("embedding_dropout", 0.0), **kw
        )
    else:
        raise ValueError(f"This evaluator supports variants hard/full, got {variant}")
    model.load_state_dict(state["model"])
    return model.to(device).eval(), args


def add_metrics(row, rank, ks):
    row["count"] += 1
    hit = rank <= max(ks)
    for k in ks:
        if rank <= k:
            row[f"HR@{k}"] += 1.0
            row[f"NDCG@{k}"] += 1.0 / torch.log2(torch.tensor(float(rank + 1))).item()
            row[f"MRR@{k}"] += 1.0 / rank


def evaluate(model, loader, groups, device, num_items, keep_seen, ks):
    vectors = torch.cat([
        model.item_encoder(torch.arange(start, min(start + 512, num_items + 1), device=device))["vectors"]
        for start in range(1, num_items + 1, 512)
    ])
    names = sorted({name for item in groups for name in item})
    totals = {
        name: {"count": 0, **{f"{m}@{k}": 0.0 for k in ks for m in ("HR", "NDCG", "MRR")}}
        for name in names
    }
    offset = 0
    with torch.no_grad():
        for sequences, targets, histories in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            hidden, _ = model.encode_sequence(sequences)
            scores = hidden @ vectors.T
            if not keep_seen:
                for row, history in enumerate(histories):
                    seen = list(set(history) - {0, int(targets[row])})
                    if seen:
                        scores[row, torch.tensor(seen, device=device) - 1] = -torch.inf
            order = scores.topk(k=max(ks), dim=-1).indices + 1
            for row in range(len(targets)):
                hits = (order[row] == targets[row]).nonzero(as_tuple=False)
                rank = int(hits[0].item() + 1) if hits.numel() else max(ks) + 1
                for name in groups[offset + row]:
                    add_metrics(totals[name], rank, ks)
            offset += len(targets)
    for row in totals.values():
        count = max(row["count"], 1)
        for key in list(row):
            if key != "count":
                row[key] /= count
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ks", default="5,10,20")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ks = tuple(int(x) for x in args.ks.split(","))
    checkpoints = parse_checkpoints(args.checkpoint)
    dataset_dir = Path(args.dataset_dir)
    rows = read_json(dataset_dir / "sequences.json")
    num_items = int(read_json(dataset_dir / "stats.json")["num_items"])
    semantic_obj = read_json(args.semantic_ids)
    hard, codes, num_tokens = build_semantic_table(semantic_obj, num_items)
    _, item_sid, _ = build_group_semantic_table(semantic_obj, num_items)
    soft, prior, reliability = build_soft_sid_table(
        hard, codes, SoftSIDConfig(top_m=4, min_overlap_slots=3, leave_one_level_out=False, tie_break_seed=2026)
    )
    frequency = build_train_item_frequency(rows, num_items)
    item_meta = read_json(dataset_dir / "item_meta.json")
    counts = build_train_target_counts(rows)
    prefix_sizes = build_prefix_sizes(item_sid, 2)
    overlap_sizes = build_overlap_sizes(item_sid, 2)
    popular, popular_meta = build_popular_token_flags(item_sid, 0.90)
    features = item_features(item_meta)
    group_args = SimpleNamespace(
        low_freq_threshold=5, high_sharing_threshold=10,
        isolated_prefix_threshold=1, isolated_overlap_threshold=1,
        popular_token_quantile=0.90, mismatch_title_jaccard=0.30,
        mismatch_min_title_overlap=2, min_overlap_slots=2,
    )
    test_data = NextItemDataset(rows, max_len=20, split="test")
    groups = [
        sample_groups(seq, int(target), counts, prefix_sizes, overlap_sizes,
                      popular, item_sid, features, group_args)
        for seq, target, _ in test_data
    ]
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval)
    results = {}
    model_args = {}
    for label, path in checkpoints.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        model, saved_args = make_model(state, hard, soft, prior, reliability, frequency, num_tokens, device)
        results[label] = evaluate(model, loader, groups, device, num_items,
                                  bool(saved_args.get("keep_seen_items", False)), ks)
        model_args[label] = saved_args
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = {
        "dataset": dataset_dir.name,
        "semantic_ids": str(args.semantic_ids),
        "checkpoints": {k: str(v) for k, v in checkpoints.items()},
        "group_counts": dict(Counter(name for row in groups for name in row)),
        "popular_token_meta": popular_meta,
        "group_definitions": {
            "low_frequency": "training target count < 5",
            "high_sharing": "two-slot prefix group size >= 10",
            "isolated_sid": "two-slot prefix group size <= 1 or overlap group size <= 1",
            "popular_token": "item-level SID hubness >= 90th percentile",
            "strict_mismatch": "title/brand-supported history item shares fewer than two SID slots",
        },
        "model_args": model_args,
        "results": results,
    }
    write_json(Path(args.output), output)
    print(json.dumps({"dataset": dataset_dir.name, "groups": output["group_counts"], "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
