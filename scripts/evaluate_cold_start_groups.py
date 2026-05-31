import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json
from qsdrec.model import QSDRec
from qsdrec.train import NextItemDataset, build_log_prior, build_semantic_table, collate_full_eval, load_mini_cluster_table


def parse_bucket_spec(spec: str):
    buckets = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith(">"):
            lower = int(raw[1:]) + 1
            buckets.append((raw, lower, None))
        elif "-" in raw:
            left, right = raw.split("-", 1)
            buckets.append((raw, int(left), int(right)))
        else:
            value = int(raw)
            buckets.append((raw, value, value))
    if not buckets:
        raise ValueError("No valid frequency buckets.")
    return buckets


def bucket_name(value: int, buckets) -> str:
    for name, lower, upper in buckets:
        if value >= lower and (upper is None or value <= upper):
            return name
    return "other"


def build_train_item_counts(sequences) -> Dict[int, int]:
    counts = Counter()
    for row in sequences:
        items = row["items"]
        if len(items) < 3:
            continue
        # Match the training split in NextItemDataset: targets are items[1 : -2].
        for item in items[1:-2]:
            counts[int(item)] += 1
    return dict(counts)


def resolve_args(saved_args: Dict[str, Any], cli_args) -> Dict[str, Any]:
    args = dict(saved_args)
    for key in [
        "dataset_dir",
        "semantic_ids",
        "device",
        "eval_batch_eval_size",
        "sem_weight",
        "batch_size",
    ]:
        value = getattr(cli_args, key, None)
        if value is not None:
            args[key] = value
    return args


def new_stats(ks: Sequence[int]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"count": 0}
    for k in ks:
        stats[f"HR@{k}"] = 0.0
        stats[f"Recall@{k}"] = 0.0
        stats[f"NDCG@{k}"] = 0.0
    return stats


def add_metrics(stats: Dict[str, Any], rank: int, ks: Sequence[int]) -> None:
    stats["count"] += 1
    for k in ks:
        if rank <= k:
            gain = 1.0 / torch.log2(torch.tensor(float(rank + 1))).item()
            stats[f"HR@{k}"] += 1.0
            stats[f"Recall@{k}"] += 1.0
            stats[f"NDCG@{k}"] += gain


def finalize_metrics(stats: Dict[str, Any], ks: Sequence[int]) -> Dict[str, Any]:
    count = stats["count"]
    row = {"count": count}
    for k in ks:
        row[f"HR@{k}"] = stats[f"HR@{k}"] / max(count, 1)
        row[f"Recall@{k}"] = stats[f"Recall@{k}"] / max(count, 1)
        row[f"NDCG@{k}"] = stats[f"NDCG@{k}"] / max(count, 1)
    return row


@torch.no_grad()
def evaluate_grouped(
    model: QSDRec,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    sem_weight: float,
    item_counts: Dict[int, int],
    buckets,
    ks: Sequence[int],
    batch_eval_size: int,
) -> Dict[str, Any]:
    model.eval()
    max_k = max(ks)
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    stats = {"all": new_stats(ks), **{name: new_stats(ks) for name, _, _ in buckets}, "other": new_stats(ks)}
    bucket_counts = Counter()

    for seq, targets in loader:
        seq = seq.to(device)
        targets = targets.to(device)
        batch_size = seq.size(0)

        score_chunks = []
        for start in range(0, num_items, batch_eval_size):
            cand = all_items[start : start + batch_eval_size]
            cand = cand.unsqueeze(0).expand(batch_size, -1)
            score_chunks.append(model(seq, cand, sem_weight=sem_weight)["score"])
        scores = torch.cat(score_chunks, dim=1)

        seen_mask = seq.gt(0)
        if seen_mask.any():
            history_rows, history_cols = seen_mask.nonzero(as_tuple=True)
            history_item_idx = seq[history_rows, history_cols] - 1
            scores[history_rows, history_item_idx] = float("-inf")

        topk_idx = scores.topk(k=max_k, dim=1).indices + 1
        matches = topk_idx.eq(targets.unsqueeze(1))

        for row_idx in range(batch_size):
            target = int(targets[row_idx].item())
            freq = int(item_counts.get(target, 0))
            bucket = bucket_name(freq, buckets)
            bucket_counts[bucket] += 1

            hit_positions = matches[row_idx].nonzero(as_tuple=False)
            rank = int(hit_positions[0].item() + 1) if hit_positions.numel() else max_k + 1
            add_metrics(stats["all"], rank, ks)
            add_metrics(stats[bucket], rank, ks)

    result = {key: finalize_metrics(value, ks) for key, value in stats.items() if value["count"] > 0}
    result["bucket_counts"] = dict(bucket_counts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--semantic-ids", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-eval-size", type=int, default=None)
    parser.add_argument("--sem-weight", type=float, default=None)
    parser.add_argument("--buckets", default="0,1-5,6-10,11-20,>20")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    saved_args = state.get("args", {})
    cfg = resolve_args(saved_args, args)

    dataset_dir = Path(cfg["dataset_dir"])
    semantic_ids = Path(cfg["semantic_ids"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg.get("batch_size", 256))
    batch_eval_size = int(cfg.get("eval_batch_eval_size", 1024))
    sem_weight = float(cfg.get("sem_weight", 1.0))

    sequences = read_json(dataset_dir / "sequences.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(semantic_ids)
    num_items = int(stats["num_items"])
    semantic_table, _, num_semantic_tokens = build_semantic_table(semantic_obj, num_items)
    semantic_token_log_prior = build_log_prior(semantic_table, num_semantic_tokens)
    mini_cluster_table, mini_cluster_log_prior = load_mini_cluster_table(
        cfg.get("mini_clusters"),
        num_items,
        semantic_table,
    )
    item_counts = build_train_item_counts(sequences)

    test_data = NextItemDataset(sequences, int(cfg.get("max_len", 50)), "test")
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_full_eval,
    )

    model = QSDRec(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        semantic_id_table=semantic_table,
        dim=int(cfg.get("dim", 64)),
        max_len=int(cfg.get("max_len", 50)),
        num_interests=int(cfg.get("num_interests", 4)),
        num_heads=int(cfg.get("num_heads", 2)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        interest_router=str(cfg.get("interest_router", "semantic")),
        prefix_level=int(cfg.get("prefix_level", 2)),
        semantic_token_log_prior=semantic_token_log_prior,
        mini_cluster_table=mini_cluster_table,
        mini_cluster_log_prior=mini_cluster_log_prior,
        hub_score_weight=float(cfg.get("hub_score_weight", 0.0)),
        hub_attn_weight=float(cfg.get("hub_attn_weight", 0.0)),
        evidence_gate=str(cfg.get("evidence_gate", "none")),
        evidence_floor=float(cfg.get("evidence_floor", 0.1)),
        evidence_recency_weight=float(cfg.get("evidence_recency_weight", 0.0)),
        evidence_hub_weight=float(cfg.get("evidence_hub_weight", 0.0)),
        evidence_cross_weight=float(cfg.get("evidence_cross_weight", 0.2)),
        prior_lift_alpha=float(cfg.get("prior_lift_alpha", 0.1)),
        prior_lift_tau=float(cfg.get("prior_lift_tau", 1.0)),
        prior_lift_eta=float(cfg.get("prior_lift_eta", 1.0)),
        hub_penalty_weight=float(cfg.get("hub_penalty_weight", 0.0)),
        semantic_fusion=str(cfg.get("semantic_fusion", "fixed")),
        fusion_floor=float(cfg.get("fusion_floor", 0.0)),
        contrastive_alpha=float(cfg.get("contrastive_alpha", 0.0)),
    )
    model.load_state_dict(state["model"], strict=False)
    model.to(device)

    buckets = parse_bucket_spec(args.buckets)
    grouped = evaluate_grouped(
        model=model,
        loader=test_loader,
        device=device,
        num_items=num_items,
        sem_weight=sem_weight,
        item_counts=item_counts,
        buckets=buckets,
        ks=(5, 10, 20),
        batch_eval_size=batch_eval_size,
    )
    output = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(dataset_dir),
        "semantic_ids": str(semantic_ids),
        "frequency_definition": "training target count, matching NextItemDataset train split items[1:-2]",
        "sem_weight": sem_weight,
        "buckets": args.buckets,
        "args": cfg,
        "grouped_metrics": grouped,
    }
    write_json(args.output, output)
    print(json.dumps(output["grouped_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
