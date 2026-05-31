import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
            buckets.append((raw, int(raw[1:]) + 1, None))
        elif "-" in raw:
            left, right = raw.split("-", 1)
            buckets.append((raw, int(left), int(right)))
        else:
            value = int(raw)
            buckets.append((raw, value, value))
    if not buckets:
        raise ValueError("No valid buckets.")
    return buckets


def bucket_name(value: int, buckets) -> str:
    for name, lower, upper in buckets:
        if value >= lower and (upper is None or value <= upper):
            return name
    return "other"


def resolve_args(saved_args: Dict[str, Any], cli_args) -> Dict[str, Any]:
    cfg = dict(saved_args)
    for key in ["dataset_dir", "semantic_ids", "device", "batch_size", "eval_batch_eval_size"]:
        value = getattr(cli_args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def load_model(checkpoint: Path, cfg: Dict[str, Any], semantic_table: torch.Tensor, num_semantic_tokens: int, num_items: int):
    state = torch.load(checkpoint, map_location="cpu")
    saved_args = dict(state.get("args", {}))
    for key in ["dataset_dir", "semantic_ids", "device", "batch_size", "eval_batch_eval_size"]:
        if key in cfg:
            saved_args[key] = cfg[key]
    semantic_token_log_prior = build_log_prior(semantic_table, num_semantic_tokens)
    mini_cluster_table, mini_cluster_log_prior = load_mini_cluster_table(
        saved_args.get("mini_clusters"),
        num_items,
        semantic_table,
    )
    model = QSDRec(
        num_items=num_items,
        num_semantic_tokens=num_semantic_tokens,
        semantic_id_table=semantic_table,
        dim=int(saved_args.get("dim", 64)),
        max_len=int(saved_args.get("max_len", 50)),
        num_interests=int(saved_args.get("num_interests", 4)),
        num_heads=int(saved_args.get("num_heads", 2)),
        num_layers=int(saved_args.get("num_layers", 2)),
        dropout=float(saved_args.get("dropout", 0.2)),
        interest_router=str(saved_args.get("interest_router", "semantic")),
        prefix_level=int(saved_args.get("prefix_level", 2)),
        semantic_token_log_prior=semantic_token_log_prior,
        mini_cluster_table=mini_cluster_table,
        mini_cluster_log_prior=mini_cluster_log_prior,
        hub_score_weight=float(saved_args.get("hub_score_weight", 0.0)),
        hub_attn_weight=float(saved_args.get("hub_attn_weight", 0.0)),
        evidence_gate=str(saved_args.get("evidence_gate", "none")),
        evidence_floor=float(saved_args.get("evidence_floor", 0.1)),
        evidence_recency_weight=float(saved_args.get("evidence_recency_weight", 0.0)),
        evidence_hub_weight=float(saved_args.get("evidence_hub_weight", 0.0)),
        evidence_cross_weight=float(saved_args.get("evidence_cross_weight", 0.2)),
        prior_lift_alpha=float(saved_args.get("prior_lift_alpha", 0.1)),
        prior_lift_tau=float(saved_args.get("prior_lift_tau", 1.0)),
        prior_lift_eta=float(saved_args.get("prior_lift_eta", 1.0)),
        hub_penalty_weight=float(saved_args.get("hub_penalty_weight", 0.0)),
        semantic_fusion=str(saved_args.get("semantic_fusion", "fixed")),
        fusion_floor=float(saved_args.get("fusion_floor", 0.0)),
        contrastive_alpha=float(saved_args.get("contrastive_alpha", 0.0)),
    )
    model.load_state_dict(state["model"], strict=False)
    return model, saved_args


def build_prefix_sizes(item_semantic_ids: Dict[int, List[int]], prefix_level: int) -> Dict[int, int]:
    groups = defaultdict(list)
    for item, sid in item_semantic_ids.items():
        groups[tuple(sid[:prefix_level])].append(item)
    sizes = {}
    for items in groups.values():
        for item in items:
            sizes[item] = len(items)
    return sizes


def build_overlap_sizes(item_semantic_ids: Dict[int, List[int]], min_overlap_slots: int) -> Dict[int, int]:
    inverted = defaultdict(list)
    for item, sid in item_semantic_ids.items():
        for slot, code in enumerate(sid):
            inverted[(slot, int(code))].append(item)

    sizes = {}
    for item, sid in item_semantic_ids.items():
        counts = Counter()
        for slot, code in enumerate(sid):
            counts.update(inverted[(slot, int(code))])
        sizes[item] = sum(1 for count in counts.values() if count >= min_overlap_slots)
    return sizes


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
            stats[f"HR@{k}"] += 1.0
            stats[f"Recall@{k}"] += 1.0
            stats[f"NDCG@{k}"] += 1.0 / torch.log2(torch.tensor(float(rank + 1))).item()


def finalize(stats: Dict[str, Any], ks: Sequence[int]) -> Dict[str, Any]:
    count = stats["count"]
    row = {"count": count}
    for k in ks:
        row[f"HR@{k}"] = stats[f"HR@{k}"] / max(count, 1)
        row[f"Recall@{k}"] = stats[f"Recall@{k}"] / max(count, 1)
        row[f"NDCG@{k}"] = stats[f"NDCG@{k}"] / max(count, 1)
    return row


@torch.no_grad()
def score_full(model: QSDRec, seq: torch.Tensor, num_items: int, sem_weight: float, chunk_size: int) -> torch.Tensor:
    batch_size = seq.size(0)
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=seq.device)
    chunks = []
    for start in range(0, num_items, chunk_size):
        cand = all_items[start : start + chunk_size].unsqueeze(0).expand(batch_size, -1)
        chunks.append(model(seq, cand, sem_weight=sem_weight)["score"])
    scores = torch.cat(chunks, dim=1)
    seen_mask = seq.gt(0)
    if seen_mask.any():
        rows, cols = seen_mask.nonzero(as_tuple=True)
        scores[rows, seq[rows, cols] - 1] = float("-inf")
    return scores


def collect_rank(topk_idx: torch.Tensor, target: int, max_k: int) -> int:
    hit = topk_idx.eq(target).nonzero(as_tuple=False)
    return int(hit[0].item() + 1) if hit.numel() else max_k + 1


@torch.no_grad()
def compare_grouped(
    base_model: QSDRec,
    comp_model: QSDRec,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    base_sem_weight: float,
    comp_sem_weight: float,
    sharing_sizes: Dict[int, int],
    buckets,
    batch_eval_size: int,
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[str, Any]:
    base_model.eval()
    comp_model.eval()
    max_k = max(ks)
    metric_keys = ["all"] + [name for name, _, _ in buckets] + ["other"]
    stats = {
        "base": {key: new_stats(ks) for key in metric_keys},
        "compare": {key: new_stats(ks) for key in metric_keys},
    }
    bucket_counts = Counter()

    for seq, targets in loader:
        seq = seq.to(device)
        targets = targets.to(device)
        base_topk = score_full(base_model, seq, num_items, base_sem_weight, batch_eval_size).topk(max_k, dim=1).indices + 1
        comp_topk = score_full(comp_model, seq, num_items, comp_sem_weight, batch_eval_size).topk(max_k, dim=1).indices + 1

        for row_idx in range(seq.size(0)):
            target = int(targets[row_idx].item())
            size = sharing_sizes.get(target, 1)
            bucket = bucket_name(size, buckets)
            bucket_counts[bucket] += 1
            base_rank = collect_rank(base_topk[row_idx], target, max_k)
            comp_rank = collect_rank(comp_topk[row_idx], target, max_k)
            add_metrics(stats["base"]["all"], base_rank, ks)
            add_metrics(stats["base"][bucket], base_rank, ks)
            add_metrics(stats["compare"]["all"], comp_rank, ks)
            add_metrics(stats["compare"][bucket], comp_rank, ks)

    result = {
        model_name: {key: finalize(value, ks) for key, value in model_stats.items() if value["count"] > 0}
        for model_name, model_stats in stats.items()
    }
    result["bucket_counts"] = dict(bucket_counts)
    result["delta_compare_minus_base"] = {}
    for key in result["compare"]:
        if key not in result["base"]:
            continue
        result["delta_compare_minus_base"][key] = {
            metric: result["compare"][key][metric] - result["base"][key][metric]
            for metric in result["compare"][key]
            if metric != "count"
        }
        result["delta_compare_minus_base"][key]["count"] = result["compare"][key]["count"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--compare-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--semantic-ids", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-eval-size", type=int, default=None)
    parser.add_argument("--sharing-mode", choices=["prefix", "overlap"], default="overlap")
    parser.add_argument("--prefix-level", type=int, default=2)
    parser.add_argument("--min-overlap-slots", type=int, default=2)
    parser.add_argument("--buckets", default="1,2-5,6-10,>10")
    args = parser.parse_args()

    state = torch.load(args.base_checkpoint, map_location="cpu")
    cfg = resolve_args(state.get("args", {}), args)
    dataset_dir = Path(cfg["dataset_dir"])
    semantic_path = Path(cfg["semantic_ids"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg.get("batch_size", 256))
    batch_eval_size = int(cfg.get("eval_batch_eval_size", 1024))

    sequences = read_json(dataset_dir / "sequences.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(semantic_path)
    num_items = int(stats["num_items"])
    semantic_table, item_semantic_ids, num_semantic_tokens = build_semantic_table(semantic_obj, num_items)

    if args.sharing_mode == "prefix":
        sharing_sizes = build_prefix_sizes(item_semantic_ids, args.prefix_level)
    else:
        sharing_sizes = build_overlap_sizes(item_semantic_ids, args.min_overlap_slots)

    base_model, base_args = load_model(Path(args.base_checkpoint), cfg, semantic_table, num_semantic_tokens, num_items)
    comp_model, comp_args = load_model(Path(args.compare_checkpoint), cfg, semantic_table, num_semantic_tokens, num_items)
    base_model.to(device)
    comp_model.to(device)

    test_data = NextItemDataset(sequences, int(base_args.get("max_len", 50)), "test")
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_full_eval)
    buckets = parse_bucket_spec(args.buckets)
    grouped = compare_grouped(
        base_model=base_model,
        comp_model=comp_model,
        loader=loader,
        device=device,
        num_items=num_items,
        base_sem_weight=float(base_args.get("sem_weight", 1.0)),
        comp_sem_weight=float(comp_args.get("sem_weight", 1.0)),
        sharing_sizes=sharing_sizes,
        buckets=buckets,
        batch_eval_size=batch_eval_size,
    )
    output = {
        "base_checkpoint": args.base_checkpoint,
        "compare_checkpoint": args.compare_checkpoint,
        "dataset_dir": str(dataset_dir),
        "semantic_ids": str(semantic_path),
        "sharing_mode": args.sharing_mode,
        "prefix_level": args.prefix_level,
        "min_overlap_slots": args.min_overlap_slots,
        "buckets": args.buckets,
        "base_args": base_args,
        "compare_args": comp_args,
        "grouped_metrics": grouped,
    }
    write_json(args.output, output)
    print(json.dumps(grouped["delta_compare_minus_base"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
