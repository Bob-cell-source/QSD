import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json
from qsdrec.model import CRSIDRec, QSDRec
from qsdrec.train import (
    NextItemDataset,
    build_behavior_neighbors,
    build_log_prior,
    build_semantic_hubness,
    build_semantic_table,
    build_soft_semantic_table,
    build_train_item_frequency,
    collate_full_eval,
    load_mini_cluster_table,
)


TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "new",
    "pack",
    "set",
    "size",
}


def parse_checkpoint_specs(specs: Sequence[str]) -> Dict[str, Path]:
    checkpoints = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Checkpoint spec must be label=path, got: {spec}")
        label, path = spec.split("=", 1)
        checkpoints[label.strip()] = Path(path)
    return checkpoints


def parse_ks(value: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if token not in TITLE_STOPWORDS and len(token) > 1
    }


def title_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def item_features(item_meta: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    features = {}
    for raw_item, meta in item_meta.items():
        item = int(raw_item)
        features[item] = {
            "title_tokens": tokenize(meta.get("title", "")),
            "brand": str(meta.get("brand", "") or "").strip().lower(),
            "categories": {str(x).strip().lower() for x in (meta.get("categories") or []) if str(x).strip()},
        }
    return features


def build_train_target_counts(sequences: List[Dict[str, Any]]) -> Dict[int, int]:
    counts = Counter()
    for row in sequences:
        items = row["items"]
        if len(items) < 3:
            continue
        for item in items[1:-2]:
            counts[int(item)] += 1
    return dict(counts)


def build_prefix_sizes(item_semantic_ids: Dict[int, List[int]], prefix_level: int) -> Dict[int, int]:
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for item, sid in item_semantic_ids.items():
        groups[tuple(sid[:prefix_level])].append(item)
    sizes = {}
    for items in groups.values():
        for item in items:
            sizes[item] = len(items)
    return sizes


def build_overlap_sizes(item_semantic_ids: Dict[int, List[int]], min_overlap_slots: int) -> Dict[int, int]:
    inverted: Dict[Tuple[int, int], List[int]] = defaultdict(list)
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


def build_popular_token_flags(
    item_semantic_ids: Dict[int, List[int]],
    quantile: float,
) -> Tuple[Dict[int, bool], Dict[str, Any]]:
    depth = len(next(iter(item_semantic_ids.values()))) if item_semantic_ids else 0
    slot_counts: List[Counter[int]] = [Counter() for _ in range(depth)]
    for sid in item_semantic_ids.values():
        for slot, code in enumerate(sid):
            slot_counts[slot][int(code)] += 1

    # Aggregate normalized per-slot token frequencies into one item-level
    # hubness score. Quantiling these item scores avoids the old "any popular
    # token" rule, which marked almost every item when the SID has many slots.
    slot_max_counts = [max(counts.values(), default=1) for counts in slot_counts]
    item_scores = {}
    for item, sid in item_semantic_ids.items():
        normalized = [
            math.log1p(slot_counts[slot][int(code)]) / math.log1p(slot_max_counts[slot])
            for slot, code in enumerate(sid)
        ]
        item_scores[item] = sum(normalized) / max(len(normalized), 1)

    values = sorted(item_scores.values())
    if values:
        idx = min(len(values) - 1, max(0, int(math.ceil(quantile * len(values))) - 1))
        threshold = values[idx]
    else:
        threshold = float("inf")
    flags = {item: score >= threshold for item, score in item_scores.items()}
    return flags, {
        "quantile": quantile,
        "item_hubness_threshold": threshold,
        "num_flagged_items": sum(flags.values()),
        "num_items": len(flags),
        "score_definition": "mean slot-wise log-normalized SID token frequency",
    }


def sid_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(int(x) == int(y) for x, y in zip(a, b))


def has_hard_sid_mismatch(
    history: Sequence[int],
    target: int,
    features: Dict[int, Dict[str, Any]],
    item_semantic_ids: Dict[int, List[int]],
    title_threshold: float,
    min_title_overlap: int,
    min_sid_overlap: int,
) -> bool:
    target_feat = features.get(target, {})
    target_brand = target_feat.get("brand", "")
    target_title = target_feat.get("title_tokens", set())
    target_sid = item_semantic_ids.get(target)
    if target_sid is None or not target_title:
        return False

    for item in history:
        feat = features.get(int(item), {})
        history_title = feat.get("title_tokens", set())
        overlap = len(target_title.intersection(history_title))
        same_brand = bool(target_brand and target_brand == feat.get("brand", ""))
        strong_title_match = overlap >= min_title_overlap and title_jaccard(target_title, history_title) >= title_threshold
        brand_supported_match = same_brand and overlap >= 1
        if not (strong_title_match or brand_supported_match):
            continue

        history_sid = item_semantic_ids.get(int(item))
        if history_sid is not None and sid_overlap(target_sid, history_sid) < min_sid_overlap:
            return True
    return False


def sample_groups(
    seq: torch.Tensor,
    target: int,
    train_counts: Dict[int, int],
    prefix_sizes: Dict[int, int],
    overlap_sizes: Dict[int, int],
    popular_token_flags: Dict[int, bool],
    item_semantic_ids: Dict[int, List[int]],
    features: Dict[int, Dict[str, Any]],
    args,
) -> List[str]:
    history = [int(x) for x in seq.tolist() if int(x) > 0]
    groups = ["overall"]
    if train_counts.get(target, 0) < args.low_freq_threshold:
        groups.append("low_frequency")
    if prefix_sizes.get(target, 1) >= args.high_sharing_threshold:
        groups.append("high_sharing")
    if (
        prefix_sizes.get(target, 1) <= args.isolated_prefix_threshold
        or overlap_sizes.get(target, 1) <= args.isolated_overlap_threshold
    ):
        groups.append("isolated_sid")
    if popular_token_flags.get(target, False):
        groups.append("popular_token")
    if has_hard_sid_mismatch(
        history,
        target,
        features,
        item_semantic_ids,
        args.mismatch_title_jaccard,
        args.mismatch_min_title_overlap,
        args.min_overlap_slots,
    ):
        groups.append("mismatch")
    return groups


def new_stats(ks: Sequence[int]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"count": 0}
    for k in ks:
        row[f"HR@{k}"] = 0.0
        row[f"Recall@{k}"] = 0.0
        row[f"NDCG@{k}"] = 0.0
    return row


def add_rank(stats: Dict[str, Any], rank: int, ks: Sequence[int]) -> None:
    stats["count"] += 1
    for k in ks:
        if rank <= k:
            gain = 1.0 / torch.log2(torch.tensor(float(rank + 1))).item()
            stats[f"HR@{k}"] += 1.0
            stats[f"Recall@{k}"] += 1.0
            stats[f"NDCG@{k}"] += gain


def finalize(stats: Dict[str, Any], ks: Sequence[int]) -> Dict[str, Any]:
    count = stats["count"]
    row = {"count": count}
    for k in ks:
        row[f"HR@{k}"] = stats[f"HR@{k}"] / max(count, 1)
        row[f"Recall@{k}"] = stats[f"Recall@{k}"] / max(count, 1)
        row[f"NDCG@{k}"] = stats[f"NDCG@{k}"] / max(count, 1)
    return row


def model_config(saved_args: Dict[str, Any], cli_args) -> Dict[str, Any]:
    cfg = dict(saved_args)
    for key in ["dataset_dir", "semantic_ids", "device", "batch_size", "eval_batch_eval_size"]:
        value = getattr(cli_args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def instantiate_model(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    sequences: List[Dict[str, Any]],
    semantic_table: torch.Tensor,
    item_semantic_ids: Dict[int, List[int]],
    num_semantic_tokens: int,
    num_items: int,
):
    variant = str(cfg.get("model_variant", "qsdrec"))
    if variant in {"crsid", "crsid_semhub", "crsid_soft"}:
        item_frequency = build_train_item_frequency(sequences, num_items)
        semantic_token_hubness, _ = build_semantic_hubness(semantic_table, num_semantic_tokens)
        soft_table = None
        soft_weight = None
        reliability = None
        if variant == "crsid_soft":
            behavior_neighbors = None
            if float(cfg.get("cr_soft_behavior_weight", 0.0)) > 0.0:
                behavior_neighbors = build_behavior_neighbors(
                    sequences=sequences,
                    num_items=num_items,
                    window_size=int(cfg.get("cr_soft_behavior_window", 5)),
                    min_count=int(cfg.get("cr_soft_behavior_min_count", 2)),
                    max_neighbors=int(cfg.get("cr_soft_max_behavior_neighbors", 50)),
                )
            soft_table, soft_weight, reliability = build_soft_semantic_table(
                semantic_table=semantic_table,
                item_semantic_ids=item_semantic_ids,
                num_items=num_items,
                top_m=int(cfg.get("cr_soft_top_m", 4)),
                min_overlap_slots=int(cfg.get("cr_soft_min_overlap_slots", 2)),
                min_support=float(cfg.get("cr_soft_min_support", 0.05)),
                support_eta=float(cfg.get("cr_soft_support_eta", 1.0)),
                hard_token_prior=float(cfg.get("cr_soft_hard_token_prior", 1.0)),
                reliability_floor=float(cfg.get("cr_soft_reliability_floor", 0.10)),
                max_neighbors=int(cfg.get("cr_soft_max_neighbors", 50)),
                lift_kappa=float(cfg.get("cr_soft_lift_kappa", 0.0)),
                lift_clip=float(cfg.get("cr_soft_lift_clip", 5.0)),
                lift_eps=float(cfg.get("cr_soft_lift_eps", 1e-6)),
                decouple_reliability=bool(cfg.get("cr_soft_decouple_reliability", False)),
                behavior_neighbors=behavior_neighbors,
                behavior_neighbor_weight=float(cfg.get("cr_soft_behavior_weight", 0.0)),
            )
        model = CRSIDRec(
            num_items=num_items,
            num_semantic_tokens=num_semantic_tokens,
            semantic_id_table=semantic_table,
            item_frequency=item_frequency,
            soft_semantic_id_table=soft_table,
            soft_semantic_id_weight=soft_weight,
            semantic_reliability=reliability,
            dim=int(cfg.get("dim", 64)),
            max_len=int(cfg.get("max_len", 50)),
            num_heads=int(cfg.get("num_heads", 2)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
            tail_tau=float(cfg.get("cr_tail_tau", 20.0)),
            residual_scale=float(cfg.get("cr_residual_scale", 1.0)),
            alpha_mode="semantic_hubness" if variant == "crsid_semhub" else "item_frequency",
            semantic_token_hubness=semantic_token_hubness,
            hub_alpha_floor=float(cfg.get("cr_hub_alpha_floor", 0.05)),
            hub_alpha_gamma=float(cfg.get("cr_hub_alpha_gamma", 1.0)),
            disable_semantic_basis=bool(cfg.get("cr_disable_semantic_basis", False)),
            disable_shared_residual=bool(cfg.get("cr_disable_shared_residual", False)),
            disable_private_residual=bool(cfg.get("cr_disable_private_residual", False)),
            alpha_override=cfg.get("cr_alpha_override"),
            alpha_frequency_transform=str(cfg.get("cr_alpha_frequency_transform", "raw")),
        )
    else:
        semantic_token_log_prior = build_log_prior(semantic_table, num_semantic_tokens)
        mini_cluster_table, mini_cluster_log_prior = load_mini_cluster_table(
            cfg.get("mini_clusters"),
            num_items,
            semantic_table,
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
    return model


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    sem_weight: float,
    sample_group_lists: List[List[str]],
    ks: Sequence[int],
    batch_eval_size: int,
) -> Dict[str, Dict[str, Any]]:
    model.eval()
    max_k = max(ks)
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: new_stats(ks))
    sample_offset = 0

    for seq, targets in loader:
        seq = seq.to(device)
        targets = targets.to(device)
        batch_size = seq.size(0)
        score_chunks = []
        for start in range(0, num_items, batch_eval_size):
            cand = all_items[start : start + batch_eval_size].unsqueeze(0).expand(batch_size, -1)
            score_chunks.append(model(seq, cand, sem_weight=sem_weight)["score"])
        scores = torch.cat(score_chunks, dim=1)

        seen_mask = seq.gt(0)
        if seen_mask.any():
            rows, cols = seen_mask.nonzero(as_tuple=True)
            scores[rows, seq[rows, cols] - 1] = float("-inf")

        topk = scores.topk(k=max_k, dim=1).indices + 1
        matches = topk.eq(targets.unsqueeze(1))
        for row_idx in range(batch_size):
            hit_positions = matches[row_idx].nonzero(as_tuple=False)
            rank = int(hit_positions[0].item() + 1) if hit_positions.numel() else max_k + 1
            for group in sample_group_lists[sample_offset + row_idx]:
                add_rank(stats[group], rank, ks)
        sample_offset += batch_size

    return {group: finalize(group_stats, ks) for group, group_stats in sorted(stats.items())}


def write_csv_rows(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "group",
        "model",
        "count",
        "NDCG@5",
        "HR@5",
        "NDCG@10",
        "HR@10",
        "NDCG@20",
        "HR@20",
        "gain_vs_lcsoft_NDCG@10",
        "lcsoft_gain_over_model_NDCG@10",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="Model checkpoint as label=path.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--main-label", default="lcsoft")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-eval-size", type=int, default=256)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--prefix-level", type=int, default=2)
    parser.add_argument("--min-overlap-slots", type=int, default=2)
    parser.add_argument("--low-freq-threshold", type=int, default=5)
    parser.add_argument("--high-sharing-threshold", type=int, default=10)
    parser.add_argument("--isolated-prefix-threshold", type=int, default=1)
    parser.add_argument("--isolated-overlap-threshold", type=int, default=1)
    parser.add_argument("--popular-token-quantile", type=float, default=0.90)
    parser.add_argument("--mismatch-title-jaccard", type=float, default=0.30)
    parser.add_argument("--mismatch-min-title-overlap", type=int, default=2)
    args = parser.parse_args()

    checkpoints = parse_checkpoint_specs(args.checkpoint)
    dataset_dir = Path(args.dataset_dir)
    semantic_path = Path(args.semantic_ids)
    output_path = Path(args.output)
    csv_path = Path(args.csv) if args.csv else output_path.with_suffix(".csv")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ks = parse_ks(args.ks)

    sequences = read_json(dataset_dir / "sequences.json")
    item_meta = read_json(dataset_dir / "item_meta.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(semantic_path)
    num_items = int(stats["num_items"])
    semantic_table, item_semantic_ids, num_semantic_tokens = build_semantic_table(semantic_obj, num_items)

    train_counts = build_train_target_counts(sequences)
    prefix_sizes = build_prefix_sizes(item_semantic_ids, args.prefix_level)
    overlap_sizes = build_overlap_sizes(item_semantic_ids, args.min_overlap_slots)
    popular_flags, popular_meta = build_popular_token_flags(item_semantic_ids, args.popular_token_quantile)
    features = item_features(item_meta)

    first_state = torch.load(next(iter(checkpoints.values())), map_location="cpu")
    first_cfg = model_config(first_state.get("args", {}), args)
    test_data = NextItemDataset(sequences, max_len=int(first_cfg.get("max_len", 50)), split="test")
    sample_group_lists = [
        sample_groups(seq, int(target), train_counts, prefix_sizes, overlap_sizes, popular_flags, item_semantic_ids, features, args)
        for seq, target in test_data
    ]
    group_counts = Counter(group for groups in sample_group_lists for group in groups)
    num_test_samples = len(test_data)
    group_rates = {group: count / max(num_test_samples, 1) for group, count in group_counts.items()}

    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_full_eval)
    model_results = {}
    model_args = {}
    for label, checkpoint in checkpoints.items():
        state = torch.load(checkpoint, map_location="cpu")
        cfg = model_config(state.get("args", {}), args)
        model = instantiate_model(state, cfg, sequences, semantic_table, item_semantic_ids, num_semantic_tokens, num_items)
        model.to(device)
        model_results[label] = evaluate_model(
            model=model,
            loader=loader,
            device=device,
            num_items=num_items,
            sem_weight=float(cfg.get("sem_weight", 1.0)),
            sample_group_lists=sample_group_lists,
            ks=ks,
            batch_eval_size=args.eval_batch_eval_size,
        )
        model_args[label] = cfg
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    main_result = model_results.get(args.main_label, {})
    for group in sorted(group_counts):
        main_ndcg = main_result.get(group, {}).get("NDCG@10")
        for label, result in model_results.items():
            metrics = result.get(group)
            if not metrics:
                continue
            ndcg = metrics.get("NDCG@10")
            row = {
                "dataset": dataset_dir.name,
                "group": group,
                "model": label,
                "count": metrics.get("count"),
                "NDCG@5": metrics.get("NDCG@5"),
                "HR@5": metrics.get("HR@5"),
                "NDCG@10": metrics.get("NDCG@10"),
                "HR@10": metrics.get("HR@10"),
                "NDCG@20": metrics.get("NDCG@20"),
                "HR@20": metrics.get("HR@20"),
            }
            if main_ndcg is not None and ndcg is not None:
                row["gain_vs_lcsoft_NDCG@10"] = ndcg - main_ndcg
                row["lcsoft_gain_over_model_NDCG@10"] = main_ndcg - ndcg
            rows.append(row)

    output = {
        "group_definition_version": "v2_strict_item_hubness_and_sid_mismatch",
        "dataset_dir": str(dataset_dir),
        "semantic_ids": str(semantic_path),
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "main_label": args.main_label,
        "group_definitions": {
            "low_frequency": f"train target count < {args.low_freq_threshold}",
            "high_sharing": f"prefix group size >= {args.high_sharing_threshold}, prefix_level={args.prefix_level}",
            "isolated_sid": (
                f"prefix group size <= {args.isolated_prefix_threshold} or "
                f"overlap group size <= {args.isolated_overlap_threshold}"
            ),
            "popular_token": (
                "target item-level SID hubness is at or above quantile "
                f"{args.popular_token_quantile}; hubness averages slot-wise log-normalized token frequencies"
            ),
            "mismatch": (
                "history contains an item-side match (same brand with >=1 informative shared title token, or "
                f">={args.mismatch_min_title_overlap} shared title tokens with title_jaccard >= "
                f"{args.mismatch_title_jaccard}) whose hard SID shares fewer than "
                f"{args.min_overlap_slots} aligned slots with the target"
            ),
        },
        "popular_token_meta": popular_meta,
        "group_counts": dict(group_counts),
        "group_rates": group_rates,
        "model_args": model_args,
        "model_results": model_results,
        "rows": rows,
    }
    write_json(output_path, output)
    write_csv_rows(rows, csv_path)
    print(
        json.dumps(
            {
                "group_definition_version": output["group_definition_version"],
                "group_counts": dict(group_counts),
                "group_rates": group_rates,
                "csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
