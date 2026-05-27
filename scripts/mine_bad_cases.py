import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json
from qsdrec.model import QSDRec
from qsdrec.train import NextItemDataset, build_semantic_table, collate_full_eval


def load_model(checkpoint: Path, cli_args, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu")
    cfg = dict(state.get("args", {}))
    if cli_args.dataset_dir:
        cfg["dataset_dir"] = cli_args.dataset_dir
    if cli_args.semantic_ids:
        cfg["semantic_ids"] = cli_args.semantic_ids

    stats = read_json(Path(cfg["dataset_dir"]) / "stats.json")
    semantic_obj = read_json(cfg["semantic_ids"])
    num_items = int(stats["num_items"])
    semantic_table, item_semantic_ids, num_semantic_tokens = build_semantic_table(semantic_obj, num_items)

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
        hub_score_weight=float(cfg.get("hub_score_weight", 0.0)),
        hub_attn_weight=float(cfg.get("hub_attn_weight", 0.0)),
        evidence_gate=str(cfg.get("evidence_gate", "none")),
        evidence_floor=float(cfg.get("evidence_floor", 0.1)),
        evidence_recency_weight=float(cfg.get("evidence_recency_weight", 0.0)),
        evidence_hub_weight=float(cfg.get("evidence_hub_weight", 0.0)),
        evidence_cross_weight=float(cfg.get("evidence_cross_weight", 0.2)),
        hub_penalty_weight=float(cfg.get("hub_penalty_weight", 0.0)),
        semantic_fusion=str(cfg.get("semantic_fusion", "fixed")),
        fusion_floor=float(cfg.get("fusion_floor", 0.0)),
        contrastive_alpha=float(cfg.get("contrastive_alpha", 0.0)),
    )
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    return model, cfg, num_items, item_semantic_ids


def build_prefix_group_sizes(item_semantic_ids: Dict[int, List[int]], prefix_level: int) -> Dict[int, int]:
    groups = defaultdict(list)
    for item, sid in item_semantic_ids.items():
        groups[tuple(sid[:prefix_level])].append(item)
    sizes = {}
    for items in groups.values():
        for item in items:
            sizes[item] = len(items)
    return sizes


def build_overlap_group_sizes(item_semantic_ids: Dict[int, List[int]], min_overlap_slots: int) -> Dict[int, int]:
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


def build_train_item_counts(sequences: List[Dict[str, Any]]) -> Dict[int, int]:
    counts = defaultdict(int)
    for row in sequences:
        items = row["items"]
        for item in items[1:-2]:
            counts[int(item)] += 1
    return dict(counts)


def item_brief(
    item_id: int,
    item_meta: Dict[str, Any],
    item_semantic_ids: Dict[int, List[int]],
    item_counts: Dict[int, int],
    prefix_sizes: Dict[int, int],
    overlap_sizes: Dict[int, int],
):
    meta = item_meta.get(str(item_id), {})
    return {
        "item_id": item_id,
        "asin": meta.get("asin"),
        "title": meta.get("title", ""),
        "brand": meta.get("brand", ""),
        "categories": meta.get("categories", []),
        "semantic_id": item_semantic_ids.get(item_id),
        "train_count": item_counts.get(item_id, 0),
        "prefix_group_size": prefix_sizes.get(item_id, 1),
        "overlap_group_size": overlap_sizes.get(item_id, 1),
    }


def rank_from_topk(topk: torch.Tensor, target: int) -> int | None:
    hit = (topk == target).nonzero(as_tuple=False)
    if hit.numel() == 0:
        return None
    return int(hit[0].item() + 1)


@torch.no_grad()
def score_topk(model: QSDRec, seq: torch.Tensor, num_items: int, sem_weight: float, chunk_size: int, top_k: int):
    batch_size = seq.size(0)
    device = seq.device
    all_items = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    score_chunks = []
    for start in range(0, num_items, chunk_size):
        cand = all_items[start : start + chunk_size]
        cand = cand.unsqueeze(0).expand(batch_size, -1)
        score_chunks.append(model(seq, cand, sem_weight=sem_weight)["score"])
    scores = torch.cat(score_chunks, dim=1)

    seen_mask = seq.gt(0)
    if seen_mask.any():
        history_rows, history_cols = seen_mask.nonzero(as_tuple=True)
        history_item_idx = seq[history_rows, history_cols] - 1
        scores[history_rows, history_item_idx] = float("-inf")

    values, indices = scores.topk(k=top_k, dim=1)
    return indices + 1, values


def classify_case(base_rank, comp_rank, top_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= top_k
    comp_hit = comp_rank is not None and comp_rank <= top_k
    if base_hit and not comp_hit:
        return "base_correct_comp_wrong"
    if comp_hit and not base_hit:
        return "comp_correct_base_wrong"
    if base_hit and comp_hit:
        return "both_correct"
    return "both_wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--compare-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--semantic-ids", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-cases", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-eval-size", type=int, default=1024)
    parser.add_argument("--prefix-level", type=int, default=2)
    parser.add_argument("--min-prefix-size", type=int, default=1)
    parser.add_argument("--sharing-mode", choices=["prefix", "overlap"], default="prefix")
    parser.add_argument("--min-overlap-slots", type=int, default=2)
    parser.add_argument("--min-overlap-size", type=int, default=1)
    parser.add_argument("--max-train-count", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, base_cfg, num_items, item_semantic_ids = load_model(Path(args.base_checkpoint), args, device)
    comp_model, comp_cfg, comp_num_items, _ = load_model(Path(args.compare_checkpoint), args, device)
    if num_items != comp_num_items:
        raise ValueError("Base and compare checkpoints use different num_items.")

    dataset_dir = Path(args.dataset_dir or base_cfg["dataset_dir"])
    sequences = read_json(dataset_dir / "sequences.json")
    item_meta = read_json(dataset_dir / "item_meta.json")
    item_counts = build_train_item_counts(sequences)
    prefix_sizes = build_prefix_group_sizes(item_semantic_ids, args.prefix_level)
    overlap_sizes = build_overlap_group_sizes(item_semantic_ids, args.min_overlap_slots)

    test_data = NextItemDataset(sequences, int(base_cfg.get("max_len", 50)), "test")
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_full_eval)

    buckets = {
        "base_correct_comp_wrong": [],
        "comp_correct_base_wrong": [],
        "both_correct": [],
        "both_wrong": [],
    }
    sample_index = 0
    for seq, targets in loader:
        seq = seq.to(device)
        targets = targets.to(device)
        base_topk, _ = score_topk(
            base_model,
            seq,
            num_items,
            float(base_cfg.get("sem_weight", 1.0)),
            args.eval_batch_eval_size,
            args.top_k,
        )
        comp_topk, _ = score_topk(
            comp_model,
            seq,
            num_items,
            float(comp_cfg.get("sem_weight", 1.0)),
            args.eval_batch_eval_size,
            args.top_k,
        )

        for row_idx in range(seq.size(0)):
            target = int(targets[row_idx].item())
            prefix_size = prefix_sizes.get(target, 1)
            overlap_size = overlap_sizes.get(target, 1)
            train_count = item_counts.get(target, 0)
            sharing_size = prefix_size if args.sharing_mode == "prefix" else overlap_size
            min_sharing_size = args.min_prefix_size if args.sharing_mode == "prefix" else args.min_overlap_size
            if sharing_size < min_sharing_size:
                sample_index += 1
                continue
            if args.max_train_count is not None and train_count > args.max_train_count:
                sample_index += 1
                continue

            base_rank = rank_from_topk(base_topk[row_idx], target)
            comp_rank = rank_from_topk(comp_topk[row_idx], target)
            kind = classify_case(base_rank, comp_rank, args.top_k)
            if len(buckets[kind]) < args.num_cases:
                history_items = [int(x) for x in seq[row_idx].detach().cpu().tolist() if int(x) > 0]
                buckets[kind].append(
                    {
                        "sample_index": sample_index,
                        "case_type": kind,
                        "base_rank_at_k": base_rank,
                        "compare_rank_at_k": comp_rank,
                        "target": item_brief(
                            target,
                            item_meta,
                            item_semantic_ids,
                            item_counts,
                            prefix_sizes,
                            overlap_sizes,
                        ),
                        "history_tail": [
                            item_brief(item, item_meta, item_semantic_ids, item_counts, prefix_sizes, overlap_sizes)
                            for item in history_items[-5:]
                        ],
                        "base_top_items": [
                            item_brief(int(item), item_meta, item_semantic_ids, item_counts, prefix_sizes, overlap_sizes)
                            for item in base_topk[row_idx].detach().cpu().tolist()[: args.top_k]
                        ],
                        "compare_top_items": [
                            item_brief(int(item), item_meta, item_semantic_ids, item_counts, prefix_sizes, overlap_sizes)
                            for item in comp_topk[row_idx].detach().cpu().tolist()[: args.top_k]
                        ],
                    }
                )
            sample_index += 1
        if all(len(v) >= args.num_cases for v in buckets.values()):
            break

    output = {
        "base_checkpoint": args.base_checkpoint,
        "compare_checkpoint": args.compare_checkpoint,
        "dataset_dir": str(dataset_dir),
        "top_k": args.top_k,
        "filters": {
            "sharing_mode": args.sharing_mode,
            "prefix_level": args.prefix_level,
            "min_prefix_size": args.min_prefix_size,
            "min_overlap_slots": args.min_overlap_slots,
            "min_overlap_size": args.min_overlap_size,
            "max_train_count": args.max_train_count,
        },
        "base_args": base_cfg,
        "compare_args": comp_cfg,
        "cases": buckets,
    }
    write_json(args.output, output)
    print(json.dumps({k: len(v) for k, v in buckets.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
