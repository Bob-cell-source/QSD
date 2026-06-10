import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json
from qsdrec.train import (
    NextItemDataset,
    build_semantic_table,
    build_soft_semantic_table,
    build_train_item_frequency,
    collate_full_eval,
)
from scripts.evaluate_lcsoft_group_benchmarks import (
    evaluate_model,
    instantiate_model,
    model_config,
)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def soft_entropy(weights: torch.Tensor) -> np.ndarray:
    values = weights[1:].clamp_min(1e-12)
    entropy = -(values * values.log()).sum(dim=-1)
    valid_slots = weights[1:].sum(dim=-1).gt(0)
    return (
        (entropy * valid_slots.float()).sum(dim=-1)
        / valid_slots.sum(dim=-1).clamp_min(1)
    ).cpu().numpy()


def quantile_bins(values: np.ndarray, num_bins: int) -> Tuple[np.ndarray, List[float]]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    internal = edges[1:-1]
    assignments = np.searchsorted(internal, values, side="right")
    return assignments, [float(value) for value in edges]


def write_rows(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "bin",
        "item_count",
        "reliability_min",
        "reliability_max",
        "reliability_mean",
        "frequency_mean",
        "frequency_median",
        "soft_entropy_mean",
        "test_count",
        "NDCG@5",
        "HR@5",
        "NDCG@10",
        "HR@10",
        "NDCG@20",
        "HR@20",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-eval-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--min-overlap-slots", type=int, default=2)
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--support-eta", type=float, default=2.0)
    parser.add_argument("--hard-token-prior", type=float, default=1.0)
    parser.add_argument("--reliability-floor", type=float, default=0.10)
    parser.add_argument("--max-neighbors", type=int, default=50)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    sequences = read_json(dataset_dir / "sequences.json")
    stats = read_json(dataset_dir / "stats.json")
    semantic_obj = read_json(args.semantic_ids)
    num_items = int(stats["num_items"])
    semantic_table, item_codes, num_tokens = build_semantic_table(semantic_obj, num_items)
    soft_ids, soft_weights, reliability_tensor = build_soft_semantic_table(
        semantic_table=semantic_table,
        item_semantic_ids=item_codes,
        num_items=num_items,
        top_m=args.top_m,
        min_overlap_slots=args.min_overlap_slots,
        min_support=args.min_support,
        support_eta=args.support_eta,
        hard_token_prior=args.hard_token_prior,
        reliability_floor=args.reliability_floor,
        max_neighbors=args.max_neighbors,
    )
    frequency_tensor = build_train_item_frequency(sequences, num_items)
    reliability = reliability_tensor[1:].cpu().numpy()
    frequency = frequency_tensor[1:].cpu().numpy()
    entropy = soft_entropy(soft_weights)
    assignments, edges = quantile_bins(reliability, args.num_bins)

    rows = []
    for bin_id in range(args.num_bins):
        mask = assignments == bin_id
        rows.append(
            {
                "bin": f"R{bin_id + 1}",
                "item_count": int(mask.sum()),
                "reliability_min": float(reliability[mask].min()) if mask.any() else None,
                "reliability_max": float(reliability[mask].max()) if mask.any() else None,
                "reliability_mean": float(reliability[mask].mean()) if mask.any() else None,
                "frequency_mean": float(frequency[mask].mean()) if mask.any() else None,
                "frequency_median": float(np.median(frequency[mask])) if mask.any() else None,
                "soft_entropy_mean": float(entropy[mask].mean()) if mask.any() else None,
            }
        )

    test_metrics: Dict[str, Dict[str, Any]] = {}
    checkpoint_args = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_args = state.get("args", {})
        cli_override = argparse.Namespace(
            dataset_dir=str(dataset_dir),
            semantic_ids=str(args.semantic_ids),
            device=args.device,
            batch_size=args.batch_size,
            eval_batch_eval_size=args.eval_batch_eval_size,
        )
        cfg = model_config(checkpoint_args, cli_override)
        model = instantiate_model(
            state,
            cfg,
            sequences,
            semantic_table,
            item_codes,
            num_tokens,
            num_items,
        )
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        model.to(device)
        test_data = NextItemDataset(sequences, int(cfg.get("max_len", 50)), "test")
        sample_groups = [[f"R{assignments[int(target) - 1] + 1}"] for _, target in test_data]
        loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_full_eval,
        )
        test_metrics = evaluate_model(
            model=model,
            loader=loader,
            device=device,
            num_items=num_items,
            sem_weight=float(cfg.get("sem_weight", 1.0)),
            sample_group_lists=sample_groups,
            ks=(5, 10, 20),
            batch_eval_size=args.eval_batch_eval_size,
        )
        for row in rows:
            metrics = test_metrics.get(row["bin"], {})
            row["test_count"] = metrics.get("count", 0)
            row.update({key: value for key, value in metrics.items() if key != "count"})

    report = {
        "dataset": dataset_dir.name,
        "num_items": num_items,
        "reliability_frequency_spearman": spearman(reliability, frequency),
        "reliability_log_frequency_spearman": spearman(reliability, np.log1p(frequency)),
        "reliability_entropy_spearman": spearman(reliability, entropy),
        "reliability_quantile_edges": edges,
        "bins": rows,
        "test_metrics": test_metrics,
        "checkpoint_args": checkpoint_args,
        "configuration": vars(args),
    }
    output = Path(args.output)
    write_json(output, report)
    write_rows(rows, Path(args.csv) if args.csv else output.with_suffix(".csv"))
    print(json.dumps({key: value for key, value in report.items() if "spearman" in key}, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
