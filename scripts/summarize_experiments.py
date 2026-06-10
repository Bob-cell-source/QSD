import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_FIELDS = [
    "model_variant",
    "sem_weight",
    "num_interests",
    "cr_tail_tau",
    "cr_residual_scale",
    "cr_residual_reg",
    "cr_hub_alpha_floor",
    "cr_hub_alpha_gamma",
    "cr_disable_semantic_basis",
    "cr_disable_shared_residual",
    "cr_disable_private_residual",
    "cr_alpha_override",
    "cr_alpha_frequency_transform",
    "cr_soft_top_m",
    "cr_soft_min_overlap_slots",
    "cr_soft_min_support",
    "cr_soft_support_eta",
    "cr_soft_hard_token_prior",
    "cr_soft_reliability_floor",
    "cr_soft_max_neighbors",
    "cr_soft_neighbor_source",
    "cr_soft_lift_kappa",
    "cr_soft_lift_clip",
    "cr_soft_lift_eps",
    "cr_soft_decouple_reliability",
    "cr_soft_behavior_weight",
    "cr_soft_behavior_window",
    "cr_soft_behavior_min_count",
    "cr_soft_max_behavior_neighbors",
    "dis_weight",
    "div_weight",
    "num_hard_neg",
    "prefix_level",
    "dim",
    "num_layers",
    "dropout",
    "batch_size",
    "num_random_neg",
    "max_len",
    "lr",
    "weight_decay",
    "train_objective",
    "grad_clip",
    "seed",
]


def load_result(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "test" in obj:
        metrics = obj.get("test", {})
        args = obj.get("args", {})
        best_valid = obj.get("best_valid_NDCG@10")
    else:
        metrics = obj
        args = {}
        best_valid = None

    row: Dict[str, Any] = {
        "exp": path.parent.name,
        "path": str(path),
        "best_valid_NDCG@10": best_valid,
    }
    row.update(metrics)
    for field in DEFAULT_FIELDS:
        row[field] = args.get(field)
    return row


def collect_results(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("test_metrics.json")):
        try:
            rows.append(load_result(path))
        except Exception as exc:
            print(f"Skip invalid result: {path} ({exc})")
    return rows


def metric_value(row: Dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return float("-inf")
    return float(value)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "-"
    return str(value)


def print_table(rows: List[Dict[str, Any]], metric: str, top_k: int) -> None:
    fields = [
        "rank",
        "exp",
        "model_variant",
        "NDCG@10",
        "HR@10",
        "NDCG@20",
        "HR@20",
        "best_valid_NDCG@10",
        "sem_weight",
        "num_interests",
        "cr_tail_tau",
        "cr_residual_scale",
        "cr_hub_alpha_floor",
        "cr_hub_alpha_gamma",
        "cr_disable_semantic_basis",
        "cr_disable_shared_residual",
        "cr_disable_private_residual",
        "cr_alpha_override",
        "cr_alpha_frequency_transform",
        "cr_soft_top_m",
        "cr_soft_min_overlap_slots",
        "cr_soft_neighbor_source",
        "cr_soft_min_support",
        "cr_soft_support_eta",
        "cr_soft_reliability_floor",
        "cr_soft_lift_kappa",
        "cr_soft_decouple_reliability",
        "cr_soft_behavior_weight",
        "cr_soft_behavior_window",
        "dis_weight",
        "div_weight",
        "num_hard_neg",
        "prefix_level",
    ]
    print("\t".join(fields))
    for rank, row in enumerate(rows[:top_k], start=1):
        out = dict(row)
        out["rank"] = rank
        print("\t".join(format_value(out.get(field)) for field in fields))

    if rows:
        best = rows[0]
        print()
        print("Best experiment:")
        print(json.dumps(best, ensure_ascii=False, indent=2))
    else:
        print("No test_metrics.json files found.")


def write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp",
        "NDCG@5",
        "HR@5",
        "Recall@5",
        "NDCG@10",
        "HR@10",
        "Recall@10",
        "NDCG@20",
        "HR@20",
        "Recall@20",
        "best_valid_NDCG@10",
        *DEFAULT_FIELDS,
        "path",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    print(f"Wrote CSV: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/office", help="Directory containing experiment outputs.")
    parser.add_argument("--metric", default="NDCG@10", help="Metric used for ranking experiments.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--csv", default=None, help="Optional path to write a CSV summary.")
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_results(root)
    rows.sort(key=lambda row: metric_value(row, args.metric), reverse=True)
    print_table(rows, metric=args.metric, top_k=args.top_k)
    if args.csv:
        write_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
