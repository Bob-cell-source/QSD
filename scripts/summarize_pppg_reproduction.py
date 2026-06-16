#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ["Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "MRR@10"]


def read_json(path: Path) -> Any:
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def collect_run(dataset: str, model: str, run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    payload = read_json(metrics_path)
    test = payload.get("test", {})
    args = payload.get("args", {})
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "seed": args.get("seed", run_dir.name.replace("seed", "")),
        "run_dir": str(run_dir),
        "best_valid_metric": payload.get("best_valid_metric", ""),
        "early_stop_metric": payload.get("early_stop_metric", ""),
    }
    for metric in METRICS:
        row[metric] = test.get(metric, "")
    return row


def mean_std(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return f"{statistics.mean(values):.6f} +/- {statistics.pstdev(values):.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fields = ["dataset", "model", "num_runs", *METRICS]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="pppg_reproduction")
    parser.add_argument("--datasets", nargs="+", default=["office", "beauty", "sports"])
    parser.add_argument("--models", nargs="+", default=["sasrec", "locorec"])
    parser.add_argument("--output-dir", default="runs/pppg_reproduction_summary")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            root = Path("runs") / dataset / args.run_tag / model
            if not root.exists():
                continue
            for run_dir in sorted(root.glob("seed*")):
                row = collect_run(dataset, model, run_dir)
                if row is not None:
                    rows.append(row)

    detail_fields = [
        "dataset",
        "model",
        "seed",
        *METRICS,
        "best_valid_metric",
        "early_stop_metric",
        "run_dir",
    ]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "detail.csv", rows, detail_fields)

    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            group = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
            if not group:
                continue
            summary: dict[str, Any] = {"dataset": dataset, "model": model, "num_runs": len(group)}
            for metric in METRICS:
                vals = [float(row[metric]) for row in group if row.get(metric) != ""]
                summary[metric] = mean_std(vals)
            summary_rows.append(summary)

    summary_fields = ["dataset", "model", "num_runs", *METRICS]
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    write_markdown(output_dir / "summary.md", summary_rows)
    print({"detail": str(output_dir / "detail.csv"), "summary": str(output_dir / "summary.csv")})


if __name__ == "__main__":
    main()
