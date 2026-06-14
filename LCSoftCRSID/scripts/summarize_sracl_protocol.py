#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ("NDCG@10", "HR@10", "NDCG@20", "HR@20")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LoCoRec SRA-CL protocol runs.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--datasets", nargs="+", default=["office", "beauty", "sports"])
    parser.add_argument("--output", default="runs/sracl_protocol_summary.csv")
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        result_files = sorted(
            (Path(args.runs_root) / dataset / "sracl_protocol").glob(
                "locorec_seed*/test_metrics.json"
            )
        )
        values = {metric: [] for metric in METRICS}
        for result_file in result_files:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            metrics = payload.get("test", payload)
            for metric in METRICS:
                values[metric].append(float(metrics[metric]))

        row = {"dataset": dataset, "num_runs": len(result_files)}
        for metric in METRICS:
            metric_values = values[metric]
            row[f"{metric}_mean"] = (
                statistics.fmean(metric_values) if metric_values else ""
            )
            row[f"{metric}_std"] = (
                statistics.stdev(metric_values) if len(metric_values) > 1 else ""
            )
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
