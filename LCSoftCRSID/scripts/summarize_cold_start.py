#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cross-dataset LC-SoftCRSID cold-start results."
    )
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cold-group", default="cold_0-5")
    parser.add_argument("--warm-group", default="warm_gt5")
    return parser.parse_args()


def metric(group: dict[str, Any], name: str) -> float | str:
    value = group.get(name)
    return "" if value is None else float(value)


def load_rows(
    result_dirs: list[Path], cold_group: str, warm_group: str
) -> list[dict[str, Any]]:
    rows = []
    for result_dir in result_dirs:
        path = result_dir / "cold_start_metrics.json"
        if not path.exists():
            print(f"Skip missing result: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = Path(payload.get("dataset_dir", result_dir.parent)).name
        model_groups = {
            result["name"]: result["groups"] for result in payload["results"]
        }
        full_cold = model_groups.get("full", {}).get(cold_group, {})
        for result in payload["results"]:
            groups = result["groups"]
            overall = groups.get("overall", {})
            cold = groups.get(cold_group, {})
            warm = groups.get(warm_group, {})
            full_ndcg = full_cold.get("NDCG@10")
            current_ndcg = cold.get("NDCG@10")
            relative_gain = ""
            if (
                result["name"] != "full"
                and full_ndcg is not None
                and current_ndcg not in (None, 0)
            ):
                relative_gain = (float(full_ndcg) - float(current_ndcg)) / float(
                    current_ndcg
                )
            rows.append(
                {
                    "dataset": dataset,
                    "model": result["name"],
                    "overall_count": overall.get("count", ""),
                    "overall_NDCG@10": metric(overall, "NDCG@10"),
                    "overall_HR@10": metric(overall, "HR@10"),
                    "cold_count": cold.get("count", ""),
                    "cold_NDCG@5": metric(cold, "NDCG@5"),
                    "cold_HR@5": metric(cold, "HR@5"),
                    "cold_NDCG@10": metric(cold, "NDCG@10"),
                    "cold_HR@10": metric(cold, "HR@10"),
                    "warm_count": warm.get("count", ""),
                    "warm_NDCG@10": metric(warm, "NDCG@10"),
                    "warm_HR@10": metric(warm, "HR@10"),
                    "full_gain_over_model_cold_NDCG@10": relative_gain,
                    "checkpoint": result["checkpoint"],
                }
            )
    return rows


def display_model(name: str) -> str:
    labels = {
        "full": "LC-SoftCRSID",
        "sasrec": "SASRec",
        "hard_sid": "Hard SID",
        "without_prior_bias": "w/o Prior Bias",
        "without_shared": "w/o Shared Residual",
        "without_private": "w/o Private Residual",
        "learnable_allocation": "Learnable Allocation",
    }
    return labels.get(name, name.replace("_", " "))


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def number(value: Any) -> str:
    return "--" if value in (None, "") else f"{float(value):.4f}"


def write_latex(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Model & Overall N@10 & Overall H@10 & Cold N@10 & Cold H@10 \\",
        r"\midrule",
    ]
    previous_dataset = None
    for row in rows:
        dataset = latex_escape(str(row["dataset"]))
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\midrule")
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                dataset,
                latex_escape(display_model(str(row["model"]))),
                number(row["overall_NDCG@10"]),
                number(row["overall_HR@10"]),
                number(row["cold_NDCG@10"]),
                number(row["cold_HR@10"]),
            )
        )
        previous_dataset = dataset
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.result_dirs, args.cold_group, args.warm_group)
    if not rows:
        raise SystemExit("No completed cold-start result was found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latex_path = args.output_dir / "table.tex"
    write_latex(rows, latex_path)
    print(csv_path)
    print(latex_path)


if __name__ == "__main__":
    main()
