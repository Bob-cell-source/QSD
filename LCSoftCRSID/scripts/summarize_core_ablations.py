#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    ("00_full_lcsoftcrsid", "Full LC-SoftCRSID"),
    ("10_hard_sid_m1", "Hard SID"),
    ("20_without_prior_bias", "w/o Prior Bias"),
    ("30_without_shared_residual", "w/o Shared Residual"),
    ("31_without_private_residual", "w/o Private Residual"),
    ("40_learnable_allocation", "Learnable Allocation"),
)

EXPECTED_CONFIG = {
    "00_full_lcsoftcrsid": {
        "soft_top_m": 4,
        "candidate_weight_mode": "prior_guided",
        "alpha_mode": "fixed",
        "disable_shared_residual": False,
        "disable_private_residual": False,
    },
    "10_hard_sid_m1": {
        "soft_top_m": 1,
        "candidate_weight_mode": "prior_guided",
        "alpha_mode": "fixed",
        "disable_shared_residual": False,
        "disable_private_residual": False,
    },
    "20_without_prior_bias": {
        "soft_top_m": 4,
        "candidate_weight_mode": "learned",
        "alpha_mode": "fixed",
        "disable_shared_residual": False,
        "disable_private_residual": False,
    },
    "30_without_shared_residual": {
        "soft_top_m": 4,
        "candidate_weight_mode": "prior_guided",
        "alpha_mode": "fixed",
        "disable_shared_residual": True,
        "disable_private_residual": False,
    },
    "31_without_private_residual": {
        "soft_top_m": 4,
        "candidate_weight_mode": "prior_guided",
        "alpha_mode": "fixed",
        "disable_shared_residual": False,
        "disable_private_residual": True,
    },
    "40_learnable_allocation": {
        "soft_top_m": 4,
        "candidate_weight_mode": "prior_guided",
        "alpha_mode": "learnable_monotonic",
        "disable_shared_residual": False,
        "disable_private_residual": False,
    },
}

COMMON_CONFIG = {
    "max_len": 50,
    "dim": 128,
    "num_heads": 2,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "num_random_negatives": 100,
    "tail_tau": 20.0,
    "soft_min_overlap_slots": 3,
    "soft_min_support": 0.05,
    "soft_reliability_floor": 0.10,
    "soft_max_neighbors": 50,
    "soft_neighbor_source": "sid_overlap",
    "disable_semantic_basis": False,
    "seed": 2026,
}

FIELDNAMES = (
    "dataset",
    "variant",
    "experiment",
    "best_valid_NDCG@10",
    "NDCG@5",
    "HR@5",
    "NDCG@10",
    "HR@10",
    "NDCG@20",
    "HR@20",
    "config_check",
    "path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and validate the six core LC-SoftCRSID ablations."
    )
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def close_enough(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-9
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_config(experiment: str, args: dict[str, Any]) -> str:
    errors = []
    expected = {**COMMON_CONFIG, **EXPECTED_CONFIG[experiment]}
    for key, value in expected.items():
        if not close_enough(args.get(key), value):
            errors.append(f"{key}={args.get(key)!r}, expected {value!r}")
    return "OK" if not errors else "; ".join(errors)


def load_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        dataset = root.parent.name
        for experiment, display_name in VARIANTS:
            path = root / experiment / "test_metrics.json"
            if not path.exists():
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": display_name,
                        "experiment": experiment,
                        "config_check": "MISSING",
                        "path": str(path),
                    }
                )
                continue

            payload = json.loads(path.read_text(encoding="utf-8"))
            test = payload["test"]
            run_args = payload.get("args", {})
            rows.append(
                {
                    "dataset": dataset,
                    "variant": display_name,
                    "experiment": experiment,
                    "best_valid_NDCG@10": payload.get("best_valid_NDCG@10"),
                    "NDCG@5": test.get("NDCG@5"),
                    "HR@5": test.get("HR@5"),
                    "NDCG@10": test.get("NDCG@10"),
                    "HR@10": test.get("HR@10"),
                    "NDCG@20": test.get("NDCG@20"),
                    "HR@20": test.get("HR@20"),
                    "config_check": validate_config(experiment, run_args),
                    "path": str(path),
                }
            )
    return rows


def metric(value: Any) -> str:
    return "--" if value is None else f"{float(value):.4f}"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex(rows: list[dict[str, Any]], output: Path) -> None:
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    lines = []
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        lines.extend(
            [
                rf"% Dataset: {latex_escape(dataset)}",
                r"\begin{tabular}{lcccc}",
                r"\toprule",
                r"Variant & NDCG@5 & HR@5 & NDCG@10 & HR@10 \\",
                r"\midrule",
            ]
        )
        for row in selected:
            lines.append(
                "{} & {} & {} & {} & {} \\\\".format(
                    latex_escape(row["variant"]),
                    metric(row.get("NDCG@5")),
                    metric(row.get("HR@5")),
                    metric(row.get("NDCG@10")),
                    metric(row.get("HR@10")),
                )
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    latex_path = args.output_dir / "table.tex"
    write_latex(rows, latex_path)

    invalid = [row for row in rows if row["config_check"] != "OK"]
    print(csv_path)
    print(latex_path)
    if invalid:
        print("Configuration audit warnings:")
        for row in invalid:
            print(
                f"- {row['dataset']}/{row['experiment']}: "
                f"{row['config_check']}"
            )


if __name__ == "__main__":
    main()
