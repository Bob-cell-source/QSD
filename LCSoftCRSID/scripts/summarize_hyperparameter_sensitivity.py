#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PARAMETER_ORDER = ("M", "delta", "tau")
VARIANTS: dict[str, tuple[str, float]] = {
    "m_1": ("M", 1),
    "m_2": ("M", 2),
    "m_8": ("M", 8),
    "delta_1": ("delta", 1),
    "delta_2": ("delta", 2),
    "delta_4": ("delta", 4),
    "tau_5": ("tau", 5),
    "tau_80": ("tau", 80),
}
DEFAULTS = {"M": 4.0, "delta": 3.0, "tau": 20.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize final-protocol LoCoRec sensitivity runs."
    )
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-group", default="cold_0-5")
    return parser.parse_args()


def metric(group: dict[str, Any], name: str) -> float | str:
    value = group.get(name)
    return "" if value is None else float(value)


def result_row(
    dataset: str,
    parameter: str,
    value: float,
    is_default: bool,
    result: dict[str, Any],
    tail_group: str,
) -> dict[str, Any]:
    groups = result["groups"]
    overall = groups.get("overall", {})
    tail = groups.get(tail_group, {})
    return {
        "dataset": dataset,
        "parameter": parameter,
        "value": value,
        "is_default": int(is_default),
        "overall_NDCG@10": metric(overall, "NDCG@10"),
        "overall_HR@10": metric(overall, "HR@10"),
        "tail_count": tail.get("count", ""),
        "tail_NDCG@10": metric(tail, "NDCG@10"),
        "tail_HR@10": metric(tail, "HR@10"),
        "checkpoint": result["checkpoint"],
    }


def load_rows(result_dirs: list[Path], tail_group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        path = result_dir / "cold_start_metrics.json"
        if not path.exists():
            print(f"Skip missing result: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = Path(payload["dataset_dir"]).name
        results = {entry["name"]: entry for entry in payload["results"]}
        default = results.get("default")
        if default is None:
            raise ValueError(f"Missing default checkpoint in {path}")

        for parameter, value in DEFAULTS.items():
            rows.append(
                result_row(dataset, parameter, value, True, default, tail_group)
            )
        for name, (parameter, value) in VARIANTS.items():
            result = results.get(name)
            if result is None:
                raise ValueError(f"Missing variant {name} in {path}")
            rows.append(
                result_row(dataset, parameter, value, False, result, tail_group)
            )

    parameter_rank = {name: rank for rank, name in enumerate(PARAMETER_ORDER)}
    rows.sort(key=lambda row: (row["dataset"], parameter_rank[row["parameter"]], row["value"]))
    return rows


def format_number(value: Any) -> str:
    return "--" if value in (None, "") else f"{float(value):.4f}"


def write_table(rows: list[dict[str, Any]], output: Path) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    by_key = {
        (str(row["dataset"]), str(row["parameter"]), float(row["value"])): row
        for row in rows
    }
    values = {
        parameter: sorted(
            {float(row["value"]) for row in rows if row["parameter"] == parameter}
        )
        for parameter in PARAMETER_ORDER
    }

    columns = "ll" + "cc" * len(datasets)
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Parameter & Value & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{{dataset.title()}}}" for dataset in datasets)
        + r" \\",
        " & & "
        + " & ".join(["Overall N@10 & Tail N@10"] * len(datasets))
        + r" \\",
        r"\midrule",
    ]
    for parameter_index, parameter in enumerate(PARAMETER_ORDER):
        for value in values[parameter]:
            value_label = f"{value:g}"
            if value == DEFAULTS[parameter]:
                value_label += " (default)"
            cells = [parameter if value == values[parameter][0] else "", value_label]
            for dataset in datasets:
                row = by_key[(dataset, parameter, value)]
                cells.extend(
                    [
                        format_number(row["overall_NDCG@10"]),
                        format_number(row["tail_NDCG@10"]),
                    ]
                )
            lines.append(" & ".join(cells) + r" \\")
        if parameter_index + 1 < len(PARAMETER_ORDER):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def write_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skip sensitivity figures.")
        return

    datasets = sorted({str(row["dataset"]) for row in rows})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["parameter"]))].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), constrained_layout=True)
    metrics = (("overall_NDCG@10", "Overall NDCG@10"), ("tail_NDCG@10", "Tail NDCG@10"))
    markers = ("o", "s", "^")
    for column, parameter in enumerate(PARAMETER_ORDER):
        for row_index, (metric_name, ylabel) in enumerate(metrics):
            axis = axes[row_index][column]
            for dataset_index, dataset in enumerate(datasets):
                points = sorted(grouped[(dataset, parameter)], key=lambda row: row["value"])
                default = next(row for row in points if row["is_default"])
                denominator = float(default[metric_name])
                xs = [float(row["value"]) for row in points]
                ys = [100.0 * float(row[metric_name]) / denominator for row in points]
                axis.plot(
                    xs,
                    ys,
                    marker=markers[dataset_index % len(markers)],
                    linewidth=1.5,
                    label=dataset.title(),
                )
            axis.axhline(100.0, color="gray", linestyle="--", linewidth=0.8)
            axis.set_xlabel(parameter)
            axis.set_ylabel(f"Relative {ylabel} (%)")
            axis.grid(alpha=0.25)
    axes[0][0].legend(frameon=False)
    fig.savefig(output_dir / "sensitivity.png", dpi=300)
    fig.savefig(output_dir / "sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.result_dirs, args.tail_group)
    if not rows:
        raise SystemExit("No completed sensitivity result was found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_table(rows, args.output_dir / "table.tex")
    write_plot(rows, args.output_dir)
    print(csv_path)


if __name__ == "__main__":
    main()

