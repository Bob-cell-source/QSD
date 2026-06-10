#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_result(path: Path, family: str) -> Dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result.get("test", {})
    config = result.get("args", {})
    alpha = result.get("learned_alpha_parameters") or {}
    return {
        "family": family,
        "experiment": path.parent.name,
        "valid_NDCG@10": result.get("best_valid_NDCG@10"),
        "NDCG@5": metrics.get("NDCG@5"),
        "HR@5": metrics.get("HR@5"),
        "NDCG@10": metrics.get("NDCG@10"),
        "HR@10": metrics.get("HR@10"),
        "NDCG@20": metrics.get("NDCG@20"),
        "HR@20": metrics.get("HR@20"),
        "neighbor_source": config.get("cr_soft_neighbor_source", "sid_overlap"),
        "overlap_slots": config.get(
            "cr_soft_min_overlap_slots", config.get("soft_min_overlap_slots")
        ),
        "tail_tau": config.get("cr_tail_tau", config.get("tail_tau")),
        "candidate_weight_mode": config.get("candidate_weight_mode"),
        "attention_kl_weight": config.get("attention_kl_weight"),
        "alpha_mode": config.get("alpha_mode"),
        "learned_prior_beta": result.get("learned_prior_beta"),
        "alpha_frequency_slope": alpha.get("frequency_slope"),
        "alpha_reliability_slope": alpha.get("reliability_slope"),
        "alpha_bias": alpha.get("bias"),
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    if args.reference and Path(args.reference).is_file():
        rows.append(load_result(Path(args.reference), "existing_main_reference"))
    for family in ("reviewer_training", "learnable_probe"):
        family_root = root / family
        for path in sorted(family_root.glob("*/test_metrics.json")):
            rows.append(load_result(path, family))

    rows.sort(
        key=lambda row: (
            row["family"],
            -(float(row["valid_NDCG@10"]) if row["valid_NDCG@10"] is not None else -1.0),
        )
    )
    fields = list(rows[0]) if rows else []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        print("\t".join(fields[:-1]))
        for row in rows:
            print("\t".join("-" if row[field] is None else str(row[field]) for field in fields[:-1]))
    else:
        print("No completed validation results found.")


if __name__ == "__main__":
    main()
