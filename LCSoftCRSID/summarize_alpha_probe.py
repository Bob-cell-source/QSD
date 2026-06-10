#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.root).glob("*/test_metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result.get("test", {})
        config = result.get("args", {})
        alpha = result.get("learned_alpha_parameters") or {}
        rows.append(
            {
                "experiment": path.parent.name,
                "alpha_mode": config.get("alpha_mode"),
                "candidate_mode": config.get("candidate_weight_mode"),
                "freq_slope": alpha.get("frequency_slope"),
                "rel_slope": alpha.get("reliability_slope"),
                "bias": alpha.get("bias"),
                "valid_NDCG@10": result.get("best_valid_NDCG@10"),
                "NDCG@10": metrics.get("NDCG@10"),
                "HR@10": metrics.get("HR@10"),
                "NDCG@20": metrics.get("NDCG@20"),
                "HR@20": metrics.get("HR@20"),
            }
        )
    rows.sort(key=lambda row: row["valid_NDCG@10"] or -1, reverse=True)
    fields = list(rows[0]) if rows else []
    if fields:
        print("\t".join(fields))
        for row in rows:
            print("\t".join("-" if row[field] is None else str(row[field]) for field in fields))


if __name__ == "__main__":
    main()
