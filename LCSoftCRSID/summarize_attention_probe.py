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
        rows.append(
            {
                "experiment": path.parent.name,
                "mode": config.get("candidate_weight_mode"),
                "kl_weight": config.get("attention_kl_weight"),
                "prior_beta": result.get("learned_prior_beta"),
                "valid_NDCG@10": result.get("best_valid_NDCG@10"),
                "NDCG@10": metrics.get("NDCG@10"),
                "HR@10": metrics.get("HR@10"),
                "NDCG@20": metrics.get("NDCG@20"),
                "HR@20": metrics.get("HR@20"),
            }
        )
    rows.sort(key=lambda row: row["valid_NDCG@10"] or -1, reverse=True)
    fields = [
        "experiment",
        "mode",
        "kl_weight",
        "prior_beta",
        "valid_NDCG@10",
        "NDCG@10",
        "HR@10",
        "NDCG@20",
        "HR@20",
    ]
    print("\t".join(fields))
    for row in rows:
        print("\t".join("-" if row[field] is None else str(row[field]) for field in fields))


if __name__ == "__main__":
    main()
