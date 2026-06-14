#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METRICS = ("NDCG@5", "HR@5", "NDCG@10", "HR@10", "NDCG@20", "HR@20")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    soft = json.loads(args.soft.read_text(encoding="utf-8"))["groups"]
    hard = json.loads(args.hard.read_text(encoding="utf-8"))["groups"]
    rows = []
    for group in soft:
        if group not in hard:
            continue
        row = {"group": group, "count": soft[group]["count"]}
        for metric in METRICS:
            soft_value = soft[group][metric]
            hard_value = hard[group][metric]
            row[f"soft_{metric}"] = soft_value
            row[f"hard_{metric}"] = hard_value
            row[f"gain_{metric}"] = soft_value - hard_value
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "count"]
    for metric in METRICS:
        fields.extend((f"soft_{metric}", f"hard_{metric}", f"gain_{metric}"))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
