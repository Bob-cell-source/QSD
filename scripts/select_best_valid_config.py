#!/usr/bin/env python3
import argparse
import json
import shlex
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select an experiment using validation NDCG@10 and emit shell assignments."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--prefix", default="BEST_")
    parser.add_argument("--fields", nargs="+", required=True)
    args = parser.parse_args()

    candidates = []
    for path in Path(args.root).rglob("test_metrics.json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        cfg = obj.get("args", {})
        if cfg.get("model_variant") != args.variant:
            continue
        metric = obj.get("best_valid_NDCG@10")
        if metric is None:
            continue
        candidates.append((float(metric), str(path), cfg))

    if not candidates:
        raise SystemExit(f"No completed {args.variant!r} experiments found under {args.root}")

    metric, path, cfg = max(candidates, key=lambda row: row[0])
    print(f"{args.prefix}VALID_NDCG={shlex.quote(str(metric))}")
    print(f"{args.prefix}RESULT_PATH={shlex.quote(path)}")
    for field in args.fields:
        if field not in cfg:
            raise SystemExit(f"Selected result has no argument named {field!r}: {path}")
        key = args.prefix + field.upper()
        print(f"{key}={shlex.quote(str(cfg[field]))}")


if __name__ == "__main__":
    main()
