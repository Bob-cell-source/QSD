#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def read_json(path: str | Path):
    with Path(path).open("rt", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-meta", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--item-ids", required=True)
    args = parser.parse_args()

    item_meta = read_json(args.item_meta)
    item_ids = [str(x) for x in read_json(args.item_ids)]
    embeddings = np.load(args.embeddings, mmap_mode="r")
    expected = sorted(item_meta.keys(), key=lambda x: int(x))

    if embeddings.shape[0] != len(item_ids):
        raise SystemExit(
            f"Embedding rows ({embeddings.shape[0]}) != item id count ({len(item_ids)})."
        )
    if item_ids != expected:
        missing = sorted(set(expected) - set(item_ids), key=int)[:10]
        extra = sorted(set(item_ids) - set(expected), key=int)[:10]
        raise SystemExit(
            "Embedding item ids do not match current item_meta ids. "
            f"missing={missing}, extra={extra}, first_item_ids={item_ids[:5]}, expected={expected[:5]}"
        )

    print(
        {
            "status": "ok",
            "num_items": len(item_ids),
            "embedding_shape": list(embeddings.shape),
        }
    )


if __name__ == "__main__":
    main()
