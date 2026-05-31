import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json


def load_embedding_map(embeddings_path: str, item_ids_path: str):
    embeddings = np.load(embeddings_path).astype("float32")
    item_ids = read_json(item_ids_path)
    if len(item_ids) != embeddings.shape[0]:
        raise ValueError("item ids length must match embedding rows")
    return {int(item_id): embeddings[idx] for idx, item_id in enumerate(item_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--item-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-token-size", type=int, default=20)
    parser.add_argument("--target-cluster-size", type=int, default=10)
    parser.add_argument("--max-clusters-per-token", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    semantic_obj = read_json(args.semantic_ids)
    sem_ids = {int(k): list(map(int, v)) for k, v in semantic_obj["semantic_ids"].items()}
    emb_map = load_embedding_map(args.embeddings, args.item_ids)
    depth = len(next(iter(sem_ids.values())))

    groups = defaultdict(list)
    for item, sid in sem_ids.items():
        for slot, code in enumerate(sid):
            groups[(slot, int(code))].append(item)

    next_cluster_id = 1
    mini_ids = {item: [0] * depth for item in sem_ids}
    cluster_meta = []

    for (slot, code), items in sorted(groups.items()):
        valid_items = [item for item in items if item in emb_map]
        if len(valid_items) < args.min_token_size:
            cluster_count = 1
        else:
            cluster_count = min(
                args.max_clusters_per_token,
                max(2, int(np.ceil(len(valid_items) / max(args.target_cluster_size, 1)))),
            )

        if cluster_count == 1:
            cluster_id = next_cluster_id
            next_cluster_id += 1
            for item in items:
                mini_ids[item][slot] = cluster_id
            cluster_meta.append(
                {
                    "slot": slot,
                    "code": code,
                    "cluster_id": cluster_id,
                    "local_cluster": 0,
                    "size": len(items),
                    "parent_size": len(items),
                }
            )
            continue

        x = np.stack([emb_map[item] for item in valid_items], axis=0)
        km = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=args.seed + slot * 100000 + code,
            batch_size=args.batch_size,
            n_init="auto",
        )
        labels = km.fit_predict(x)
        local_to_global = {}
        for local in range(cluster_count):
            local_to_global[local] = next_cluster_id
            next_cluster_id += 1

        for item, label in zip(valid_items, labels):
            mini_ids[item][slot] = local_to_global[int(label)]

        # Items missing embeddings fall back to the largest local cluster.
        sizes = np.bincount(labels, minlength=cluster_count)
        fallback = int(sizes.argmax())
        for item in items:
            if mini_ids[item][slot] == 0:
                mini_ids[item][slot] = local_to_global[fallback]

        for local in range(cluster_count):
            cluster_meta.append(
                {
                    "slot": slot,
                    "code": code,
                    "cluster_id": local_to_global[local],
                    "local_cluster": local,
                    "size": int(sizes[local]),
                    "parent_size": len(items),
                }
            )

    output = {
        "source_semantic_ids": str(args.semantic_ids),
        "source_embeddings": str(args.embeddings),
        "source_item_ids": str(args.item_ids),
        "min_token_size": args.min_token_size,
        "target_cluster_size": args.target_cluster_size,
        "max_clusters_per_token": args.max_clusters_per_token,
        "num_mini_clusters": next_cluster_id - 1,
        "mini_cluster_ids": {str(item): codes for item, codes in sorted(mini_ids.items())},
        "cluster_meta": cluster_meta,
    }
    write_json(args.output, output)
    print(
        {
            "num_items": len(mini_ids),
            "num_mini_clusters": next_cluster_id - 1,
            "split_parent_tokens": sum(1 for m in cluster_meta if m["parent_size"] >= args.min_token_size),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
