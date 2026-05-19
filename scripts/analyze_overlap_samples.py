import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json
from qsdrec.train import NextItemDataset


def build_train_item_counts(sequences: List[Dict[str, Any]]) -> Dict[int, int]:
    counts = Counter()
    for row in sequences:
        items = row["items"]
        if len(items) < 3:
            continue
        counts.update(int(item) for item in items[1:-2])
    return dict(counts)


def build_prefix_sizes(item_semantic_ids: Dict[int, List[int]], prefix_level: int) -> Dict[int, int]:
    groups = defaultdict(list)
    for item, sid in item_semantic_ids.items():
        groups[tuple(sid[:prefix_level])].append(item)
    sizes = {}
    for items in groups.values():
        for item in items:
            sizes[item] = len(items)
    return sizes


def build_overlap_neighbors(item_semantic_ids: Dict[int, List[int]], min_overlap_slots: int) -> Dict[int, List[int]]:
    inverted = defaultdict(list)
    for item, sid in item_semantic_ids.items():
        for slot, code in enumerate(sid):
            inverted[(slot, int(code))].append(item)

    neighbors = {}
    for item, sid in item_semantic_ids.items():
        counts = Counter()
        for slot, code in enumerate(sid):
            counts.update(inverted[(slot, int(code))])
        neighbors[item] = sorted(x for x, count in counts.items() if count >= min_overlap_slots)
    return neighbors


def bucket_name(value: int, buckets):
    for name, lower, upper in buckets:
        if value >= lower and (upper is None or value <= upper):
            return name
    return "other"


def parse_buckets(spec: str):
    buckets = []
    for raw in spec.split(","):
        raw = raw.strip()
        if raw.startswith(">"):
            buckets.append((raw, int(raw[1:]) + 1, None))
        elif "-" in raw:
            left, right = raw.split("-", 1)
            buckets.append((raw, int(left), int(right)))
        else:
            value = int(raw)
            buckets.append((raw, value, value))
    return buckets


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def words(text: str):
    return set(re.findall(r"[a-z0-9][a-z0-9+\-']{1,}", norm_text(text)))


def brief(item: int, item_meta: Dict[str, Any], item_semantic_ids: Dict[int, List[int]], counts: Dict[int, int], prefix_sizes: Dict[int, int], overlap_neighbors: Dict[int, List[int]]):
    meta = item_meta.get(str(item), {})
    return {
        "item_id": item,
        "title": meta.get("title", ""),
        "brand": meta.get("brand", ""),
        "categories": meta.get("categories", []),
        "semantic_id": item_semantic_ids.get(item),
        "train_count": counts.get(item, 0),
        "prefix_group_size": prefix_sizes.get(item, 1),
        "overlap_group_size": len(overlap_neighbors.get(item, [])),
    }


def item_similarity(target: int, neighbor: int, item_meta: Dict[str, Any], item_semantic_ids: Dict[int, List[int]], prefix_level: int):
    t_sid = item_semantic_ids[target]
    n_sid = item_semantic_ids[neighbor]
    overlap_slots = [idx for idx, (a, b) in enumerate(zip(t_sid, n_sid)) if a == b]
    t_meta = item_meta.get(str(target), {})
    n_meta = item_meta.get(str(neighbor), {})
    same_brand = norm_text(t_meta.get("brand")) and norm_text(t_meta.get("brand")) == norm_text(n_meta.get("brand"))
    t_terms = words(t_meta.get("title", ""))
    n_terms = words(n_meta.get("title", ""))
    jaccard = len(t_terms & n_terms) / max(len(t_terms | n_terms), 1)
    return {
        "item_id": neighbor,
        "title": n_meta.get("title", ""),
        "brand": n_meta.get("brand", ""),
        "semantic_id": n_sid,
        "same_prefix": t_sid[:prefix_level] == n_sid[:prefix_level],
        "overlap_slots": overlap_slots,
        "same_brand": bool(same_brand),
        "title_jaccard": jaccard,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-level", type=int, default=2)
    parser.add_argument("--min-overlap-slots", type=int, default=2)
    parser.add_argument("--buckets", default="1,2-5,6-10,11-20,21-50,>50")
    parser.add_argument("--examples-per-bucket", type=int, default=5)
    parser.add_argument("--neighbors-per-example", type=int, default=8)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    sequences = read_json(dataset_dir / "sequences.json")
    item_meta = read_json(dataset_dir / "item_meta.json")
    semantic_obj = read_json(args.semantic_ids)
    item_semantic_ids = {int(k): list(map(int, v)) for k, v in semantic_obj["semantic_ids"].items()}
    counts = build_train_item_counts(sequences)
    prefix_sizes = build_prefix_sizes(item_semantic_ids, args.prefix_level)
    overlap_neighbors = build_overlap_neighbors(item_semantic_ids, args.min_overlap_slots)
    buckets = parse_buckets(args.buckets)

    test_data = NextItemDataset(sequences, max_len=50, split="test")
    stats = {
        name: {
            "count": 0,
            "avg_train_count": 0.0,
            "avg_prefix_size": 0.0,
            "avg_overlap_size": 0.0,
            "prefix_size_1_count": 0,
        }
        for name, _, _ in buckets
    }
    stats["other"] = {
        "count": 0,
        "avg_train_count": 0.0,
        "avg_prefix_size": 0.0,
        "avg_overlap_size": 0.0,
        "prefix_size_1_count": 0,
    }
    examples = defaultdict(list)

    for _, target in test_data:
        target = int(target)
        overlap_size = len(overlap_neighbors.get(target, []))
        prefix_size = prefix_sizes.get(target, 1)
        bucket = bucket_name(overlap_size, buckets)
        row = stats[bucket]
        row["count"] += 1
        row["avg_train_count"] += counts.get(target, 0)
        row["avg_prefix_size"] += prefix_size
        row["avg_overlap_size"] += overlap_size
        row["prefix_size_1_count"] += int(prefix_size == 1)

        if len(examples[bucket]) < args.examples_per_bucket:
            neighbors = [x for x in overlap_neighbors.get(target, []) if x != target]
            neighbors.sort(
                key=lambda x: (
                    item_similarity(target, x, item_meta, item_semantic_ids, args.prefix_level)["same_prefix"],
                    item_similarity(target, x, item_meta, item_semantic_ids, args.prefix_level)["same_brand"],
                    item_similarity(target, x, item_meta, item_semantic_ids, args.prefix_level)["title_jaccard"],
                ),
                reverse=True,
            )
            examples[bucket].append(
                {
                    "target": brief(target, item_meta, item_semantic_ids, counts, prefix_sizes, overlap_neighbors),
                    "overlap_neighbors": [
                        item_similarity(target, n, item_meta, item_semantic_ids, args.prefix_level)
                        for n in neighbors[: args.neighbors_per_example]
                    ],
                }
            )

    for row in stats.values():
        count = max(row["count"], 1)
        row["avg_train_count"] /= count
        row["avg_prefix_size"] /= count
        row["avg_overlap_size"] /= count
        row["prefix_size_1_ratio"] = row["prefix_size_1_count"] / count

    output = {
        "dataset_dir": str(dataset_dir),
        "semantic_ids": args.semantic_ids,
        "prefix_level": args.prefix_level,
        "min_overlap_slots": args.min_overlap_slots,
        "buckets": args.buckets,
        "stats": stats,
        "examples": dict(examples),
    }
    write_json(args.output, output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
