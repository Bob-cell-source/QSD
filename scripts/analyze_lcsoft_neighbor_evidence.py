import argparse
import csv
import itertools
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json
from qsdrec.train import build_semantic_table, build_soft_semantic_table


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(value: str) -> set[str]:
    return set(TOKEN_RE.findall((value or "").lower()))


def title_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def reservoir_add(
    samples: List[Tuple[int, int]],
    pair: Tuple[int, int],
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if len(samples) < limit:
        samples.append(pair)
        return
    position = rng.randrange(seen)
    if position < limit:
        samples[position] = pair


def load_ordered_embeddings(path: Path, item_ids_path: Path, num_items: int) -> np.ndarray:
    embeddings = np.load(path).astype("float32")
    item_ids = [int(item) for item in read_json(item_ids_path)]
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("Embedding rows and item IDs differ.")
    row_for_item = {item: row for row, item in enumerate(item_ids)}
    if set(row_for_item) != set(range(1, num_items + 1)):
        raise ValueError("Embedding item IDs must cover all internal item IDs.")
    ordered = embeddings[[row_for_item[item] for item in range(1, num_items + 1)]]
    norms = np.linalg.norm(ordered, axis=1, keepdims=True)
    return ordered / np.maximum(norms, 1e-12)


def find_train_cooccurrence(
    sequences: Sequence[Dict[str, Any]],
    window: int,
    queried_pairs: set[Tuple[int, int]],
) -> set[Tuple[int, int]]:
    hits: set[Tuple[int, int]] = set()
    for row in sequences:
        items = [int(item) for item in row["items"][:-2]]
        for left, item in enumerate(items):
            for other in items[left + 1 : left + 1 + window]:
                if item != other:
                    pair = (min(item, other), max(item, other))
                    if pair in queried_pairs:
                        hits.add(pair)
    return hits


def metadata_features(item_meta: Dict[str, Any], num_items: int) -> Dict[int, Dict[str, Any]]:
    result = {}
    for item in range(1, num_items + 1):
        meta = item_meta.get(str(item), item_meta.get(item, {}))
        categories = [
            str(category).strip().lower()
            for category in (meta.get("categories") or [])
            if str(category).strip()
        ]
        result[item] = {
            "title": tokenize(str(meta.get("title", ""))),
            "brand": str(meta.get("brand", "") or "").strip().lower(),
            "categories": set(categories[1:] if len(categories) > 1 else categories),
            "leaf_category": categories[-1] if categories else "",
        }
    return result


def summarize_pairs(
    pairs: Iterable[Tuple[int, int]],
    embeddings: np.ndarray,
    features: Dict[int, Dict[str, Any]],
    cooccurrence: set[Tuple[int, int]],
) -> Dict[str, float | int]:
    values = list(pairs)
    if not values:
        return {"sampled_pairs": 0}
    cosine = []
    title = []
    same_brand = 0
    same_leaf_category = 0
    category_jaccard_values = []
    behavior_overlap = 0
    brand_comparable = 0
    category_comparable = 0
    for left, right in values:
        cosine.append(float(np.dot(embeddings[left - 1], embeddings[right - 1])))
        left_feat = features[left]
        right_feat = features[right]
        title.append(title_jaccard(left_feat["title"], right_feat["title"]))
        if left_feat["brand"] and right_feat["brand"]:
            brand_comparable += 1
            same_brand += int(left_feat["brand"] == right_feat["brand"])
        if left_feat["categories"] and right_feat["categories"]:
            category_comparable += 1
            intersection = left_feat["categories"] & right_feat["categories"]
            union = left_feat["categories"] | right_feat["categories"]
            category_jaccard_values.append(len(intersection) / max(len(union), 1))
            same_leaf_category += int(
                bool(left_feat["leaf_category"])
                and left_feat["leaf_category"] == right_feat["leaf_category"]
            )
        behavior_overlap += int((left, right) in cooccurrence)
    return {
        "sampled_pairs": len(values),
        "mean_text_cosine": sum(cosine) / len(cosine),
        "mean_title_jaccard": sum(title) / len(title),
        "same_brand_rate": same_brand / max(brand_comparable, 1),
        "mean_category_jaccard": sum(category_jaccard_values) / max(category_comparable, 1),
        "same_leaf_category_rate": same_leaf_category / max(category_comparable, 1),
        "train_cooccurrence_rate": behavior_overlap / len(values),
        "brand_comparable_pairs": brand_comparable,
        "category_comparable_pairs": category_comparable,
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "overlap_slots",
        "total_pairs",
        "sampled_pairs",
        "mean_text_cosine",
        "mean_title_jaccard",
        "same_brand_rate",
        "mean_category_jaccard",
        "same_leaf_category_rate",
        "train_cooccurrence_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--embedding-item-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--max-pairs-per-overlap", type=int, default=50000)
    parser.add_argument("--behavior-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    stats = read_json(dataset_dir / "stats.json")
    sequences = read_json(dataset_dir / "sequences.json")
    item_meta = read_json(dataset_dir / "item_meta.json")
    semantic_obj = read_json(args.semantic_ids)
    num_items = int(stats["num_items"])
    semantic_table, item_codes, _ = build_semantic_table(semantic_obj, num_items)
    depth = semantic_table.size(1)
    embeddings = load_ordered_embeddings(
        Path(args.embeddings), Path(args.embedding_item_ids), num_items
    )
    features = metadata_features(item_meta, num_items)
    inverted: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for item, codes in item_codes.items():
        for slot, code in enumerate(codes):
            inverted[(slot, int(code))].append(item)

    slot_posting_ordered_ops = sum(len(items) ** 2 for items in inverted.values())
    slot_posting_pair_ops = sum(len(items) * (len(items) - 1) // 2 for items in inverted.values())
    max_slot_bucket = max((len(items) for items in inverted.values()), default=0)

    pair_index: Dict[Tuple[Tuple[int, int], ...], int] = Counter()
    for codes in item_codes.values():
        for slots in itertools.combinations(range(depth), 2):
            signature = tuple((slot, int(codes[slot])) for slot in slots)
            pair_index[signature] += 1
    pair_combo_ops = sum(size * (size - 1) // 2 for size in pair_index.values())
    pair_combo_ordered_ops = sum(size * size for size in pair_index.values())
    max_pair_bucket = max(pair_index.values(), default=0)

    rng = random.Random(args.seed)
    sampled: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    total_pairs: Counter[int] = Counter()
    scan_start = time.perf_counter()
    for item in range(1, num_items + 1):
        counts: Counter[int] = Counter()
        for slot, code in enumerate(item_codes[item]):
            counts.update(inverted[(slot, int(code))])
        for neighbor, overlap in counts.items():
            if neighbor <= item:
                continue
            overlap = int(overlap)
            total_pairs[overlap] += 1
            reservoir_add(
                sampled[overlap],
                (item, neighbor),
                total_pairs[overlap],
                args.max_pairs_per_overlap,
                rng,
            )
    pair_scan_seconds = time.perf_counter() - scan_start

    soft_start = time.perf_counter()
    soft_ids, soft_weights, reliability = build_soft_semantic_table(
        semantic_table=semantic_table,
        item_semantic_ids=item_codes,
        num_items=num_items,
        top_m=4,
        min_overlap_slots=2,
        min_support=0.05,
        support_eta=2.0,
        hard_token_prior=1.0,
        reliability_floor=0.10,
        max_neighbors=50,
    )
    soft_build_seconds = time.perf_counter() - soft_start
    soft_table_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (soft_ids, soft_weights, reliability)
    )

    random_pairs = set()
    target_random = min(args.max_pairs_per_overlap, sum(total_pairs.values()))
    while len(random_pairs) < target_random:
        left = rng.randint(1, num_items)
        right = rng.randint(1, num_items)
        if left != right:
            random_pairs.add((min(left, right), max(left, right)))

    queried_pairs = set(random_pairs)
    for pairs in sampled.values():
        queried_pairs.update(pairs)
    cooccurrence = find_train_cooccurrence(
        sequences,
        args.behavior_window,
        queried_pairs,
    )

    rows = []
    for overlap in sorted(total_pairs):
        row = {
            "overlap_slots": overlap,
            "total_pairs": total_pairs[overlap],
            **summarize_pairs(sampled[overlap], embeddings, features, cooccurrence),
        }
        rows.append(row)

    report = {
        "dataset": dataset_dir.name,
        "num_items": num_items,
        "sid_depth": depth,
        "neighbor_quality": rows,
        "random_pair_baseline": summarize_pairs(random_pairs, embeddings, features, cooccurrence),
        "scalability": {
            "naive_ordered_slot_comparisons": num_items * num_items * depth,
            "slot_posting_ordered_traversals": slot_posting_ordered_ops,
            "slot_posting_unordered_pair_traversals": slot_posting_pair_ops,
            "slot_posting_fraction_of_naive": slot_posting_ordered_ops
            / max(num_items * num_items * depth, 1),
            "max_single_slot_bucket": max_slot_bucket,
            "two_slot_combination_pair_traversals": pair_combo_ops,
            "two_slot_combination_ordered_traversals": pair_combo_ordered_ops,
            "two_slot_combination_fraction_of_naive": pair_combo_ordered_ops
            / max(num_items * num_items * depth, 1),
            "max_two_slot_bucket": max_pair_bucket,
            "pair_scan_seconds": pair_scan_seconds,
            "formal_soft_sid_build_seconds": soft_build_seconds,
            "formal_soft_sid_tensor_bytes": soft_table_bytes,
        },
        "configuration": vars(args),
    }
    output = Path(args.output)
    write_json(output, report)
    write_csv(rows, Path(args.csv) if args.csv else output.with_suffix(".csv"))
    print(json.dumps(report["scalability"], indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
