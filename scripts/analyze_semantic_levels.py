import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.io_utils import read_json, write_json


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
    "new",
    "pack",
    "set",
    "pcs",
    "piece",
    "pieces",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9+\-']{1,}", normalize_text(text))
    return [tok for tok in tokens if tok not in STOPWORDS and not tok.isdigit()]


def flatten_categories(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [normalize_text(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, list):
                out.extend(flatten_categories(item))
            else:
                text = normalize_text(item)
                if text:
                    out.append(text)
        return out
    return [normalize_text(value)]


def top_with_ratio(counter: Counter, total: int, top_k: int) -> List[Dict[str, Any]]:
    rows = []
    for key, count in counter.most_common(top_k):
        rows.append({"value": key, "count": count, "ratio": count / max(total, 1)})
    return rows


def distinctive_terms(
    group_counter: Counter,
    global_counter: Counter,
    group_total: int,
    global_total: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    rows = []
    for term, count in group_counter.items():
        if count < 2:
            continue
        group_rate = count / max(group_total, 1)
        global_rate = global_counter.get(term, 0) / max(global_total, 1)
        score = math.log((group_rate + 1e-9) / (global_rate + 1e-9)) * math.log1p(count)
        rows.append(
            {
                "value": term,
                "count": count,
                "ratio": group_rate,
                "global_ratio": global_rate,
                "score": score,
            }
        )
    rows.sort(key=lambda x: (x["score"], x["count"]), reverse=True)
    return rows[:top_k]


def build_item_features(item_meta: Dict[str, Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Counter, int]:
    features = {}
    global_terms = Counter()
    global_term_total = 0
    for raw_item, meta in item_meta.items():
        item = int(raw_item)
        title = normalize_text(meta.get("title", ""))
        brand = normalize_text(meta.get("brand", ""))
        categories = flatten_categories(meta.get("categories", []))
        terms = tokenize(title)
        global_terms.update(set(terms))
        global_term_total += len(set(terms))
        features[item] = {
            "title": title,
            "brand": brand,
            "categories": categories,
            "terms": terms,
        }
    return features, global_terms, global_term_total


def group_items_by_prefix(semantic_ids: Dict[str, List[int]], level: int) -> Dict[Tuple[int, ...], List[int]]:
    groups = defaultdict(list)
    for raw_item, sid in semantic_ids.items():
        if len(sid) >= level:
            groups[tuple(int(x) for x in sid[:level])].append(int(raw_item))
    return dict(groups)


def sample_titles(items: Iterable[int], features: Dict[int, Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    rows = []
    for item in list(items)[:max_samples]:
        feat = features.get(item, {})
        rows.append(
            {
                "item_id": item,
                "title": feat.get("title", ""),
                "brand": feat.get("brand", ""),
                "categories": feat.get("categories", [])[:3],
            }
        )
    return rows


def analyze_level(
    groups: Dict[Tuple[int, ...], List[int]],
    features: Dict[int, Dict[str, Any]],
    global_terms: Counter,
    global_term_total: int,
    top_groups: int,
    top_terms: int,
    min_size: int,
    max_samples: int,
) -> Dict[str, Any]:
    rows = []
    kept = [(prefix, items) for prefix, items in groups.items() if len(items) >= min_size]
    kept.sort(key=lambda x: len(x[1]), reverse=True)

    for prefix, items in kept[:top_groups]:
        brand_counter = Counter()
        category_counter = Counter()
        term_counter = Counter()
        term_total = 0
        for item in items:
            feat = features.get(item, {})
            brand = feat.get("brand", "")
            if brand:
                brand_counter[brand] += 1
            category_counter.update(feat.get("categories", []))
            terms = set(feat.get("terms", []))
            term_counter.update(terms)
            term_total += len(terms)

        dominant_brand_ratio = brand_counter.most_common(1)[0][1] / len(items) if brand_counter else 0.0
        dominant_category_ratio = category_counter.most_common(1)[0][1] / len(items) if category_counter else 0.0
        rows.append(
            {
                "prefix": list(prefix),
                "size": len(items),
                "dominant_brand_ratio": dominant_brand_ratio,
                "dominant_category_ratio": dominant_category_ratio,
                "top_brands": top_with_ratio(brand_counter, len(items), top_terms),
                "top_categories": top_with_ratio(category_counter, len(items), top_terms),
                "distinctive_title_terms": distinctive_terms(
                    term_counter,
                    global_terms,
                    term_total,
                    global_term_total,
                    top_terms,
                ),
                "samples": sample_titles(items, features, max_samples),
            }
        )

    avg_size = sum(len(items) for items in groups.values()) / max(len(groups), 1)
    singleton = sum(1 for items in groups.values() if len(items) == 1)
    return {
        "num_groups": len(groups),
        "avg_size": avg_size,
        "max_size": max((len(items) for items in groups.values()), default=0),
        "singleton_groups": singleton,
        "singleton_ratio": singleton / max(len(groups), 1),
        "reported_groups": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain semantic ID hierarchy by prefix-level metadata coherence and keywords."
    )
    parser.add_argument("--semantic-ids", required=True)
    parser.add_argument("--item-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()

    semantic_obj = read_json(args.semantic_ids)
    item_meta = read_json(args.item_meta)
    semantic_ids = semantic_obj["semantic_ids"]
    codebook_sizes = semantic_obj.get("codebook_sizes", [])

    features, global_terms, global_term_total = build_item_features(item_meta)
    result = {
        "semantic_ids": str(args.semantic_ids),
        "item_meta": str(args.item_meta),
        "codebook_sizes": codebook_sizes,
        "note": (
            "Levels are RQ-KMeans residual code prefixes. They are not predefined taxonomy fields; "
            "this report interprets them by metadata coherence inside each prefix group."
        ),
        "levels": {},
    }

    for level in range(1, len(codebook_sizes) + 1):
        groups = group_items_by_prefix(semantic_ids, level)
        result["levels"][str(level)] = analyze_level(
            groups=groups,
            features=features,
            global_terms=global_terms,
            global_term_total=global_term_total,
            top_groups=args.top_groups,
            top_terms=args.top_terms,
            min_size=args.min_size,
            max_samples=args.max_samples,
        )

    write_json(args.output, result)

    for level, row in result["levels"].items():
        print(
            f"level={level} groups={row['num_groups']} avg_size={row['avg_size']:.2f} "
            f"max_size={row['max_size']} singleton_ratio={row['singleton_ratio']:.3f}"
        )
        for group in row["reported_groups"][:3]:
            terms = ", ".join(x["value"] for x in group["distinctive_title_terms"][:5])
            cats = ", ".join(x["value"] for x in group["top_categories"][:3])
            print(f"  prefix={group['prefix']} size={group['size']} cats=[{cats}] terms=[{terms}]")


if __name__ == "__main__":
    main()
