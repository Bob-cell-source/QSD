import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .io_utils import iter_json_records, write_json


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_clean_text(v) for v in value)
    return str(value).replace("\n", " ").strip()


def load_meta(meta_path: str | Path) -> Dict[str, Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    for row in iter_json_records(meta_path):
        asin = row.get("asin")
        if not asin:
            continue
        cats = row.get("categories") or []
        flat_cats = []
        if cats and isinstance(cats, list):
            for chain in cats:
                if isinstance(chain, list):
                    flat_cats.extend(str(x) for x in chain)
                else:
                    flat_cats.append(str(chain))
        items[asin] = {
            "asin": asin,
            "title": _clean_text(row.get("title")),
            "description": _clean_text(row.get("description")),
            "brand": _clean_text(row.get("brand")),
            "categories": flat_cats,
        }
    return items


def load_interactions(
    reviews_path: str | Path,
    min_rating: float,
) -> Dict[str, List[Tuple[int, str, float]]]:
    by_user: Dict[str, List[Tuple[int, str, float]]] = defaultdict(list)
    for row in iter_json_records(reviews_path):
        user = row.get("reviewerID")
        asin = row.get("asin")
        rating = float(row.get("overall", 0.0) or 0.0)
        ts = int(row.get("unixReviewTime", 0) or 0)
        if not user or not asin or rating < min_rating:
            continue
        by_user[user].append((ts, asin, rating))
    for user in list(by_user):
        by_user[user].sort(key=lambda x: (x[0], x[1]))
    return by_user


def filter_k_core(
    by_user: Dict[str, List[Tuple[int, str, float]]],
    meta: Dict[str, Dict[str, Any]],
    min_user_inter: int,
    min_item_inter: int,
    max_rounds: int = 20,
) -> Dict[str, List[Tuple[int, str, float]]]:
    valid_items = set(meta)
    by_user = {
        u: [x for x in seq if x[1] in valid_items]
        for u, seq in by_user.items()
    }
    for _ in range(max_rounds):
        by_user = {u: seq for u, seq in by_user.items() if len(seq) >= min_user_inter}
        item_count = defaultdict(int)
        for seq in by_user.values():
            for _, asin, _ in seq:
                item_count[asin] += 1
        keep_items = {i for i, c in item_count.items() if c >= min_item_inter}
        changed = False
        new_by_user = {}
        for u, seq in by_user.items():
            new_seq = [x for x in seq if x[1] in keep_items]
            if len(new_seq) != len(seq):
                changed = True
            if len(new_seq) >= min_user_inter:
                new_by_user[u] = new_seq
        if not changed and len(new_by_user) == len(by_user):
            return new_by_user
        by_user = new_by_user
    return by_user


def build_dataset(
    reviews_path: str | Path,
    meta_path: str | Path,
    output_dir: str | Path,
    min_user_inter: int = 5,
    min_item_inter: int = 5,
    min_rating: float = 0.0,
) -> None:
    output_dir = Path(output_dir)
    meta = load_meta(meta_path)
    by_user = load_interactions(reviews_path, min_rating=min_rating)
    by_user = filter_k_core(by_user, meta, min_user_inter, min_item_inter)

    users = sorted(by_user)
    item_asins = sorted({asin for seq in by_user.values() for _, asin, _ in seq})
    user2id = {u: idx + 1 for idx, u in enumerate(users)}
    item2id = {asin: idx + 1 for idx, asin in enumerate(item_asins)}

    sequences = []
    for user in users:
        seq = [item2id[asin] for _, asin, _ in by_user[user]]
        timestamps = [ts for ts, _, _ in by_user[user]]
        sequences.append({"user_id": user2id[user], "items": seq, "timestamps": timestamps})

    item_meta = {
        str(item2id[asin]): {
            "asin": asin,
            "title": meta.get(asin, {}).get("title", ""),
            "brand": meta.get(asin, {}).get("brand", ""),
            "categories": meta.get(asin, {}).get("categories", []),
            "description": meta.get(asin, {}).get("description", ""),
        }
        for asin in item_asins
    }

    write_json(output_dir / "sequences.json", sequences)
    write_json(output_dir / "item_meta.json", item_meta)
    write_json(output_dir / "user2id.json", user2id)
    write_json(output_dir / "item2id.json", item2id)
    stats = {
        "num_users": len(users),
        "num_items": len(item_asins),
        "num_interactions": sum(len(x["items"]) for x in sequences),
        "min_user_inter": min_user_inter,
        "min_item_inter": min_item_inter,
        "min_rating": min_rating,
    }
    write_json(output_dir / "stats.json", stats)
    print(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-user-inter", type=int, default=5)
    parser.add_argument("--min-item-inter", type=int, default=5)
    parser.add_argument("--min-rating", type=float, default=0.0)
    args = parser.parse_args()
    build_dataset(
        args.reviews,
        args.meta,
        args.output_dir,
        args.min_user_inter,
        args.min_item_inter,
        args.min_rating,
    )


if __name__ == "__main__":
    main()
