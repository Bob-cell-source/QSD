import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .io_utils import iter_json_records, write_json

# 清洗并统一文本字段格式
def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_clean_text(v) for v in value)
    return str(value).replace("\n", " ").strip()


def _pick_item_key(row: Dict[str, Any], item_key: str) -> str:
    if item_key == "asin":
        return _clean_text(row.get("asin"))
    if item_key == "parent_asin":
        return _clean_text(row.get("parent_asin"))
    return _clean_text(row.get("asin") or row.get("parent_asin"))


def _pick_user(row: Dict[str, Any]) -> str:
    return _clean_text(row.get("reviewerID") or row.get("user_id"))


def _pick_rating(row: Dict[str, Any]) -> float:
    return float(row.get("overall", row.get("rating", 0.0)) or 0.0)


def _pick_timestamp(row: Dict[str, Any]) -> int:
    ts = int(row.get("unixReviewTime", row.get("timestamp", 0)) or 0)
    if ts > 10_000_000_000:
        ts //= 1000
    return ts


def detect_meta_item_key(meta_path: str | Path) -> str:
    asin_count = 0
    parent_count = 0
    for idx, row in enumerate(iter_json_records(meta_path)):
        if row.get("asin"):
            asin_count += 1
        if row.get("parent_asin"):
            parent_count += 1
        if idx >= 999:
            break
    if asin_count > 0:
        return "asin"
    if parent_count > 0:
        return "parent_asin"
    return "auto"

# 加载并规范化商品文本及类别元数据
def load_meta(meta_path: str | Path, item_key: str = "auto") -> Dict[str, Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    for row in iter_json_records(meta_path):
        item_id = _pick_item_key(row, item_key)
        if not item_id:
            continue
        cats = row.get("categories") or []
        flat_cats = []
        if cats and isinstance(cats, list):
            for chain in cats:
                if isinstance(chain, list):
                    flat_cats.extend(str(x) for x in chain)
                else:
                    flat_cats.append(str(chain))
        desc_parts = []
        if row.get("description"):
            desc_parts.append(row.get("description"))
        if row.get("features"):
            desc_parts.append(row.get("features"))
        items[item_id] = {
            "asin": item_id,
            "raw_asin": _clean_text(row.get("asin")),
            "parent_asin": _clean_text(row.get("parent_asin")),
            "title": _clean_text(row.get("title")),
            "description": _clean_text(desc_parts),
            "brand": _clean_text(row.get("brand") or row.get("store")),
            "categories": flat_cats,
        }
    return items


def _review_text(row: Dict[str, Any], max_chars: int = 1200) -> str:
    text = _clean_text([row.get("title"), row.get("text")])
    return text[:max_chars]

# 加载用户交互序列，按需从评论中构建商品备用元数据
def load_interactions_with_review_meta(
    reviews_path: str | Path,
    min_rating: float,
    item_key: str = "auto",
    collect_item_meta: bool = False,
    max_review_texts_per_item: int = 3,
) -> Tuple[Dict[str, List[Tuple[int, str, float]]], Dict[str, Dict[str, Any]]]:
    by_user: Dict[str, List[Tuple[int, str, float]]] = defaultdict(list)
    review_meta: Dict[str, Dict[str, Any]] = {}
    for row in iter_json_records(reviews_path):
        user = _pick_user(row)
        asin = _pick_item_key(row, item_key)
        rating = _pick_rating(row)
        ts = _pick_timestamp(row)
        if not user or not asin or rating < min_rating:
            continue
        by_user[user].append((ts, asin, rating))
        if collect_item_meta:
            meta = review_meta.setdefault(
                asin,
                {
                    "asin": asin,
                    "raw_asin": _clean_text(row.get("asin")),
                    "parent_asin": _clean_text(row.get("parent_asin")),
                    "title": "",
                    "description": "",
                    "brand": "",
                    "categories": [],
                    "_num_review_texts": 0,
                },
            )
            if not meta["title"]:
                meta["title"] = _clean_text(row.get("title"))
            if meta["_num_review_texts"] < max_review_texts_per_item:
                snippet = _review_text(row)
                if snippet:
                    meta["description"] = _clean_text([meta["description"], snippet])
                    meta["_num_review_texts"] += 1
    for user in list(by_user):
        by_user[user].sort(key=lambda x: (x[0], x[1]))
    for meta in review_meta.values():
        meta.pop("_num_review_texts", None)
    return by_user, review_meta


def load_interactions(
    reviews_path: str | Path,
    min_rating: float,
    item_key: str = "auto",
) -> Dict[str, List[Tuple[int, str, float]]]:
    by_user, _ = load_interactions_with_review_meta(
        reviews_path,
        min_rating=min_rating,
        item_key=item_key,
        collect_item_meta=False,
    )
    return by_user

# 迭代执行用户-物品 k-core 过滤
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
    item_key: str = "auto",
    allow_missing_meta: bool = False,
) -> None:
    output_dir = Path(output_dir)
    resolved_item_key = detect_meta_item_key(meta_path) if item_key == "auto" else item_key
    meta = load_meta(meta_path, item_key=resolved_item_key)
    by_user, review_meta = load_interactions_with_review_meta(
        reviews_path,
        min_rating=min_rating,
        item_key=resolved_item_key,
        collect_item_meta=allow_missing_meta,
    )
    if allow_missing_meta:
        for asin, fallback_meta in review_meta.items():
            meta.setdefault(asin, fallback_meta)
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
            "raw_asin": meta.get(asin, {}).get("raw_asin", ""),
            "parent_asin": meta.get(asin, {}).get("parent_asin", ""),
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
        "item_key": resolved_item_key,
        "allow_missing_meta": allow_missing_meta,
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
    parser.add_argument(
        "--item-key",
        choices=["auto", "asin", "parent_asin"],
        default="auto",
        help="Item identifier field. Use parent_asin for Amazon Review 2023 metadata without asin.",
    )
    parser.add_argument(
        "--allow-missing-meta",
        action="store_true",
        help="Keep review items missing from product metadata and build fallback item text from review title/text.",
    )
    args = parser.parse_args()
    build_dataset(
        args.reviews,
        args.meta,
        args.output_dir,
        args.min_user_inter,
        args.min_item_inter,
        args.min_rating,
        args.item_key,
        args.allow_missing_meta,
    )


if __name__ == "__main__":
    main()
