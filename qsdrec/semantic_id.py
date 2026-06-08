import argparse
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from .io_utils import read_json, write_json


SPACE_RE = re.compile(r"\s+")

# 合并连续空白符并去除首尾空白
def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()

# 根据特定标记截断描述，并限制最大词数
def trim_description(text: str, max_words: int = 160) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    for marker in [
        "What's in the Box",
        "What’s in the Box",
        "Package Contents",
        "Pros:",
        "CONS:",
        "Cons:",
    ]:
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos].strip()
    words = text.split()
    return " ".join(words[:max_words])

# 拼接商品标题、品牌、类别和描述，构造编码器输入文本
def build_item_text(meta: Dict, max_desc_words: int = 160) -> str:
    title = normalize_text(meta.get("title", ""))
    brand = normalize_text(meta.get("brand", ""))
    categories = [normalize_text(x) for x in (meta.get("categories") or []) if normalize_text(x)]
    desc = trim_description(meta.get("description", ""), max_words=max_desc_words)

    parts = []
    if title:
        parts.append(f"Title: {title}.")
    if brand:
        parts.append(f"Brand: {brand}.")
    if categories:
        parts.append(f"Category: {' > '.join(categories)}.")
    if desc:
        parts.append(f"Description: {desc}")
    return " ".join(parts)

# 加载物品文本和对应id
def load_item_texts(item_meta_path: str | Path, max_desc_words: int = 160) -> Tuple[List[str], List[str]]:
    item_meta = read_json(item_meta_path)
    item_ids = sorted(item_meta.keys(), key=lambda x: int(x))
    texts = [build_item_text(item_meta[item_id], max_desc_words=max_desc_words) for item_id in item_ids]
    return item_ids, texts

# 使用文本嵌入模型对物品文本进行编码
def encode_texts(
    texts: Sequence[str],
    encoder_model: str,
    batch_size: int = 64,
    max_length: int = 512,
    device: str | None = None,
):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Encoding requires numpy and torch.") from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(encoder_model, device=device)
        model.max_seq_length = max_length
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")
    except (ImportError, OSError, ValueError) as exc:
        warnings.warn(
            "SentenceTransformer encoding is unavailable or incompatible; "
            f"falling back to transformers mean pooling. Original error: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        model = AutoModel.from_pretrained(encoder_model)
        model.to(device)
        model.eval()

        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                outputs.append(pooled.cpu())
        return torch.cat(outputs, dim=0).numpy().astype("float32")
# 统计文本 token 长度及超过编码长度限制的比例
def inspect_text_lengths(
    texts: Sequence[str],
    encoder_model: str,
    max_length: int,
    device: str | None = None,
) -> None:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Length inspection requires numpy and sentence-transformers.") from exc

    model = SentenceTransformer(encoder_model, device=device)
    tokenizer = model.tokenizer

    lengths = []
    for text in texts:
        token_ids = tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        lengths.append(len(token_ids))

    arr = np.array(lengths)
    over_limit = arr > max_length

    print(
        {
            "num_texts": int(len(arr)),
            "max_token_length": int(arr.max()) if len(arr) else 0,
            "avg_token_length": round(float(arr.mean()), 2) if len(arr) else 0.0,
            "p90_token_length": int(np.percentile(arr, 90)) if len(arr) else 0,
            "p95_token_length": int(np.percentile(arr, 95)) if len(arr) else 0,
            "num_over_limit": int(over_limit.sum()),
            "over_limit_rate": round(float(over_limit.mean()), 6) if len(arr) else 0.0,
        }
    )

    if over_limit.any():
        print("Examples over limit:")
        shown = 0
        for i, length in enumerate(arr):
            if length > max_length:
                print(
                    {
                        "item_index": i,
                        "token_length": int(length),
                        "text_preview": texts[i][:200],
                    }
                )
                shown += 1
                if shown >= 5:
                    break

# 汇总语义 ID 前缀分组规模及重复编码物品比例
def summarize_sizes(sizes: List[int]) -> Dict[str, float | int]:
    if not sizes:
        return {
            "unique_groups": 0,
            "avg_size": 0.0,
            "max_size": 0,
            "groups_gt_1": 0,
            "groups_gt_5": 0,
            "groups_gt_10": 0,
            "collision_item_rate": 0.0,
        }
    n_items = sum(sizes)
    unique = len(sizes)
    return {
        "unique_groups": unique,
        "avg_size": round(mean(sizes), 4),
        "max_size": max(sizes),
        "groups_gt_1": sum(1 for s in sizes if s > 1),
        "groups_gt_5": sum(1 for s in sizes if s > 5),
        "groups_gt_10": sum(1 for s in sizes if s > 10),
        "collision_item_rate": round((n_items - unique) / max(n_items, 1), 6),
    }

# 分析各层语义 ID 前缀的分组规模和冲突情况
def analyze_semantic_ids(
    semantic_id_path: str | Path,
    output_path: str | Path,
    top_groups: int = 20,
) -> None:
    obj = read_json(semantic_id_path)
    sem_ids = {str(k): list(map(int, v)) for k, v in obj["semantic_ids"].items()}
    if not sem_ids:
        raise ValueError("No semantic IDs found.")
    depth = len(next(iter(sem_ids.values())))
    rows = []
    for level in range(1, depth + 1):
        groups = Counter(tuple(v[:level]) for v in sem_ids.values())
        summary = summarize_sizes(list(groups.values()))
        summary["level"] = level
        rows.append(summary)

    full_groups = defaultdict(list)
    for item_id, sid in sem_ids.items():
        full_groups[tuple(sid)].append(item_id)
    biggest = sorted(full_groups.items(), key=lambda x: len(x[1]), reverse=True)[:top_groups]

    report = {
        "num_items": len(sem_ids),
        "depth": depth,
        "levels": rows,
        "top_full_collision_groups": [
            {"semantic_id": list(k), "size": len(v), "item_ids": v[:50]} for k, v in biggest if len(v) > 1
        ],
    }
    write_json(output_path, report)
    print("level\tunique\tavg_size\tmax_size\tgroups>1\tcollision_rate")
    for row in rows:
        print(
            f"{row['level']}\t{row['unique_groups']}\t{row['avg_size']}\t"
            f"{row['max_size']}\t{row['groups_gt_1']}\t{row['collision_item_rate']}"
        )

# 逐层训练残差 K-Means 码本，并将各层聚类编号组成语义 ID
def build_rq_kmeans_from_embeddings(
    embeddings,
    item_ids: Sequence[str],
    codebook_sizes: Sequence[int],
    seed: int = 2026,
):
    try:
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise RuntimeError("RQ-KMeans requires numpy and scikit-learn.") from exc

    emb = np.asarray(embeddings, dtype="float32")
    if len(item_ids) != emb.shape[0]:
        raise ValueError("item_ids length must match embedding rows.")

    residual = emb.copy()
    all_codes = []
    for level, size in enumerate(codebook_sizes):
        km = MiniBatchKMeans(
            n_clusters=int(size),
            random_state=seed + level,
            batch_size=min(4096, max(256, emb.shape[0] // 10)),
            n_init="auto",
        )
        codes = km.fit_predict(residual)
        centers = km.cluster_centers_.astype("float32")
        residual = residual - centers[codes]
        all_codes.append(codes.tolist())

    semantic_ids = {
        item_id: [all_codes[level][idx] for level in range(len(codebook_sizes))]
        for idx, item_id in enumerate(item_ids)
    }
    return semantic_ids

# 编排文本构造、向量编码和 RQ-KMeans，为每个商品生成语义 ID
def build_semantic_ids_with_encoder(
    item_meta_path: str | Path,
    output_path: str | Path,
    encoder_model: str,
    codebook_sizes: Sequence[int],
    batch_size: int = 64,
    max_length: int = 512,
    max_desc_words: int = 160,
    seed: int = 2026,
    device: str | None = None,
    save_embeddings: str | Path | None = None,
    save_item_ids: str | Path | None = None,
) -> None:
    item_ids, texts = load_item_texts(item_meta_path, max_desc_words=max_desc_words)

    inspect_text_lengths(
        texts=texts,
        encoder_model=encoder_model,
        max_length=max_length,
        device=device,
    )

    embeddings = encode_texts(
        texts,
        encoder_model=encoder_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    if save_embeddings is not None:
        import numpy as np

        Path(save_embeddings).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_embeddings, embeddings)
    if save_item_ids is not None:
        write_json(save_item_ids, list(item_ids))

    semantic_ids = build_rq_kmeans_from_embeddings(
        embeddings=embeddings,
        item_ids=item_ids,
        codebook_sizes=codebook_sizes,
        seed=seed,
    )
    write_json(
        output_path,
        {
            "method": "encoder_rq_kmeans",
            "encoder_model": encoder_model,
            "codebook_sizes": list(map(int, codebook_sizes)),
            "semantic_ids": semantic_ids,
        },
    )
    print(
        {
            "num_items": len(item_ids),
            "embedding_dim": int(embeddings.shape[1]),
            "codebook_sizes": list(map(int, codebook_sizes)),
            "encoder_model": encoder_model,
        }
    )


def parse_codebook_sizes(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--item-meta", required=True)
    p_build.add_argument("--output", required=True)
    p_build.add_argument("--encoder-model", required=True)
    p_build.add_argument("--codebook-sizes", default="64,128,256,512")
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--max-length", type=int, default=512)
    p_build.add_argument("--max-desc-words", type=int, default=160)
    p_build.add_argument("--seed", type=int, default=2026)
    p_build.add_argument("--device", default=None)
    p_build.add_argument("--save-embeddings", default=None)
    p_build.add_argument("--save-item-ids", default=None)

    p_rq = sub.add_parser("rq-kmeans")
    p_rq.add_argument("--embeddings", required=True)
    p_rq.add_argument("--item-ids", required=True)
    p_rq.add_argument("--output", required=True)
    p_rq.add_argument("--codebook-sizes", default="64,128,256,512")
    p_rq.add_argument("--seed", type=int, default=2026)

    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--semantic-ids", required=True)
    p_analyze.add_argument("--output", required=True)
    p_analyze.add_argument("--top-groups", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "build":
        build_semantic_ids_with_encoder(
            item_meta_path=args.item_meta,
            output_path=args.output,
            encoder_model=args.encoder_model,
            codebook_sizes=parse_codebook_sizes(args.codebook_sizes),
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_desc_words=args.max_desc_words,
            seed=args.seed,
            device=args.device,
            save_embeddings=args.save_embeddings,
            save_item_ids=args.save_item_ids,
        )
    elif args.cmd == "rq-kmeans":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("rq-kmeans mode requires numpy.") from exc
        embeddings = np.load(args.embeddings).astype("float32")
        item_ids = [str(x) for x in read_json(args.item_ids)]
        semantic_ids = build_rq_kmeans_from_embeddings(
            embeddings=embeddings,
            item_ids=item_ids,
            codebook_sizes=parse_codebook_sizes(args.codebook_sizes),
            seed=args.seed,
        )
        write_json(
            args.output,
            {
                "method": "rq_kmeans",
                "codebook_sizes": parse_codebook_sizes(args.codebook_sizes),
                "semantic_ids": semantic_ids,
            },
        )
    elif args.cmd == "analyze":
        analyze_semantic_ids(args.semantic_ids, args.output, args.top_groups)


if __name__ == "__main__":
    main()
