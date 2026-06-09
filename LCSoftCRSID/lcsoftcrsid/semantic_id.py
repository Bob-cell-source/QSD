import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .io import read_json, write_json


SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def trim_description(text: str, max_words: int) -> str:
    text = normalize_text(text)
    for marker in [
        "What's in the Box",
        "What’s in the Box",
        "Package Contents",
        "Pros:",
        "CONS:",
        "Cons:",
    ]:
        position = text.find(marker)
        if position >= 0:
            text = text[:position].strip()
    return " ".join(text.split()[:max_words])


def build_item_text(metadata: Dict, max_description_words: int) -> str:
    title = normalize_text(metadata.get("title", ""))
    brand = normalize_text(metadata.get("brand", ""))
    categories = [normalize_text(value) for value in metadata.get("categories", []) if normalize_text(value)]
    description = trim_description(metadata.get("description", ""), max_description_words)
    fields = []
    if title:
        fields.append(f"Title: {title}.")
    if brand:
        fields.append(f"Brand: {brand}.")
    if categories:
        fields.append(f"Category: {' > '.join(categories)}.")
    if description:
        fields.append(f"Description: {description}")
    return " ".join(fields)


def load_item_texts(path: str | Path, max_description_words: int) -> Tuple[List[str], List[str]]:
    metadata = read_json(path)
    item_ids = sorted(metadata, key=lambda value: int(value))
    texts = [build_item_text(metadata[item_id], max_description_words) for item_id in item_ids]
    return item_ids, texts


def encode_texts(
    texts: Sequence[str],
    encoder_model: str,
    batch_size: int,
    max_length: int,
    device: str | None,
):
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
    except (ImportError, OSError, ValueError):
        import torch
        from transformers import AutoModel, AutoTokenizer

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        model = AutoModel.from_pretrained(encoder_model).to(resolved_device)
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = tokenizer(
                    list(texts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(resolved_device) for key, value in tokens.items()}
                hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                outputs.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(outputs, dim=0).numpy().astype("float32")


def residual_kmeans(
    embeddings,
    item_ids: Sequence[str],
    codebook_sizes: Sequence[int],
    seed: int,
) -> Dict[str, List[int]]:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    residual = np.asarray(embeddings, dtype="float32").copy()
    all_codes = []
    for level, codebook_size in enumerate(codebook_sizes):
        kmeans = MiniBatchKMeans(
            n_clusters=int(codebook_size),
            random_state=seed + level,
            batch_size=min(4096, max(256, residual.shape[0] // 10)),
            n_init="auto",
        )
        codes = kmeans.fit_predict(residual)
        residual -= kmeans.cluster_centers_.astype("float32")[codes]
        all_codes.append(codes.tolist())
    return {
        item_id: [all_codes[level][index] for level in range(len(codebook_sizes))]
        for index, item_id in enumerate(item_ids)
    }


def parse_sizes(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard Semantic IDs for LC-SoftCRSID.")
    parser.add_argument("--item-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--codebook-sizes", default="64,128,256,512")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-description-words", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    item_ids, texts = load_item_texts(args.item_meta, args.max_description_words)
    embeddings = encode_texts(texts, args.encoder_model, args.batch_size, args.max_length, args.device)
    sizes = parse_sizes(args.codebook_sizes)
    semantic_ids = residual_kmeans(embeddings, item_ids, sizes, args.seed)
    write_json(
        args.output,
        {
            "method": "encoder_rq_kmeans",
            "encoder_model": args.encoder_model,
            "codebook_sizes": sizes,
            "semantic_ids": semantic_ids,
        },
    )
