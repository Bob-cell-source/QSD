from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import warnings

import torch


@dataclass(frozen=True)
class SoftSIDConfig:
    top_m: int = 4
    min_overlap_slots: int = 3
    min_support: float = 0.05
    reliability_floor: float = 0.10
    max_neighbors: int = 50
    candidate_construction: str = "local_prior"


def build_semantic_table(
    semantic_obj: Dict,
    num_items: int,
) -> Tuple[torch.Tensor, Dict[int, List[int]], int]:
    """Map slot-specific RQ codes into one non-overlapping token vocabulary."""
    raw = {int(key): list(map(int, value)) for key, value in semantic_obj["semantic_ids"].items()}
    codebook_sizes = list(map(int, semantic_obj["codebook_sizes"]))
    offsets = [1]
    for size in codebook_sizes[:-1]:
        offsets.append(offsets[-1] + size)

    table = torch.zeros(num_items + 1, len(codebook_sizes), dtype=torch.long)
    item_codes: Dict[int, List[int]] = {}
    for item in range(1, num_items + 1):
        codes = raw.get(item)
        if codes is None:
            raise ValueError(f"Semantic ID is missing for item {item}.")
        if len(codes) != len(codebook_sizes):
            raise ValueError(f"Item {item} has {len(codes)} codes; expected {len(codebook_sizes)}.")
        table[item] = torch.tensor(
            [offsets[slot] + int(code) for slot, code in enumerate(codes)],
            dtype=torch.long,
        )
        item_codes[item] = codes
    return table, item_codes, sum(codebook_sizes)


def build_soft_sid_table(
    semantic_table: torch.Tensor,
    item_codes: Dict[int, List[int]],
    config: SoftSIDConfig,
    base_neighbors: Optional[Dict[int, List[int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct local-consistent candidate tokens and item reliability."""
    if config.candidate_construction not in {"local_prior", "uniform_topk"}:
        raise ValueError(
            f"Unsupported candidate construction: {config.candidate_construction}"
        )
    num_items = semantic_table.size(0) - 1
    depth = semantic_table.size(1)
    top_m = max(int(config.top_m), 1)

    inverted: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for item, codes in item_codes.items():
        for slot, code in enumerate(codes):
            inverted[(slot, int(code))].append(item)

    soft_ids = torch.zeros(num_items + 1, depth, top_m, dtype=torch.long)
    soft_weights = torch.zeros(num_items + 1, depth, top_m, dtype=torch.float)
    reliability = torch.zeros(num_items + 1, dtype=torch.float)

    for item in range(1, num_items + 1):
        if base_neighbors is None:
            overlap_counts: Counter[int] = Counter()
            for slot, code in enumerate(item_codes[item]):
                overlap_counts.update(inverted[(slot, int(code))])

            neighbors = [
                (neighbor, overlap)
                for neighbor, overlap in overlap_counts.items()
                if neighbor != item and overlap >= config.min_overlap_slots
            ]
            neighbors.sort(key=lambda row: (row[1], -row[0]), reverse=True)
            neighbor_ids = [neighbor for neighbor, _ in neighbors]
        else:
            neighbor_ids = [
                int(neighbor)
                for neighbor in base_neighbors.get(item, [])
                if int(neighbor) != item and 0 < int(neighbor) <= num_items
            ]
        if config.max_neighbors > 0:
            neighbor_ids = neighbor_ids[: config.max_neighbors]
        denominator = max(len(neighbor_ids), 1)

        item_reliability = 0.0
        for slot in range(depth):
            hard_token = int(semantic_table[item, slot])
            counts = Counter(int(semantic_table[neighbor, slot]) for neighbor in neighbor_ids)
            if config.candidate_construction == "uniform_topk":
                neighbor_tokens = sorted(
                    (
                        (token, count)
                        for token, count in counts.items()
                        if token > 0 and token != hard_token
                    ),
                    key=lambda row: (-row[1], row[0]),
                )
                selected_tokens = [hard_token]
                selected_tokens.extend(
                    token for token, _ in neighbor_tokens[: max(top_m - 1, 0)]
                )
                uniform_weight = 1.0 / len(selected_tokens)
                for rank, token in enumerate(selected_tokens):
                    soft_ids[item, slot, rank] = token
                    soft_weights[item, slot, rank] = uniform_weight
                slot_consensus = max(counts.values(), default=0) / denominator
                item_reliability += slot_consensus
                continue

            candidates = {
                token: float(count)
                for token, count in counts.items()
                if token > 0 and count / denominator >= config.min_support
            }

            # The original hard token remains a stable anchor while local
            # candidates repair overly rigid code assignments.
            candidates[hard_token] = candidates.get(hard_token, 0.0) + denominator
            ranked = sorted(
                (
                    (token, weight_count, max(weight_count, 1e-6) ** 2)
                    for token, weight_count in candidates.items()
                ),
                key=lambda row: (row[2], row[0] == hard_token),
                reverse=True,
            )[:top_m]

            scores = torch.tensor([score for _, _, score in ranked], dtype=torch.float)
            weights = scores / scores.sum().clamp_min(1e-6)
            slot_reliability = 0.0
            for rank, ((token, _, _), weight) in enumerate(zip(ranked, weights.tolist())):
                soft_ids[item, slot, rank] = token
                soft_weights[item, slot, rank] = weight
                slot_reliability += weight * counts.get(token, 0) / denominator
            item_reliability += slot_reliability

        reliability[item] = max(
            float(config.reliability_floor),
            item_reliability / max(depth, 1),
        )

    return soft_ids, soft_weights, reliability.clamp(0.0, 1.0)


def build_text_knn_neighbors(
    embeddings_path: str | Path,
    item_ids_path: str | Path,
    num_items: int,
    max_neighbors: int,
    chunk_size: int = 256,
) -> Tuple[Dict[int, List[int]], Dict[str, float | int | str]]:
    """Build cosine text neighbors for a controlled neighborhood baseline."""
    import json

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("text_knn neighbors require numpy.") from exc

    embeddings = np.load(embeddings_path).astype("float32")
    with Path(item_ids_path).open("r", encoding="utf-8") as handle:
        item_ids = [int(item) for item in json.load(handle)]
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("Embedding rows and embedding item IDs must have the same length.")
    if set(item_ids) != set(range(1, num_items + 1)):
        raise ValueError("Text embedding item IDs must cover every internal item ID exactly once.")

    row_for_item = {item: row for row, item in enumerate(item_ids)}
    ordered = embeddings[[row_for_item[item] for item in range(1, num_items + 1)]]
    norms = np.linalg.norm(ordered, axis=1, keepdims=True)
    ordered = ordered / np.maximum(norms, 1e-12)
    k = min(max(int(max_neighbors), 1) + 1, num_items)
    start = time.perf_counter()
    backend = "chunked_exact_cosine"

    try:
        import faiss

        index = faiss.IndexFlatIP(ordered.shape[1])
        index.add(ordered)
        _, indices = index.search(ordered, k)
        backend = "faiss_flat_ip"
    except ImportError:
        warnings.warn(
            "faiss is unavailable; text-kNN uses exact chunked cosine search. "
            "Install faiss for larger item collections.",
            RuntimeWarning,
        )
        indices = np.empty((num_items, k), dtype="int64")
        chunk_size = max(int(chunk_size), 1)
        for begin in range(0, num_items, chunk_size):
            end = min(begin + chunk_size, num_items)
            scores = ordered[begin:end] @ ordered.T
            local = np.argpartition(scores, -k, axis=1)[:, -k:]
            local_scores = np.take_along_axis(scores, local, axis=1)
            order = np.argsort(local_scores, axis=1)[:, ::-1]
            indices[begin:end] = np.take_along_axis(local, order, axis=1)

    neighbors: Dict[int, List[int]] = {}
    for item in range(1, num_items + 1):
        rows = indices[item - 1]
        values = [int(row) + 1 for row in rows if int(row) + 1 != item]
        neighbors[item] = values[:max_neighbors]
    return neighbors, {
        "backend": backend,
        "num_items": num_items,
        "embedding_dim": int(ordered.shape[1]),
        "max_neighbors": int(max_neighbors),
        "elapsed_seconds": time.perf_counter() - start,
    }


def build_train_item_frequency(sequences: List[Dict], num_items: int) -> torch.Tensor:
    """Count only training interactions; validation/test targets are excluded."""
    frequency = torch.zeros(num_items + 1, dtype=torch.float)
    for row in sequences:
        for item in row["items"][:-2]:
            item = int(item)
            if 0 < item <= num_items:
                frequency[item] += 1.0
    return frequency
