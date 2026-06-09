from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class SoftSIDConfig:
    top_m: int = 4
    min_overlap_slots: int = 2
    min_support: float = 0.05
    support_eta: float = 2.0
    hard_token_prior: float = 1.0
    reliability_floor: float = 0.10
    max_neighbors: int = 50


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct local-consistent candidate tokens and item reliability."""
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
        overlap_counts: Counter[int] = Counter()
        for slot, code in enumerate(item_codes[item]):
            overlap_counts.update(inverted[(slot, int(code))])

        neighbors = [
            (neighbor, overlap)
            for neighbor, overlap in overlap_counts.items()
            if neighbor != item and overlap >= config.min_overlap_slots
        ]
        neighbors.sort(key=lambda row: (row[1], -row[0]), reverse=True)
        if config.max_neighbors > 0:
            neighbors = neighbors[: config.max_neighbors]
        neighbor_ids = [neighbor for neighbor, _ in neighbors]
        denominator = max(len(neighbor_ids), 1)

        item_reliability = 0.0
        for slot in range(depth):
            hard_token = int(semantic_table[item, slot])
            counts = Counter(int(semantic_table[neighbor, slot]) for neighbor in neighbor_ids)
            candidates = {
                token: float(count)
                for token, count in counts.items()
                if token > 0 and count / denominator >= config.min_support
            }

            # The original hard token remains a stable anchor while local
            # candidates repair overly rigid code assignments.
            candidates[hard_token] = candidates.get(hard_token, 0.0) + max(
                1.0,
                float(config.hard_token_prior) * denominator,
            )
            ranked = sorted(
                (
                    (token, weight_count, max(weight_count, 1e-6) ** config.support_eta)
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


def build_train_item_frequency(sequences: List[Dict], num_items: int) -> torch.Tensor:
    """Count only training interactions; validation/test targets are excluded."""
    frequency = torch.zeros(num_items + 1, dtype=torch.float)
    for row in sequences:
        for item in row["items"][:-2]:
            item = int(item)
            if 0 < item <= num_items:
                frequency[item] += 1.0
    return frequency
