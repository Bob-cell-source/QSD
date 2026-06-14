from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class SoftSIDConfig:
    top_m: int = 4
    loo_min_overlap_slots: int = 2
    min_support: float = 0.05
    max_neighbors: int = 50
    tie_break_seed: int = 2026


def _stable_tie_break(item: int, neighbor: int, seed: int) -> int:
    value = (
        (item * 0x9E3779B185EBCA87)
        ^ (neighbor * 0xC2B2AE3D27D4EB4F)
        ^ seed
    ) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def build_semantic_table(
    semantic_obj: Dict,
    num_items: int,
) -> Tuple[torch.Tensor, Dict[int, List[int]], int]:
    raw = {
        int(item): [int(code) for code in codes]
        for item, codes in semantic_obj["semantic_ids"].items()
    }
    codebook_sizes = [int(size) for size in semantic_obj["codebook_sizes"]]
    offsets = [1]
    for size in codebook_sizes[:-1]:
        offsets.append(offsets[-1] + size)

    table = torch.zeros(num_items + 1, len(codebook_sizes), dtype=torch.long)
    item_codes: Dict[int, List[int]] = {}
    for item in range(1, num_items + 1):
        codes = raw.get(item)
        if codes is None or len(codes) != len(codebook_sizes):
            raise ValueError(f"Invalid Semantic ID for item {item}.")
        table[item] = torch.tensor(
            [offsets[slot] + code for slot, code in enumerate(codes)],
            dtype=torch.long,
        )
        item_codes[item] = codes
    return table, item_codes, sum(codebook_sizes)


def build_soft_sid_table(
    semantic_table: torch.Tensor,
    item_codes: Dict[int, List[int]],
    config: SoftSIDConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_items = semantic_table.size(0) - 1
    depth = semantic_table.size(1)
    inverted: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for item, codes in item_codes.items():
        for slot, code in enumerate(codes):
            inverted[(slot, code)].append(item)

    soft_ids = torch.zeros(num_items + 1, depth, config.top_m, dtype=torch.long)
    priors = torch.zeros(num_items + 1, depth, config.top_m, dtype=torch.float)
    local_consistency = torch.zeros(num_items + 1, dtype=torch.float)
    hard_consistency = torch.zeros(num_items + 1, dtype=torch.float)

    for item in range(1, num_items + 1):
        item_local_consistency = 0.0
        item_hard_consistency = 0.0
        valid_consistency_slots = 0
        for slot in range(depth):
            overlap_counts: Counter[int] = Counter()
            for context_slot, code in enumerate(item_codes[item]):
                if context_slot != slot:
                    overlap_counts.update(inverted[(context_slot, code)])
            neighbors = [
                (neighbor, overlap)
                for neighbor, overlap in overlap_counts.items()
                if neighbor != item
                and overlap >= config.loo_min_overlap_slots
            ]
            neighbors.sort(
                key=lambda row: (
                    row[1],
                    _stable_tie_break(
                        item,
                        row[0],
                        config.tie_break_seed + slot * 0x9E3779B1,
                    ),
                ),
                reverse=True,
            )
            neighbor_ids = [
                neighbor for neighbor, _ in neighbors[: config.max_neighbors]
            ]
            neighbor_count = len(neighbor_ids)
            hard_token = int(semantic_table[item, slot])
            counts = Counter(
                int(semantic_table[neighbor, slot]) for neighbor in neighbor_ids
            )
            if neighbor_count == 0:
                soft_ids[item, slot, 0] = hard_token
                priors[item, slot, 0] = 1.0
                continue

            support = {
                token: count / neighbor_count
                for token, count in counts.items()
                if token > 0 and count / neighbor_count >= config.min_support
            }
            non_hard = sorted(
                (
                    (token, value)
                    for token, value in support.items()
                    if token != hard_token
                ),
                key=lambda row: (row[1], row[0]),
                reverse=True,
            )[: max(config.top_m - 1, 0)]
            hard_support = counts.get(hard_token, 0) / neighbor_count
            # Preserve the hard token as an anchor without the previous
            # neighbor-count-sized pseudo count. One virtual observation is
            # used only when the LOO neighborhood gives it no support.
            hard_anchor = max(hard_support, 1.0 / neighbor_count)
            ranked = [(hard_token, hard_anchor), *non_hard]
            ranked = sorted(
                ranked,
                key=lambda row: (row[1], row[0] == hard_token, row[0]),
                reverse=True,
            )[: config.top_m]
            scores = torch.tensor(
                [max(value, 1e-8) ** 2 for _, value in ranked],
                dtype=torch.float,
            )
            weights = scores / scores.sum().clamp_min(1e-6)
            for rank, ((token, _), weight) in enumerate(
                zip(ranked, weights.tolist())
            ):
                soft_ids[item, slot, rank] = token
                priors[item, slot, rank] = weight

            item_local_consistency += max(counts.values()) / neighbor_count
            item_hard_consistency += hard_support
            valid_consistency_slots += 1

        if valid_consistency_slots:
            local_consistency[item] = (
                item_local_consistency / valid_consistency_slots
            )
            hard_consistency[item] = (
                item_hard_consistency / valid_consistency_slots
            )
    return (
        soft_ids,
        priors,
        local_consistency.clamp(0.0, 1.0),
        hard_consistency.clamp(0.0, 1.0),
    )


def build_train_item_frequency(sequences: List[Dict], num_items: int) -> torch.Tensor:
    frequency = torch.zeros(num_items + 1, dtype=torch.float)
    for row in sequences:
        for item in row["items"][:-2]:
            item = int(item)
            if 0 < item <= num_items:
                frequency[item] += 1.0
    return frequency
