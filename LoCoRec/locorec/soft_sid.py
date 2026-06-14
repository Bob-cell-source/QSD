from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class SoftSIDConfig:
    top_m: int = 4
    min_overlap_slots: int = 3
    min_support: float = 0.05
    consistency_floor: float = 0.10
    max_neighbors: int = 50
    tie_break_seed: int = 2026
    leave_one_level_out: bool = True


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_items = semantic_table.size(0) - 1
    depth = semantic_table.size(1)
    inverted: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for item, codes in item_codes.items():
        for slot, code in enumerate(codes):
            inverted[(slot, code)].append(item)

    soft_ids = torch.zeros(num_items + 1, depth, config.top_m, dtype=torch.long)
    priors = torch.zeros(num_items + 1, depth, config.top_m, dtype=torch.float)
    local_consistency = torch.zeros(num_items + 1, dtype=torch.float)

    for item in range(1, num_items + 1):
        item_consistency = 0.0
        valid_consistency_levels = 0
        for slot in range(depth):
            overlap_counts: Counter[int] = Counter()
            for context_slot, code in enumerate(item_codes[item]):
                if config.leave_one_level_out and context_slot == slot:
                    continue
                overlap_counts.update(inverted[(context_slot, code)])
            neighbors = [
                (neighbor, overlap)
                for neighbor, overlap in overlap_counts.items()
                if neighbor != item and overlap >= config.min_overlap_slots
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
            denominator = max(len(neighbor_ids), 1)
            hard_token = int(semantic_table[item, slot])
            counts = Counter(
                int(semantic_table[neighbor, slot]) for neighbor in neighbor_ids
            )
            candidates = {
                token: float(count)
                for token, count in counts.items()
                if token > 0 and count / denominator >= config.min_support
            }
            candidates[hard_token] = candidates.get(hard_token, 0.0) + denominator
            ranked = sorted(
                (
                    (token, max(count, 1e-6) ** 2)
                    for token, count in candidates.items()
                ),
                key=lambda row: (row[1], row[0] == hard_token),
                reverse=True,
            )[: config.top_m]
            scores = torch.tensor([score for _, score in ranked], dtype=torch.float)
            weights = scores / scores.sum().clamp_min(1e-6)
            for rank, ((token, _), weight) in enumerate(zip(ranked, weights.tolist())):
                soft_ids[item, slot, rank] = token
                priors[item, slot, rank] = weight

            if neighbor_ids:
                if config.leave_one_level_out:
                    # Reliability is the held-out conditional concentration.
                    # It is independent of the hard anchor and candidate prior.
                    item_consistency += max(counts.values()) / len(neighbor_ids)
                else:
                    item_consistency += sum(
                        weight * counts.get(token, 0) / denominator
                        for (token, _), weight in zip(ranked, weights.tolist())
                    )
                valid_consistency_levels += 1

        local_consistency[item] = max(
            config.consistency_floor,
            item_consistency / max(valid_consistency_levels, 1),
        )
    return soft_ids, priors, local_consistency.clamp(0.0, 1.0)


def build_train_item_frequency(sequences: List[Dict], num_items: int) -> torch.Tensor:
    frequency = torch.zeros(num_items + 1, dtype=torch.float)
    for row in sequences:
        for item in row["items"][:-2]:
            item = int(item)
            if 0 < item <= num_items:
                frequency[item] += 1.0
    return frequency
