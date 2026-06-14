import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class NextItemDataset(Dataset):
    def __init__(self, sequences: List[Dict], max_len: int, split: str) -> None:
        self.max_len = max_len
        self.samples: List[Tuple[List[int], int, Tuple[int, ...]]] = []
        for row in sequences:
            items = [int(item) for item in row["items"]]
            if len(items) < 3:
                continue
            if split == "train":
                train_items = tuple(items[:-2])
                self.samples.extend(
                    (items[:index], items[index], train_items)
                    for index in range(1, len(items) - 2)
                )
            elif split == "valid":
                history = tuple(items[:-2])
                self.samples.append((list(history), items[-2], history))
            elif split == "test":
                history = tuple(items[:-1])
                self.samples.append((list(history), items[-1], history))
            else:
                raise ValueError(f"Unsupported split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Tuple[int, ...]]:
        sequence, target, full_context = self.samples[index]
        sequence = sequence[-self.max_len :]
        padded = [0] * (self.max_len - len(sequence)) + sequence
        return torch.tensor(padded, dtype=torch.long), target, full_context


class RandomNegativeSampler:
    def __init__(self, num_items: int, num_negatives: int) -> None:
        self.num_items = num_items
        self.num_negatives = num_negatives

    def sample(self, target: int, known_train_positives: Sequence[int]) -> List[int]:
        candidates = [target]
        used = {target, *known_train_positives}
        while len(candidates) <= self.num_negatives:
            negative = random.randint(1, self.num_items)
            if negative not in used:
                candidates.append(negative)
                used.add(negative)
        return candidates


def collate_train(batch, sampler: RandomNegativeSampler):
    sequences, targets, train_positives = zip(*batch)
    candidate_rows = []
    for target, positives in zip(targets, train_positives):
        candidate_rows.append(sampler.sample(int(target), positives))
    return torch.stack(sequences), torch.tensor(candidate_rows, dtype=torch.long)


def collate_eval(batch):
    sequences, targets, full_histories = zip(*batch)
    return (
        torch.stack(sequences),
        torch.tensor(targets, dtype=torch.long),
        full_histories,
    )
