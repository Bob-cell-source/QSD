import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class NextItemDataset(Dataset):
    def __init__(self, sequences: List[Dict], max_len: int, split: str) -> None:
        self.max_len = max_len
        self.samples: List[Tuple[List[int], int]] = []
        for row in sequences:
            items = list(map(int, row["items"]))
            if len(items) < 3:
                continue
            if split == "train":
                for index in range(1, len(items) - 2):
                    self.samples.append((items[:index], items[index]))
            elif split == "valid":
                self.samples.append((items[:-2], items[-2]))
            elif split == "test":
                self.samples.append((items[:-1], items[-1]))
            else:
                raise ValueError(f"Unsupported split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sequence, target = self.samples[index]
        sequence = sequence[-self.max_len :]
        padded = [0] * (self.max_len - len(sequence)) + sequence
        return torch.tensor(padded, dtype=torch.long), target


class RandomNegativeSampler:
    def __init__(self, num_items: int, num_negatives: int) -> None:
        self.num_items = num_items
        self.num_negatives = num_negatives

    def sample(self, target: int, seen: set[int]) -> List[int]:
        candidates = [target]
        used = {target, *seen}
        while len(candidates) < 1 + self.num_negatives:
            negative = random.randint(1, self.num_items)
            if negative not in used:
                candidates.append(negative)
                used.add(negative)
        return candidates


def collate_train(batch, sampler: RandomNegativeSampler):
    sequences, targets = zip(*batch)
    sequence_tensor = torch.stack(sequences)
    candidates = []
    for sequence, target in zip(sequences, targets):
        seen = {int(item) for item in sequence.tolist() if int(item) > 0}
        candidates.append(sampler.sample(int(target), seen))
    return sequence_tensor, torch.tensor(candidates, dtype=torch.long)


def collate_eval(batch):
    sequences, targets = zip(*batch)
    return torch.stack(sequences), torch.tensor(targets, dtype=torch.long)
