"""One-shot parameter untying probe, inspired by structural refinement.

This is NOT an implementation of SteepGS/Splitting Steepest Descent: linear
embedding lookup does not satisfy their nonlinear-primitive assumptions.
Split selection uses a first-order, group-size-weighted quadratic surrogate.
"""
from copy import deepcopy

import torch
from torch import nn

from .model import SharingItemEncoder


class RefinementItemEncoder(SharingItemEncoder):
    def __init__(self, hard, num_tokens, dim):
        super().__init__('hard', hard, num_tokens, None, None, dim)
        self.register_buffer('lookup_table', hard.clone())
        self.captured = None

    def forward(self, items):
        value = self.private_embedding(items) + self.shared_embedding(self.lookup_table[items]).mean(-2)
        if self.captured is not None and value.requires_grad:
            def capture(gradient):
                # Includes both history and candidate paths. Divide by L to get
                # each level's contribution to its shared token gradient.
                with torch.no_grad():
                    self.captured.index_add_(0, items.reshape(-1),
                                             gradient.reshape(-1, self.dim) / self.lookup_table.shape[1])
            value.register_hook(capture)
        return {'vectors': self.output_norm(value) * items.ne(0).unsqueeze(-1)}


def partition_score(grad_a, grad_b, left, right):
    """Cross-view predicted advantage of untying under a quadratic surrogate.

    For a group C, min_delta G_C.delta + |C| ||delta||^2/(2 eta)
    gives delta=-eta G_C/|C|. The first-order advantage on a separate
    gradient estimate is proportional to the expression below. With A=B
    this is nonnegative by construction; independent B can be negative.
    This does not guarantee AdamW or held-out recommendation improvement.
    """
    a0, a1 = grad_a[left].sum(0), grad_a[right].sum(0)
    b0, b1 = grad_b[left].sum(0), grad_b[right].sum(0)
    return float(a0.dot(b0) / len(left) + a1.dot(b1) / len(right)
                 - (a0 + a1).dot(b0 + b1) / (len(left) + len(right)))


def propose_partitions(hard, gradient, frequency, budget=32, min_group=8, seed=1707):
    """Fit splits using view A only; do not look at audit view B or validation."""
    hard, gradient, frequency = hard.cpu(), gradient.double().cpu(), frequency.cpu()
    candidates = []
    for level in range(hard.shape[1]):
        for token in hard[1:, level].unique().tolist():
            members = torch.where(hard[:, level].eq(token))[0]
            if len(members) < min_group:
                continue
            centered = gradient[members] - gradient[members].mean(0)
            if centered.square().sum() <= 1e-24:
                continue
            # Exact PCA is inexpensive for the small catalog probe. Balanced
            # splitting prevents the learned arm from buying singleton IDs.
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            order = (centered @ vh[0]).argsort(stable=True)
            left, right = members[order[:len(members)//2]], members[order[len(members)//2:]]
            score = partition_score(gradient, gradient, left, right)
            candidates.append({'level': level, 'parent': token,
                               'left': left.tolist(), 'right': right.tolist(),
                               'fit_score': score, 'frequency': float(frequency[members].sum())})
    ranked = sorted(candidates, key=lambda r: (-r['fit_score'], r['parent']))
    count = min(budget, len(ranked))
    learned = deepcopy(ranked[:count])
    generator = torch.Generator().manual_seed(seed)

    def shuffle_members(rows):
        out = deepcopy(rows)
        for row in out:
            members = torch.tensor(sorted(row['left'] + row['right']))
            members = members[torch.randperm(len(members), generator=generator)].tolist()
            nleft = len(row['left'])
            row['left'], row['right'] = members[:nleft], members[nleft:]
        return out

    random_ids = torch.randperm(len(ranked), generator=generator)[:count].tolist()
    return {
        'no_split': [],
        'gradient_split': learned,
        'same_parent_random': shuffle_members(learned),
        'random_split': shuffle_members([ranked[i] for i in random_ids]),
        'frequency_split': shuffle_members(sorted(candidates, key=lambda r: (-r['frequency'], r['parent']))[:count]),
    }


@torch.no_grad()
def apply_partitions(model, optimizer, partitions):
    """Copy parent rows and Adam moments; preserve the function at split time."""
    if not partitions:
        return
    enc = model.item_encoder
    old = enc.shared_embedding
    parents = torch.tensor([row['parent'] for row in partitions], device=old.weight.device)
    weight = torch.cat([old.weight.detach(), old.weight.detach()[parents]], 0)
    new = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
    if len(set((r['level'], r['parent']) for r in partitions)) != len(partitions):
        raise ValueError('A parent can be split only once in this probe')
    for index, row in enumerate(partitions):
        members = row['left'] + row['right']
        actual = torch.where(enc.hard_sid_table[:, row['level']].eq(row['parent']))[0].tolist()
        if sorted(members) != actual or not row['left'] or not row['right']:
            raise ValueError('Partition must cover exactly the original parent membership')
        enc.lookup_table[row['right'], row['level']] = old.num_embeddings + index
    for group in optimizer.param_groups:
        group['params'] = [new.weight if p is old.weight else p for p in group['params']]
    if old.weight in optimizer.state:
        previous = optimizer.state.pop(old.weight)
        optimizer.state[new.weight] = {
            key: torch.cat([value, value[parents]], 0) if torch.is_tensor(value) and value.shape == old.weight.shape
            else deepcopy(value) for key, value in previous.items()
        }
    enc.shared_embedding = new
