import math

import torch
from torch import nn
from torch.nn import functional as F

from LoCoRec.locorec.model import CausalTransformerEncoder


def prefix_statistics(hard_sid_table):
    """Exclude padding; IDs are local to each depth, with padding group 0."""
    if hard_sid_table.ndim != 2 or hard_sid_table.size(0) < 3 or hard_sid_table.size(1) < 1:
        raise ValueError("Require at least two items and one SID level.")
    if hard_sid_table[0].ne(0).any() or hard_sid_table[1:].le(0).any():
        raise ValueError("Only padding row 0 may contain zero SID tokens.")
    n, depth = hard_sid_table.size(0) - 1, hard_sid_table.size(1)
    groups = torch.zeros_like(hard_sid_table)
    entropies, counts = [], []
    for r in range(1, depth + 1):
        _, inverse, count = torch.unique(
            hard_sid_table[1:, :r], dim=0, return_inverse=True, return_counts=True
        )
        groups[1:, r - 1] = inverse + 1
        p = count.double() / n
        entropies.append(-(p * p.log()).sum())
        counts.append(count.numel())
    entropy = torch.stack(entropies)
    cost = torch.cat([(entropy / math.log(n)).clamp(0, 1), entropy.new_ones(1)]).float()
    return groups, entropy, cost, counts


class MultiResolutionItemEncoder(nn.Module):
    def __init__(self, num_items, num_semantic_tokens, hard_sid_table, dim):
        super().__init__()
        if hard_sid_table.size(0) != num_items + 1 or hard_sid_table.max() > num_semantic_tokens:
            raise ValueError("SID table does not match the catalog/token vocabulary.")
        groups, entropy, cost, _ = prefix_statistics(hard_sid_table)
        self.register_buffer("hard_sid_table", hard_sid_table.long())
        self.register_buffer("prefix_group_id", groups)
        self.register_buffer("prefix_entropy", entropy)
        self.register_buffer("resolution_cost", cost)
        self.semantic_basis_embedding = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.shared_residual_embedding = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.private_residual_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        # Preserve the existing LoCoRec projection (including its bias) for transfer.
        self.basis_projection = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            for table in (self.semantic_basis_embedding, self.shared_residual_embedding,
                          self.private_residual_embedding):
                table.weight[0].zero_()

    def levels(self, items):
        tokens = self.hard_sid_table[items]
        return self.basis_projection(self.semantic_basis_embedding(tokens)) + self.shared_residual_embedding(tokens)

    def exact(self, items):
        vectors = self.output_norm(self.levels(items).mean(-2) + self.private_residual_embedding(items))
        return vectors * items.ne(0).unsqueeze(-1)

    def forward(self, items):
        levels = self.levels(items)
        divisors = torch.arange(1, levels.size(-2) + 1, device=items.device, dtype=levels.dtype)
        shared = levels.cumsum(-2) / divisors.unsqueeze(-1)
        shared = torch.cat([shared, shared[..., -1:, :]], dim=-2)
        private = self.private_residual_embedding(items).unsqueeze(-2)
        vectors = self.output_norm(shared + self.resolution_cost.unsqueeze(-1) * private)
        return vectors * items.ne(0).unsqueeze(-1).unsqueeze(-1)


class CCSR(nn.Module):
    def __init__(self, num_items, num_semantic_tokens, hard_sid_table, dim=128,
                 max_len=50, num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.depth = hard_sid_table.size(1)
        self.item_encoder = MultiResolutionItemEncoder(num_items, num_semantic_tokens, hard_sid_table, dim)
        self.sequence_encoder = CausalTransformerEncoder(dim, max_len, num_heads, num_layers, dropout)
        self.stop_head = nn.Linear(dim, self.depth)
        with torch.no_grad():
            self.stop_head.weight.zero_()
            q = 1.0 / torch.arange(self.depth + 1, 1, -1, dtype=torch.float)
            self.stop_head.bias.copy_(torch.logit(q))

    def encode_sequence(self, sequence):
        return self.sequence_encoder(sequence, self.item_encoder.exact(sequence))

    def stopping_distribution(self, h, uniform=False, fixed_resolution=None):
        if fixed_resolution is not None:
            if not 1 <= fixed_resolution <= self.depth + 1:
                raise ValueError("Fixed resolution must be in 1..L+1.")
            pi = h.new_zeros(h.size(0), self.depth + 1)
            pi[:, fixed_resolution - 1] = 1
            return pi
        if uniform:
            return h.new_full((h.size(0), self.depth + 1), 1 / (self.depth + 1))
        q = self.stop_head(h).sigmoid()
        survival = torch.cat([torch.ones_like(q[:, :1]), (1 - q).cumprod(-1)], -1)
        return torch.cat([survival[:, :-1] * q, survival[:, -1:]], -1)

    def forward(self, sequence, candidates, uniform=False, fixed_resolution=None):
        h = self.encode_sequence(sequence)
        pi = self.stopping_distribution(h, uniform, fixed_resolution)
        vectors = self.item_encoder(candidates)  # B, C, R, d
        scores = torch.einsum("bd,bcrd->brc", h, vectors) / math.sqrt(self.dim)
        return {"scores": scores, "pi": pi}

    def objective(self, output, lambda_res=0.0):
        scores, pi = output["scores"], output["pi"]
        loss_r = -F.log_softmax(scores, dim=-1)[..., 0]
        pred = (pi * loss_r).sum(-1).mean()
        res = (pi * self.item_encoder.resolution_cost).sum(-1).mean()
        return {"loss": pred + lambda_res * res, "loss_pred": pred,
                "loss_res": res, "loss_per_resolution": loss_r}

    def score_catalog(self, h, vectors, pi, mode="hard"):
        """vectors: C,R,d. Hard scoring groups users before matrix multiplication."""
        if mode == "hard":
            selected = pi.argmax(-1)
            scores = h.new_empty(h.size(0), vectors.size(0))
            for r in range(self.depth + 1):
                rows = selected.eq(r)
                if rows.any():
                    scores[rows] = h[rows] @ vectors[:, r].T
            return scores / math.sqrt(self.dim)
        if mode == "soft":
            return torch.einsum("br,bd,crd->bc", pi, h, vectors) / math.sqrt(self.dim)
        r = int(mode) - 1
        if not 0 <= r <= self.depth:
            raise ValueError("Resolution must be in 1..L+1.")
        return (h @ vectors[:, r].T) / math.sqrt(self.dim)


def load_hard_checkpoint(model, checkpoint):
    """Strict transfer of all reusable parameters, never soft weights or SID buffers."""
    state = checkpoint["model"]
    table = state.get("item_encoder.hard_sid_table")
    if table is None:
        tokens = state.get("item_encoder.soft_sid_table")
        prior = state.get("item_encoder.candidate_prior")
        if tokens is None or prior is None or tokens.shape != prior.shape:
            raise ValueError("Checkpoint must contain verifiable Hard SID assignments.")
        active = (prior > 0) & (tokens != 0)
        if not active[1:].sum(-1).eq(1).all():
            raise ValueError("Soft SID checkpoint rejected: expected exactly one active token per level.")
        table = (tokens * active).sum(-1)
    if not torch.equal(table.cpu(), model.item_encoder.hard_sid_table.cpu()):
        raise ValueError("Checkpoint SID/item mapping differs from the current dataset.")
    desired = dict(model.named_parameters())
    keys = [key for key in desired if not key.startswith("stop_head.")]
    missing = [key for key in keys if key not in state or state[key].shape != desired[key].shape]
    if missing:
        raise ValueError(f"Incompatible Hard SID checkpoint; missing/mismatched parameters: {missing}")
    with torch.no_grad():
        for key in keys:
            desired[key].copy_(state[key])
    return {"loaded_parameters": keys, "ignored_keys": sorted(set(state) - set(keys))}
