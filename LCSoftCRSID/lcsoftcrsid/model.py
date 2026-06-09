import math
from typing import Dict, Literal, Tuple

import torch
from torch import nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.transpose(-1, -2)
        hidden = self.dropout1(self.conv1(hidden))
        hidden = self.relu(hidden)
        hidden = self.dropout2(self.conv2(hidden))
        return hidden.transpose(-1, -2)


class CausalTransformerEncoder(nn.Module):
    """SASRec-style causal encoder over externally constructed item vectors."""

    def __init__(
        self,
        dim: int,
        max_len: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max_len + 1, dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.attention_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_norms.append(nn.LayerNorm(dim, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(dim, num_heads, dropout=dropout))
            self.ffn_norms.append(nn.LayerNorm(dim, eps=1e-8))
            self.ffn_layers.append(PointWiseFeedForward(dim, dropout))
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dim = dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.position_embedding.weight[0].zero_()

    def forward(self, sequence: torch.Tensor, item_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = sequence.shape
        positions = torch.arange(1, length + 1, device=sequence.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1) * sequence.ne(0)
        hidden = item_vectors * math.sqrt(self.dim) + self.position_embedding(positions)
        hidden = self.input_dropout(hidden)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequence.device),
            diagonal=1,
        )

        for attention_norm, attention, ffn_norm, ffn in zip(
            self.attention_norms,
            self.attention_layers,
            self.ffn_norms,
            self.ffn_layers,
        ):
            transposed = hidden.transpose(0, 1)
            query = attention_norm(transposed)
            attention_output, _ = attention(query, query, query, attn_mask=causal_mask)
            hidden = (transposed + attention_output).transpose(0, 1)
            hidden = hidden + ffn(ffn_norm(hidden))
            hidden = hidden * sequence.ne(0).unsqueeze(-1)

        hidden = self.output_norm(hidden)
        return hidden[:, -1], hidden


class LCSoftCRSIDItemEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_semantic_tokens: int,
        soft_sid_table: torch.Tensor,
        soft_sid_weights: torch.Tensor,
        semantic_reliability: torch.Tensor,
        item_frequency: torch.Tensor,
        dim: int,
        dropout: float,
        tail_tau: float,
        residual_scale: float,
        frequency_transform: Literal["raw", "log"] = "raw",
        disable_semantic_basis: bool = False,
        disable_shared_residual: bool = False,
        disable_private_residual: bool = False,
    ) -> None:
        super().__init__()
        if frequency_transform not in {"raw", "log"}:
            raise ValueError(f"Unsupported frequency transform: {frequency_transform}")
        if soft_sid_table.shape != soft_sid_weights.shape:
            raise ValueError("Soft SID token and weight tables must have identical shapes.")

        self.register_buffer("soft_sid_table", soft_sid_table.long())
        self.register_buffer("soft_sid_weights", soft_sid_weights.float())
        self.register_buffer("semantic_reliability", semantic_reliability.float().clamp(0.0, 1.0))

        frequency = item_frequency.float().clamp_min(0.0)
        if frequency_transform == "log":
            frequency = torch.log1p(frequency)
        calibrated_tau = float(tail_tau) * self.semantic_reliability.clamp_min(1e-6)
        private_weight = frequency / (frequency + calibrated_tau)
        private_weight[0] = 0.0
        self.register_buffer("private_weight", private_weight.clamp(0.0, 1.0))

        self.semantic_basis_embedding = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.shared_residual_embedding = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.private_residual_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.basis_projection = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)
        self.disable_semantic_basis = disable_semantic_basis
        self.disable_shared_residual = disable_shared_residual
        self.disable_private_residual = disable_private_residual
        self.dim = dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.semantic_basis_embedding.weight[0].zero_()
            self.shared_residual_embedding.weight[0].zero_()
            self.private_residual_embedding.weight[0].zero_()

    def semantic_pool(self, embedding: nn.Embedding, items: torch.Tensor) -> torch.Tensor:
        tokens = self.soft_sid_table[items]
        weights = self.soft_sid_weights[items].unsqueeze(-1)
        candidate_pool = (embedding(tokens) * weights).sum(dim=-2)
        slot_mask = self.soft_sid_weights[items].sum(dim=-1, keepdim=True).gt(0).float()
        return (candidate_pool * slot_mask).sum(dim=-2) / slot_mask.sum(dim=-2).clamp_min(1.0)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        zeros = items.new_zeros((*items.shape, self.dim), dtype=torch.float)
        basis = zeros if self.disable_semantic_basis else self.basis_projection(
            self.semantic_pool(self.semantic_basis_embedding, items)
        )
        shared = zeros if self.disable_shared_residual else self.semantic_pool(
            self.shared_residual_embedding, items
        )
        private = zeros if self.disable_private_residual else self.private_residual_embedding(items)

        alpha = self.private_weight[items].unsqueeze(-1)
        residual = alpha * private + (1.0 - alpha) * shared
        output = self.output_norm(basis + self.residual_scale * residual)
        return self.dropout(output) * items.ne(0).unsqueeze(-1)


class LCSoftCRSID(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_semantic_tokens: int,
        soft_sid_table: torch.Tensor,
        soft_sid_weights: torch.Tensor,
        semantic_reliability: torch.Tensor,
        item_frequency: torch.Tensor,
        dim: int = 128,
        max_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        tail_tau: float = 20.0,
        residual_scale: float = 1.0,
        frequency_transform: Literal["raw", "log"] = "raw",
        disable_semantic_basis: bool = False,
        disable_shared_residual: bool = False,
        disable_private_residual: bool = False,
    ) -> None:
        super().__init__()
        self.item_encoder = LCSoftCRSIDItemEncoder(
            num_items=num_items,
            num_semantic_tokens=num_semantic_tokens,
            soft_sid_table=soft_sid_table,
            soft_sid_weights=soft_sid_weights,
            semantic_reliability=semantic_reliability,
            item_frequency=item_frequency,
            dim=dim,
            dropout=dropout,
            tail_tau=tail_tau,
            residual_scale=residual_scale,
            frequency_transform=frequency_transform,
            disable_semantic_basis=disable_semantic_basis,
            disable_shared_residual=disable_shared_residual,
            disable_private_residual=disable_private_residual,
        )
        self.sequence_encoder = CausalTransformerEncoder(
            dim=dim,
            max_len=max_len,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, sequence: torch.Tensor, candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
        sequence_vectors = self.item_encoder(sequence)
        user_vector, _ = self.sequence_encoder(sequence, sequence_vectors)
        candidate_vectors = self.item_encoder(candidates)
        scores = torch.einsum("bd,bcd->bc", user_vector, candidate_vectors)
        return {"score": scores}
