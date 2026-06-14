import math
from typing import Dict, Tuple

import torch
from torch import nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.transpose(-1, -2)
        hidden = self.dropout1(self.conv1(hidden)).relu()
        hidden = self.dropout2(self.conv2(hidden))
        return hidden.transpose(-1, -2)


class CausalTransformerEncoder(nn.Module):
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
            self.attention_layers.append(
                nn.MultiheadAttention(dim, num_heads, dropout=dropout)
            )
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

    def forward(self, sequence: torch.Tensor, item_vectors: torch.Tensor) -> torch.Tensor:
        batch_size, length = sequence.shape
        positions = torch.arange(1, length + 1, device=sequence.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1) * sequence.ne(0)
        hidden = item_vectors * math.sqrt(self.dim) + self.position_embedding(positions)
        hidden = self.input_dropout(hidden)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequence.device),
            diagonal=1,
        )
        padding_mask = sequence.eq(0)
        for attention_norm, attention, ffn_norm, ffn in zip(
            self.attention_norms,
            self.attention_layers,
            self.ffn_norms,
            self.ffn_layers,
        ):
            transposed = hidden.transpose(0, 1)
            query = attention_norm(transposed)
            attention_output, _ = attention(
                query,
                query,
                query,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            attention_output = torch.nan_to_num(attention_output)
            hidden = (transposed + attention_output).transpose(0, 1)
            hidden = hidden + ffn(ffn_norm(hidden))
            hidden = hidden * sequence.ne(0).unsqueeze(-1)
        return self.output_norm(hidden)[:, -1]


class LoCoRecItemEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_semantic_tokens: int,
        soft_sid_table: torch.Tensor,
        candidate_prior: torch.Tensor,
        local_consistency: torch.Tensor,
        item_frequency: torch.Tensor,
        dim: int = 128,
        dropout: float = 0.2,
        tail_tau: float = 20.0,
        residual_scale: float = 1.0,
        gate_correction_scale: float = 0.3,
        gate_private_margin: float = 0.05,
    ) -> None:
        super().__init__()
        if soft_sid_table.shape != candidate_prior.shape:
            raise ValueError("Soft SID tokens and candidate priors must align.")
        self.register_buffer("soft_sid_table", soft_sid_table.long())
        self.register_buffer("candidate_prior", candidate_prior.float())
        self.register_buffer(
            "local_consistency", local_consistency.float().clamp(0.0, 1.0)
        )
        frequency = item_frequency.float().clamp_min(0.0)
        self.register_buffer("item_frequency", frequency)
        self.register_buffer("has_item_evidence", frequency.gt(0))
        shared_evidence = tail_tau * self.local_consistency.clamp_min(1e-6)
        alpha_prior = frequency / (frequency + shared_evidence)
        alpha_prior[0] = 0.0
        self.register_buffer("alpha_prior", alpha_prior.clamp(0.0, 1.0))

        prior_entropy = -(
            self.candidate_prior
            * torch.log(self.candidate_prior.clamp_min(1e-8))
        ).sum(dim=-1)
        prior_entropy = prior_entropy.mean(dim=-1) / math.log(
            max(self.candidate_prior.size(-1), 2)
        )
        active_ratio = self.candidate_prior.gt(0).float().mean(dim=(-1, -2))
        log_frequency = torch.log1p(frequency)
        mean = log_frequency[1:].mean()
        std = log_frequency[1:].std(unbiased=False).clamp_min(1e-6)
        gate_features = torch.stack(
            [
                (log_frequency - mean) / std,
                self.local_consistency,
                prior_entropy,
                active_ratio,
            ],
            dim=-1,
        )
        gate_features[0].zero_()
        self.register_buffer("gate_features", gate_features, persistent=False)

        self.semantic_basis_embedding = nn.Embedding(
            num_semantic_tokens + 1, dim, padding_idx=0
        )
        self.shared_residual_embedding = nn.Embedding(
            num_semantic_tokens + 1, dim, padding_idx=0
        )
        self.private_residual_embedding = nn.Embedding(
            num_items + 1, dim, padding_idx=0
        )
        self.basis_projection = nn.Linear(dim, dim)
        self.selector_embedding = nn.Embedding(
            num_semantic_tokens + 1, dim, padding_idx=0
        )
        self.selector_query = nn.Linear(dim, dim, bias=False)
        self.selector_key = nn.Linear(dim, dim, bias=False)
        self.prior_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)
        self.gate_correction_scale = float(gate_correction_scale)
        self.gate_private_margin = float(gate_private_margin)
        self.dim = dim
        self._reset_parameters()
        gate_hidden = max(8, dim // 8)
        self.residual_gate = nn.Sequential(
            nn.Linear(4, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2),
        )
        with torch.no_grad():
            self.residual_gate[-1].weight.zero_()
            self.residual_gate[-1].bias.zero_()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.selector_embedding.weight[0].zero_()
            self.semantic_basis_embedding.weight[0].zero_()
            self.shared_residual_embedding.weight[0].zero_()
            self.private_residual_embedding.weight[0].zero_()

    def candidate_weights(
        self, items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.soft_sid_table[items]
        prior = self.candidate_prior[items]
        candidate_mask = tokens.ne(0) & prior.gt(0)
        valid_slot = candidate_mask.any(dim=-1, keepdim=True)

        selector_vectors = self.selector_embedding(tokens)
        query_seed = (selector_vectors * prior.unsqueeze(-1)).sum(dim=-2)
        query = self.selector_query(query_seed).unsqueeze(-2)
        keys = self.selector_key(selector_vectors)
        beta = torch.nn.functional.softplus(self.prior_beta_raw)
        logits = (query * keys).sum(dim=-1) / math.sqrt(self.dim)
        logits = logits + beta * torch.log(prior.clamp_min(1e-8))
        logits = logits.masked_fill(~candidate_mask, -1e9)
        logits = torch.where(valid_slot, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1) * candidate_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        log_weights = torch.log(weights.clamp_min(1e-8))
        valid = valid_slot.squeeze(-1).float()
        denominator = valid.sum().clamp_min(1.0)
        entropy = (
            -(weights * log_weights * candidate_mask).sum(dim=-1) * valid
        ).sum() / denominator
        return weights, entropy

    def semantic_pool(
        self,
        embedding: nn.Embedding,
        items: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.soft_sid_table[items]
        candidate_pool = (embedding(tokens) * weights.unsqueeze(-1)).sum(dim=-2)
        slot_mask = weights.sum(dim=-1, keepdim=True).gt(0).float()
        return (candidate_pool * slot_mask).sum(dim=-2) / slot_mask.sum(
            dim=-2
        ).clamp_min(1.0)

    @staticmethod
    def _bernoulli_kl(value: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        value = value.clamp(1e-8, 1.0 - 1e-8)
        prior = prior.clamp(1e-8, 1.0 - 1e-8)
        return value * torch.log(value / prior) + (1.0 - value) * torch.log(
            (1.0 - value) / (1.0 - prior)
        )

    def fusion_weights(
        self, items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_item = items.ne(0)
        alpha_prior = self.alpha_prior[items]
        correction = self.gate_correction_scale * torch.tanh(
            self.residual_gate(self.gate_features[items])
        )
        gamma_prior = alpha_prior.new_full(
            alpha_prior.shape,
            self.residual_scale / (1.0 + self.residual_scale),
        ).clamp(1e-6, 1.0 - 1e-6)
        safe_alpha_prior = alpha_prior.clamp(1e-6, 1.0 - 1e-6)
        gamma = torch.sigmoid(torch.logit(gamma_prior) + correction[..., 0])
        alpha = torch.sigmoid(
            torch.logit(safe_alpha_prior) + correction[..., 1]
        )
        alpha = alpha * self.has_item_evidence[items]
        effective_scale = gamma / (1.0 - gamma).clamp_min(1e-6)
        weights = torch.stack(
            [
                torch.ones_like(gamma),
                effective_scale * (1.0 - alpha),
                effective_scale * alpha,
            ],
            dim=-1,
        ) * valid_item.unsqueeze(-1)

        gamma_kl = self._bernoulli_kl(gamma, gamma_prior)
        alpha_kl = self._bernoulli_kl(alpha, safe_alpha_prior)
        alpha_mask = valid_item & self.has_item_evidence[items]
        gate_kl = (
            (gamma_kl * valid_item).sum() + (alpha_kl * alpha_mask).sum()
        ) / (valid_item.sum() + alpha_mask.sum()).clamp_min(1)
        private_excess = torch.relu(
            alpha - alpha_prior - self.gate_private_margin
        )
        private_penalty = (
            private_excess.square()
            * (1.0 - alpha_prior).detach()
            * valid_item
        ).sum() / valid_item.sum().clamp_min(1)
        return weights, gate_kl.clamp_min(0.0), private_penalty

    def forward(self, items: torch.Tensor) -> Dict[str, torch.Tensor]:
        candidate_weights, attention_entropy = self.candidate_weights(items)
        basis = self.basis_projection(
            self.semantic_pool(
                self.semantic_basis_embedding, items, candidate_weights
            )
        )
        shared = self.semantic_pool(
            self.shared_residual_embedding, items, candidate_weights
        )
        private = self.private_residual_embedding(items)
        fusion_weights, gate_kl, private_penalty = self.fusion_weights(items)
        vectors = self.output_norm(
            fusion_weights[..., 0:1] * basis
            + fusion_weights[..., 1:2] * shared
            + fusion_weights[..., 2:3] * private
        )
        vectors = self.dropout(vectors) * items.ne(0).unsqueeze(-1)
        reduce_dims = tuple(range(fusion_weights.dim() - 1))
        valid = items.ne(0).float()
        denominator = valid.sum().clamp_min(1.0)
        gate_mean = (fusion_weights * valid.unsqueeze(-1)).sum(dim=reduce_dims)
        gate_mean = gate_mean / denominator
        return {
            "vectors": vectors,
            "attention_entropy": attention_entropy,
            "gate_kl": gate_kl,
            "private_penalty": private_penalty,
            "gate_mean": gate_mean,
        }

    @torch.no_grad()
    def statistics(self) -> Dict[str, float]:
        items = torch.arange(1, self.item_frequency.numel(), device=self.item_frequency.device)
        weights, gate_kl, private_penalty = self.fusion_weights(items)
        correction = self.gate_correction_scale * torch.tanh(
            self.residual_gate(self.gate_features[items])
        )
        gamma_prior = weights.new_full(
            items.shape,
            self.residual_scale / (1.0 + self.residual_scale),
        )
        gamma = torch.sigmoid(torch.logit(gamma_prior) + correction[..., 0])
        alpha = torch.sigmoid(
            torch.logit(self.alpha_prior[items].clamp(1e-6, 1.0 - 1e-6))
            + correction[..., 1]
        ) * self.has_item_evidence[items]
        return {
            "semantic_basis": float(weights[:, 0].mean().cpu()),
            "shared_residual": float(weights[:, 1].mean().cpu()),
            "private_residual": float(weights[:, 2].mean().cpu()),
            "residual_share_gamma": float(gamma.mean().cpu()),
            "shared_private_alpha": float(alpha.mean().cpu()),
            "effective_residual_scale": float((gamma / (1.0 - gamma)).mean().cpu()),
            "catalog_kl": float(gate_kl.cpu()),
            "catalog_private_penalty": float(private_penalty.cpu()),
        }


class LoCoRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_semantic_tokens: int,
        soft_sid_table: torch.Tensor,
        candidate_prior: torch.Tensor,
        local_consistency: torch.Tensor,
        item_frequency: torch.Tensor,
        dim: int = 128,
        max_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        tail_tau: float = 20.0,
        residual_scale: float = 1.0,
        gate_correction_scale: float = 0.3,
        gate_private_margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.sequence_encoder = CausalTransformerEncoder(
            dim=dim,
            max_len=max_len,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.item_encoder = LoCoRecItemEncoder(
            num_items=num_items,
            num_semantic_tokens=num_semantic_tokens,
            soft_sid_table=soft_sid_table,
            candidate_prior=candidate_prior,
            local_consistency=local_consistency,
            item_frequency=item_frequency,
            dim=dim,
            dropout=dropout,
            tail_tau=tail_tau,
            residual_scale=residual_scale,
            gate_correction_scale=gate_correction_scale,
            gate_private_margin=gate_private_margin,
        )

    def encode_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output = self.item_encoder(sequence)
        return self.sequence_encoder(sequence, output["vectors"]), output

    def score_candidates(
        self, user_vector: torch.Tensor, candidates: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output = self.item_encoder(candidates)
        scores = torch.einsum("bd,bcd->bc", user_vector, output["vectors"])
        return scores, output

    def forward(self, sequence: torch.Tensor, candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
        user_vector, sequence_output = self.encode_sequence(sequence)
        scores, candidate_output = self.score_candidates(user_vector, candidates)
        return {
            "score": scores,
            "attention_entropy": 0.5
            * (
                sequence_output["attention_entropy"]
                + candidate_output["attention_entropy"]
            ),
            "gate_kl": 0.5
            * (sequence_output["gate_kl"] + candidate_output["gate_kl"]),
            "private_penalty": 0.5
            * (
                sequence_output["private_penalty"]
                + candidate_output["private_penalty"]
            ),
            "gate_mean": 0.5
            * (sequence_output["gate_mean"] + candidate_output["gate_mean"]),
        }
