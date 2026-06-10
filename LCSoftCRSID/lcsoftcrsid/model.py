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
        alpha_mode: Literal["fixed", "learnable_monotonic"] = "fixed",
        candidate_weight_mode: Literal["fixed", "learned", "prior_guided"] = "fixed",
        prior_beta_init: float = 1.0,
        disable_semantic_basis: bool = False,
        disable_shared_residual: bool = False,
        disable_private_residual: bool = False,
    ) -> None:
        super().__init__()
        if frequency_transform not in {"raw", "log"}:
            raise ValueError(f"Unsupported frequency transform: {frequency_transform}")
        if alpha_mode not in {"fixed", "learnable_monotonic"}:
            raise ValueError(f"Unsupported alpha mode: {alpha_mode}")
        if candidate_weight_mode not in {"fixed", "learned", "prior_guided"}:
            raise ValueError(f"Unsupported candidate weight mode: {candidate_weight_mode}")
        if soft_sid_table.shape != soft_sid_weights.shape:
            raise ValueError("Soft SID token and weight tables must have identical shapes.")

        self.register_buffer("soft_sid_table", soft_sid_table.long())
        self.register_buffer("soft_sid_weights", soft_sid_weights.float())
        self.register_buffer("semantic_reliability", semantic_reliability.float().clamp(0.0, 1.0))

        raw_frequency = item_frequency.float().clamp_min(0.0)
        frequency = raw_frequency
        if frequency_transform == "log":
            frequency = torch.log1p(frequency)
        self.register_buffer("item_frequency", frequency)
        self.register_buffer("has_item_evidence", raw_frequency.gt(0))
        calibrated_tau = float(tail_tau) * self.semantic_reliability.clamp_min(1e-6)
        private_weight = frequency / (frequency + calibrated_tau)
        private_weight[0] = 0.0
        self.register_buffer("private_weight", private_weight.clamp(0.0, 1.0))
        if alpha_mode == "learnable_monotonic":
            inverse_softplus_one = math.log(math.expm1(1.0))
            self.alpha_frequency_raw = nn.Parameter(torch.tensor(inverse_softplus_one))
            self.alpha_reliability_raw = nn.Parameter(torch.tensor(inverse_softplus_one))
            self.alpha_bias = nn.Parameter(torch.tensor(-math.log(max(float(tail_tau), 1e-8))))
        else:
            self.register_parameter("alpha_frequency_raw", None)
            self.register_parameter("alpha_reliability_raw", None)
            self.register_parameter("alpha_bias", None)

        self.semantic_basis_embedding = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.shared_residual_embedding = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.private_residual_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.basis_projection = nn.Linear(dim, dim)
        if candidate_weight_mode == "fixed":
            self.selector_embedding = None
            self.selector_query = None
            self.selector_key = None
            self.prior_beta_raw = None
        else:
            self.selector_embedding = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
            self.selector_query = nn.Linear(dim, dim, bias=False)
            self.selector_key = nn.Linear(dim, dim, bias=False)
            if candidate_weight_mode == "prior_guided":
                beta = max(float(prior_beta_init), 1e-6)
                self.prior_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(beta))))
            else:
                self.register_parameter("prior_beta_raw", None)
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)
        self.alpha_mode = alpha_mode
        self.candidate_weight_mode = candidate_weight_mode
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
            if self.selector_embedding is not None:
                self.selector_embedding.weight[0].zero_()

    def candidate_weights(self, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.soft_sid_table[items]
        prior = self.soft_sid_weights[items]
        candidate_mask = tokens.ne(0) & prior.gt(0)
        valid_slot = candidate_mask.any(dim=-1, keepdim=True)

        if self.candidate_weight_mode == "fixed":
            zero = prior.new_tensor(0.0)
            log_prior = torch.log(prior.clamp_min(1e-8))
            slot_entropy = -(prior * log_prior * candidate_mask).sum(dim=-1)
            valid = valid_slot.squeeze(-1).float()
            entropy = (slot_entropy * valid).sum() / valid.sum().clamp_min(1.0)
            return prior, zero, entropy

        selector_vectors = self.selector_embedding(tokens)
        query_seed = (selector_vectors * prior.unsqueeze(-1)).sum(dim=-2)
        query = self.selector_query(query_seed).unsqueeze(-2)
        keys = self.selector_key(selector_vectors)
        logits = (query * keys).sum(dim=-1) / math.sqrt(self.dim)
        if self.candidate_weight_mode == "prior_guided":
            beta = torch.nn.functional.softplus(self.prior_beta_raw)
            logits = logits + beta * torch.log(prior.clamp_min(1e-8))

        logits = logits.masked_fill(~candidate_mask, -1e9)
        logits = torch.where(valid_slot, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1) * candidate_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        log_weights = torch.log(weights.clamp_min(1e-8))
        log_prior = torch.log(prior.clamp_min(1e-8))
        slot_kl = (weights * (log_weights - log_prior) * candidate_mask).sum(dim=-1)
        slot_entropy = -(weights * log_weights * candidate_mask).sum(dim=-1)
        valid = valid_slot.squeeze(-1).float()
        denominator = valid.sum().clamp_min(1.0)
        kl = (slot_kl * valid).sum() / denominator
        entropy = (slot_entropy * valid).sum() / denominator
        return weights, kl.clamp_min(0.0), entropy

    def semantic_pool(
        self,
        embedding: nn.Embedding,
        items: torch.Tensor,
        candidate_weights: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.soft_sid_table[items]
        weights = candidate_weights.unsqueeze(-1)
        candidate_pool = (embedding(tokens) * weights).sum(dim=-2)
        slot_mask = candidate_weights.sum(dim=-1, keepdim=True).gt(0).float()
        return (candidate_pool * slot_mask).sum(dim=-2) / slot_mask.sum(dim=-2).clamp_min(1.0)

    def residual_alpha(self, items: torch.Tensor) -> torch.Tensor:
        if self.alpha_mode == "fixed":
            return self.private_weight[items].unsqueeze(-1)

        frequency_slope = torch.nn.functional.softplus(self.alpha_frequency_raw)
        reliability_slope = torch.nn.functional.softplus(self.alpha_reliability_raw)
        frequency = self.item_frequency[items]
        reliability = self.semantic_reliability[items]
        logits = (
            frequency_slope * torch.log(frequency.clamp_min(1e-8))
            - reliability_slope * torch.log(reliability.clamp_min(1e-8))
            + self.alpha_bias
        )
        alpha = torch.sigmoid(logits)
        alpha = alpha * self.has_item_evidence[items]
        return alpha.unsqueeze(-1)

    def encode_with_aux(self, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_weights, attention_kl, attention_entropy = self.candidate_weights(items)
        zeros = items.new_zeros((*items.shape, self.dim), dtype=torch.float)
        basis = zeros if self.disable_semantic_basis else self.basis_projection(
            self.semantic_pool(self.semantic_basis_embedding, items, candidate_weights)
        )
        shared = zeros if self.disable_shared_residual else self.semantic_pool(
            self.shared_residual_embedding, items, candidate_weights
        )
        private = zeros if self.disable_private_residual else self.private_residual_embedding(items)

        alpha = self.residual_alpha(items)
        residual = alpha * private + (1.0 - alpha) * shared
        output = self.output_norm(basis + self.residual_scale * residual)
        output = self.dropout(output) * items.ne(0).unsqueeze(-1)
        return output, attention_kl, attention_entropy

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        return self.encode_with_aux(items)[0]

    def prior_beta(self) -> float | None:
        if self.prior_beta_raw is None:
            return None
        return float(torch.nn.functional.softplus(self.prior_beta_raw).detach().cpu())

    def alpha_parameters(self) -> Dict[str, float] | None:
        if self.alpha_mode == "fixed":
            return None
        return {
            "frequency_slope": float(
                torch.nn.functional.softplus(self.alpha_frequency_raw).detach().cpu()
            ),
            "reliability_slope": float(
                torch.nn.functional.softplus(self.alpha_reliability_raw).detach().cpu()
            ),
            "bias": float(self.alpha_bias.detach().cpu()),
        }


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
        alpha_mode: Literal["fixed", "learnable_monotonic"] = "fixed",
        candidate_weight_mode: Literal["fixed", "learned", "prior_guided"] = "fixed",
        prior_beta_init: float = 1.0,
        disable_semantic_basis: bool = False,
        disable_shared_residual: bool = False,
        disable_private_residual: bool = False,
    ) -> None:
        super().__init__()
        # Construct the common sequence encoder first so its initialization is
        # identical across fixed and learnable candidate-weight variants.
        self.sequence_encoder = CausalTransformerEncoder(
            dim=dim,
            max_len=max_len,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
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
            alpha_mode=alpha_mode,
            candidate_weight_mode=candidate_weight_mode,
            prior_beta_init=prior_beta_init,
            disable_semantic_basis=disable_semantic_basis,
            disable_shared_residual=disable_shared_residual,
            disable_private_residual=disable_private_residual,
        )
    def forward(self, sequence: torch.Tensor, candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
        sequence_vectors, sequence_kl, sequence_entropy = self.item_encoder.encode_with_aux(sequence)
        user_vector, _ = self.sequence_encoder(sequence, sequence_vectors)
        candidate_vectors, candidate_kl, candidate_entropy = self.item_encoder.encode_with_aux(candidates)
        scores = torch.einsum("bd,bcd->bc", user_vector, candidate_vectors)
        return {
            "score": scores,
            "attention_kl": 0.5 * (sequence_kl + candidate_kl),
            "attention_entropy": 0.5 * (sequence_entropy + candidate_entropy),
        }
