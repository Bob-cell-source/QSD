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
        alpha_mode: Literal["fixed", "learnable_monotonic"] = "fixed",
        fusion_mode: Literal[
            "fixed", "prior_guided_gate", "hierarchical_residual_gate"
        ] = "fixed",
        residual_scale: float = 1.0,
        gate_correction_scale: float = 1.0,
        gate_private_margin: float = 0.0,
        candidate_weight_mode: Literal[
            "fixed", "learned", "prior_guided", "neighborhood_learned"
        ] = "prior_guided",
        disable_semantic_basis: bool = False,
        disable_shared_residual: bool = False,
        disable_private_residual: bool = False,
    ) -> None:
        super().__init__()
        if alpha_mode not in {"fixed", "learnable_monotonic"}:
            raise ValueError(f"Unsupported alpha mode: {alpha_mode}")
        if fusion_mode not in {
            "fixed",
            "prior_guided_gate",
            "hierarchical_residual_gate",
        }:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive.")
        if candidate_weight_mode not in {
            "fixed",
            "learned",
            "prior_guided",
            "neighborhood_learned",
        }:
            raise ValueError(f"Unsupported candidate weight mode: {candidate_weight_mode}")
        if soft_sid_table.shape != soft_sid_weights.shape:
            raise ValueError("Soft SID token and weight tables must have identical shapes.")

        self.register_buffer("soft_sid_table", soft_sid_table.long())
        self.register_buffer("soft_sid_weights", soft_sid_weights.float())
        self.register_buffer("semantic_reliability", semantic_reliability.float().clamp(0.0, 1.0))

        raw_frequency = item_frequency.float().clamp_min(0.0)
        self.register_buffer("item_frequency", raw_frequency)
        self.register_buffer("has_item_evidence", raw_frequency.gt(0))
        calibrated_tau = float(tail_tau) * self.semantic_reliability.clamp_min(1e-6)
        private_weight = raw_frequency / (raw_frequency + calibrated_tau)
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
                beta = 1.0
                self.prior_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(beta))))
            else:
                self.register_parameter("prior_beta_raw", None)
        if fusion_mode != "fixed":
            prior = self.soft_sid_weights.clamp_min(0.0)
            prior_entropy = -(prior * torch.log(prior.clamp_min(1e-8))).sum(dim=-1)
            max_entropy = math.log(max(prior.size(-1), 2))
            prior_entropy = prior_entropy.mean(dim=-1) / max(max_entropy, 1e-8)
            active_ratio = prior.gt(0).float().mean(dim=(-1, -2))
            log_frequency = torch.log1p(raw_frequency)
            valid_frequency = log_frequency[1:]
            frequency_mean = valid_frequency.mean()
            frequency_std = valid_frequency.std(unbiased=False).clamp_min(1e-6)
            gate_features = torch.stack(
                [
                    (log_frequency - frequency_mean) / frequency_std,
                    self.semantic_reliability,
                    prior_entropy,
                    active_ratio,
                ],
                dim=-1,
            )
            gate_features[0].zero_()
            self.register_buffer("gate_features", gate_features, persistent=False)
            self.fusion_gate = None
        else:
            self.register_buffer("gate_features", torch.empty(0), persistent=False)
            self.fusion_gate = None
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.alpha_mode = alpha_mode
        self.fusion_mode = fusion_mode
        self.residual_scale = float(residual_scale)
        self.gate_correction_scale = float(gate_correction_scale)
        self.gate_private_margin = float(gate_private_margin)
        self.candidate_weight_mode = candidate_weight_mode
        self.disable_semantic_basis = disable_semantic_basis
        self.disable_shared_residual = disable_shared_residual
        self.disable_private_residual = disable_private_residual
        self.dim = dim
        self._reset_parameters()
        if fusion_mode != "fixed":
            # Build the additional module after initializing the common model
            # so fixed and gated variants share identical initial parameters.
            gate_hidden = max(8, dim // 8)
            self.fusion_gate = nn.Sequential(
                nn.Linear(4, gate_hidden),
                nn.GELU(),
                nn.Linear(
                    gate_hidden,
                    2 if fusion_mode == "hierarchical_residual_gate" else 3,
                ),
            )
            with torch.no_grad():
                self.fusion_gate[-1].weight.zero_()
                self.fusion_gate[-1].bias.zero_()

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

    def fusion_weights(
        self, items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = self.residual_alpha(items).squeeze(-1)
        valid_item = items.ne(0)

        if self.fusion_mode == "hierarchical_residual_gate":
            correction = self.gate_correction_scale * torch.tanh(
                self.fusion_gate(self.gate_features[items])
            )
            gamma_prior = alpha.new_full(
                alpha.shape,
                self.residual_scale / (1.0 + self.residual_scale),
            ).clamp(1e-6, 1.0 - 1e-6)
            alpha_prior = alpha.clamp(1e-6, 1.0 - 1e-6)
            gamma = torch.sigmoid(
                torch.logit(gamma_prior) + correction[..., 0]
            )
            learned_alpha = torch.sigmoid(
                torch.logit(alpha_prior) + correction[..., 1]
            )
            learned_alpha = learned_alpha * self.has_item_evidence[items]

            # gamma is a normalized residual share. Converting it to odds
            # preserves the original b + lambda*r parameterization exactly.
            effective_scale = gamma / (1.0 - gamma).clamp_min(1e-6)
            basis_weight = torch.ones_like(gamma)
            shared_weight = effective_scale * (1.0 - learned_alpha)
            private_weight = effective_scale * learned_alpha
            if self.disable_semantic_basis:
                basis_weight = torch.zeros_like(basis_weight)
            if self.disable_shared_residual:
                shared_weight = torch.zeros_like(shared_weight)
            if self.disable_private_residual:
                private_weight = torch.zeros_like(private_weight)
            weights = torch.stack(
                [basis_weight, shared_weight, private_weight], dim=-1
            ) * valid_item.unsqueeze(-1)

            def bernoulli_kl(value: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
                value = value.clamp(1e-8, 1.0 - 1e-8)
                prior = prior.clamp(1e-8, 1.0 - 1e-8)
                return value * torch.log(value / prior) + (1.0 - value) * torch.log(
                    (1.0 - value) / (1.0 - prior)
                )

            gamma_kl = bernoulli_kl(gamma, gamma_prior)
            alpha_kl = bernoulli_kl(learned_alpha, alpha_prior)
            alpha_mask = valid_item & self.has_item_evidence[items]
            gate_kl = (
                (gamma_kl * valid_item).sum()
                + (alpha_kl * alpha_mask).sum()
            ) / (valid_item.sum() + alpha_mask.sum()).clamp_min(1)

            tail_weight = (1.0 - alpha).detach()
            private_excess = torch.relu(
                learned_alpha - alpha - self.gate_private_margin
            )
            private_penalty = (
                private_excess.square() * tail_weight * valid_item
            ).sum() / valid_item.sum().clamp_min(1)
            return weights, gate_kl.clamp_min(0.0), private_penalty

        basis_prior = torch.ones_like(alpha)
        shared_prior = 1.0 - alpha
        private_prior = alpha
        prior = torch.stack([basis_prior, shared_prior, private_prior], dim=-1)

        component_mask = torch.ones_like(prior, dtype=torch.bool)
        if self.disable_semantic_basis:
            component_mask[..., 0] = False
        if self.disable_shared_residual:
            component_mask[..., 1] = False
        if self.disable_private_residual:
            component_mask[..., 2] = False
        component_mask[..., 2] &= self.has_item_evidence[items]
        component_mask &= valid_item.unsqueeze(-1)

        prior = prior * component_mask
        if self.fusion_mode == "fixed":
            zero = prior.new_tensor(0.0)
            return prior, zero, zero

        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        correction = self.gate_correction_scale * torch.tanh(
            self.fusion_gate(self.gate_features[items])
        )
        logits = torch.log(prior.clamp_min(1e-8)) + correction
        logits = logits.masked_fill(~component_mask, -1e9)
        logits = torch.where(valid_item.unsqueeze(-1), logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1) * component_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        log_weights = torch.log(weights.clamp_min(1e-8))
        log_prior = torch.log(prior.clamp_min(1e-8))
        item_kl = (weights * (log_weights - log_prior) * component_mask).sum(dim=-1)
        gate_kl = (item_kl * valid_item).sum() / valid_item.sum().clamp_min(1)

        # Low-frequency items have weak private-ID supervision. Penalize only
        # the amount by which the learned private weight exceeds its prior.
        tail_weight = (1.0 - alpha).detach()
        private_excess = torch.relu(
            weights[..., 2] - prior[..., 2] - self.gate_private_margin
        )
        private_penalty = (
            private_excess.square() * tail_weight * valid_item
        ).sum() / valid_item.sum().clamp_min(1)
        return weights, gate_kl.clamp_min(0.0), private_penalty

    def encode_with_aux(
        self, items: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        candidate_weights, attention_kl, attention_entropy = self.candidate_weights(items)
        zeros = items.new_zeros((*items.shape, self.dim), dtype=torch.float)
        basis = zeros if self.disable_semantic_basis else self.basis_projection(
            self.semantic_pool(self.semantic_basis_embedding, items, candidate_weights)
        )
        shared = zeros if self.disable_shared_residual else self.semantic_pool(
            self.shared_residual_embedding, items, candidate_weights
        )
        private = zeros if self.disable_private_residual else self.private_residual_embedding(items)

        fusion_weights, gate_kl, gate_private_penalty = self.fusion_weights(items)
        output = self.output_norm(
            fusion_weights[..., 0:1] * basis
            + fusion_weights[..., 1:2] * shared
            + fusion_weights[..., 2:3] * private
        )
        output = self.dropout(output) * items.ne(0).unsqueeze(-1)
        valid = items.ne(0).float()
        gate_mean = (fusion_weights * valid.unsqueeze(-1)).sum(
            dim=tuple(range(fusion_weights.ndim - 1))
        ) / valid.sum().clamp_min(1.0)
        return (
            output,
            attention_kl,
            attention_entropy,
            gate_kl,
            gate_private_penalty,
            gate_mean,
        )

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        return self.encode_with_aux(items)[0]

    @torch.no_grad()
    def gate_statistics(self) -> Dict[str, float] | None:
        if self.fusion_mode == "fixed":
            return None
        items = torch.arange(1, self.item_frequency.numel(), device=self.item_frequency.device)
        weights, gate_kl, private_penalty = self.fusion_weights(items)
        mean = weights.mean(dim=0)
        statistics = {
            "semantic_basis": float(mean[0].cpu()),
            "shared_residual": float(mean[1].cpu()),
            "private_residual": float(mean[2].cpu()),
            "catalog_kl": float(gate_kl.cpu()),
            "catalog_private_penalty": float(private_penalty.cpu()),
        }
        if self.fusion_mode == "hierarchical_residual_gate":
            correction = self.gate_correction_scale * torch.tanh(
                self.fusion_gate(self.gate_features[items])
            )
            alpha_prior = self.residual_alpha(items).squeeze(-1)
            gamma_prior_value = self.residual_scale / (1.0 + self.residual_scale)
            gamma_prior = alpha_prior.new_full(
                alpha_prior.shape, gamma_prior_value
            ).clamp(1e-6, 1.0 - 1e-6)
            gamma = torch.sigmoid(torch.logit(gamma_prior) + correction[..., 0])
            learned_alpha = torch.sigmoid(
                torch.logit(alpha_prior.clamp(1e-6, 1.0 - 1e-6))
                + correction[..., 1]
            ) * self.has_item_evidence[items]
            statistics.update(
                {
                    "residual_share_gamma": float(gamma.mean().cpu()),
                    "shared_private_alpha": float(learned_alpha.mean().cpu()),
                    "effective_residual_scale": float(
                        (gamma / (1.0 - gamma).clamp_min(1e-6)).mean().cpu()
                    ),
                }
            )
        return statistics

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
        alpha_mode: Literal["fixed", "learnable_monotonic"] = "fixed",
        fusion_mode: Literal[
            "fixed", "prior_guided_gate", "hierarchical_residual_gate"
        ] = "fixed",
        residual_scale: float = 1.0,
        gate_correction_scale: float = 1.0,
        gate_private_margin: float = 0.0,
        candidate_weight_mode: Literal[
            "fixed", "learned", "prior_guided", "neighborhood_learned"
        ] = "prior_guided",
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
            alpha_mode=alpha_mode,
            fusion_mode=fusion_mode,
            residual_scale=residual_scale,
            gate_correction_scale=gate_correction_scale,
            gate_private_margin=gate_private_margin,
            candidate_weight_mode=candidate_weight_mode,
            disable_semantic_basis=disable_semantic_basis,
            disable_shared_residual=disable_shared_residual,
            disable_private_residual=disable_private_residual,
        )
    def forward(self, sequence: torch.Tensor, candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
        (
            sequence_vectors,
            sequence_kl,
            sequence_entropy,
            sequence_gate_kl,
            sequence_private_penalty,
            sequence_gate_mean,
        ) = self.item_encoder.encode_with_aux(sequence)
        user_vector, _ = self.sequence_encoder(sequence, sequence_vectors)
        (
            candidate_vectors,
            candidate_kl,
            candidate_entropy,
            candidate_gate_kl,
            candidate_private_penalty,
            candidate_gate_mean,
        ) = self.item_encoder.encode_with_aux(candidates)
        scores = torch.einsum("bd,bcd->bc", user_vector, candidate_vectors)
        return {
            "score": scores,
            "attention_kl": 0.5 * (sequence_kl + candidate_kl),
            "attention_entropy": 0.5 * (sequence_entropy + candidate_entropy),
            "gate_kl": 0.5 * (sequence_gate_kl + candidate_gate_kl),
            "gate_private_penalty": 0.5
            * (sequence_private_penalty + candidate_private_penalty),
            "gate_mean": 0.5 * (sequence_gate_mean + candidate_gate_mean),
        }

    def full_catalog_forward(
        self, sequence: torch.Tensor, catalog_items: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        (
            sequence_vectors,
            sequence_kl,
            sequence_entropy,
            sequence_gate_kl,
            sequence_private_penalty,
            sequence_gate_mean,
        ) = self.item_encoder.encode_with_aux(sequence)
        user_vector, _ = self.sequence_encoder(sequence, sequence_vectors)
        (
            catalog_vectors,
            catalog_kl,
            catalog_entropy,
            catalog_gate_kl,
            catalog_private_penalty,
            catalog_gate_mean,
        ) = self.item_encoder.encode_with_aux(catalog_items)
        scores = user_vector @ catalog_vectors.transpose(0, 1)
        return {
            "score": scores,
            "attention_kl": 0.5 * (sequence_kl + catalog_kl),
            "attention_entropy": 0.5 * (sequence_entropy + catalog_entropy),
            "gate_kl": 0.5 * (sequence_gate_kl + catalog_gate_kl),
            "gate_private_penalty": 0.5
            * (sequence_private_penalty + catalog_private_penalty),
            "gate_mean": 0.5 * (sequence_gate_mean + catalog_gate_mean),
        }
