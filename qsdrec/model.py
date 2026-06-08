import math
from typing import Dict, Literal, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(-1, -2)
        y = self.conv1(y)
        y = self.dropout1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.dropout2(y)
        return y.transpose(-1, -2)


class SASRecEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        dim: int,
        max_len: int,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.dim = dim
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layernorms.append(nn.LayerNorm(dim, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)
            )
            self.forward_layernorms.append(nn.LayerNorm(dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(dim, dropout))
        self.norm = nn.LayerNorm(dim, eps=1e-8)
        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)
            self.pos_emb.weight[0].fill_(0)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = seq.shape
        pos = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        pos = pos * seq.ne(0).long()
        x = self.item_emb(seq) * math.sqrt(self.dim)
        x = x + self.pos_emb(pos)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        h = x
        for attn_norm, attn, ffn_norm, ffn in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.forward_layernorms,
            self.forward_layers,
        ):
            h_t = h.transpose(0, 1)
            q = attn_norm(h_t)
            attn_out, _ = attn(q, q, q, attn_mask=causal_mask)
            h = (h_t + attn_out).transpose(0, 1)
            h = h + ffn(ffn_norm(h))
            h = h * seq.ne(0).unsqueeze(-1)
        h = self.norm(h)
        # Sequences are left padded, so the newest non-padding item is at the
        # final position for every non-empty training/eval sample.
        last = h[:, -1]
        return last, h


class SASRecDynamicEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        max_len: int,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len + 1, dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.dim = dim
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layernorms.append(nn.LayerNorm(dim, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)
            )
            self.forward_layernorms.append(nn.LayerNorm(dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(dim, dropout))
        self.norm = nn.LayerNorm(dim, eps=1e-8)
        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.pos_emb.weight[0].fill_(0)

    def forward(self, seq: torch.Tensor, item_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = seq.shape
        pos = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        pos = pos * seq.ne(0).long()
        x = item_repr * math.sqrt(self.dim)
        x = x + self.pos_emb(pos)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        h = x
        for attn_norm, attn, ffn_norm, ffn in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.forward_layernorms,
            self.forward_layers,
        ):
            h_t = h.transpose(0, 1)
            q = attn_norm(h_t)
            attn_out, _ = attn(q, q, q, attn_mask=causal_mask)
            h = (h_t + attn_out).transpose(0, 1)
            h = h + ffn(ffn_norm(h))
            h = h * seq.ne(0).unsqueeze(-1)
        h = self.norm(h)
        return h[:, -1], h


class GRU4RecDynamicEncoder(nn.Module):
    """GRU4Rec encoder that consumes externally constructed item representations."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-8)
        self.dim = dim
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, seq: torch.Tensor, item_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = seq.shape
        lengths = seq.ne(0).sum(dim=1).clamp_min(1)

        # Dataset sequences are left padded. GRU packing expects valid tokens
        # first, so move each non-padding suffix to the beginning.
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        source = length - lengths.unsqueeze(1) + positions
        valid = positions < lengths.unsqueeze(1)
        source = source.clamp(max=length - 1)
        gather_index = source.unsqueeze(-1).expand(-1, -1, self.dim)
        right_padded = item_repr.gather(1, gather_index)
        right_padded = self.input_dropout(right_padded) * valid.unsqueeze(-1)

        packed = pack_padded_sequence(
            right_padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=length,
        )
        output = self.norm(output) * valid.unsqueeze(-1)
        last = self.norm(hidden[-1])
        return last, output


class GRU4Rec(nn.Module):
    """ID-only GRU4Rec baseline with tied input and candidate embeddings."""

    def __init__(
        self,
        num_items: int,
        dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.encoder = GRU4RecDynamicEncoder(dim, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)

    def item_representation(self, items: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.item_emb(items)) * items.ne(0).unsqueeze(-1)

    def forward(self, seq: torch.Tensor, candidates: torch.Tensor, sem_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        del sem_weight
        seq_repr = self.item_representation(seq)
        user_repr, _ = self.encoder(seq, seq_repr)
        cand_repr = self.item_representation(candidates)
        score = torch.einsum("bd,bcd->bc", user_repr, cand_repr)
        return {
            "score": score,
            "id_score": score,
            "sem_score": torch.zeros_like(score),
            "amateur_sem_score": torch.zeros_like(score),
            "hub_loss": score.new_tensor(0.0),
            "residual_l2": score.new_tensor(0.0),
        }

    @staticmethod
    def diversity_loss(queries: torch.Tensor) -> torch.Tensor:
        return queries.new_tensor(0.0)


class CRSIDRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_semantic_tokens: int,
        semantic_id_table: torch.Tensor,
        item_frequency: torch.Tensor,
        soft_semantic_id_table: torch.Tensor | None = None,
        soft_semantic_id_weight: torch.Tensor | None = None,
        semantic_reliability: torch.Tensor | None = None,
        dim: int = 64,
        max_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        tail_tau: float = 20.0,
        residual_scale: float = 1.0,
        alpha_mode: Literal["item_frequency", "semantic_hubness"] = "item_frequency",
        semantic_token_hubness: torch.Tensor | None = None,
        hub_alpha_floor: float = 0.05,
        hub_alpha_gamma: float = 1.0,
        disable_semantic_basis: bool = False,
        disable_shared_residual: bool = False,
        disable_private_residual: bool = False,
        alpha_override: float | None = None,
        alpha_frequency_transform: Literal["raw", "log"] = "raw",
        encoder_type: Literal["sasrec", "gru4rec"] = "sasrec",
    ) -> None:
        super().__init__()
        if alpha_mode not in {"item_frequency", "semantic_hubness"}:
            raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")
        if alpha_frequency_transform not in {"raw", "log"}:
            raise ValueError(f"Unsupported alpha_frequency_transform: {alpha_frequency_transform}")
        if encoder_type == "sasrec":
            self.encoder = SASRecDynamicEncoder(dim, max_len, num_heads, num_layers, dropout)
        elif encoder_type == "gru4rec":
            self.encoder = GRU4RecDynamicEncoder(dim, num_layers, dropout)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.register_buffer("semantic_id_table", semantic_id_table.long())
        if soft_semantic_id_table is not None and soft_semantic_id_weight is not None:
            if soft_semantic_id_table.shape != soft_semantic_id_weight.shape:
                raise ValueError("soft_semantic_id_table and soft_semantic_id_weight must have the same shape.")
            self.register_buffer("soft_semantic_id_table", soft_semantic_id_table.long())
            self.register_buffer("soft_semantic_id_weight", soft_semantic_id_weight.float())
            self.use_soft_semantic_ids = True
        else:
            self.register_buffer("soft_semantic_id_table", torch.zeros(1, 1, 1, dtype=torch.long))
            self.register_buffer("soft_semantic_id_weight", torch.zeros(1, 1, 1, dtype=torch.float))
            self.use_soft_semantic_ids = False
        if semantic_reliability is None:
            semantic_reliability = torch.ones_like(item_frequency, dtype=torch.float)
            semantic_reliability[0] = 0.0
        self.register_buffer("semantic_reliability", semantic_reliability.float().clamp(0.0, 1.0))
        if semantic_token_hubness is None:
            semantic_token_hubness = torch.zeros(num_semantic_tokens + 1, dtype=torch.float)
        self.register_buffer("semantic_token_hubness", semantic_token_hubness.float())
        freq = item_frequency.float().clamp_min(0.0)
        if alpha_frequency_transform == "log":
            freq = torch.log1p(freq)
        tau = float(tail_tau) * self.semantic_reliability.clamp_min(1e-6)
        alpha = freq / (freq + tau)
        alpha[0] = 0.0
        self.register_buffer("item_residual_alpha", alpha.clamp(0.0, 1.0))
        self.semantic_basis_emb = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.semantic_residual_emb = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.item_residual_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.basis_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = residual_scale
        self.alpha_mode = alpha_mode
        self.hub_alpha_floor = hub_alpha_floor
        self.hub_alpha_gamma = hub_alpha_gamma
        self.disable_semantic_basis = disable_semantic_basis
        self.disable_shared_residual = disable_shared_residual
        self.disable_private_residual = disable_private_residual
        self.alpha_override = alpha_override
        self.alpha_frequency_transform = alpha_frequency_transform
        self.encoder_type = encoder_type
        self.dim = dim
        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.semantic_basis_emb.weight[0].fill_(0)
            self.semantic_residual_emb.weight[0].fill_(0)
            self.item_residual_emb.weight[0].fill_(0)

    def semantic_pool(self, token_emb: nn.Embedding, items: torch.Tensor) -> torch.Tensor:
        if self.use_soft_semantic_ids:
            sid = self.soft_semantic_id_table[items]
            weight = self.soft_semantic_id_weight[items].unsqueeze(-1)
            tok = token_emb(sid)
            pooled = (tok * weight).sum(dim=-2)
            slot_weight = self.soft_semantic_id_weight[items].sum(dim=-1, keepdim=True)
            slot_mask = slot_weight.gt(0).float()
            return (pooled * slot_mask).sum(dim=-2) / slot_mask.sum(dim=-2).clamp_min(1.0)

        sid = self.semantic_id_table[items]
        tok = token_emb(sid)
        mask = sid.ne(0).float().unsqueeze(-1)
        return (tok * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)

    def residual_alpha(self, items: torch.Tensor) -> torch.Tensor:
        if self.alpha_override is not None:
            alpha = items.new_full(items.shape, float(self.alpha_override), dtype=torch.float)
            return alpha.clamp(0.0, 1.0).unsqueeze(-1)

        if self.alpha_mode == "item_frequency":
            return self.item_residual_alpha[items].unsqueeze(-1)

        sid = self.semantic_id_table[items]
        mask = sid.ne(0).float()
        token_hub = self.semantic_token_hubness[sid]
        hub = (token_hub * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        if self.hub_alpha_gamma != 1.0:
            hub = hub.clamp(0.0, 1.0).pow(self.hub_alpha_gamma)
        alpha = self.hub_alpha_floor + (1.0 - self.hub_alpha_floor) * hub
        return alpha.clamp(0.0, 1.0).unsqueeze(-1)

    def item_representation(self, items: torch.Tensor) -> torch.Tensor:
        if self.disable_semantic_basis:
            basis = items.new_zeros((*items.shape, self.dim), dtype=torch.float)
        else:
            basis = self.basis_proj(self.semantic_pool(self.semantic_basis_emb, items))

        if self.disable_shared_residual:
            shared_residual = items.new_zeros((*items.shape, self.dim), dtype=torch.float)
        else:
            shared_residual = self.semantic_pool(self.semantic_residual_emb, items)

        if self.disable_private_residual:
            private_residual = items.new_zeros((*items.shape, self.dim), dtype=torch.float)
        else:
            private_residual = self.item_residual_emb(items)

        alpha = self.residual_alpha(items)
        residual = alpha * private_residual + (1.0 - alpha) * shared_residual
        out = self.out_norm(basis + self.residual_scale * residual)
        return self.dropout(out) * items.ne(0).unsqueeze(-1)

    def forward(self, seq: torch.Tensor, candidates: torch.Tensor, sem_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        seq_repr = self.item_representation(seq)
        h_id, _ = self.encoder(seq, seq_repr)
        cand_repr = self.item_representation(candidates)
        score = torch.einsum("bd,bcd->bc", h_id, cand_repr)
        residual_l2 = self.item_residual_emb(candidates).pow(2).mean()
        return {
            "score": score,
            "id_score": score,
            "sem_score": torch.zeros_like(score),
            "amateur_sem_score": torch.zeros_like(score),
            "hub_loss": score.new_tensor(0.0),
            "residual_l2": residual_l2,
        }

    @staticmethod
    def diversity_loss(queries: torch.Tensor) -> torch.Tensor:
        return queries.new_tensor(0.0)


class QSDRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_semantic_tokens: int,
        semantic_id_table: torch.Tensor,
        dim: int = 64,
        max_len: int = 50,
        num_interests: int = 4,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        interest_router: Literal["semantic", "prefix"] = "semantic",
        prefix_level: int = 2,
        semantic_token_hubness: torch.Tensor | None = None,
        semantic_item_hubness: torch.Tensor | None = None,
        hub_score_weight: float = 0.0,
        hub_attn_weight: float = 0.0,
        evidence_gate: Literal[
            "none",
            "history_overlap",
            "reliability",
            "hub_reliability",
            "strength",
            "strength_idf",
            "cross_strength_idf",
            "learnable",
            "prior_lift",
            "mini_lift",
        ] = "none",
        evidence_floor: float = 0.1,
        evidence_recency_weight: float = 0.0,
        evidence_hub_weight: float = 0.0,
        evidence_cross_weight: float = 0.2,
        semantic_token_log_prior: torch.Tensor | None = None,
        mini_cluster_table: torch.Tensor | None = None,
        mini_cluster_log_prior: torch.Tensor | None = None,
        prior_lift_alpha: float = 0.1,
        prior_lift_tau: float = 1.0,
        prior_lift_eta: float = 1.0,
        hub_penalty_weight: float = 0.0,
        semantic_fusion: Literal["fixed", "evidence_coverage"] = "fixed",
        fusion_floor: float = 0.0,
        contrastive_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if interest_router not in {"semantic", "prefix"}:
            raise ValueError(f"Unsupported interest_router: {interest_router}")
        if evidence_gate not in {
            "none",
            "history_overlap",
            "reliability",
            "hub_reliability",
            "strength",
            "strength_idf",
            "cross_strength_idf",
            "learnable",
            "prior_lift",
            "mini_lift",
        }:
            raise ValueError(f"Unsupported evidence_gate: {evidence_gate}")
        if semantic_fusion not in {"fixed", "evidence_coverage"}:
            raise ValueError(f"Unsupported semantic_fusion: {semantic_fusion}")
        self.encoder = SASRecEncoder(num_items, dim, max_len, num_heads, num_layers, dropout)
        self.register_buffer("semantic_id_table", semantic_id_table.long())
        if semantic_token_hubness is None:
            semantic_token_hubness = torch.zeros(num_semantic_tokens + 1, dtype=torch.float)
        if semantic_item_hubness is None:
            semantic_item_hubness = torch.zeros(num_items + 1, dtype=torch.float)
        if semantic_token_log_prior is None:
            semantic_token_log_prior = torch.zeros(num_semantic_tokens + 1, dtype=torch.float)
        if mini_cluster_table is None:
            mini_cluster_table = semantic_id_table.long()
        if mini_cluster_log_prior is None:
            mini_cluster_log_prior = semantic_token_log_prior.float()
        self.register_buffer("semantic_token_hubness", semantic_token_hubness.float())
        self.register_buffer("semantic_item_hubness", semantic_item_hubness.float())
        self.register_buffer("semantic_token_log_prior", semantic_token_log_prior.float())
        self.register_buffer("mini_cluster_table", mini_cluster_table.long())
        self.register_buffer("mini_cluster_log_prior", mini_cluster_log_prior.float())
        self.semantic_emb = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
        self.query_proto = nn.Parameter(torch.randn(num_interests, dim) * 0.02)
        self.id_to_query = nn.Linear(dim, dim)
        self.query_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.beta = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.sem_proj = nn.Linear(dim, dim)
        self.prefix_router = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.prefix_proj = nn.Linear(dim, dim)
        self.evidence_mlp = nn.Sequential(
            nn.Linear(4, max(8, dim // 8)),
            nn.GELU(),
            nn.Linear(max(8, dim // 8), 1),
        )
        self.num_interests = num_interests
        self.dim = dim
        self.interest_router = interest_router
        self.prefix_level = prefix_level
        self.hub_score_weight = hub_score_weight
        self.hub_attn_weight = hub_attn_weight
        self.evidence_gate = evidence_gate
        self.evidence_floor = evidence_floor
        self.evidence_recency_weight = evidence_recency_weight
        self.evidence_hub_weight = evidence_hub_weight
        self.evidence_cross_weight = evidence_cross_weight
        self.prior_lift_alpha = prior_lift_alpha
        self.prior_lift_tau = prior_lift_tau
        self.prior_lift_eta = prior_lift_eta
        self.hub_penalty_weight = hub_penalty_weight
        self.semantic_fusion = semantic_fusion
        self.fusion_floor = fusion_floor
        self.contrastive_alpha = contrastive_alpha

    def sequence_semantic_memory(self, seq: torch.Tensor) -> torch.Tensor:
        sid = self.semantic_id_table[seq]  # [B, L, F]
        tok = self.semantic_emb(sid)  # [B, L, F, D]
        mask = sid.ne(0).float().unsqueeze(-1)
        return (tok * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

    def sequence_semantic_profile(self, seq: torch.Tensor) -> torch.Tensor:
        memory = self.sequence_semantic_memory(seq)
        mask = seq.ne(0).float().unsqueeze(-1)
        return (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def user_queries(self, seq: torch.Tensor, h_id: torch.Tensor) -> torch.Tensor:
        batch = seq.size(0)
        q = self.query_proto.unsqueeze(0).expand(batch, -1, -1)
        q = q + self.id_to_query(h_id).unsqueeze(1)
        memory = self.sequence_semantic_memory(seq)
        memory = torch.cat([memory, h_id.unsqueeze(1)], dim=1)
        key_padding = torch.cat(
            [seq.eq(0), torch.zeros(batch, 1, dtype=torch.bool, device=seq.device)],
            dim=1,
        )
        q, _ = self.query_attn(q, memory, memory, key_padding_mask=key_padding)
        return q

    def slot_evidence_features(self, seq: torch.Tensor, candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
        cand_sid = self.semantic_id_table[candidates]  # [B, C, F]
        hist_sid = self.semantic_id_table[seq]  # [B, L, F]
        hist_mask = seq.ne(0).unsqueeze(-1)
        match = cand_sid.unsqueeze(2).eq(hist_sid.unsqueeze(1)) & hist_mask.unsqueeze(1)
        pos = torch.linspace(0.0, 1.0, seq.size(1), device=seq.device).view(1, 1, -1, 1)
        if self.evidence_recency_weight > 0:
            recency = torch.exp(-self.evidence_recency_weight * (1.0 - pos))
        else:
            recency = torch.ones_like(pos)
        same_count = (match.float() * recency).sum(dim=2)
        same_strength = 1.0 - torch.exp(-same_count)
        latest_support = (match.float() * pos).amax(dim=2)
        specificity = (1.0 - self.semantic_token_hubness[cand_sid]).clamp(0.0, 1.0)

        cross_strength = torch.zeros_like(same_strength)
        if self.evidence_gate in {"cross_strength_idf", "learnable"}:
            cand_by_slot = cand_sid.unsqueeze(3).unsqueeze(4)  # [B, C, F, 1, 1]
            hist_by_slot = hist_sid.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L, F]
            cross_match = cand_by_slot.eq(hist_by_slot)
            cross_match = cross_match & hist_mask.unsqueeze(1).unsqueeze(1)
            slots = cand_sid.size(-1)
            slot_eye = torch.eye(slots, dtype=torch.bool, device=seq.device).view(1, 1, slots, 1, slots)
            cross_match = cross_match & ~slot_eye
            cross_pos = pos.unsqueeze(2)
            cross_count = (cross_match.float() * cross_pos.new_ones(cross_pos.shape) * recency.unsqueeze(2)).sum(dim=(3, 4))
            cross_strength = 1.0 - torch.exp(-cross_count)

        return {
            "same_binary": match.any(dim=2).float(),
            "same_strength": same_strength,
            "cross_strength": cross_strength,
            "latest_support": latest_support,
            "specificity": specificity,
        }

    def slot_evidence(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        features = self.slot_evidence_features(seq, candidates)
        if self.evidence_gate == "history_overlap":
            return features["same_binary"].clamp_min(self.evidence_floor)

        support = features["same_strength"]
        if self.evidence_gate in {"strength_idf", "hub_reliability"}:
            support = support * features["specificity"]
        elif self.evidence_gate == "cross_strength_idf":
            support = (support + self.evidence_cross_weight * features["cross_strength"]).clamp_max(1.0)
            support = support * features["specificity"]
        elif self.evidence_gate == "learnable":
            mlp_input = torch.stack(
                [
                    features["same_strength"],
                    features["cross_strength"],
                    features["specificity"],
                    features["latest_support"],
                ],
                dim=-1,
            )
            support = torch.sigmoid(self.evidence_mlp(mlp_input).squeeze(-1))

        evidence = self.evidence_floor + (1.0 - self.evidence_floor) * support
        if self.evidence_gate == "hub_reliability":
            cand_sid = self.semantic_id_table[candidates]
            token_hub = self.semantic_token_hubness[cand_sid]
            evidence = evidence * (1.0 - self.evidence_hub_weight * token_hub)
        return evidence.clamp_min(self.evidence_floor)

    def evidence_coverage(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        features = self.slot_evidence_features(seq, candidates)
        return features["same_binary"].mean(dim=-1)

    def prior_lift(self, seq: torch.Tensor, candidates: torch.Tensor, use_mini: bool = False) -> torch.Tensor:
        if use_mini:
            cand_key = self.mini_cluster_table[candidates]
            hist_key = self.mini_cluster_table[seq]
            log_prior_table = self.mini_cluster_log_prior
        else:
            cand_key = self.semantic_id_table[candidates]
            hist_key = self.semantic_id_table[seq]
            log_prior_table = self.semantic_token_log_prior

        hist_mask = seq.ne(0).unsqueeze(-1)
        match = cand_key.unsqueeze(2).eq(hist_key.unsqueeze(1)) & hist_mask.unsqueeze(1)
        pos = torch.linspace(0.0, 1.0, seq.size(1), device=seq.device).view(1, 1, -1, 1)
        if self.evidence_recency_weight > 0:
            recency = torch.exp(-self.evidence_recency_weight * (1.0 - pos))
        else:
            recency = torch.ones_like(pos)

        counts = (match.float() * recency).sum(dim=2)
        hist_total = (hist_mask.float() * recency.squeeze(1)).sum(dim=(1, 2)).view(-1, 1, 1)
        num_keys = max(int(log_prior_table.numel() - 1), 1)
        log_user_prob = torch.log(counts + self.prior_lift_alpha)
        log_user_prob = log_user_prob - torch.log(hist_total + self.prior_lift_alpha * num_keys)
        log_global_prob = log_prior_table[cand_key]
        lift = log_user_prob - self.prior_lift_tau * log_global_prob
        return lift.clamp(min=-8.0, max=8.0)

    def amateur_semantic_score(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        user_sem = self.sequence_semantic_profile(seq)
        sid = self.semantic_id_table[candidates]
        tok = self.sem_proj(self.semantic_emb(sid))
        mask = sid.ne(0).float().unsqueeze(-1)
        cand_sem = (tok * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return torch.einsum("bd,bcd->bc", user_sem, cand_sem)

    def semantic_score(self, queries: torch.Tensor, candidates: torch.Tensor, seq: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sid = self.semantic_id_table[candidates]  # [B, C, F]
        tok = self.semantic_emb(sid)  # [B, C, F, D]
        tok = self.sem_proj(tok)
        q = queries.unsqueeze(1)  # [B, 1, K, D]
        attn_logits = torch.einsum("bckd,bcfd->bckf", q.expand(-1, candidates.size(1), -1, -1), tok)
        attn_logits = attn_logits / math.sqrt(self.dim)
        token_hub = self.semantic_token_hubness[sid].unsqueeze(2)  # [B, C, 1, F]
        if self.hub_attn_weight > 0:
            attn_logits = attn_logits - self.hub_attn_weight * token_hub
        if self.evidence_gate == "prior_lift" and seq is not None:
            lift = self.prior_lift(seq, candidates, use_mini=False).unsqueeze(2)
            attn_logits = attn_logits + self.prior_lift_eta * lift
        elif self.evidence_gate == "mini_lift" and seq is not None:
            lift = self.prior_lift(seq, candidates, use_mini=True).unsqueeze(2)
            attn_logits = attn_logits + self.prior_lift_eta * lift
        attn = torch.softmax(attn_logits, dim=-1)
        if self.evidence_gate not in {"none", "prior_lift", "mini_lift"} and seq is not None:
            evidence = self.slot_evidence(seq, candidates).unsqueeze(2)  # [B, C, 1, F]
            attn = attn * evidence
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            evidence = torch.ones_like(attn)
        resp = torch.einsum("bckf,bcfd->bckd", attn, tok)
        q_exp = q.expand(-1, candidates.size(1), -1, -1)
        if self.interest_router == "prefix":
            prefix_sid = sid[:, :, : self.prefix_level]
            prefix_tok = self.semantic_emb(prefix_sid)
            prefix_mask = prefix_sid.ne(0).float().unsqueeze(-1)
            prefix = (prefix_tok * prefix_mask).sum(dim=2) / prefix_mask.sum(dim=2).clamp_min(1.0)
            prefix = self.prefix_proj(prefix).unsqueeze(2).expand(-1, -1, queries.size(1), -1)
            beta_in = torch.cat([q_exp, prefix, q_exp * prefix], dim=-1)
            beta = torch.softmax(self.prefix_router(beta_in).squeeze(-1), dim=-1)
        else:
            beta_in = torch.cat([q_exp, resp, q_exp * resp], dim=-1)
            beta = torch.softmax(self.beta(beta_in).squeeze(-1), dim=-1)
        dots = (q_exp * resp).sum(dim=-1)
        score = (beta * dots).sum(dim=-1)
        if self.hub_penalty_weight > 0:
            token_hub_by_query = (attn * token_hub).sum(dim=-1)
            hub_penalty = (beta * token_hub_by_query).sum(dim=-1)
            score = score - self.hub_penalty_weight * hub_penalty
        if self.hub_score_weight > 0:
            score = score - self.hub_score_weight * self.semantic_item_hubness[candidates]
        hub_loss = (attn * token_hub).sum(dim=-1).mean()
        return score, {
            "attn": attn,
            "beta": beta,
            "evidence": evidence,
            "hub_loss": hub_loss,
        }

    def forward(self, seq: torch.Tensor, candidates: torch.Tensor, sem_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        h_id, _ = self.encoder(seq)
        cand_emb = self.encoder.item_emb(candidates)
        id_score = torch.einsum("bd,bcd->bc", h_id, cand_emb)
        queries = self.user_queries(seq, h_id)
        sem_score, sem_aux = self.semantic_score(queries, candidates, seq=seq)
        amateur_score = None
        if self.contrastive_alpha > 0:
            amateur_score = self.amateur_semantic_score(seq, candidates)
            sem_score = sem_score - self.contrastive_alpha * amateur_score
        if self.semantic_fusion == "evidence_coverage":
            coverage = self.evidence_coverage(seq, candidates)
            sem_scale = sem_weight * (self.fusion_floor + (1.0 - self.fusion_floor) * coverage)
            score = id_score + sem_scale * sem_score
        else:
            score = id_score + sem_weight * sem_score
        return {
            "score": score,
            "id_score": id_score,
            "sem_score": sem_score,
            "amateur_sem_score": amateur_score if amateur_score is not None else torch.zeros_like(sem_score),
            "hub_loss": sem_aux["hub_loss"],
            "queries": queries,
        }

    @staticmethod
    def diversity_loss(queries: torch.Tensor) -> torch.Tensor:
        q = torch.nn.functional.normalize(queries, dim=-1)
        sim = torch.matmul(q, q.transpose(1, 2)).abs()
        eye = torch.eye(sim.size(-1), device=sim.device).unsqueeze(0)
        return (sim * (1.0 - eye)).mean()
