import math
from typing import Dict, Literal, Tuple

import torch
from torch import nn


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
        evidence_gate: Literal["none", "history_overlap", "reliability", "hub_reliability"] = "none",
        evidence_floor: float = 0.1,
        evidence_recency_weight: float = 0.0,
        evidence_hub_weight: float = 0.0,
        contrastive_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if interest_router not in {"semantic", "prefix"}:
            raise ValueError(f"Unsupported interest_router: {interest_router}")
        if evidence_gate not in {"none", "history_overlap", "reliability", "hub_reliability"}:
            raise ValueError(f"Unsupported evidence_gate: {evidence_gate}")
        self.encoder = SASRecEncoder(num_items, dim, max_len, num_heads, num_layers, dropout)
        self.register_buffer("semantic_id_table", semantic_id_table.long())
        if semantic_token_hubness is None:
            semantic_token_hubness = torch.zeros(num_semantic_tokens + 1, dtype=torch.float)
        if semantic_item_hubness is None:
            semantic_item_hubness = torch.zeros(num_items + 1, dtype=torch.float)
        self.register_buffer("semantic_token_hubness", semantic_token_hubness.float())
        self.register_buffer("semantic_item_hubness", semantic_item_hubness.float())
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

    def slot_evidence(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        cand_sid = self.semantic_id_table[candidates]  # [B, C, F]
        hist_sid = self.semantic_id_table[seq]  # [B, L, F]
        hist_mask = seq.ne(0).unsqueeze(-1)
        match = cand_sid.unsqueeze(2).eq(hist_sid.unsqueeze(1)) & hist_mask.unsqueeze(1)
        if self.evidence_gate == "history_overlap":
            evidence = match.any(dim=2).float()
            return evidence.clamp_min(self.evidence_floor)

        pos = torch.linspace(0.0, 1.0, seq.size(1), device=seq.device).view(1, 1, -1, 1)
        recency = 1.0 + self.evidence_recency_weight * pos
        weighted_count = (match.float() * recency).sum(dim=2)
        support = 1.0 - torch.exp(-weighted_count)
        evidence = self.evidence_floor + (1.0 - self.evidence_floor) * support
        if self.evidence_gate == "hub_reliability":
            token_hub = self.semantic_token_hubness[cand_sid]
            evidence = evidence * (1.0 - self.evidence_hub_weight * token_hub)
        return evidence.clamp_min(self.evidence_floor)

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
        attn = torch.softmax(attn_logits, dim=-1)
        if self.evidence_gate != "none" and seq is not None:
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
