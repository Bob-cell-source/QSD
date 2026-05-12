import math
from typing import Dict, Tuple

import torch
from torch import nn


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
        self.pos_emb = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        pad_mask = seq.eq(0)
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
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
    ) -> None:
        super().__init__()
        self.encoder = SASRecEncoder(num_items, dim, max_len, num_heads, num_layers, dropout)
        self.register_buffer("semantic_id_table", semantic_id_table.long())
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
        self.num_interests = num_interests
        self.dim = dim

    def sequence_semantic_memory(self, seq: torch.Tensor) -> torch.Tensor:
        sid = self.semantic_id_table[seq]  # [B, L, F]
        tok = self.semantic_emb(sid)  # [B, L, F, D]
        mask = sid.ne(0).float().unsqueeze(-1)
        return (tok * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

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

    def semantic_score(self, queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        sid = self.semantic_id_table[candidates]  # [B, C, F]
        tok = self.semantic_emb(sid)  # [B, C, F, D]
        tok = self.sem_proj(tok)
        q = queries.unsqueeze(1)  # [B, 1, K, D]
        attn_logits = torch.einsum("bckd,bcfd->bckf", q.expand(-1, candidates.size(1), -1, -1), tok)
        attn_logits = attn_logits / math.sqrt(self.dim)
        attn = torch.softmax(attn_logits, dim=-1)
        resp = torch.einsum("bckf,bcfd->bckd", attn, tok)
        q_exp = q.expand(-1, candidates.size(1), -1, -1)
        beta_in = torch.cat([q_exp, resp, q_exp * resp], dim=-1)
        beta = torch.softmax(self.beta(beta_in).squeeze(-1), dim=-1)
        dots = (q_exp * resp).sum(dim=-1)
        return (beta * dots).sum(dim=-1)

    def forward(self, seq: torch.Tensor, candidates: torch.Tensor, sem_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        h_id, _ = self.encoder(seq)
        cand_emb = self.encoder.item_emb(candidates)
        id_score = torch.einsum("bd,bcd->bc", h_id, cand_emb)
        queries = self.user_queries(seq, h_id)
        sem_score = self.semantic_score(queries, candidates)
        score = id_score + sem_weight * sem_score
        return {
            "score": score,
            "id_score": id_score,
            "sem_score": sem_score,
            "queries": queries,
        }

    @staticmethod
    def diversity_loss(queries: torch.Tensor) -> torch.Tensor:
        q = torch.nn.functional.normalize(queries, dim=-1)
        sim = torch.matmul(q, q.transpose(1, 2)).abs()
        eye = torch.eye(sim.size(-1), device=sim.device).unsqueeze(0)
        return (sim * (1.0 - eye)).mean()
