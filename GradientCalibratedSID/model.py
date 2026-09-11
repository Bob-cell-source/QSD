import math
import torch
from torch import nn

from LoCoRec.locorec.model import CausalTransformerEncoder


class SIDItemEncoder(nn.Module):
    def __init__(self, hard_ids, num_tokens, dim):
        super().__init__()
        self.register_buffer('hard_sid_table', hard_ids.clone())
        self.register_buffer('candidate_ids', hard_ids.unsqueeze(-1).clone())
        self.register_buffer('sharing_weight', hard_ids.ne(0).float().unsqueeze(-1))
        # Level-offset token IDs implement disjoint level embedding tables.
        self.sid_embedding = nn.Embedding(num_tokens + 1, dim, padding_idx=0)
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        nn.init.xavier_normal_(self.sid_embedding.weight)
        with torch.no_grad():
            self.sid_embedding.weight[0].zero_()

    @torch.no_grad()
    def set_lookup(self, candidate_ids, sharing_weight):
        if candidate_ids.shape != sharing_weight.shape or candidate_ids.shape[:2] != self.hard_sid_table.shape:
            raise ValueError('Lookup dimensions must match catalog and SID levels')
        if candidate_ids.min() < 0 or candidate_ids.max() >= self.sid_embedding.num_embeddings:
            raise ValueError('Candidate token outside embedding table')
        if not torch.isfinite(sharing_weight).all() or sharing_weight.lt(0).any():
            raise ValueError('Weights must be finite and nonnegative')
        if sharing_weight[candidate_ids.eq(0)].ne(0).any():
            raise ValueError('Padding candidate weight must be zero')
        torch.testing.assert_close(sharing_weight[1:].sum(-1), torch.ones_like(sharing_weight[1:].sum(-1)))
        if candidate_ids[0].ne(0).any() or sharing_weight[0].ne(0).any():
            raise ValueError('Padding item must have no active candidates')
        device = self.hard_sid_table.device
        self.candidate_ids = candidate_ids.detach().clone().to(device)
        self.sharing_weight = sharing_weight.detach().clone().to(device)

    def hard_pre_ln(self, items):
        return self.sid_embedding(self.hard_sid_table[items]).mean(-2)

    def pre_ln(self, items):
        embeddings = self.sid_embedding(self.candidate_ids[items])
        return (embeddings * self.sharing_weight[items].unsqueeze(-1)).sum(-2).mean(-2)

    def forward(self, items):
        # Exactly one item LayerNorm, after all weighted level lookups are pooled.
        value = self.output_norm(self.pre_ln(items))
        return {'vectors': value * items.ne(0).unsqueeze(-1)}


class GCSS(nn.Module):
    def __init__(self, hard_ids, num_tokens, dim=128, max_len=50, heads=2, layers=2, dropout=.2):
        super().__init__()
        self.config = dict(dim=dim, max_len=max_len, heads=heads, layers=layers, dropout=dropout)
        self.item_encoder = SIDItemEncoder(hard_ids, num_tokens, dim)
        self.sequence_encoder = CausalTransformerEncoder(dim, max_len, heads, layers, dropout)

    def encode_items(self, items):
        unique, inverse = items.unique(return_inverse=True)
        return self.item_encoder(unique)['vectors'][inverse]

    def encode_sequence(self, sequence):
        return self.sequence_encoder(sequence, self.encode_items(sequence)), {}

    def forward(self, sequence, candidates):
        h, _ = self.encode_sequence(sequence)
        return {'score': torch.einsum('bd,bcd->bc', h, self.encode_items(candidates))}


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'gcss-v1':
        raise ValueError('Requires a pure GCSS checkpoint; old private/gated models are incompatible')
    state = checkpoint['model']
    model = GCSS(state['item_encoder.hard_sid_table'], len(state['item_encoder.sid_embedding.weight']) - 1,
                 **checkpoint['config'])
    model.item_encoder.set_lookup(state['item_encoder.candidate_ids'], state['item_encoder.sharing_weight'])
    model.load_state_dict(state, strict=True)
    return model.to(device), checkpoint
