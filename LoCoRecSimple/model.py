import math
import torch
from torch import nn
from torch.nn import functional as F
from LoCoRec.locorec.model import CausalTransformerEncoder, LoCoRec


class GlobalCorrection(nn.Module):
    """Two global corrections, with the original bounded correction parameterization."""
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, features):
        return self.logits.expand(*features.shape[:-1], 2)


def configure_fusion_control(model, variant):
    """Apply after matched initialization so all untouched parameters are identical."""
    enc = model.item_encoder
    if variant == 'full_global':
        # Equal initial catalog-average alpha; remove all item-dependent priors.
        mean_alpha = enc.alpha_prior[1:].mean()
        enc.alpha_prior[1:] = mean_alpha
        enc.residual_gate = GlobalCorrection()
    elif variant == 'full_equal':
        # weights [1, 1, 1]: gamma=2/3 gives residual scale 2; alpha=1/2.
        enc.alpha_prior[1:] = .5
        enc.residual_scale = 2.
    elif variant not in ('full', 'full_fixed'):
        raise ValueError(variant)


class SharingItemEncoder(nn.Module):
    def __init__(self, variant, hard_table, num_tokens, soft_table, prior, dim):
        super().__init__()
        if variant not in ('id', 'hard', 'soft'):
            raise ValueError(variant)
        self.variant, self.dim = variant, dim
        self.register_buffer('hard_sid_table', hard_table)
        self.private_embedding = nn.Embedding(len(hard_table), dim, padding_idx=0)
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        if variant != 'id':
            self.shared_embedding = nn.Embedding(num_tokens + 1, dim, padding_idx=0)
        if variant == 'soft':
            self.register_buffer('soft_sid_table', soft_table)
            self.register_buffer('candidate_prior', prior)
            self.selector_embedding = nn.Embedding(num_tokens + 1, dim, padding_idx=0)
            self.selector_query = nn.Linear(dim, dim, bias=False)
            self.selector_key = nn.Linear(dim, dim, bias=False)
            self.prior_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    module.weight[0].zero_()

    def assignment(self, items):
        tokens, prior = self.soft_sid_table[items], self.candidate_prior[items]
        embeddings = self.selector_embedding(tokens)
        query = self.selector_query((embeddings * prior[..., None]).sum(-2))
        keys = self.selector_key(embeddings)
        scores = (query[..., None, :] * keys).sum(-1) / math.sqrt(self.dim)
        scores += F.softplus(self.prior_beta_raw) * prior.clamp_min(1e-8).log()
        mask = tokens.ne(0) & prior.gt(0)
        weights = scores.masked_fill(~mask, -1e9).softmax(-1) * mask
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)

    def forward(self, items):
        value = self.private_embedding(items)
        if self.variant == 'hard':
            value = value + self.shared_embedding(self.hard_sid_table[items]).mean(-2)
        elif self.variant == 'soft':
            weights = self.assignment(items)
            shared = (self.shared_embedding(self.soft_sid_table[items]) * weights[..., None]).sum(-2).mean(-2)
            value = value + shared
        vectors = self.output_norm(value) * items.ne(0).unsqueeze(-1)
        return {'vectors': vectors}


class SimpleSharing(nn.Module):
    def __init__(self, variant, hard_table, num_tokens, soft_table, prior, dim=128,
                 max_len=50, num_heads=2, num_layers=2, dropout=.2,
                 embedding_dropout=0.0):
        super().__init__()
        self.sequence_encoder = CausalTransformerEncoder(dim, max_len, num_heads, num_layers, dropout)
        self.item_encoder = SharingItemEncoder(variant, hard_table, num_tokens, soft_table, prior, dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def encode_sequence(self, sequence):
        unique, inverse = sequence.unique(return_inverse=True)
        vectors = self.item_encoder(unique)['vectors'][inverse]
        vectors = self.embedding_dropout(vectors)
        return self.sequence_encoder(sequence, vectors), {}

    def forward(self, sequence, candidates):
        h, _ = self.encode_sequence(sequence)
        unique, inverse = candidates.unique(return_inverse=True)
        vectors = self.item_encoder(unique)['vectors'][inverse]
        return {'score': torch.einsum('bd,bcd->bc', h, vectors),
                'gate_kl': h.new_zeros(()), 'private_penalty': h.new_zeros(())}


def initialize_matched(model, seed, dim, max_len, heads, layers, dropout, num_items, num_tokens):
    """Same encoder/private/shared/selector starts independent of variant allocation order."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = CausalTransformerEncoder(dim, max_len, heads, layers, dropout)
        model.sequence_encoder.load_state_dict(encoder.state_dict())
        enc = model.item_encoder
        private = getattr(enc, 'private_embedding', getattr(enc, 'private_residual_embedding', None))
        torch.manual_seed(seed + 1)
        with torch.no_grad():
            nn.init.xavier_normal_(private.weight)
            private.weight[0].zero_()
        shared = getattr(enc, 'shared_embedding', getattr(enc, 'shared_residual_embedding', None))
        if shared is not None:
            torch.manual_seed(seed + 2)
            with torch.no_grad():
                nn.init.xavier_normal_(shared.weight)
                shared.weight[0].zero_()
        if hasattr(enc, 'selector_embedding'):
            torch.manual_seed(seed + 3)
            with torch.no_grad():
                for module in (enc.selector_embedding, enc.selector_query, enc.selector_key):
                    nn.init.xavier_normal_(module.weight)
                enc.selector_embedding.weight[0].zero_()
