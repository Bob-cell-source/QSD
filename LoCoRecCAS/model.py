import math
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from LoCoRec.locorec.model import CausalTransformerEncoder
from LoCoRec.locorec.soft_sid import _stable_tie_break


def build_scope(hard_table, max_neighbors=50, min_overlap=3, seed=2026):
    """Equations 2–7: all-level overlap, full local union, hard anchor, squared prior.

    No support cutoff or top-M truncation. The empty neighborhood is hard-only.
    """
    n, depth = hard_table.shape[0] - 1, hard_table.shape[1]
    if n < 2 or depth < 1 or hard_table[0].ne(0).any() or hard_table[1:].le(0).any():
        raise ValueError('Invalid Hard SID table (0 is reserved for padding).')
    if max_neighbors < 1 or not 1 <= min_overlap <= depth:
        raise ValueError('Require H >= 1 and 1 <= delta <= L.')
    codes = hard_table.cpu().tolist()
    inverted = defaultdict(list)
    for item in range(1, n + 1):
        for level, code in enumerate(codes[item]):
            inverted[level, code].append(item)
    rows, neighborhood_sizes = [], []
    for item in range(1, n + 1):
        overlap = Counter()
        for level, code in enumerate(codes[item]):
            overlap.update(inverted[level, code])
        neighbors = [j for j, count in overlap.items() if j != item and count >= min_overlap]
        neighbors.sort(key=lambda j: (-overlap[j], _stable_tie_break(item, j, seed)))
        neighbors = neighbors[:max_neighbors]
        neighborhood_sizes.append(len(neighbors))
        levels = []
        for level in range(depth):
            counts = Counter(codes[j][level] for j in neighbors)
            anchor = codes[item][level]
            tokens = [anchor] + sorted(set(counts) - {anchor})
            mass = [(counts[k] / max(len(neighbors), 1) + int(k == anchor)) ** 2 for k in tokens]
            levels.append((tokens, [m / sum(mass) for m in mass]))
        rows.append(levels)
    width = max(len(tokens) for levels in rows for tokens, _ in levels)
    tokens = torch.zeros(n + 1, depth, width, dtype=torch.long)
    prior = torch.zeros_like(tokens, dtype=torch.float)
    for item, levels in enumerate(rows, 1):
        for level, (ids, probabilities) in enumerate(levels):
            tokens[item, level, :len(ids)] = torch.tensor(ids)
            prior[item, level, :len(ids)] = torch.tensor(probabilities)
    metadata = {'neighbors_mean': sum(neighborhood_sizes) / n,
                'empty_neighborhoods': neighborhood_sizes.count(0), 'candidate_width': width,
                'active_candidates_mean': float(tokens[1:].ne(0).sum(-1).float().mean()),
                'min_overlap': min_overlap, 'max_neighbors': max_neighbors,
                'neighborhood_definition': 'all-level overlap, excludes self; no top-M truncation'}
    return tokens, prior, metadata


class ScopeItemEncoder(nn.Module):
    def __init__(self, hard_table, num_tokens, scope_tokens, prior, dim):
        super().__init__()
        if scope_tokens.shape != prior.shape or scope_tokens.shape[:2] != hard_table.shape:
            raise ValueError('Scope/prior shape mismatch.')
        if not torch.equal(scope_tokens[..., 0], hard_table):
            raise ValueError('The hard anchor must occupy candidate slot 0.')
        if (prior < 0).any() or not torch.allclose(prior[1:].sum(-1), torch.ones_like(prior[1:].sum(-1))):
            raise ValueError('Invalid scope prior.')
        self.register_buffer('hard_sid_table', hard_table.long())
        self.register_buffer('scope_tokens', scope_tokens.long())
        self.register_buffer('candidate_prior', prior.float())
        self.depth, self.dim = hard_table.size(1), dim
        self.semantic_basis_embedding = nn.Embedding(num_tokens + 1, dim, padding_idx=0)
        self.shared_residual_embedding = nn.Embedding(num_tokens + 1, dim, padding_idx=0)
        self.private_residual_embedding = nn.Embedding(len(hard_table), dim, padding_idx=0)
        self.basis_projection = nn.Linear(dim, dim)
        self.selector_embedding = nn.Embedding(num_tokens + 1, dim, padding_idx=0)
        self.selector_query = nn.Linear(dim, dim, bias=False)
        self.selector_key = nn.Linear(dim, dim, bias=False)
        self.prior_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        self.output_norm = nn.LayerNorm(dim, eps=1e-8)
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            for emb in (self.semantic_basis_embedding, self.shared_residual_embedding,
                        self.private_residual_embedding, self.selector_embedding):
                emb.weight[0].zero_()

    def assignment(self, items):
        tokens, prior = self.scope_tokens[items], self.candidate_prior[items]
        embeddings = self.selector_embedding(tokens)
        # Explicit definition of the item-level query in Eq. 8.
        query = self.selector_query((embeddings * prior[..., None]).sum(-2))
        key = self.selector_key(embeddings)
        logits = (query.unsqueeze(-2) * key).sum(-1) / math.sqrt(self.dim)
        logits = logits + F.softplus(self.prior_beta_raw) * prior.clamp_min(1e-8).log()
        mask = tokens.ne(0) & prior.gt(0)
        logits = logits.masked_fill(~mask, -1e9)
        weights = logits.softmax(-1) * mask
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)

    def components(self, items):
        """Return base=b+p and level residuals s, all context-independent."""
        tokens, weights = self.scope_tokens[items], self.assignment(items)
        basis = (self.semantic_basis_embedding(tokens) * weights[..., None]).sum(-2).mean(-2)
        basis = self.basis_projection(basis)
        shared = (self.shared_residual_embedding(tokens) * weights[..., None]).sum(-2)
        mask = items.ne(0).unsqueeze(-1)
        base = (basis + self.private_residual_embedding(items)) * mask
        return base, shared * mask.unsqueeze(-1)

    def static(self, items):
        base, shared = self.components(items)
        return self.output_norm(base + shared.mean(-2)) * items.ne(0).unsqueeze(-1)


class LoCoRecCAS(nn.Module):
    def __init__(self, hard_table, num_tokens, scope_tokens, prior, dim=128, max_len=50,
                 num_heads=2, num_layers=2, dropout=.2, gate_init=.9):
        super().__init__()
        if not 0 < gate_init < 1:
            raise ValueError('gate_init must be strictly between 0 and 1.')
        self.dim, self.depth = dim, hard_table.size(1)
        self.item_encoder = ScopeItemEncoder(hard_table, num_tokens, scope_tokens, prior, dim)
        self.sequence_encoder = CausalTransformerEncoder(dim, max_len, num_heads, num_layers, dropout)
        self.context_head = nn.Linear(dim, self.depth)
        self.global_logits = nn.Parameter(torch.full((self.depth,), math.log(gate_init / (1 - gate_init))))
        with torch.no_grad():
            self.context_head.weight.zero_()
            self.context_head.bias.copy_(self.global_logits)

    def encode_sequence(self, sequence):
        # Deduplicate items to avoid repeated local assignment evaluation in a batch.
        unique, inverse = sequence.unique(return_inverse=True)
        vectors = self.item_encoder.static(unique)[inverse]
        return self.sequence_encoder(sequence, vectors)

    def gates(self, h, mode='context'):
        if mode == 'static':
            return h.new_ones(len(h), self.depth)
        if mode == 'none':
            return h.new_zeros(len(h), self.depth)
        if mode == 'half':
            return h.new_full((len(h), self.depth), .5)
        if mode == 'global':
            return self.global_logits.sigmoid().expand(len(h), -1)
        if mode == 'context':
            return self.context_head(h).sigmoid()
        raise ValueError(f'Unknown gate mode: {mode}')

    def score_components(self, h, base, shared, gates):
        """Sampled B,C,... or shared catalog C,...; preserve candidate-dependent LN."""
        if base.ndim == 2:
            base, shared = base.unsqueeze(0), shared.unsqueeze(0)
        mixed = base + (shared * gates[:, None, :, None]).sum(-2) / self.depth
        vectors = self.item_encoder.output_norm(mixed)
        return (h[:, None] * vectors).sum(-1) / math.sqrt(self.dim)

    @torch.no_grad()
    def utility_targets(self, h, base, shared, temperature):
        if temperature <= 0:
            raise ValueError('Utility temperature must be positive.')
        all_gates = h.new_ones(len(h), self.depth)
        all_loss = -self.score_components(h, base, shared, all_gates).log_softmax(-1)[:, 0]
        differences = []
        for level in range(self.depth):
            gates = all_gates.clone()
            gates[:, level] = 0
            removed_loss = -self.score_components(h, base, shared, gates).log_softmax(-1)[:, 0]
            differences.append(removed_loss - all_loss)
        delta = torch.stack(differences, -1)
        return (delta / temperature).sigmoid(), delta

    def forward(self, sequence, candidates, mode='context', lambda_util=1., temperature=.1):
        h = self.encode_sequence(sequence)
        unique, inverse = candidates.unique(return_inverse=True)
        base, shared = self.item_encoder.components(unique)
        base, shared = base[inverse], shared[inverse]
        gates = self.gates(h, mode)
        rec = -self.score_components(h, base, shared, gates).log_softmax(-1)[:, 0].mean()
        util = rec.new_zeros(())
        targets, delta = None, None
        if mode == 'context' and lambda_util > 0:
            targets, delta = self.utility_targets(h, base, shared, temperature)
            util = F.binary_cross_entropy_with_logits(self.context_head(h), targets)
        return {'loss': rec + lambda_util * util, 'loss_rec': rec, 'loss_util': util,
                'gates': gates, 'targets': targets, 'delta': delta}


def transfer_locorec(model, checkpoint, source_hard_table):
    if not torch.equal(source_hard_table.cpu(), model.item_encoder.hard_sid_table.cpu()):
        raise ValueError('Source checkpoint and current SID mappings differ.')
    source = checkpoint['model']
    target = dict(model.named_parameters())
    keys = [k for k in target if k != 'global_logits' and not k.startswith('context_head.')]
    missing = [k for k in keys if k not in source or source[k].shape != target[k].shape]
    if missing:
        raise ValueError(f'Checkpoint parameters are incompatible: {missing}')
    with torch.no_grad():
        for key in keys:
            target[key].copy_(source[key])
    return {'loaded_parameters': keys, 'ignored_keys': sorted(set(source) - set(keys)),
            'scope': 'rebuilt from current methodology; old scope buffers and gates not transferred'}


class FrozenCatalog:
    """Exact LayerNorm scoring from centered components and their Gram matrices.

    This retains the user-item-dependent variance; it is NOT Eq. 30's invalid
    additive score expansion. Only appropriate while the item encoder is frozen.
    """
    @torch.no_grad()
    def __init__(self, encoder, chunk_size=128):
        parts = []
        for start in range(0, len(encoder.hard_sid_table), chunk_size):
            items = torch.arange(start, min(start + chunk_size, len(encoder.hard_sid_table)),
                                 device=encoder.hard_sid_table.device)
            base, shared = encoder.components(items)
            components = torch.cat([base[:, None], shared / encoder.depth], dim=1)
            parts.append(components - components.mean(-1, keepdim=True))
        self.centered = torch.cat(parts).detach()
        self.gram = (self.centered @ self.centered.transpose(-1, -2)) / encoder.dim
        self.gamma = encoder.output_norm.weight.detach().clone()
        self.beta = encoder.output_norm.bias.detach().clone()
        self.eps, self.dim = encoder.output_norm.eps, encoder.dim

    def features(self, h, items):
        weighted = h * self.gamma
        if items.ndim == 1:
            dot = torch.einsum('bd,ckd->bck', weighted, self.centered[items])
        else:
            dot = torch.einsum('bd,bckd->bck', weighted, self.centered[items])
        return dot

    def score(self, dot, items, gates, h=None):
        coeff = torch.cat([torch.ones_like(gates[:, :1]), gates], -1)
        numerator = (dot * coeff[:, None]).sum(-1)
        gram = self.gram[items]
        variance = torch.einsum('bk,ckj,bj->bc', coeff, gram, coeff) if items.ndim == 1 else torch.einsum('bk,bckj,bj->bc', coeff, gram, coeff)
        scores = numerator * torch.rsqrt(variance.clamp_min(0) + self.eps)
        if h is not None:
            scores = scores + (h * self.beta).sum(-1, keepdim=True)
        # h dot beta is constant across candidates and can be omitted for CE/ranking.
        return scores / math.sqrt(self.dim)
