"""Sparse posting-list semantic neighborhoods; no N x N similarity matrix."""
from collections import Counter, defaultdict
import numpy as np
import torch


def build_candidates(hard_ids, text_embeddings, max_neighbors=50, max_candidates=4):
    """Inputs and saved artifact exclude padding item; neighbor IDs are 1-based."""
    if max_neighbors < 0 or max_candidates < 1:
        raise ValueError('H must be nonnegative and M positive')
    hard = np.asarray(hard_ids, dtype=np.int64)
    text = np.asarray(text_embeddings, dtype=np.float64)
    if hard.ndim != 2 or text.ndim != 2 or len(hard) != len(text) or (hard <= 0).any():
        raise ValueError('Expected positive level-offset SID IDs and aligned text rows')
    if not np.isfinite(text).all():
        raise ValueError('Non-finite frozen text embeddings')
    norms = np.linalg.norm(text, axis=-1, keepdims=True)
    if (norms == 0).any():
        raise ValueError('Cosine tie-break requires nonzero text embeddings')
    text = text / norms
    n, depth = hard.shape
    postings = defaultdict(list)
    for i in range(n):
        for level, token in enumerate(hard[i]):
            postings[level, int(token)].append(i)
    candidate_ids = np.zeros((n, depth, max_candidates), dtype=np.int64)
    support = np.zeros_like(candidate_ids, dtype=np.float32)
    mask = np.zeros_like(candidate_ids, dtype=bool)
    neighbors = np.zeros((n, max_neighbors), dtype=np.int64)
    for i in range(n):
        overlap = Counter()
        for level, token in enumerate(hard[i]):
            overlap.update(postings[level, int(token)])
        overlap.pop(i, None)
        indices = np.array(list(overlap), dtype=np.int64)
        if len(indices):
            affinity = text[indices] @ text[i]
            counts = np.array([overlap[int(j)] for j in indices])
            # Lexicographic (-overlap, -text cosine); item index only resolves
            # exact ties of BOTH meaningful scores for deterministic output.
            order = np.lexsort((indices, -affinity, -counts))
            selected = indices[order[:max_neighbors]]
        else:
            selected = np.empty(0, dtype=np.int64)
        neighbors[i, :len(selected)] = selected + 1
        for level in range(depth):
            token = int(hard[i, level])
            counts = Counter([token, *map(int, hard[selected, level])])
            first_rank = {}
            for rank, j in enumerate(selected):
                first_rank.setdefault(int(hard[j, level]), rank)
            other = sorted((k for k in counts if k != token), key=lambda k: (-counts[k], first_rank[k]))
            chosen = [token, *other[:max_candidates - 1]]
            candidate_ids[i, level, :len(chosen)] = chosen
            support[i, level, :len(chosen)] = [counts[k] / (len(selected) + 1) for k in chosen]
            mask[i, level, :len(chosen)] = True
    return {'hard_ids': torch.from_numpy(hard.copy()), 'candidate_ids': torch.from_numpy(candidate_ids),
            'candidate_support': torch.from_numpy(support), 'candidate_mask': torch.from_numpy(mask),
            'neighbor_ids': torch.from_numpy(neighbors)}
