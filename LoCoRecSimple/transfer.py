"""Fixed-SID training operator probe; not a claim of established novelty.

For level-wise group-mean projectors Pi_l, C=mean_l Pi_l.
K=I+eta*(C-diag(C)) preserves the diagonal while transferring other-item
gradients. C is PSD with eigenvalues in [0,1]. For eta in [0,1],
K is PSD with eigenvalues in [1-eta,1+eta]. These facts concern K and
plain gradient descent, not a descent guarantee after Adam/clipping.
"""
import torch


class SIDGradientTransfer:
    def __init__(self, hard_table, strength=.5):
        if not 0 <= strength <= 1:
            raise ValueError('strength must be in [0,1]')
        self.tokens = hard_table[1:].long()
        if self.tokens.ndim != 2 or self.tokens.le(0).any():
            raise ValueError('nonpadding items must have positive level-offset tokens')
        self.strength = strength
        self.counts = torch.bincount(self.tokens.flatten()).clamp_min(1)
        self.diagonal = self.counts[self.tokens].double().reciprocal().mean(-1)

    def __call__(self, gradient):
        if self.strength == 0:
            return gradient
        tokens = self.tokens.to(gradient.device)
        count = self.counts.to(device=gradient.device,dtype=gradient.dtype)
        values = gradient[1:]
        pooled = gradient.new_zeros((len(count), gradient.shape[-1]))
        pooled.index_add_(0, tokens.flatten(), values[:,None,:].expand(-1,tokens.shape[1],-1).reshape(-1,gradient.shape[-1]))
        pooled = pooled / count[:,None]
        cross = pooled[tokens].mean(1) - self.diagonal.to(gradient)[:,None]*values
        result = gradient.clone()
        result[1:] = values + self.strength*cross
        return result
