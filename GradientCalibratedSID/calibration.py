import torch


@torch.no_grad()
def compatibility(artifact, grad_signature):
    hard = artifact['hard_ids'].cpu()
    ids, mask = artifact['candidate_ids'].cpu(), artifact['candidate_mask'].cpu()
    neighbors = artifact['neighbor_ids'].cpu()
    gradient = grad_signature.detach().float().cpu()
    if gradient.ndim != 2 or len(gradient) != len(hard):
        raise ValueError('Gradient signature must have shape [N,d], without padding')
    if not torch.isfinite(gradient).all() or gradient.norm(dim=-1).gt(1 + 1e-5).any():
        raise ValueError('Signature must be the unnormalized mean of unit sample gradients')
    result = torch.zeros_like(artifact['candidate_support'], device='cpu')
    for i in range(len(hard)):
        group_ids = neighbors[i][neighbors[i].gt(0)] - 1
        if group_ids.eq(i).any():
            raise ValueError('Self must not occur in a gradient neighborhood')
        for level in range(hard.shape[1]):
            for slot in torch.where(mask[i, level])[0].tolist():
                group = group_ids[hard[group_ids, level].eq(ids[i, level, slot])]
                if len(group):
                    # Includes zero signatures in the group mean; no extra
                    # confidence weighting or prototype normalization.
                    prototype = gradient[group].mean(0)
                    result[i, level, slot] = gradient[i].dot(prototype)
    if result.abs().gt(1 + 1e-5).any():
        raise ValueError('Compatibility out of bounds')
    return result


@torch.no_grad()
def sharing_weights(support, mask, comp, lambda_grad):
    if not (support.shape == mask.shape == comp.shape):
        raise ValueError('Calibration shapes differ')
    if not torch.isfinite(comp).all() or not torch.isfinite(support).all() or support[mask].le(0).any():
        raise ValueError('Active support must be positive and inputs finite')
    if not torch.isfinite(torch.tensor(lambda_grad)):
        raise ValueError('Non-finite lambda')
    logits = support.clamp_min(1e-12).log() + lambda_grad * comp
    logits = logits.masked_fill(~mask, -torch.inf)
    active = mask.any(-1, keepdim=True)
    logits = torch.where(active, logits, torch.zeros_like(logits))
    return torch.softmax(logits, -1) * mask


@torch.no_grad()
def shuffled_compatibility(comp, mask, seed):
    """Permute valid c slots within each item/level; preserve its value multiset."""
    result = comp.detach().clone().cpu()
    valid = mask.cpu()
    generator = torch.Generator().manual_seed(seed)
    for i in range(len(result)):
        for level in range(result.shape[1]):
            slots = torch.where(valid[i, level])[0]
            if len(slots) > 1:
                order = slots[torch.randperm(len(slots), generator=generator)]
                result[i, level, slots] = comp[i, level, order]
    return result


def pad_item(tensor):
    return torch.cat([torch.zeros_like(tensor[:1]), tensor], dim=0)
