"""Positive-target-only gradient profiling: frozen model, per-example leaf."""
import math
from contextlib import contextmanager
import torch


@contextmanager
def profiling_precision():
    # cuDNN's TF32 pointwise convolutions can take different numeric paths for
    # batches and individual examples. Profile in FP32, then restore training.
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@profiling_precision()
def positive_gradients(model, history, targets, negative_ids):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Profiling requires eval mode and every model parameter frozen')
    if targets.le(0).any():
        raise ValueError('Padding cannot be a positive target')
    with torch.no_grad():
        h, _ = model.encode_sequence(history)
        negative = model.encode_items(negative_ids)
        pre_ln = model.item_encoder.hard_pre_ln(targets)
    leaf = pre_ln.detach().requires_grad_(True)
    positive = model.item_encoder.output_norm(leaf)
    scale = math.sqrt(model.config['dim'])
    pos_score = (h.detach() * positive).sum(-1, keepdim=True) / scale
    neg_score = (h.detach()[:, None] * negative.detach()).sum(-1) / scale
    logits = torch.cat([pos_score, neg_score], -1)
    # Sum, not mean: each row is exactly d L_x / d e_i, irrespective of batch size.
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(targets), dtype=torch.long, device=targets.device), reduction='sum')
    gradient = torch.autograd.grad(loss, leaf, only_inputs=True)[0]
    return gradient.detach()


@profiling_precision()
def fullsoftmax_positive_gradients(model, history, targets):
    """Gradient of full-catalog CE wrt only each row's positive pre-LN leaf."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Profiling requires eval mode and every model parameter frozen')
    if targets.le(0).any():
        raise ValueError('Padding cannot be a positive target')
    with torch.no_grad():
        h, _ = model.encode_sequence(history)
        catalog = model.encode_items(torch.arange(1, len(model.item_encoder.hard_sid_table), device=history.device))
        pre_ln = model.item_encoder.hard_pre_ln(targets)
    leaf = pre_ln.detach().requires_grad_(True)
    positive = model.item_encoder.output_norm(leaf)
    logits = (h.detach() @ catalog.detach().T) / math.sqrt(model.config['dim'])
    row = torch.arange(len(targets), device=history.device)
    logits = logits.clone()
    logits[row, targets - 1] = (h.detach() * positive).sum(-1) / math.sqrt(model.config['dim'])
    loss = torch.nn.functional.cross_entropy(logits, targets - 1, reduction='sum')
    return torch.autograd.grad(loss, leaf, only_inputs=True)[0].detach()


def profile_fullsoftmax(model, loader, device, eps=1e-12):
    model.eval(); model.requires_grad_(False); model.zero_grad(set_to_none=True)
    n=len(model.item_encoder.hard_sid_table)-1; grad_sum=torch.zeros(n+1,model.config['dim'],dtype=torch.float64); grad_count=torch.zeros(n+1,dtype=torch.long)
    examples=valid_examples=0
    for history, candidates in loader:
        history,candidates=history.to(device),candidates.to(device)
        targets = candidates[:,0] if candidates.ndim > 1 else candidates
        gradient=fullsoftmax_positive_gradients(model,history,targets)
        valid_examples += aggregate_directions(grad_sum,grad_count,targets,gradient,eps); examples += len(history)
    signature=(grad_sum/grad_count.clamp_min(1)[:,None])[1:].float()
    if not torch.isfinite(signature).all() or signature.norm(dim=-1).gt(1+1e-6).any(): raise ValueError('Invalid mean directional signature')
    if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Profiling polluted parameters')
    return {'grad_signature':signature,'grad_count':grad_count[1:],'examples':examples,'valid_examples':valid_examples}


def aggregate_directions(grad_sum, grad_count, item_ids, gradient, eps=1e-12):
    values = gradient.detach().double().cpu()
    if not torch.isfinite(values).all():
        raise ValueError('Non-finite positive-target gradient')
    item_ids = item_ids.detach().long().cpu()
    norm = values.norm(dim=-1, keepdim=True)
    valid = norm.squeeze(-1).gt(eps)
    normalized = values / (norm + eps)
    grad_sum.index_add_(0, item_ids[valid], normalized[valid])
    grad_count.index_add_(0, item_ids[valid], torch.ones_like(item_ids[valid]))
    return int(valid.sum())


def profile(model, loader, device, eps=1e-12):
    if model.item_encoder.candidate_ids.shape[-1] != 1:
        raise ValueError('Only the fixed pure Hard SID checkpoint may be profiled')
    model.eval()
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    n = len(model.item_encoder.hard_sid_table) - 1
    grad_sum = torch.zeros(n + 1, model.config['dim'], dtype=torch.float64)
    grad_count = torch.zeros(n + 1, dtype=torch.long)
    examples, valid_examples = 0, 0
    for history, candidates in loader:
        history, candidates = history.to(device), candidates.to(device)
        gradient = positive_gradients(model, history, candidates[:,0], candidates[:,1:])
        valid_examples += aggregate_directions(grad_sum, grad_count, candidates[:,0], gradient, eps)
        examples += len(history)
    # Do NOT normalize this mean: its norm contains directional consistency.
    signature = (grad_sum / grad_count.clamp_min(1)[:,None])[1:].float()
    if not torch.isfinite(signature).all() or signature.norm(dim=-1).gt(1 + 1e-6).any():
        raise ValueError('Invalid mean directional signature')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Profiling polluted checkpoint parameter gradients')
    return {'grad_signature': signature, 'grad_count': grad_count[1:],
            'examples': examples, 'valid_examples': valid_examples}
