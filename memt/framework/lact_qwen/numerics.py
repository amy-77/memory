"""Opt-in numerical safeguards; no changes to the default training path."""

import torch


@torch.no_grad()
def clip_grad_norm_fp64(parameters, max_norm, error_if_nonfinite=True):
    """FP64 norm reduction avoids overflow when individual FP32 gradients are finite."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0, dtype=torch.float64)
    norms = torch.stack([torch.linalg.vector_norm(g.detach(), dtype=torch.float64) for g in grads])
    total = torch.linalg.vector_norm(norms)
    if error_if_nonfinite and not bool(torch.isfinite(total)):
        raise RuntimeError("Non-finite gradient norm after FP64 reduction")
    coefficient = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(coefficient.to(grad.dtype))
    return total
