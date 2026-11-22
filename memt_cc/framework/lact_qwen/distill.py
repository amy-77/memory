"""Self-distillation target for the memory branch.

The teacher is the same frozen base model with full attention and no memory. At every position
it shows what the output should be when the context beyond the window is available, so
KL(teacher || student) is a dense training signal that is close to zero wherever the window
already suffices and large exactly where history is needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ce_and_kl(
    logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_head: torch.nn.Module,
    labels: torch.Tensor,
    chunk: int = 1024,
    top_k: int = 0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token cross entropy and KL(teacher || student), both averaged over labelled positions.

    logits: [B, L, V] student logits. teacher_hidden: [B, L, H] final (normed) hidden states of the
    teacher; its logits are produced chunk by chunk so the full [L, V] teacher matrix never exists.
    labels: [B, L] with -100 where no loss is taken. Position i predicts labels[:, i + 1].
    """
    if temperature <= 0 or top_k < 0 or top_k > logits.shape[-1]:
        raise ValueError("invalid distillation temperature or top_k")
    n = logits.shape[1] - 1
    target = labels[:, 1:]
    mask = target != -100
    count = mask.sum().clamp(min=1)
    ce = logits.new_zeros((), dtype=torch.float32)
    kl = logits.new_zeros((), dtype=torch.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        m = mask[:, a:b]
        if not bool(m.any()):
            continue
        student_logits = logits[:, a:b][m].float()
        log_s = F.log_softmax(student_logits, dim=-1)
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_hidden[:, a:b][m]).float() / temperature
            if top_k:
                values, indices = teacher_logits.topk(top_k, dim=-1)
                log_t = F.log_softmax(values, dim=-1)
            else:
                log_t = F.log_softmax(teacher_logits, dim=-1)
        ce = ce + F.nll_loss(log_s, target[:, a:b][m], reduction="sum")
        log_s_kd = log_s if temperature == 1 else F.log_softmax(student_logits / temperature, dim=-1)
        if top_k:
            # Only the teacher is renormalized on its Top-K support (MemT Eq. in Sec. 3.3).
            log_s_kd = log_s_kd.gather(-1, indices)
        kl = kl + temperature**2 * (log_t.exp() * (log_t - log_s_kd)).sum()
    return ce / count, kl / count
