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
    topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token cross entropy and KL(teacher || student), both averaged over labelled positions.

    logits: [B, L, V] student logits. teacher_hidden: [B, L, H] final (normed) hidden states of the
    teacher; its logits are produced chunk by chunk so the full [L, V] teacher matrix never exists.
    labels: [B, L] with -100 where no loss is taken. Position i predicts labels[:, i + 1].
    topk > 0: the teacher distribution is cut to its topk most likely tokens and renormalised
    (MemT uses 64); the student's probabilities stay those of its full softmax.
    """
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
        log_s = F.log_softmax(logits[:, a:b][m].float(), dim=-1)
        with torch.no_grad():
            log_t = F.log_softmax(teacher_head(teacher_hidden[:, a:b][m]).float(), dim=-1)
        ce = ce + F.nll_loss(log_s, target[:, a:b][m], reduction="sum")
        if topk > 0:
            top, idx = log_t.topk(topk, dim=-1)
            log_t, log_s_at = F.log_softmax(top, dim=-1), log_s.gather(-1, idx)
        else:
            log_s_at = log_s
        kl = kl + (log_t.exp() * (log_t - log_s_at)).sum()
    return ce / count, kl / count
