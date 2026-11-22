"""Training loss computed from the last hidden state in checkpointed chunks.

A 32K-token sequence has 32767 x 151936 logits: about 20 GB in float32 before the softmax and its
gradient. Here the LM head and the loss run on `chunk` positions at a time inside
torch.utils.checkpoint, so only one chunk of logits is alive in the forward or the backward pass
(the same idea as flash-linear-attention's FusedLinearCrossEntropyLoss, which LaCT trains with).

The wrapper returns sums, not means, so the caller can weight every supervised token equally across
sequences of different length, accumulation steps and ranks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_terms(h, y, t, head, teacher_head, top_k: int, temperature: float):
    logits = head(h).float()
    ce = F.cross_entropy(logits, y, reduction="sum")
    if t is None:
        return ce, ce.new_zeros(())
    with torch.no_grad():
        t_logits = teacher_head(t).float() / temperature
        if top_k:
            # MemT Sec. 3.3 / 3.6: the teacher is renormalised on its Top-K tokens, the student is not.
            t_logits, index = t_logits.topk(top_k, dim=-1)
        log_t = F.log_softmax(t_logits, dim=-1)
    log_s = F.log_softmax(logits / temperature, dim=-1) if temperature != 1 else F.log_softmax(logits, dim=-1)
    if top_k:
        log_s = log_s.gather(-1, index)
    kl = temperature**2 * (log_t.exp() * (log_t - log_s)).sum()
    return ce, kl


def chunked_ce_kl(hidden, head, labels, teacher_hidden=None, teacher_head=None,
                  chunk: int = 2048, top_k: int = 0, temperature: float = 1.0):
    """Returns (sum of next-token CE, sum of KL(teacher || student), number of supervised tokens).

    hidden: [B, L, H] final hidden states; position i predicts labels[:, i + 1]; labels == -100 are skipped.
    """
    target = labels[:, 1:].reshape(-1)
    keep = target != -100
    count = keep.sum()
    zero = hidden.sum() * 0.0  # keeps the graph alive when nothing is supervised
    if int(count) == 0:
        return zero.float(), zero.float(), count
    h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    t = None if teacher_hidden is None else teacher_hidden[:, :-1].reshape(-1, teacher_hidden.shape[-1])
    if not bool(keep.all()):
        h, target = h[keep], target[keep]
        t = None if t is None else t[keep]
    ce, kl = zero.float(), zero.float()
    for a in range(0, target.numel(), chunk):
        tc = None if t is None else t[a : a + chunk]
        if h.requires_grad:
            c, k = checkpoint(_chunk_terms, h[a : a + chunk], target[a : a + chunk], tc, head, teacher_head,
                              top_k, temperature, use_reentrant=False)
        else:
            c, k = _chunk_terms(h[a : a + chunk], target[a : a + chunk], tc, head, teacher_head, top_k, temperature)
        ce, kl = ce + c, kl + k
    return ce, kl, count


class StudentWithLoss(torch.nn.Module):
    """Student forward + loss in one module, so DistributedDataParallel sees a single forward."""

    def __init__(self, student, chunk: int = 2048, top_k: int = 0, temperature: float = 1.0):
        super().__init__()
        self.student = student
        self.chunk, self.top_k, self.temperature = chunk, top_k, temperature

    def forward(self, input_ids, labels, teacher_hidden=None, teacher_head=None):
        hidden = self.student.model(input_ids=input_ids, use_cache=False).last_hidden_state
        return chunked_ce_kl(hidden, self.student.lm_head, labels, teacher_hidden, teacher_head,
                             self.chunk, self.top_k, self.temperature)
