"""Checkpointed token chunks: never materialize the complete [B, L, V] logits.

The wrapper's entire student forward runs through DDP. Checkpointing each LM-head
chunk also prevents autograd from retaining all chunks' softmaxes until backward.
The frozen teacher is intentionally outside the DDP module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def hidden_ce_and_kl(hidden, head, labels, *, teacher_hidden=None, teacher_head=None,
                     teacher_topk=None, chunk=128, top_k=0, temperature=1.0):
    if chunk < 1 or temperature <= 0 or not 0 <= top_k <= head.weight.shape[0]:
        raise ValueError("invalid loss chunk size, temperature, or top_k")
    if (teacher_hidden is None) != (teacher_head is None):
        raise ValueError("teacher hidden states and head must be supplied together")
    if teacher_topk is not None and teacher_hidden is not None:
        raise ValueError("use either online teacher or cached Top-K targets")
    target = labels[:, 1:].reshape(-1)
    mask = target != -100
    # Selecting hidden states is cheap compared with retaining vocabulary logits.
    student = hidden[:, :-1].reshape(-1, hidden.shape[-1])[mask]
    target = target[mask]
    teacher = (teacher_hidden[:, :-1].reshape(-1, teacher_hidden.shape[-1])[mask]
               if teacher_hidden is not None else None)
    if target.numel() == 0:
        zero = hidden.sum() * 0.0
        return zero, zero

    if teacher_topk is not None:
        expected = (target.numel(), top_k)
        if top_k < 1 or any(tuple(x.shape) != expected for x in teacher_topk):
            raise ValueError("cached targets must align to supervised next-token positions")

    def chunk_loss(h, y, t, cached_indices, cached_log_probs):
        logits = head(h).float()
        if t is None and cached_indices is None:
            ce = F.cross_entropy(logits, y, reduction="sum")
            return ce, ce.new_zeros(())
        log_s = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_s, y, reduction="sum")
        if cached_indices is not None:
            indices, log_t = cached_indices, cached_log_probs.detach().float()
        else:
            with torch.no_grad():
                t_logits = teacher_head(t).float() / temperature
                if top_k:
                    values, indices = t_logits.topk(top_k, dim=-1)
                    log_t = F.log_softmax(values, dim=-1)
                else:
                    log_t = F.log_softmax(t_logits, dim=-1)
        if temperature != 1:
            log_s = F.log_softmax(logits / temperature, dim=-1)
        if top_k:
            log_s = log_s.gather(-1, indices)
        kl = temperature**2 * (log_t.exp() * (log_t - log_s)).sum()
        return ce, kl

    ce, kl = hidden.new_zeros((), dtype=torch.float32), hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, target.numel(), chunk):
        end = start + chunk
        t = None if teacher is None else teacher[start:end]
        indices, probs = (None, None) if teacher_topk is None else (v[start:end] for v in teacher_topk)
        c, k = checkpoint(chunk_loss, student[start:end], target[start:end], t, indices, probs,
                          use_reentrant=False)
        ce, kl = ce + c, kl + k
    return ce / target.numel(), kl / target.numel()


class MemoryTrainingLoss(torch.nn.Module):
    def __init__(self, student, *, chunk=128, top_k=0, temperature=1.0):
        super().__init__()
        self.student = student
        self.chunk, self.top_k, self.temperature = chunk, top_k, temperature

    def forward(self, input_ids, labels, attention_mask=None, teacher_hidden=None, teacher_head=None,
                teacher_topk=None):
        hidden = self.student.model(input_ids=input_ids, attention_mask=attention_mask,
                                    use_cache=False).last_hidden_state
        return hidden_ce_and_kl(hidden, self.student.lm_head, labels,
                                teacher_hidden=teacher_hidden, teacher_head=teacher_head,
                                teacher_topk=teacher_topk,
                                chunk=self.chunk, top_k=self.top_k, temperature=self.temperature)
