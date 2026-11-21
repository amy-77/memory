"""Selective writing: per-token importance for the memory write.

Each write segment (a turn, or a fixed chunk) is written with one step. Without selection every token
of the segment enters that step with the learning rate the model predicts for it. Here the rate of
token t is multiplied by a weight w_t >= 0 with mean ~1 over the segment, so the step is spent on the
tokens that matter. The weights come from scores that cost one small matrix product per layer:

query     (SnapKV's observation window, but the window is the QUESTION, and Select-TTT's idea of
          relevance to the query): score_t = how strongly the question's read vectors q hit the key
          k_t that token t writes. This is the similarity the memory itself reads with (it returns
          the value written under keys close to the read vector), so no extra forward is needed and
          the question's q are computed once and reused for every segment.
agnostic  (H2O / KVzip: keep what later tokens look at): score_t = the largest (KVzip) or summed
          (H2O) attention that later tokens of the same segment pay to token t, computed with the
          branch's own q and k. Needs no question.
hybrid    alpha * query + (1 - alpha) * agnostic, both z-scored inside the segment.
oracle    weights are given (MemoryContext.oracle_weight); an upper bound for the scoring rule.

With `keep` > 0 the weights are a hard SnapKV-style choice instead: the top share of tokens (scores
max-pooled over neighbours) is written, the rest is not.

Turning scores into weights follows Select-TTT: w = min(cap, n * softmax(z / tau)) inside a segment
of n tokens, z = z-scored scores. With `seg_gain` the size of each segment's step is scaled the same
way across segments (Muon gives every write the same size, so token weights alone only change the
direction of a write, not how much room a segment takes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

SELECT_MODES = ("none", "query", "agnostic", "hybrid", "oracle")


@dataclass
class SelectConfig:
    mode: str = "none"
    tau: float = 0.5  # temperature on the z-scored token scores; smaller = sharper
    cap: float = 10.0  # largest multiplier (Select-TTT's M)
    floor: float = 0.0  # smallest multiplier
    topk_frac: float = 0.1  # share of question tokens averaged per key (Select-TTT's rho)
    pool: int = 1  # SnapKV: max-pool the scores over this many neighbouring tokens (odd)
    alpha: float = 0.5  # hybrid: weight of the query score
    agnostic_reduce: str = "max"  # "max" (KVzip) or "sum" (H2O)
    agnostic_tau: float = 0.05  # temperature of the within-segment attention
    seg_gain: bool = False  # also scale each segment's step size by its relevance
    seg_tau: float = 0.5
    seg_cap: float = 4.0
    score_layer: int = -1  # >= 0: layers above this one reuse its weights instead of scoring again
    # SnapKV-style hard selection: > 0 keeps this share of a segment's tokens (the highest scores after
    # max-pooling over `pool` neighbours, so a hit brings its surrounding tokens along) at weight 1 and
    # writes the rest with weight `floor`. 0 = soft weights from the softmax above.
    keep: float = 0.0
    # Which question tokens score the keys: 0 = all of them, N > 0 = only the last N (the positions the
    # answer is read from; earlier question tokens mostly match the wording of the cue, not the value).
    query_last: int = 0
    # "center": a token takes the best score among its neighbours on both sides (SnapKV).
    # "forward": a token takes the best score among itself and the `pool - 1` tokens BEFORE it, so the
    # tokens that follow a hit are kept (in "X is VALUE" the cue comes first, the value after it).
    pool_mode: str = "center"

    def __post_init__(self):
        if self.mode not in SELECT_MODES:
            raise ValueError(f"select mode must be one of {SELECT_MODES}, got {self.mode}")
        if self.agnostic_reduce not in ("max", "sum"):
            raise ValueError("agnostic_reduce must be 'max' or 'sum'")
        if self.pool_mode not in ("center", "forward"):
            raise ValueError("pool_mode must be 'center' or 'forward'")
        if self.pool_mode == "center" and self.pool % 2 == 0:
            raise ValueError("pool must be odd with pool_mode 'center'")


def _heads_mean(x: torch.Tensor, b: int, h: int) -> torch.Tensor:
    """[B * H, L] -> [B, L]."""
    return x.view(b, h, -1).mean(dim=1)


@torch.no_grad()
def last_tokens(qmask: torch.Tensor, n: int) -> torch.Tensor:
    """Keep only the last n True entries of every row of a [B, Nq] mask."""
    if n <= 0:
        return qmask
    after = qmask.flip(1).long().cumsum(dim=1).flip(1)  # number of valid tokens from here to the end
    return qmask & (after <= n)


def query_scores(k: torch.Tensor, qq: torch.Tensor, qmask: torch.Tensor, b: int, h: int, topk_frac: float):
    """k: [BH, L, d] keys of the history; qq: [BH, Nq, d] read vectors of the question tokens;
    qmask: [B, Nq] True on real question tokens. Returns [B, L]."""
    sim = k.float() @ qq.float().transpose(1, 2)  # [BH, L, Nq]
    mask = qmask.repeat_interleave(h, dim=0)[:, None, :]
    sim = sim.masked_fill(~mask, float("-inf"))
    kq = max(1, math.ceil(topk_frac * int(qmask.sum(dim=1).min())))
    top = sim.topk(kq, dim=-1).values.mean(dim=-1)
    return _heads_mean(top, b, h)


@torch.no_grad()
def agnostic_scores(q: torch.Tensor, k: torch.Tensor, b: int, h: int, tau: float, reduce: str):
    """Attention that later tokens of the segment pay to each token. q, k: [BH, n, d]. Returns [B, n]."""
    n = q.shape[1]
    if n < 2:
        return q.new_zeros(b, n, dtype=torch.float32)
    att = (q.float() @ k.float().transpose(1, 2)) / tau  # [BH, n(query), n(key)]
    later = torch.ones(n, n, dtype=torch.bool, device=q.device).tril(-1)  # query j > key t
    att = att.masked_fill(~later, float("-inf"))
    att = torch.softmax(att[:, 1:], dim=-1)  # row 0 has no earlier key
    s = att.amax(dim=1) if reduce == "max" else att.sum(dim=1)
    return _heads_mean(s, b, h)


def _zscore(s: torch.Tensor) -> torch.Tensor:
    return (s - s.mean(dim=-1, keepdim=True)) / (s.std(dim=-1, keepdim=True) + 1e-6)


def scores_to_weights(s: torch.Tensor, cfg: SelectConfig) -> torch.Tensor:
    """s: [B, n] scores of one segment -> multipliers [B, n] with mean <= 1."""
    n = s.shape[-1]
    if n < 2:
        return torch.ones_like(s)
    if cfg.pool > 1 and cfg.pool_mode == "center":
        s = F.max_pool1d(s[:, None], cfg.pool, stride=1, padding=cfg.pool // 2)[:, 0]
    elif cfg.pool > 1:
        padded = F.pad(s[:, None], (cfg.pool - 1, 0), value=float("-inf"))
        s = F.max_pool1d(padded, cfg.pool, stride=1)[:, 0]
    if cfg.keep > 0:
        m = max(1, math.ceil(cfg.keep * n))
        thr = s.topk(m, dim=-1).values[:, -1:]
        return torch.where(s >= thr, torch.ones_like(s), torch.full_like(s, cfg.floor))
    w = torch.softmax(_zscore(s) / cfg.tau, dim=-1) * n
    return w.clamp(min=cfg.floor, max=cfg.cap)


@torch.no_grad()
def select_weights(
    cfg: SelectConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    segments: Sequence[tuple[int, int]],
    flags: Sequence[bool],
    b: int,
    h: int,
    query_span: Optional[tuple[int, int]] = None,
    query_mask: Optional[torch.Tensor] = None,
    oracle_weight: Optional[torch.Tensor] = None,
    query_q: Optional[torch.Tensor] = None,
    query_prefix: int = 0,
):
    """Returns (weights [B, L], gains [B, n_segments] or None). Tokens outside written segments get 1.

    Where the question's read vectors come from (first that applies):
      query_prefix > 0  every written segment starts with the question (that many tokens); its keys are
                        scored against the question tokens of the same segment, which are not written.
      query_q           [BH, Nq, d], taken beforehand from a forward of the question alone and reused
                        for every segment; query_mask [B, Nq] marks real tokens.
      query_span        the question sits in this token range of the same forward (reference setting:
                        its vectors have seen the end of the history).
    """
    length = q.shape[1]
    w = torch.ones(b, length, device=q.device, dtype=torch.float32)
    if cfg.mode == "oracle":
        if oracle_weight is None:
            raise ValueError("select mode 'oracle' needs MemoryContext.oracle_weight")
        return oracle_weight.to(q.device, torch.float32), None

    need_query = cfg.mode in ("query", "hybrid")
    if need_query:
        if query_prefix > 0:
            s_query = torch.zeros(b, length, device=q.device, dtype=torch.float32)
            pmask = last_tokens(torch.ones(b, query_prefix, dtype=torch.bool, device=q.device), cfg.query_last)
            for (s, e), do_update in zip(segments, flags):
                if do_update and e - s > query_prefix:
                    s_query[:, s + query_prefix : e] = query_scores(
                        k[:, s + query_prefix : e], q[:, s : s + query_prefix], pmask, b, h, cfg.topk_frac
                    )
        elif query_q is not None:
            if query_mask is None:
                raise ValueError("query_q needs query_mask")
            qm = last_tokens(query_mask.to(q.device), cfg.query_last)
            s_query = query_scores(k, query_q.to(q.device), qm, b, h, cfg.topk_frac)
        elif query_span is not None and query_mask is not None:
            qs, qe = query_span
            qm = last_tokens(query_mask.to(q.device), cfg.query_last)
            s_query = query_scores(k, q[:, qs:qe], qm, b, h, cfg.topk_frac)
        else:
            raise ValueError(f"select mode '{cfg.mode}' needs a question: query_prefix, query_q or query_span")

    seg_scores = []
    for (s, e), do_update in zip(segments, flags):
        if not do_update:
            seg_scores.append(None)
            continue
        c0 = s + query_prefix if (query_prefix and e - s > query_prefix) else s  # content starts here
        parts = []
        if need_query:
            parts.append((cfg.alpha if cfg.mode == "hybrid" else 1.0, s_query[:, c0:e]))
        if cfg.mode in ("agnostic", "hybrid"):
            a = agnostic_scores(q[:, c0:e], k[:, c0:e], b, h, cfg.agnostic_tau, cfg.agnostic_reduce)
            parts.append(((1.0 - cfg.alpha) if cfg.mode == "hybrid" else 1.0, a))
        sc = parts[0][1] if len(parts) == 1 else sum(c * _zscore(p) for c, p in parts)
        w[:, c0:e] = scores_to_weights(sc, cfg)
        if c0 > s:
            w[:, s:c0] = 0.0  # the copy of the question is not written
        # how relevant the segment is as a whole: mean of its top scores (raw query score if there is one)
        base = s_query[:, c0:e] if need_query else sc
        top = max(1, math.ceil(cfg.topk_frac * (e - c0)))
        seg_scores.append(base.topk(min(top, e - c0), dim=-1).values.mean(dim=-1))

    gains = None
    if cfg.seg_gain:
        idx = [i for i, sc in enumerate(seg_scores) if sc is not None]
        gains = torch.ones(b, len(segments), device=q.device, dtype=torch.float32)
        if len(idx) > 1:
            stacked = torch.stack([seg_scores[i] for i in idx], dim=1)  # [B, n_written]
            g = torch.softmax(_zscore(stacked) / cfg.seg_tau, dim=-1) * len(idx)
            gains[:, idx] = g.clamp(max=cfg.seg_cap)
    return w, gains


# ----------------------------------------------------------------------------------------------
# Input-level scoring: computed once from token ids, reused by every layer and every later step.
# ----------------------------------------------------------------------------------------------
def rarity(question_ids: Sequence[int], counts: torch.Tensor, total: float) -> torch.Tensor:
    """Per question token, in [0, 1]: 1 for the rarest token of the question, ~0 for common ones.
    counts[token] / total is the token's frequency in whatever text the rarity is taken from."""
    q = torch.as_tensor(list(question_ids), dtype=torch.long)
    idf = torch.log((total + 1.0) / (counts[q].float() + 1.0)).clamp_min(0.0)
    return idf / idf.max().clamp_min(1e-9)


@torch.no_grad()
def lexical_weights(
    ids: torch.Tensor,
    question_ids: Sequence[int],
    rar: torch.Tensor,
    boost: float = 10.0,
    pool: int = 8,
    pool_mode: str = "forward",
    min_rarity: float = 0.5,
) -> torch.Tensor:
    """Write multipliers >= 1 for a token sequence `ids` [L].

    A token that equals a rare token of the question (rarity >= min_rarity; e.g. the id being asked
    about, not "the" or "is") scores that rarity; the score is spread over `pool` tokens (forward: the
    tokens after the hit, where the value usually stands). Weight = 1 + (boost - 1) * score, so text
    that does not mention what the question asks about is written exactly as before.
    """
    q = torch.as_tensor(list(question_ids), dtype=torch.long, device=ids.device)
    r = rar.to(ids.device).float()
    r = torch.where(r >= min_rarity, r, torch.zeros_like(r))
    hit = (ids[:, None] == q[None, :]).float() * r[None, :]  # [L, Nq]
    s = hit.max(dim=1).values
    if pool > 1:
        if pool_mode == "forward":
            s = F.max_pool1d(F.pad(s[None, None], (pool - 1, 0)), pool, stride=1)[0, 0]
        else:
            s = F.max_pool1d(s[None, None], pool | 1, stride=1, padding=(pool | 1) // 2)[0, 0]
    return 1.0 + (boost - 1.0) * s
