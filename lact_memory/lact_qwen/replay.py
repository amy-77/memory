"""Replay banks that protect earlier memory writes.

Modes
-----
none        no replay.
oracle_kv   store the real past (k, v) pairs and re-write a sample of them together
            with the current segment. Breaks the "history is gone" constraint, so it
            is only an upper bound.
oracle_key  store the real past keys; after the write, pull the outputs on a sample
            of them back towards their pre-write values.
gmm         same correction as oracle_key, but keys are sampled from a per-head
            diagonal Gaussian mixture fitted online to past keys.
scored_kv   keep only the `keep_per_segment` highest-scoring (k, v) pairs of every written
            segment (scores from selective writing, see select.py) in a bank of at most
            `bank_size` pairs, and re-write the `num_samples` best of them with each new
            segment. A small KV store chosen like a compressed KV cache (SnapKV / KVzip),
            used to refresh the memory instead of being attended to.
pseudo      pseudo-rehearsal (Robins 1995) without random probes. Before a segment is written,
            a set of probe vectors p is read from the memory as it is, y = f_W(p), and the pairs
            (p -> y) go into the SAME update as the segment's own (k, v). No past value is stored.
            Probes (`probe_src`): "query" = the read vectors of the question (what will be asked
            of the memory later), "bank" = the highest-scoring keys of earlier segments (keys
            only, chosen by their inner product with the question as in SnapKV), "both".
            `pseudo_target` "delta" rehearses what the writes so far added, f_W(p) - f_W0(p)
            (nothing to rehearse before the first write); "full" rehearses f_W(p).

The reference outputs are taken from the weights right before the current write.
At those weights the replay loss has zero gradient, so the correction must be a
separate step taken on the candidate weights after the write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .fast_weight import FastWeightState, fw_apply, fw_descent_direction, l2_norm, renorm_rows

REPLAY_MODES = ("none", "oracle_kv", "oracle_key", "gmm", "scored_kv", "pseudo")


@dataclass
class ReplayConfig:
    mode: str = "none"
    num_samples: int = 256
    strength: float = 1.0
    clip_ratio: float = 0.5
    selection: str = "coverage"
    temperature: float = 0.1
    gmm_components: int = 32
    bank_size: int = 65536
    kv_lr_scale: float = 1.0
    keep_per_segment: int = 32
    probe_src: str = "query"
    pseudo_target: str = "delta"

    def __post_init__(self):
        if self.mode not in REPLAY_MODES:
            raise ValueError(f"replay mode must be one of {REPLAY_MODES}, got {self.mode}")
        if self.selection not in ("uniform", "coverage"):
            raise ValueError("selection must be 'uniform' or 'coverage'")
        if self.probe_src not in ("query", "bank", "both"):
            raise ValueError("probe_src must be 'query', 'bank' or 'both'")
        if self.pseudo_target not in ("delta", "full"):
            raise ValueError("pseudo_target must be 'delta' or 'full'")


class ReplayBank:
    """Per-sequence replay state. Lives for one forward pass or one dialogue session."""

    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.lrs: Optional[torch.Tensor] = None
        self.scores: Optional[torch.Tensor] = None
        # mode 'pseudo': question read vectors [BH, Nq, d] + mask [BH, Nq] (set by the memory branch),
        # and the fast weights the sequence started from (set at the first segment)
        self.query_q: Optional[torch.Tensor] = None
        self.query_valid: Optional[torch.Tensor] = None
        self.init_weights = None
        self.mu: Optional[torch.Tensor] = None
        self.ex2: Optional[torch.Tensor] = None
        self.count: Optional[torch.Tensor] = None

    @property
    def empty(self) -> bool:
        if self.cfg.mode == "gmm":
            return self.count is None or bool((self.count.sum(dim=-1) == 0).all())
        return self.keys is None

    # ---------------------------------------------------------------- sampling
    def _selection_logits(self, candidates: torch.Tensor, current_keys: torch.Tensor) -> torch.Tensor:
        """Lower logits for history that the current segment already covers."""
        bh, n, _ = candidates.shape
        if self.cfg.selection == "uniform":
            return candidates.new_zeros(bh, n)
        seg_dir = l2_norm(current_keys.float().mean(dim=1))
        overlap = torch.einsum("bnd,bd->bn", l2_norm(candidates), seg_dir)
        return -overlap / self.cfg.temperature

    def _sample_indices(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, self.cfg.num_samples, replacement=True)

    @torch.no_grad()
    def sample_keys(self, current_keys: torch.Tensor) -> Optional[torch.Tensor]:
        if self.empty:
            return None
        if self.cfg.mode == "oracle_key":
            idx = self._sample_indices(self._selection_logits(self.keys, current_keys))
            return torch.gather(self.keys, 1, idx[..., None].expand(-1, -1, self.keys.shape[-1]))

        logits = self._selection_logits(self.mu, current_keys)
        logits = logits + torch.log(self.count.clamp_min(1e-6))
        logits = logits.masked_fill(self.count <= 0, float("-inf"))
        comp = self._sample_indices(logits)
        d = self.mu.shape[-1]
        mu = torch.gather(self.mu, 1, comp[..., None].expand(-1, -1, d))
        var = (torch.gather(self.ex2, 1, comp[..., None].expand(-1, -1, d)) - mu.pow(2)).clamp_min(1e-6)
        return l2_norm(mu + var.sqrt() * torch.randn_like(mu))

    # ------------------------------------------------------------------ write
    def _pseudo_pairs(self, k, v, lr, state):
        """Probes and what the memory returns for them right now."""
        cfg = self.cfg
        probes, valid = [], []
        if cfg.probe_src in ("query", "both") and self.query_q is not None:
            probes.append(self.query_q.float())
            valid.append(self.query_valid.float())
        if cfg.probe_src in ("bank", "both") and self.keys is not None:
            idx = self.scores.topk(min(cfg.num_samples, self.scores.shape[1]), dim=1).indices
            probes.append(torch.gather(self.keys, 1, idx[..., None].expand(-1, -1, self.keys.shape[-1])))
            valid.append(torch.ones_like(idx, dtype=torch.float32))
        if not probes or state is None:
            return k, v, lr
        p, ok = torch.cat(probes, dim=1).to(k.device), torch.cat(valid, dim=1).to(k.device)
        with torch.no_grad():
            y = fw_apply(*state.weights(), p)
            if cfg.pseudo_target == "delta":
                if self.init_weights is None:
                    return k, v, lr
                y = y - fw_apply(*self.init_weights, p)
            if float(y.abs().max()) == 0.0:
                return k, v, lr
        # each probe is written at the segment's mean rate times kv_lr_scale
        rate = lr.float().mean(dim=1, keepdim=True) * cfg.kv_lr_scale * ok[..., None]
        return (
            torch.cat([k, p.to(k.dtype)], dim=1),
            torch.cat([v, y.to(v.dtype)], dim=1),
            torch.cat([lr, rate.to(lr.dtype)], dim=1),
        )

    def augment_write(self, k: torch.Tensor, v: torch.Tensor, lr: torch.Tensor, state: Optional[FastWeightState] = None):
        if self.cfg.mode == "pseudo":
            return self._pseudo_pairs(k, v, lr, state)
        if self.cfg.mode not in ("oracle_kv", "scored_kv") or self.empty:
            return k, v, lr
        with torch.no_grad():
            if self.cfg.mode == "scored_kv":
                idx = self.scores.topk(min(self.cfg.num_samples, self.scores.shape[1]), dim=1).indices
            else:
                idx = self._sample_indices(self._selection_logits(self.keys, k))
            d = self.keys.shape[-1]
            rk = torch.gather(self.keys, 1, idx[..., None].expand(-1, -1, d))
            rv = torch.gather(self.values, 1, idx[..., None].expand(-1, -1, self.values.shape[-1]))
            rl = torch.gather(self.lrs, 1, idx[..., None].expand(-1, -1, self.lrs.shape[-1]))
        rl = rl * self.cfg.kv_lr_scale
        return (
            torch.cat([k, rk.to(k.dtype)], dim=1),
            torch.cat([v, rv.to(v.dtype)], dim=1),
            torch.cat([lr, rl.to(lr.dtype)], dim=1),
        )

    def correct(
        self,
        prev: FastWeightState,
        cand: FastWeightState,
        current_keys: torch.Tensor,
        ref_norms: Sequence[torch.Tensor],
    ) -> FastWeightState:
        if self.cfg.mode not in ("oracle_key", "gmm"):
            return cand
        keys = self.sample_keys(current_keys)
        if keys is None:
            return cand
        keys = keys.to(cand.w0.dtype)

        with torch.no_grad():
            v_ref = fw_apply(*prev.weights(), keys)
        residual = v_ref - fw_apply(*cand.weights(), keys)

        scale = self.cfg.strength / keys.shape[1]
        lr = torch.full_like(keys[..., :1], scale)
        a0, a1, a2 = fw_descent_direction(*cand.weights(), keys, residual, lr, lr, lr)

        new = []
        for a, w_c, w_p, ref in zip((a0, a1, a2), cand.weights(), prev.weights(), ref_norms):
            if self.cfg.clip_ratio > 0:
                budget = self.cfg.clip_ratio * (w_c - w_p).flatten(1).norm(dim=1)
                size = a.flatten(1).norm(dim=1)
                a = a * (budget / (size + 1e-8)).clamp(max=1.0)[:, None, None]
            new.append(renorm_rows(w_c + a, ref))
        return FastWeightState(new[0], new[1], new[2], cand.m0, cand.m1, cand.m2, cand.replay, cand.raw)

    # ---------------------------------------------------------------- observe
    @torch.no_grad()
    def observe(self, k: torch.Tensor, v: torch.Tensor, lr: torch.Tensor, score: Optional[torch.Tensor] = None):
        mode = self.cfg.mode
        if mode == "none":
            return
        k = k.detach().float()
        if mode == "pseudo":
            if self.cfg.probe_src == "query":
                return
            score = lr.detach().float().mean(dim=-1) if score is None else score.detach().float()
            idx = score.topk(min(self.cfg.keep_per_segment, k.shape[1]), dim=1).indices
            k, score = torch.gather(k, 1, idx[..., None].expand(-1, -1, k.shape[-1])), torch.gather(score, 1, idx)
            if self.keys is not None:
                k, score = torch.cat([self.keys, k], dim=1), torch.cat([self.scores, score], dim=1)
            if k.shape[1] > self.cfg.bank_size:
                idx = score.topk(self.cfg.bank_size, dim=1).indices
                k, score = torch.gather(k, 1, idx[..., None].expand(-1, -1, k.shape[-1])), torch.gather(score, 1, idx)
            self.keys, self.scores = k, score
            return
        if mode == "scored_kv":
            v, lr = v.detach().float(), lr.detach().float()
            score = lr.mean(dim=-1) if score is None else score.detach().float()
            m = min(self.cfg.keep_per_segment, k.shape[1])
            idx = score.topk(m, dim=1).indices

            def take(t):
                return torch.gather(t, 1, idx[..., None].expand(-1, -1, t.shape[-1]))

            k, v, lr, score = take(k), take(v), take(lr), torch.gather(score, 1, idx)
            if self.keys is not None:
                k, v = torch.cat([self.keys, k], dim=1), torch.cat([self.values, v], dim=1)
                lr, score = torch.cat([self.lrs, lr], dim=1), torch.cat([self.scores, score], dim=1)
            if k.shape[1] > self.cfg.bank_size:
                idx = score.topk(self.cfg.bank_size, dim=1).indices
                k, v, lr, score = take(k), take(v), take(lr), torch.gather(score, 1, idx)
            self.keys, self.values, self.lrs, self.scores = k, v, lr, score
            return
        if mode in ("oracle_kv", "oracle_key"):
            self.keys = k if self.keys is None else torch.cat([self.keys, k], dim=1)
            if mode == "oracle_kv":
                v = v.detach().float()
                lr = lr.detach().float()
                self.values = v if self.values is None else torch.cat([self.values, v], dim=1)
                self.lrs = lr if self.lrs is None else torch.cat([self.lrs, lr], dim=1)
            n = self.keys.shape[1]
            if n > self.cfg.bank_size:
                keep = torch.randperm(n, device=k.device)[: self.cfg.bank_size].sort().values
                self.keys = self.keys[:, keep]
                if mode == "oracle_kv":
                    self.values = self.values[:, keep]
                    self.lrs = self.lrs[:, keep]
            return
        self._gmm_observe(k)

    def _gmm_observe(self, k: torch.Tensor):
        bh, n, d = k.shape
        kc = self.cfg.gmm_components
        if self.mu is None:
            pick = torch.randint(0, n, (kc,), device=k.device) if n < kc else torch.randperm(n, device=k.device)[:kc]
            self.mu = k[:, pick].clone()
            self.ex2 = self.mu.pow(2).clone()
            self.count = torch.zeros(bh, kc, device=k.device)

        assign = torch.einsum("bnd,bkd->bnk", k, l2_norm(self.mu)).argmax(dim=-1)
        onehot = torch.nn.functional.one_hot(assign, kc).float()
        n_c = onehot.sum(dim=1)
        sum_c = torch.einsum("bnk,bnd->bkd", onehot, k)
        sq_c = torch.einsum("bnk,bnd->bkd", onehot, k.pow(2))

        total = self.count + n_c
        denom = total.clamp_min(1.0)[..., None]
        old = self.count[..., None]
        seen = (total > 0)[..., None]
        self.mu = torch.where(seen, (old * self.mu + sum_c) / denom, self.mu)
        self.ex2 = torch.where(seen, (old * self.ex2 + sq_c) / denom, self.ex2)
        self.count = total
