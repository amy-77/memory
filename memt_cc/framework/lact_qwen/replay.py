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

The reference outputs are taken from the weights right before the current write.
At those weights the replay loss has zero gradient, so the correction must be a
separate step taken on the candidate weights after the write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .fast_weight import FastWeightState, fw_apply, fw_descent_direction, l2_norm, renorm_rows

REPLAY_MODES = ("none", "oracle_kv", "oracle_key", "gmm")


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

    def __post_init__(self):
        if self.mode not in REPLAY_MODES:
            raise ValueError(f"replay mode must be one of {REPLAY_MODES}, got {self.mode}")
        if self.selection not in ("uniform", "coverage"):
            raise ValueError("selection must be 'uniform' or 'coverage'")


class ReplayBank:
    """Per-sequence replay state. Lives for one forward pass or one dialogue session."""

    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.lrs: Optional[torch.Tensor] = None
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
    def augment_write(self, k: torch.Tensor, v: torch.Tensor, lr: torch.Tensor):
        if self.cfg.mode != "oracle_kv" or self.empty:
            return k, v, lr
        with torch.no_grad():
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
    def observe(self, k: torch.Tensor, v: torch.Tensor, lr: torch.Tensor):
        mode = self.cfg.mode
        if mode == "none":
            return
        k = k.detach().float()
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
