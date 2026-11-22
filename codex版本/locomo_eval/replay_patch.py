"""Robins-style pseudo-rehearsal for the MemT matrix memory, at inference only.

Before a chunk is written, the probe queries q~ (the memory queries of the QUESTION's own tokens at this layer,
not random vectors) read the current state: y~ = q~ S_old. The chunk's (k, v) pairs and the rehearsal pairs
(q~ -> y~) then go into ONE delta-rule update: the rehearsal pairs are appended after the chunk's tokens as extra
pseudo-tokens with key u = sqrt(a) q~ (unit length), value z = sqrt(a) y~ and no decay. With a = 1 the update
leaves q~^T S_new = y~ for the last pair (exactly for orthogonal probes), i.e. reading with the question's queries
returns what it returned before the chunk was written; with a < 1 it moves the read-out a of the way back.

Per head, the r probes are the question tokens whose read-out is strongest on the current state (the content the
question already retrieves). No rehearsal while the state is still empty (nothing to protect yet).

Usage: REPLAY["q"] = {layer_idx: unit queries [b*h, n_question_tokens, d]}; REPLAY["k"] = r; REPLAY["alpha"] = a.
Capture the question queries with REPLAY["capture"] = (start, end) during a forward of the chunk that holds the
question (positions relative to that chunk); they land in REPLAY["captured"][layer_idx].
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

from lact_qwen import memory_branch as MB
from lact_qwen.matrix_memory import MAX_DECAY, delta_write
from lact_qwen.fast_weight import l2_norm

REPLAY = {"q": None, "k": 16, "alpha": 1.0, "capture": None, "captured": {}, "stats": {"writes": 0, "replayed": 0,
          "restore_err": 0.0, "restore_n": 0}}

_orig_read = MB.LaCTMemoryBranch._read_matrix


def _question_queries(self, xq):
    h, d = self.cfg.num_heads, self.cfg.head_dim
    q = F.linear(xq, self.qkv.weight[: h * d])
    if self.cfg.qkv_act == "silu":
        q = F.silu(q)
    return l2_norm(rearrange(q.float(), "b l (h d) -> (b h) l d", h=h, d=d))


def _read_matrix(self, x):
    cap = REPLAY["capture"]
    if cap is not None:
        s, e = cap
        REPLAY["captured"][self.layer_idx] = _question_queries(self, x[:, s:e].detach())
    return _orig_read(self, x)


def _matrix_write(self, state, x):
    h, d = self.cfg.num_heads, self.cfg.head_dim
    kv = F.linear(x, self.qkv.weight[h * d:])
    if self.cfg.qkv_act == "silu":
        kv = F.silu(kv)
    k, v = rearrange(kv.float(), "b l (n h d) -> n (b h) l d", n=2, h=h, d=d)
    erase_logits = self.erase_proj(x).float()
    erase = (torch.exp(0.5 * F.logsigmoid(erase_logits)) if self.cfg.stable_matrix_gates
             else torch.sigmoid(erase_logits).sqrt())
    write = torch.sigmoid(self.write_proj(x).float())
    decay = F.softplus(self.forget_up(self.forget_down(x)).float())
    decay = decay.view(*decay.shape[:2], h, d) * self.forget_scale.float().exp()[:, None]
    erase, write, decay = (rearrange(t, "b l (h d) -> (b h) l d", h=h, d=d) for t in (erase, write, decay.flatten(2)))
    with torch.autocast(device_type=x.device.type, enabled=False):
        u = erase * l2_norm(k)
        z = write * v
        lam = -decay.clamp(max=MAX_DECAY)
        qd = REPLAY["q"].get(self.layer_idx) if REPLAY["q"] else None
        REPLAY["stats"]["writes"] += 1
        if qd is not None and REPLAY["k"] > 0 and float(state.abs().max()) > 0:
            y = qd @ state                                             # [bh, nq, dv] read-out before the write
            r = min(REPLAY["k"], qd.shape[1])
            idx = y.norm(dim=-1).topk(r, dim=-1).indices              # strongest read-outs per head
            qs = qd.gather(1, idx[..., None].expand(-1, -1, qd.shape[-1]))
            ys = y.gather(1, idx[..., None].expand(-1, -1, y.shape[-1]))
            a = math.sqrt(REPLAY["alpha"])
            u = torch.cat([u, a * qs], dim=1)
            z = torch.cat([z, a * ys], dim=1)
            lam = torch.cat([lam, torch.zeros_like(qs)], dim=1)
            new = delta_write(state, u, z, lam, self.cfg.write_block)
            err = ((qs @ new) - ys).norm() / ys.norm().clamp_min(1e-9)
            REPLAY["stats"]["replayed"] += 1
            REPLAY["stats"]["restore_err"] += float(err); REPLAY["stats"]["restore_n"] += 1
            return new
        return delta_write(state, u, z, lam, self.cfg.write_block)


def install():
    MB.LaCTMemoryBranch._read_matrix = _read_matrix
    MB.LaCTMemoryBranch._matrix_write = _matrix_write
