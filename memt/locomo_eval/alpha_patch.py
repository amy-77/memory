"""Per-token write coefficient alpha in [0, 1] for the MemT matrix memory, at inference only.

The delta-rule step of a token is  S <- D S + u (z - u^T D S)^T  with u = sqrt(beta) k, z = omega v.
Scaling the step by alpha keeps it a delta rule:  u -> sqrt(alpha) u,  z -> sqrt(alpha) z,  i.e. the token erases
and writes alpha times as strongly. With ALPHA["decay"] the forget step is scaled too (lambda -> alpha lambda),
so alpha = 0 leaves the state exactly unchanged (no write, no erase, no decay), as LiveMem's write_mask does.
alpha = 1 everywhere gives the unpatched update (checked by alpha_selftest()).

Set ALPHA["w"] = tensor [tokens of the chunk being written] before each forward_chunk(update=True); None = all 1.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

from lact_qwen import memory_branch as MB
from lact_qwen.matrix_memory import MAX_DECAY, delta_write
from lact_qwen.fast_weight import l2_norm

ALPHA = {"w": None, "decay": True}
_orig_write = MB.LaCTMemoryBranch._matrix_write


def _matrix_write(self, state, x):
    w = ALPHA["w"]
    if w is None:
        return _orig_write(self, state, x)
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
        a = w.to(x.device, torch.float32).view(1, -1, 1).expand(erase.shape[0], -1, 1)   # [(b h), l, 1], b = 1
        u = erase * l2_norm(k) * a.sqrt()
        z = write * v * a.sqrt()
        lam = -decay.clamp(max=MAX_DECAY)
        if ALPHA["decay"]:
            lam = lam * a
        return delta_write(state, u, z, lam, self.cfg.write_block)


def install():
    MB.LaCTMemoryBranch._matrix_write = _matrix_write


def alpha_selftest(branch, x):
    """alpha = 1 must reproduce the unpatched write."""
    st = torch.randn(branch.cfg.num_heads, branch.cfg.head_dim, branch.cfg.head_dim, device=x.device) * 0.1
    ALPHA["w"] = None
    ref = _orig_write(branch, st, x)
    ALPHA["w"] = torch.ones(x.shape[1])
    got = _matrix_write(branch, st, x)
    ALPHA["w"] = None
    return float((got - ref).abs().max())
