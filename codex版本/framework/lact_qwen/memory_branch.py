from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .fast_weight import FastWeightState, UpdateConfig, fixed_segments, fw_apply, l2_norm, segmented_lact
from .matrix_memory import MAX_DECAY, delta_write
from .replay import ReplayBank, ReplayConfig


def inv_softplus(x: float) -> float:
    return x + math.log(-math.expm1(-x))


@dataclass
class MemoryConfig:
    num_heads: int = 4
    head_dim: int = 512
    inter_multi: float = 1.0
    base_lr: float = 0.01
    use_muon: bool = True
    use_momentum: bool = True
    chunk_size: int = 2048
    gate_init: float = 0.0
    layers: str = "all"
    # fast-weight normalisation of the write, see UpdateConfig.norm
    norm: str = "post"
    # True: the branch outputs f_W(q) - f_W_init(q), i.e. only what the writes changed.
    # It is exactly zero before the first write, so it cannot act as a fixed adapter.
    delta_readout: bool = False
    # Initial fast weights w0 / w2.
    #   "random":           full-rank Gaussian (the original choice of this repo).
    #   "lowrank_identity": W = W_left @ W_right + 0.5 * I with rank fw_rank, as in the official
    #                       LaCT configs (w0_w2_low_rank=32, fw_init_gain=0.5).
    fw_init: str = "random"
    fw_rank: int = 32
    fw_init_gain: float = 0.5
    # How the normalised memory output is scaled before o_proj.
    #   "gate":  one learned scalar per head, initialised to gate_init (0 keeps the base model exact).
    #   "token": silu(Linear(hidden)) per token and head, as LaCT's learnable_ttt_scale. It is not
    #            zero at initialisation, so a fresh branch perturbs the base model.
    out_scale: str = "gate"
    # Opt-in MemT-Nano output gate: two linear factors, per-token/per-channel SoftCapSiLU.
    output_gate_rank: int = 128
    output_gate_softcap: float = 2.0
    # Rotate the fast-weight q / k by token position (RoPE-style, pairs over the fast-weight head
    # dimension). LaCT does this; here it is off by default because it ties a key to its distance
    # from the query.
    rope: bool = False
    rope_theta: float = 1_000_000.0
    # Initial q/k/v projection.
    #   "random":    small Gaussian (the original choice of this repo).
    #   "attention": copy the layer's own q_proj / k_proj / v_proj (KV heads repeated as in
    #                grouped-query attention), so the memory starts out storing what the base
    #                attention uses. Needs num_heads * head_dim == attention heads * head dim.
    qkv_init: str = "random"
    # Activation on the q/k/v projection: "silu" (LaCT) or "none" (keeps a copied projection intact).
    qkv_act: str = "silu"
    # Where the output joins the base model.
    #   "post":       own output projection, added next to the attention output (original).
    #   "pre_o_proj": no own projection; the output is added to the input of the attention's
    #                 frozen o_proj, like extra attention-head outputs (as in LaCT and MemT).
    inject: str = "post"
    # Only with inject == 'pre_o_proj'. The attention output before o_proj differs in size by a factor
    # of about 50 between the first and last layers of Qwen3-0.6B (RMS 0.06 to 3.4), so one gate value
    # cannot suit them all.
    #   "none":      output = gate * normalised memory readout.
    #   "attention": that, times the RMS of the attention output of the same token and head group
    #                (no gradient through the RMS); the gate is then a size relative to attention.
    out_ref: str = "none"
    # LoRA on the frozen base layer's own linear maps (0 = off). The low-rank factors are stored
    # inside the memory branch, so they are trained, saved and loaded with it. B starts at zero,
    # so a fresh model still reproduces the base model.
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    # What holds the memory.
    #   "mlp":    LaCT. A small MLP per head; a write is one gradient step per chunk of chunk_size tokens.
    #   "matrix": MemT-style. One [head_dim, head_dim] matrix per head, zero at the start; every token
    #             is written into it with a gated delta rule (see matrix_memory.py). The tokens of a
    #             chunk read the matrix as it was at the start of the chunk. No write, no output.
    core: str = "mlp"
    # core == 'matrix': tokens per solve inside a write (speed/memory only, the result is the same).
    write_block: int = 256
    # Optional compatible write kernel. Reads remain at the storage-step boundary.
    matrix_backend: str = "torch"
    # Algebraically equivalent erase gate without sqrt(sigmoid(x)) underflow in backward.
    stable_matrix_gates: bool = False
    # core == 'matrix': the forget gate is Linear(hidden, forget_rank) -> Linear(forget_rank, heads * head_dim);
    # at the start each key channel forgets with its own half-life, drawn log-uniformly from this range (tokens).
    forget_rank: int = 64
    forget_halflife_min: float = 2048.0
    forget_halflife_max: float = 32768.0
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryConfig":
        d = dict(d)
        replay = ReplayConfig(**d.pop("replay", {}))
        return cls(replay=replay, **d)

    def layer_indices(self, num_layers: int) -> list[int]:
        spec = self.layers
        if spec == "all":
            return list(range(num_layers))
        if spec.startswith("every:"):
            step = int(spec.split(":", 1)[1])
            return [i for i in range(num_layers) if i % step == step - 1]
        return sorted(int(x) for x in spec.split(","))


class MemoryContext:
    """Shared per-forward instructions for every memory branch in the model.

    segments / update_flags: how the sequence is cut into write units (shared across
    the batch). If unset, the sequence is cut into fixed chunks of chunk_size.
    session: when True, branches start from and write back to self.states, so a
    dialogue can be fed turn by turn.
    position_offset: position of the first token of this forward within the whole
    stream; only used when MemoryConfig.rope is on (set it when feeding turn by turn).
    """

    def __init__(self):
        self.segments: Optional[list[tuple[int, int]]] = None
        self.update_flags: Optional[list[bool]] = None
        self.session: bool = False
        # per layer: a FastWeightState (core 'mlp') or the state matrices [B * H, d, d] (core 'matrix')
        self.states: dict[int, FastWeightState] = {}
        self.position_offset: int = 0

    def set_segments(self, segments, update_flags=None):
        self.segments = list(segments)
        self.update_flags = list(update_flags) if update_flags is not None else [True] * len(self.segments)

    def clear_segments(self):
        self.segments = None
        self.update_flags = None

    def start_session(self):
        self.session = True
        self.states = {}
        self.position_offset = 0

    def end_session(self):
        self.session = False
        self.states = {}
        self.position_offset = 0


def rotate_by_position(t: torch.Tensor, offset: int, theta: float) -> torch.Tensor:
    """RoPE over the last dimension. t: [BH, L, d] with even d."""
    length, d = t.shape[1], t.shape[2]
    pos = torch.arange(length, device=t.device, dtype=torch.float32) + float(offset)
    inv_freq = theta ** (-torch.arange(0, d, 2, device=t.device, dtype=torch.float32) / d)
    angle = pos[:, None] * inv_freq[None, :]
    cos, sin = angle.cos().to(t.dtype), angle.sin().to(t.dtype)
    a, b = t[..., 0::2], t[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class LaCTMemoryBranch(nn.Module):
    def __init__(self, hidden_size: int, cfg: MemoryConfig, layer_idx: int, ctx: MemoryContext):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.ctx = ctx
        h, d = cfg.num_heads, cfg.head_dim
        dh = int(d * cfg.inter_multi)

        self.qkv = nn.Linear(hidden_size, 3 * h * d, bias=False)
        if cfg.core not in ("mlp", "matrix"):
            raise ValueError("core must be 'mlp' or 'matrix'")
        self.matrix = cfg.core == "matrix"
        if self.matrix:
            if cfg.rope or cfg.replay.mode != "none":
                raise ValueError("core='matrix' does not support rope or replay")
            self._init_matrix(hidden_size)
        else:
            self._init_mlp(hidden_size, dh)

        self.out_norm = nn.RMSNorm(d, eps=1e-5)
        self._init_output(hidden_size)
        nn.init.normal_(self.qkv.weight, std=0.02)

    def _init_matrix(self, hidden_size: int):
        cfg = self.cfg
        n = cfg.num_heads * cfg.head_dim
        # erase / write gates, one value per head and channel; they start equal (about 0.5 everywhere)
        self.erase_proj = nn.Linear(hidden_size, n, bias=False)
        self.write_proj = nn.Linear(hidden_size, n, bias=False)
        nn.init.normal_(self.erase_proj.weight, std=0.02)
        with torch.no_grad():
            self.write_proj.weight.copy_(self.erase_proj.weight)
        # forget gate: decay = exp(forget_scale[head]) * softplus(forget_up(forget_down(x))) per key channel.
        # forget_up starts with zero weight, so at first the decay is the same for every token and is
        # set by the bias: ln 2 / half-life.
        self.forget_down = nn.Linear(hidden_size, cfg.forget_rank, bias=False)
        self.forget_up = nn.Linear(cfg.forget_rank, n)
        self.forget_scale = nn.Parameter(torch.zeros(cfg.num_heads))
        nn.init.normal_(self.forget_down.weight, std=0.02)
        lo, hi = math.log(cfg.forget_halflife_min), math.log(cfg.forget_halflife_max)
        decay = math.log(2.0) / torch.exp(torch.rand(n) * (hi - lo) + lo)
        with torch.no_grad():
            self.forget_up.weight.zero_()
            self.forget_up.bias.copy_(torch.log(torch.expm1(decay)))

    def _init_mlp(self, hidden_size: int, dh: int):
        cfg = self.cfg
        h, d = cfg.num_heads, cfg.head_dim
        self.lr_proj = nn.Linear(hidden_size, 3 * h, bias=False)
        self.momentum_proj = nn.Linear(hidden_size, h) if cfg.use_momentum else None
        nn.init.normal_(self.lr_proj.weight, std=0.02)

        if cfg.fw_init == "random":
            self.w0 = nn.Parameter(torch.randn(h, dh, d) / math.sqrt(d))
            self.w2 = nn.Parameter(torch.randn(h, dh, d) / math.sqrt(d))
        elif cfg.fw_init == "lowrank_identity":
            r, g = cfg.fw_rank, cfg.fw_init_gain
            self.w0_left = nn.Parameter(torch.randn(h, dh, r) * g / math.sqrt(r))
            self.w0_right = nn.Parameter(torch.randn(h, r, d) * g / math.sqrt(d))
            self.w2_left = nn.Parameter(torch.randn(h, dh, r) * g / math.sqrt(r))
            self.w2_right = nn.Parameter(torch.randn(h, r, d) * g / math.sqrt(d))
            self.register_buffer("fw_eye", 0.5 * torch.eye(dh, d).unsqueeze(0), persistent=False)
        else:
            raise ValueError("fw_init must be 'random' or 'lowrank_identity'")
        self.w1 = nn.Parameter(torch.randn(h, d, dh) / math.sqrt(dh))
        self.base_lr_inv = inv_softplus(cfg.base_lr)
        if cfg.norm not in ("post", "pre"):
            raise ValueError("norm must be 'post' or 'pre'")
        if cfg.rope and d % 2:
            raise ValueError("rope needs an even head_dim")
        self.update_cfg = UpdateConfig(use_muon=cfg.use_muon, use_momentum=cfg.use_momentum, norm=cfg.norm)

    def _init_output(self, hidden_size: int):
        cfg = self.cfg
        h, d = cfg.num_heads, cfg.head_dim
        if cfg.out_scale == "gate":
            self.gate = nn.Parameter(torch.full((h,), float(cfg.gate_init)))
        elif cfg.out_scale == "token":
            self.scale_proj = nn.Linear(hidden_size, h)
            nn.init.normal_(self.scale_proj.weight, std=0.02)
            nn.init.zeros_(self.scale_proj.bias)
        elif cfg.out_scale == "memt":
            if cfg.output_gate_rank <= 0 or cfg.output_gate_softcap <= 0:
                raise ValueError("MemT gate rank and softcap must be positive")
            self.output_gate_down = nn.Linear(hidden_size, cfg.output_gate_rank, bias=False)
            self.output_gate_up = nn.Linear(cfg.output_gate_rank, h * d, bias=False)
            nn.init.normal_(self.output_gate_down.weight, std=0.02)
            nn.init.normal_(self.output_gate_up.weight, std=0.001)
        else:
            raise ValueError("out_scale must be 'gate', 'token', or 'memt'")
        self._last_scale: Optional[torch.Tensor] = None

        if cfg.out_ref not in ("none", "attention"):
            raise ValueError("out_ref must be 'none' or 'attention'")
        if cfg.out_ref == "attention" and cfg.inject != "pre_o_proj":
            raise ValueError("out_ref='attention' needs inject='pre_o_proj'")
        if cfg.inject not in ("post", "pre_o_proj"):
            raise ValueError("inject must be 'post' or 'pre_o_proj'")
        if cfg.qkv_act not in ("silu", "none"):
            raise ValueError("qkv_act must be 'silu' or 'none'")
        if cfg.qkv_init not in ("random", "attention"):
            raise ValueError("qkv_init must be 'random' or 'attention'")
        if cfg.inject == "post":
            self.o_proj = nn.Linear(h * d, hidden_size, bias=False)
        # with inject == 'pre_o_proj': output of this forward, picked up by the hook on the attention o_proj
        self.pending: Optional[torch.Tensor] = None
        if cfg.inject == "post":
            nn.init.normal_(self.o_proj.weight, std=0.02)

    def fast_weights(self):
        """Initial (slow) fast weights w0, w1, w2 as float32, each [H, out, in]."""
        if self.cfg.fw_init == "lowrank_identity":
            w0 = self.w0_left.float() @ self.w0_right.float() + self.fw_eye
            w2 = self.w2_left.float() @ self.w2_right.float() + self.fw_eye
        else:
            w0, w2 = self.w0.float(), self.w2.float()
        return w0, self.w1.float(), w2

    def scale_stat(self) -> torch.Tensor:
        """Mean absolute output scale, for logging (the gate, or the last per-token scales)."""
        if self.cfg.out_scale == "gate":
            return self.gate.detach().abs().mean()
        if self._last_scale is None:
            return torch.zeros((), device=self.qkv.weight.device)
        return self._last_scale

    def _initial_state(self, batch: int, weights=None) -> FastWeightState:
        w0, w1, w2 = weights if weights is not None else self.fast_weights()
        replay = ReplayBank(self.cfg.replay) if self.cfg.replay.mode != "none" else None
        return FastWeightState(w0.repeat(batch, 1, 1), w1.repeat(batch, 1, 1), w2.repeat(batch, 1, 1), replay=replay)

    def _ref_norms(self, batch: int, weights=None):
        w0, w1, w2 = weights if weights is not None else self.fast_weights()
        return tuple(w.norm(dim=2, keepdim=True).repeat(batch, 1, 1) for w in (w0, w1, w2))

    def _segments(self, length: int):
        ctx = self.ctx
        if ctx.segments is None:
            segments = fixed_segments(length, self.cfg.chunk_size)
            return segments, [True] * len(segments)
        if ctx.segments[-1][1] != length:
            raise ValueError(f"segments end at {ctx.segments[-1][1]} but sequence has {length} tokens")
        return ctx.segments, ctx.update_flags

    def _matrix_write(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Write the tokens x [b, l, hidden] into state [b * h, d, d]."""
        h, d = self.cfg.num_heads, self.cfg.head_dim
        kv = F.linear(x, self.qkv.weight[h * d :])
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
            z, lam = write * v, -decay.clamp(max=MAX_DECAY)
            if self.cfg.matrix_backend == "fla_kda":
                from .kda_backend import kda_write
                return kda_write(state, u, z, lam)
            if self.cfg.matrix_backend != "torch":
                raise ValueError(f"unknown matrix backend: {self.cfg.matrix_backend}")
            return delta_write(state, u, z, lam, self.cfg.write_block)

    def _read_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Chunk by chunk: read the state left by the earlier chunks, then write this chunk into it."""
        b, length, _ = x.shape
        h, d = self.cfg.num_heads, self.cfg.head_dim
        q = F.linear(x, self.qkv.weight[: h * d])
        if self.cfg.qkv_act == "silu":
            q = F.silu(q)
        q = l2_norm(rearrange(q.float(), "b l (h d) -> (b h) l d", h=h, d=d))

        ctx = self.ctx
        segments, flags = self._segments(length)
        state = ctx.states.get(self.layer_idx) if ctx.session else None
        if state is None:
            state = torch.zeros(b * h, d, d, device=x.device, dtype=torch.float32)
        outputs = []
        for (s, e), do_update in zip(segments, flags):
            with torch.autocast(device_type=x.device.type, enabled=False):
                outputs.append(q[:, s:e] @ state / math.sqrt(d))
            if do_update:
                state = self._matrix_write(state, x[:, s:e])
        if ctx.session:
            ctx.states[self.layer_idx] = state.detach()
        return torch.cat(outputs, dim=1)

    def _read_mlp(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        h, d = self.cfg.num_heads, self.cfg.head_dim

        qkv = self.qkv(x)
        if self.cfg.qkv_act == "silu":
            qkv = F.silu(qkv)
        q, k, v = rearrange(qkv, "b l (n h d) -> n (b h) l d", n=3, h=h, d=d)
        q, k = l2_norm(q), l2_norm(k)

        ctx = self.ctx
        if self.cfg.rope:
            q = rotate_by_position(q, ctx.position_offset, self.cfg.rope_theta)
            k = rotate_by_position(k, ctx.position_offset, self.cfg.rope_theta)

        lr = self.lr_proj(x).float()
        lr = F.softplus(lr + self.base_lr_inv)
        lr = rearrange(lr, "b l (h n) -> (b h) l n", h=h, n=3)

        momentum = None
        if self.momentum_proj is not None:
            momentum = torch.sigmoid(self.momentum_proj(x).float())
            momentum = rearrange(momentum, "b l h -> (b h) l 1")

        segments, flags = self._segments(length)

        weights = self.fast_weights()
        state = ctx.states.get(self.layer_idx) if ctx.session else None
        if state is None:
            state = self._initial_state(b, weights)

        out, state = segmented_lact(
            state, q, k, v, lr, momentum, segments, flags, self._ref_norms(b, weights), self.update_cfg
        )
        if ctx.session:
            ctx.states[self.layer_idx] = state.detach()
        if self.cfg.delta_readout:
            init = self._initial_state(b, weights)
            out = out - fw_apply(*init.weights(), q)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        h, d = self.cfg.num_heads, self.cfg.head_dim
        out = self._read_matrix(x) if self.matrix else self._read_mlp(x)

        out = self.out_norm(out.float())
        out = rearrange(out, "(b h) l d -> b l h d", b=b, h=h)
        if self.cfg.out_scale == "token":
            scale = F.silu(self.scale_proj(x).float())  # [b, l, h]
            self._last_scale = scale.detach().abs().mean()
            out = out * scale[..., None]
        elif self.cfg.out_scale == "memt":
            gate = self.output_gate_up(self.output_gate_down(x)).float()
            cap = self.cfg.output_gate_softcap
            scale = cap * torch.tanh(F.silu(gate) / cap)
            self._last_scale = scale.detach().abs().mean()
            out = out * scale.view(b, length, h, d)
        else:
            out = out * self.gate.float()[:, None]
        out = out.reshape(b, length, h * d)
        if self.cfg.inject == "pre_o_proj":
            return out.to(x.dtype)
        return self.o_proj(out).to(x.dtype)
