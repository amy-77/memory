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
from .select import SelectConfig, select_weights


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
        # Diagnostic: is the memory about this text, or would any text do?
        # recorded: set to {} to collect, per layer, the memory each chunk was read with.
        # borrowed: such a collection from another forward; every chunk then reads that memory
        # instead of its own and nothing is written.
        self.recorded: Optional[dict[int, list]] = None
        self.borrowed: Optional[dict[int, list]] = None
        # Selective writing (see select.py). query_span: token range [start, end) that holds the
        # question(s); query_mask [B, end - start]: True on question tokens (answers must be False).
        self.select: Optional[SelectConfig] = None
        self.query_span: Optional[tuple[int, int]] = None
        self.query_mask: Optional[torch.Tensor] = None
        self.oracle_weight: Optional[torch.Tensor] = None
        # Question given before the history: capture_query=True makes every branch store the read vectors
        # of the tokens it sees in query_q[layer] ([BH, Nq, d]) and write nothing; a later forward scores
        # its segments against them. query_prefix: the question is the first N tokens of every segment.
        self.capture_query: bool = False
        # capture_span=(s, e): an ordinary forward also stores the read vectors of tokens [s, e) in span_q[layer]
        # (the question as it is read in context, after the history). bank_scores [B, L]: per-token relevance
        # used only to choose which keys enter the replay bank (does not change the write itself).
        self.capture_span: Optional[tuple[int, int]] = None
        self.span_q: dict[int, torch.Tensor] = {}
        self.bank_scores: Optional[torch.Tensor] = None
        self.query_q: dict[int, torch.Tensor] = {}
        self.query_prefix: int = 0
        # Diagnostic: set to {} to collect, per layer, the raw query score of every token [B, L]
        # (needs query_q and query_mask; nothing else changes).
        self.debug_scores: Optional[dict[int, torch.Tensor]] = None
        self.last_weights: dict[int, torch.Tensor] = {}  # per layer, the multipliers of the last forward
        self.last_gains: dict[int, torch.Tensor] = {}

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
        else:
            raise ValueError("out_scale must be 'gate' or 'token'")
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
        lf = getattr(ctx, "layer_update_flags", None)  # {layer: flags}: a layer's own choice of which segments it writes
        if lf is not None and self.layer_idx in lf and not ctx.capture_query:
            return ctx.segments, lf[self.layer_idx]
        return ctx.segments, ctx.update_flags

    def _matrix_write(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Write the tokens x [b, l, hidden] into state [b * h, d, d]."""
        h, d = self.cfg.num_heads, self.cfg.head_dim
        kv = F.linear(x, self.qkv.weight[h * d :])
        if self.cfg.qkv_act == "silu":
            kv = F.silu(kv)
        k, v = rearrange(kv.float(), "b l (n h d) -> n (b h) l d", n=2, h=h, d=d)
        # sqrt(sigmoid(a)) written as exp(logsigmoid(a) / 2): the plain form has an infinite slope
        # when the sigmoid underflows to 0, which turns the whole gradient into NaN.
        erase = torch.exp(0.5 * F.logsigmoid(self.erase_proj(x).float()))
        write = torch.sigmoid(self.write_proj(x).float())
        decay = F.softplus(self.forget_up(self.forget_down(x)).float())
        decay = decay.view(*decay.shape[:2], h, d) * self.forget_scale.float().exp()[:, None]
        erase, write, decay = (rearrange(t, "b l (h d) -> (b h) l d", h=h, d=d) for t in (erase, write, decay.flatten(2)))
        with torch.autocast(device_type=x.device.type, enabled=False):
            u = erase * l2_norm(k)
            return delta_write(state, u, write * v, -decay.clamp(max=MAX_DECAY), self.cfg.write_block)

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
        borrowed = None if ctx.borrowed is None else ctx.borrowed[self.layer_idx]
        for i, ((s, e), do_update) in enumerate(zip(segments, flags)):
            if borrowed is not None:
                state = borrowed[i]
            if ctx.recorded is not None:
                ctx.recorded.setdefault(self.layer_idx, []).append(state)
            with torch.autocast(device_type=x.device.type, enabled=False):
                outputs.append(q[:, s:e] @ state / math.sqrt(d))
            if do_update and borrowed is None:
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

        if ctx.capture_query:
            ctx.query_q[self.layer_idx] = q.detach()
        kp = getattr(ctx, "capture_key_pos", None)  # positions whose keys are stored per layer (key-identity diagnostic)
        if kp is not None and not ctx.capture_query:
            ctx.key_store[self.layer_idx] = k[:, kp.to(k.device)].detach().float().cpu()
        if ctx.capture_span is not None and not ctx.capture_query:
            ctx.span_q[self.layer_idx] = q[:, ctx.capture_span[0] : ctx.capture_span[1]].detach()
            segments, flags = [(0, length)], [False]

        if ctx.debug_scores is not None and not ctx.capture_query and self.layer_idx in ctx.query_q:
            # Audit of "can the question find the keys that hold its answer": several similarities between the
            # question's read vectors and every token's key, each [B, L] (see scripts/audit_scores.py).
            from .select import last_tokens

            qa = ctx.query_q[self.layer_idx].to(q.device).float()  # question alone [BH, Nq, d]
            qm = ctx.query_mask.to(q.device)
            w0, _, w2 = self.fast_weights()
            w0, w2 = w0.float(), w2.float()

            def phi(x):  # the feature map through which a LaCT write to w1 is later read: f(x) = w1 (silu(w0 x) * (w2 x))
                xb = x.view(b, h, x.shape[1], d)
                g = F.silu(torch.einsum("bhld,hed->bhle", xb, w0)) * torch.einsum("bhld,hed->bhle", xb, w2)
                return F.normalize(g, dim=-1).reshape(b * h, x.shape[1], -1)

            def sim(kk, qq, mask):  # best-matching question token, mean over heads -> [B, L]
                sgl = kk @ qq.transpose(1, 2)
                sgl = sgl.masked_fill(~mask.repeat_interleave(h, dim=0)[:, None, :], float("-inf"))
                return sgl.amax(dim=-1).view(b, h, -1).mean(dim=1).cpu()

            kf, pk = k.float(), None
            for name, n_last in (("alone_all", 0), ("alone_last1", 1)):
                m = last_tokens(qm, n_last)
                ctx.debug_scores[(self.layer_idx, "cos_" + name)] = sim(kf, qa, m)
                pk = phi(kf) if pk is None else pk
                ctx.debug_scores[(self.layer_idx, "kernel_" + name)] = sim(pk, phi(qa), m)
            tail = getattr(ctx, "debug_tailpos", None)  # [B] position of the last question token in THIS forward
            if tail is not None:
                idx = tail.to(q.device).repeat_interleave(h)[:, None, None].expand(-1, 1, d)
                qt = torch.gather(q.float(), 1, idx)  # the read vector actually used for the first answer token
                one = torch.ones(b, 1, dtype=torch.bool, device=q.device)
                ctx.debug_scores[(self.layer_idx, "cos_incontext_last1")] = sim(kf, qt, one)
                ctx.debug_scores[(self.layer_idx, "kernel_incontext_last1")] = sim(pk, phi(qt), one)
            span = getattr(ctx, "debug_span", None)  # (start, end) of the question tokens in THIS forward
            if span is not None:
                qc = q.float()[:, span[0] : span[1]]  # every question token as it is read in context [BH, Nq, d]
                mall = last_tokens(qm, 0)
                ctx.debug_scores[(self.layer_idx, "cos_incontext_all")] = sim(kf, qc, mall)  # best-matching question token
                sg = kf @ qc.transpose(1, 2)
                mm = mall.repeat_interleave(h, dim=0)[:, None, :].to(sg.dtype)
                ctx.debug_scores[(self.layer_idx, "cos_incontext_mean")] = (
                    ((sg * mm).sum(-1) / mm.sum(-1).clamp_min(1)).view(b, h, -1).mean(dim=1).cpu()
                )

        scores = gains = None
        sel = ctx.select
        if sel is not None and sel.mode != "none" and not ctx.capture_query:
            reuse = sel.score_layer >= 0 and self.layer_idx > sel.score_layer and sel.score_layer in ctx.last_weights
            if reuse:
                w, g = ctx.last_weights[sel.score_layer], ctx.last_gains.get(sel.score_layer)
            else:
                w, g = select_weights(
                    sel, q, k, segments, flags, b, h, ctx.query_span, ctx.query_mask, ctx.oracle_weight,
                    ctx.query_q.get(self.layer_idx), ctx.query_prefix,
                )
            ctx.last_weights[self.layer_idx] = w
            if g is not None:
                ctx.last_gains[self.layer_idx] = g
                gains = g.repeat_interleave(h, dim=0)
            scores = w.repeat_interleave(h, dim=0)
            lr = lr * scores[..., None].to(lr.dtype)

        ltw = getattr(ctx, "layer_token_weights", None)  # {layer: [B, L]}: this layer's own choice of which keys it writes
        if ltw is not None and self.layer_idx in ltw and not ctx.capture_query:
            lr = lr * ltw[self.layer_idx].to(q.device).repeat_interleave(h, dim=0)[..., None].to(lr.dtype)

        if scores is None and ctx.bank_scores is not None and not ctx.capture_query:
            bs = ctx.bank_scores[self.layer_idx] if isinstance(ctx.bank_scores, dict) else ctx.bank_scores
            scores = bs.to(q.device).float().repeat_interleave(h, dim=0)

        weights = self.fast_weights()
        state = ctx.states.get(self.layer_idx) if ctx.session else None
        if state is None:
            state = self._initial_state(b, weights)

        if state.replay is not None and state.replay.cfg.mode == "pseudo" and not ctx.capture_query:
            # the question's read vectors are the rehearsal probes (taken from a forward of the question
            # alone, or from where the question sits in this forward)
            qq, qm = ctx.query_q.get(self.layer_idx), ctx.query_mask
            if qq is None and ctx.query_span is not None:
                qq = q[:, ctx.query_span[0] : ctx.query_span[1]].detach()
            if qq is not None and qm is not None:
                state.replay.query_q = qq
                state.replay.query_valid = qm.to(qq.device).repeat_interleave(h, dim=0)

        out, state = segmented_lact(
            state, q, k, v, lr, momentum, segments, flags, self._ref_norms(b, weights), self.update_cfg,
            None if ctx.recorded is None else ctx.recorded.setdefault(self.layer_idx, []),
            None if ctx.borrowed is None else ctx.borrowed[self.layer_idx],
            scores=scores, gains=gains,
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
        else:
            out = out * self.gate.float()[:, None]
        out = out.reshape(b, length, h * d)
        if self.cfg.inject == "pre_o_proj":
            return out.to(x.dtype)
        return self.o_proj(out).to(x.dtype)
