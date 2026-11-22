"""Attach LaCT memory branches to a Hugging Face Qwen3 model.

Each selected decoder layer computes

    h' = h + SelfAttn(norm(h)) + Memory(norm(h))
    out = h' + MLP(norm(h'))

The memory output is multiplied by a per-head gate that starts at zero, so a fresh
model reproduces the base model exactly. The base weights keep their original
parameter names; the new ones live under ``model.layers.{i}.memory``.
"""

from __future__ import annotations

import json
import math
import os
import types
from typing import Optional

import torch
import torch.nn as nn

from .memory_branch import LaCTMemoryBranch, MemoryConfig, MemoryContext

MEMORY_PREFIX = "memory."
CONFIG_NAME = "memory_config.json"
WEIGHTS_NAME = "memory_weights.pt"


def _decoder_layers(model: nn.Module):
    base = getattr(model, "model", model)
    return base.layers


def _sink_overlay(num_sinks: int):
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
        return (kv_idx < num_sinks) & (kv_idx <= q_idx)

    return inner_mask


def _segment_overlay(segment: int, num_sinks: int, config=None):
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
        ids = getattr(config, "attention_segment_ids", None)
        if ids is not None:
            # explicit steps (set_attention_segments): a token sees only tokens of its own step
            same = ids[q_idx] == ids[kv_idx]
        else:
            same = q_idx // segment == kv_idx // segment
        return same | (kv_idx < num_sinks)

    return inner_mask


def set_attention_segments(model: nn.Module, segments=None):
    """Attention steps for the next forward(s): a list of (start, end) covering the sequence, or None for
    the fixed grid of config.attention_segment tokens. Needed when steps are not all the same length,
    e.g. a history whose last step is short followed by the query step (MemT Sec. 3.4 / 3.5)."""
    cfg = model.config
    if segments is None:
        cfg.attention_segment_ids = None
        return
    if not getattr(cfg, "attention_segment", 0):
        raise ValueError("explicit attention segments need set_attention_window(..., segment > 0)")
    if any(e - s > cfg.sliding_window for s, e in segments):
        raise ValueError("an attention step is longer than the sliding window")
    ids = torch.empty(segments[-1][1], dtype=torch.long, device=next(model.parameters()).device)
    for i, (s, e) in enumerate(segments):
        ids[s:e] = i
    cfg.attention_segment_ids = ids


def _install_sink_masks():
    """Make Qwen3's sliding-window mask also admit the first ``config.attention_sinks`` tokens,
    and with ``config.attention_segment`` restrict it to the query's own block of that many tokens."""
    import transformers.models.qwen3.modeling_qwen3 as qwen3

    if getattr(qwen3.create_sliding_window_causal_mask, "_lact_sinks", False):
        return
    original = qwen3.create_sliding_window_causal_mask

    def create_sliding_window_causal_mask(config, *args, **kwargs):
        sinks = getattr(config, "attention_sinks", 0)
        if sinks:
            cache = kwargs.get("past_key_values")
            if cache is not None and cache.get_seq_length() > 0:
                raise NotImplementedError("the sliding KV cache evicts sink tokens; run the whole sequence in one forward")
            kwargs["or_mask_function"] = _sink_overlay(sinks)
        segment = getattr(config, "attention_segment", 0)
        if segment:
            kwargs["and_mask_function"] = _segment_overlay(segment, sinks, config)
        return original(config, *args, **kwargs)

    create_sliding_window_causal_mask._lact_sinks = True
    qwen3.create_sliding_window_causal_mask = create_sliding_window_causal_mask


def set_attention_window(model: nn.Module, window: Optional[int], sinks: int = 0, segment: int = 0):
    """Restrict every self-attention layer to the last ``window`` tokens (None = full).

    ``sinks`` > 0 additionally keeps the first ``sinks`` tokens visible to every query
    (StreamingLLM attention sinks). Without them a model trained with full attention
    breaks down once its first tokens leave the window.

    ``segment`` > 0 cuts the sequence into blocks of that many tokens and lets a token attend only
    inside its own block (and to the sinks): the start of a block sees no earlier text at all, so
    everything older has to come from memory (MemT trains this way). Use ``window >= segment``
    and a memory chunk size that divides ``segment``.
    """
    cfg = model.config
    n = cfg.num_hidden_layers
    if segment and (window is None or window < segment):
        raise ValueError("attention segments need a sliding window at least as long as the segment")
    cfg.attention_segment = int(segment)
    cfg.attention_segment_ids = None
    if sinks or segment:
        if window is None:
            raise ValueError("attention sinks need a sliding window")
        if str(getattr(cfg, "_attn_implementation", "")).startswith("flash"):
            raise ValueError("flash attention ignores custom masks; use sdpa or eager with attention sinks")
        _install_sink_masks()
    cfg.attention_sinks = int(sinks)
    if window is None:
        cfg.layer_types = ["full_attention"] * n
        cfg.use_sliding_window = False
        cfg.sliding_window = None
    else:
        cfg.layer_types = ["sliding_attention"] * n
        cfg.use_sliding_window = True
        cfg.sliding_window = int(window)
        cfg.max_window_layers = n
    for layer in _decoder_layers(model):
        attn = layer.self_attn
        attn.layer_type = cfg.layer_types[attn.layer_idx]
        attn.sliding_window = cfg.sliding_window
    base = getattr(model, "model", model)
    base.has_sliding_layers = window is not None


def _forward_with_memory(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    residual = hidden_states
    normed = self.input_layernorm(hidden_states)
    pre = self.memory.cfg.inject == "pre_o_proj"
    if pre:
        # computed first; the hook on self_attn.o_proj adds it to the attention output before o_proj
        self.memory.pending = self.memory(normed)
    attn_out, _ = self.self_attn(
        hidden_states=normed,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + attn_out
    if not pre:
        hidden_states = hidden_states + self.memory(normed)

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states


def _lora_hook(branch: LaCTMemoryBranch, name: str, scale: float):
    def hook(module, inputs, output):
        a, b = branch.lora_a[name], branch.lora_b[name]
        delta = nn.functional.linear(nn.functional.linear(inputs[0].to(a.dtype), a), b)
        return output + (delta * scale).to(output.dtype)

    return hook


def _attach_lora(layer: nn.Module, branch: LaCTMemoryBranch, cfg: MemoryConfig, device):
    """Low-rank updates W + (alpha / r) * B @ A for the layer's attention and MLP linears.

    The base weights stay frozen and keep their names; A and B are parameters of the branch.
    """
    r = cfg.lora_rank
    targets = {t for t in cfg.lora_targets.split(",") if t}
    branch.lora_a, branch.lora_b = nn.ParameterDict(), nn.ParameterDict()
    for parent in (layer.self_attn, layer.mlp):
        for name, mod in parent.named_children():
            if name not in targets or not isinstance(mod, nn.Linear):
                continue
            a = torch.empty(r, mod.in_features, device=device)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            branch.lora_a[name] = nn.Parameter(a)
            branch.lora_b[name] = nn.Parameter(torch.zeros(mod.out_features, r, device=device))
            mod.register_forward_hook(_lora_hook(branch, name, cfg.lora_alpha / r))
    if not len(branch.lora_a):
        raise ValueError(f"no linear layer matches lora_targets={cfg.lora_targets!r}")


def _copy_attention_qkv(branch: LaCTMemoryBranch, attn: nn.Module):
    """Initialise the branch's q/k/v projection from the attention's q_proj / k_proj / v_proj."""
    n_q = attn.q_proj.out_features
    if branch.cfg.num_heads * branch.cfg.head_dim != n_q:
        raise ValueError(
            f"qkv_init='attention' needs num_heads * head_dim == {n_q} "
            f"(got {branch.cfg.num_heads} * {branch.cfg.head_dim})"
        )
    if attn.q_proj.bias is not None:
        raise ValueError("qkv_init='attention' expects attention projections without bias")
    rep, dh = attn.num_key_value_groups, attn.head_dim

    def expand(w):  # [kv_heads * dh, hidden] -> [heads * dh, hidden], each KV head repeated in place
        return w.view(-1, dh, w.shape[1]).repeat_interleave(rep, dim=0).reshape(n_q, w.shape[1])

    with torch.no_grad():
        w = torch.cat([attn.q_proj.weight, expand(attn.k_proj.weight), expand(attn.v_proj.weight)], dim=0)
        branch.qkv.weight.copy_(w.to(branch.qkv.weight.dtype))


def _inject_hook(branch: LaCTMemoryBranch):
    def hook(module, args):
        pending, branch.pending = branch.pending, None
        if pending is None:
            return None
        x = args[0]
        if branch.cfg.out_ref == "attention":
            h, d = branch.cfg.num_heads, branch.cfg.head_dim
            shape = (*x.shape[:-1], h, d)
            ref = x.detach().float().view(shape).pow(2).mean(-1, keepdim=True).sqrt()
            pending = (pending.float().view(shape) * ref).view(x.shape)
        return (x + pending.to(x.dtype), *args[1:])

    return hook


def attach_memory(model: nn.Module, cfg: MemoryConfig) -> MemoryContext:
    """Add a memory branch to the selected layers. Returns the shared context."""
    if getattr(model, "lact_ctx", None) is not None:
        raise RuntimeError("memory branches are already attached")
    ctx = MemoryContext()
    hidden = model.config.hidden_size
    layers = _decoder_layers(model)
    ref = next(model.parameters())
    for idx in cfg.layer_indices(len(layers)):
        layer = layers[idx]
        branch = LaCTMemoryBranch(hidden, cfg, idx, ctx).to(device=ref.device, dtype=torch.float32)
        layer.memory = branch
        if cfg.qkv_init == "attention":
            _copy_attention_qkv(branch, layer.self_attn)
        if cfg.inject == "pre_o_proj":
            if cfg.num_heads * cfg.head_dim != layer.self_attn.o_proj.in_features:
                raise ValueError(
                    f"inject='pre_o_proj' needs num_heads * head_dim == {layer.self_attn.o_proj.in_features}"
                )
            layer.self_attn.o_proj.register_forward_pre_hook(_inject_hook(branch))
        if cfg.lora_rank > 0:
            _attach_lora(layer, branch, cfg, ref.device)
        layer.forward = types.MethodType(_forward_with_memory, layer)
    model.lact_ctx = ctx
    model.lact_cfg = cfg
    return ctx


def memory_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if f".{MEMORY_PREFIX}" in n]


def freeze_base(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = f".{MEMORY_PREFIX}" in n


def memory_state_dict(model: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if f".{MEMORY_PREFIX}" in n}


def save_memory(model: nn.Module, out_dir: str, extra: Optional[dict] = None):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(memory_state_dict(model), os.path.join(out_dir, WEIGHTS_NAME))
    meta = {
        "memory": model.lact_cfg.to_dict(),
        "attention_window": model.config.sliding_window,
        "attention_sinks": getattr(model.config, "attention_sinks", 0),
        "attention_segment": getattr(model.config, "attention_segment", 0),
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
        json.dump(meta, f, indent=2)


def load_memory_config(ckpt_dir: str) -> tuple[MemoryConfig, Optional[int], int]:
    """Returns (memory config, attention window, attention sinks) the checkpoint was trained with."""
    with open(os.path.join(ckpt_dir, CONFIG_NAME)) as f:
        meta = json.load(f)
    return MemoryConfig.from_dict(meta["memory"]), meta.get("attention_window"), meta.get("attention_sinks", 0)


def load_memory_weights(model: nn.Module, ckpt_dir: str):
    state = torch.load(os.path.join(ckpt_dir, WEIGHTS_NAME), map_location="cpu")
    params = dict(model.named_parameters())
    missing = [n for n in params if f".{MEMORY_PREFIX}" in n and n not in state]
    if missing:
        raise KeyError(f"checkpoint is missing memory parameters, e.g. {missing[:3]}")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t.to(params[n].dtype))


def set_replay(model: nn.Module, **overrides):
    """Change replay settings at eval time (e.g. compare modes on one checkpoint)."""
    rcfg = model.lact_cfg.replay
    for k, v in overrides.items():
        if not hasattr(rcfg, k):
            raise AttributeError(f"ReplayConfig has no field {k}")
        setattr(rcfg, k, v)
    rcfg.__post_init__()
