from __future__ import annotations

import argparse
from typing import Optional

import torch

from .matrix_memory import MAX_DECAY
from .memory_branch import MemoryConfig
from .modeling import attach_memory, load_memory_config, load_memory_weights, set_attention_window, set_replay
from .replay import REPLAY_MODES, ReplayConfig



def _decay_cap(text):
    return None if str(text).lower() == "none" else float(text)

def add_model_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("model")
    g.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF id or local path of the base Qwen3 model")
    g.add_argument("--tokenizer", default=None, help="defaults to --model")
    g.add_argument("--tiny-random", action="store_true", help="random 2-layer Qwen3 for smoke tests")
    g.add_argument("--window", type=int, default=2048, help="attention window in tokens; 0 = full attention")
    g.add_argument("--sinks", type=int, default=0, help="first N tokens stay visible beyond the window (attention sinks)")
    g.add_argument(
        "--attn-segment",
        type=int,
        default=0,
        help="attention only sees the current block of this many tokens (plus sinks); 0 = plain sliding "
        "window. With --memory <checkpoint> and 0, the value stored in the checkpoint is used",
    )
    g.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2", "flex_attention", "eager"])
    g.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    m = p.add_argument_group("memory branch")
    m.add_argument("--memory", default="none", help="'none', 'new', or a checkpoint directory")
    m.add_argument("--mem-heads", type=int, default=4)
    m.add_argument("--mem-head-dim", type=int, default=512)
    m.add_argument("--mem-inter-multi", type=float, default=1.0)
    m.add_argument("--mem-base-lr", type=float, default=0.01)
    m.add_argument("--mem-chunk", type=int, default=2048)
    m.add_argument(
        "--mem-core",
        default="mlp",
        choices=["mlp", "matrix"],
        help="mlp: LaCT fast-weight MLP, one gradient step per chunk; matrix: MemT-style state matrix, "
        "every token written with a gated delta rule (starts at zero, so no write means no output)",
    )
    m.add_argument("--mem-write-block", type=int, default=256, help="--mem-core matrix: tokens per solve")
    m.add_argument("--mem-matrix-backend", choices=["torch", "fla_kda"], default=None,
                   help="optional CUDA KDA write kernel; default uses checkpoint value or torch")
    m.add_argument("--mem-stable-matrix-gates", action="store_true",
                   help="evaluate sqrt(sigmoid(erase_logits)) through log-sigmoid for finite extreme-logit gradients")
    m.add_argument("--mem-forget-rank", type=int, default=64, help="--mem-core matrix: width of the forget gate")
    # 矩阵记忆与 MemT-1.0 的对齐(默认值 = 归档的 4B 训练设置;见 MemoryConfig)
    m.add_argument("--mem-decay-cap", type=_decay_cap, default=MAX_DECAY,
                   help="bound on the per-token decay; 'none' = no bound (MemT-1.0)")
    m.add_argument("--mem-write-impl", choices=["scaled", "stable", "auto"], default="scaled",
                   help="matrix write algorithm (same recurrence); 'auto' or 'stable' are needed without a decay cap")
    m.add_argument("--mem-out-norm-eps", type=float, default=1e-5, help="eps of the RMSNorm on the memory read")
    m.add_argument("--mem-qk-norm", choices=["lact", "memt"], default="lact",
                   help="q/k normalisation: x/(|x|+1e-5) or MemT's x*rsqrt(sum x^2 + 1e-6)")
    m.add_argument("--mem-memt-rounding", action="store_true",
                   help="round normalised q/k, sqrt(erase), u, write gate and read to the projection dtype as MemT-1.0-Nano")
    m.add_argument("--mem-layers", default="all", help="'all', 'every:N', or comma-separated indices")
    m.add_argument("--no-muon", action="store_true")
    m.add_argument("--mem-norm", default="post", choices=["post", "pre"], help="fast-weight write normalisation")
    m.add_argument("--mem-delta-readout", action="store_true", help="output only what the writes changed")
    m.add_argument("--no-momentum", action="store_true")
    m.add_argument(
        "--mem-fw-init",
        default="random",
        choices=["random", "lowrank_identity"],
        help="initial w0/w2: full-rank Gaussian, or LaCT's low-rank + 0.5*I",
    )
    m.add_argument("--mem-fw-rank", type=int, default=32, help="rank for --mem-fw-init lowrank_identity")
    m.add_argument(
        "--mem-out-scale",
        default="gate",
        choices=["gate", "token", "memt"],
        help="scale of the memory output: one scalar gate per head, or LaCT's per-token silu(Linear(h))",
    )
    m.add_argument("--mem-gate-init", type=float, default=0.0, help="initial gate for --mem-out-scale gate")
    m.add_argument("--mem-output-gate-rank", type=int, default=128)
    m.add_argument("--mem-output-gate-softcap", type=float, default=2.0)
    m.add_argument("--mem-rope", action="store_true", help="rotate fast-weight q/k by token position")
    m.add_argument("--mem-rope-theta", type=float, default=1_000_000.0)
    m.add_argument(
        "--mem-qkv-init",
        default="random",
        choices=["random", "attention"],
        help="initial q/k/v projection: Gaussian, or a copy of the layer's attention q/k/v projections",
    )
    m.add_argument("--mem-qkv-act", default="silu", choices=["silu", "none"])
    m.add_argument(
        "--mem-out-ref",
        default="none",
        choices=["none", "attention"],
        help="attention: with --mem-inject pre_o_proj, scale the memory output by the RMS of the "
        "attention output at the same token, so --mem-gate-init is a size relative to attention",
    )
    m.add_argument(
        "--mem-inject",
        default="post",
        choices=["post", "pre_o_proj"],
        help="post: own output projection next to the attention output; pre_o_proj: add the memory "
        "output before the attention's frozen o_proj (needs heads * head_dim == o_proj input width)",
    )
    m.add_argument(
        "--mem-lora-rank",
        type=int,
        default=0,
        help="rank of LoRA on the frozen base attention/MLP linears, trained with the branch (0 = off)",
    )
    m.add_argument("--mem-lora-alpha", type=float, default=16.0)
    m.add_argument(
        "--mem-lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )

    r = p.add_argument_group("replay")
    r.add_argument("--replay", default=None, choices=REPLAY_MODES, help="override the replay mode")
    r.add_argument("--replay-samples", type=int, default=None)
    r.add_argument("--replay-strength", type=float, default=None)
    r.add_argument("--replay-clip", type=float, default=None)
    r.add_argument("--replay-selection", default=None, choices=["uniform", "coverage"])
    r.add_argument("--replay-components", type=int, default=None)


def _replay_overrides(args) -> dict:
    mapping = {
        "mode": args.replay,
        "num_samples": args.replay_samples,
        "strength": args.replay_strength,
        "clip_ratio": args.replay_clip,
        "selection": args.replay_selection,
        "gmm_components": args.replay_components,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def memory_config_from_args(args) -> MemoryConfig:
    return MemoryConfig(
        num_heads=args.mem_heads,
        head_dim=args.mem_head_dim,
        inter_multi=args.mem_inter_multi,
        base_lr=args.mem_base_lr,
        use_muon=not args.no_muon,
        use_momentum=not args.no_momentum,
        chunk_size=args.mem_chunk,
        core=args.mem_core,
        write_block=args.mem_write_block,
        stable_matrix_gates=args.mem_stable_matrix_gates,
        decay_cap=args.mem_decay_cap,
        write_impl=args.mem_write_impl,
        out_norm_eps=args.mem_out_norm_eps,
        qk_norm=args.mem_qk_norm,
        memt_rounding=args.mem_memt_rounding,
        forget_rank=args.mem_forget_rank,
        layers=args.mem_layers,
        norm=args.mem_norm,
        delta_readout=args.mem_delta_readout,
        fw_init=args.mem_fw_init,
        fw_rank=args.mem_fw_rank,
        out_scale=args.mem_out_scale,
        output_gate_rank=args.mem_output_gate_rank,
        output_gate_softcap=args.mem_output_gate_softcap,
        gate_init=args.mem_gate_init,
        rope=args.mem_rope,
        rope_theta=args.mem_rope_theta,
        qkv_init=args.mem_qkv_init,
        qkv_act=args.mem_qkv_act,
        inject=args.mem_inject,
        out_ref=args.mem_out_ref,
        lora_rank=args.mem_lora_rank,
        lora_alpha=args.mem_lora_alpha,
        lora_targets=args.mem_lora_targets,
        replay=ReplayConfig(**_replay_overrides(args)),
    )


def load_tokenizer(args):
    from transformers import AutoTokenizer

    name = args.tokenizer or ("Qwen/Qwen3-1.7B" if args.tiny_random else args.model)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _tiny_model(vocab_size: int):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=65536,
    )
    return Qwen3ForCausalLM(cfg)


def load_model(args, tokenizer=None, for_training: bool = False):
    """Returns (model, memory_config or None)."""
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.tiny_random:
        vocab = len(tokenizer) if tokenizer is not None else 151669
        model = _tiny_model(vocab).to(dtype)
        model.config._attn_implementation = args.attn_impl
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation=args.attn_impl)

    window: Optional[int] = args.window if args.window and args.window > 0 else None
    mem_cfg = None
    if args.memory == "new":
        mem_cfg = memory_config_from_args(args)
    elif args.memory != "none":
        mem_cfg, ckpt_window, ckpt_sinks = load_memory_config(args.memory)
        if window != ckpt_window:
            print(f"note: checkpoint was trained with window={ckpt_window}, evaluating with window={window}")
        if args.sinks != ckpt_sinks:
            print(f"note: checkpoint was trained with sinks={ckpt_sinks}, evaluating with sinks={args.sinks}")

    segment = getattr(args, "attn_segment", 0)
    if not segment and args.memory not in ("new", "none"):
        import json
        import os

        with open(os.path.join(args.memory, "memory_config.json")) as f:
            segment = json.load(f).get("attention_segment", 0)
        if segment:
            print(f"note: using attention segment {segment} from the checkpoint")
    set_attention_window(model, window, args.sinks, segment)
    model.to(args.device)
    if mem_cfg is not None:
        if getattr(args, "mem_matrix_backend", None) is not None:
            mem_cfg.matrix_backend = args.mem_matrix_backend
        attach_memory(model, mem_cfg)
        if args.memory not in ("new", "none"):
            load_memory_weights(model, args.memory)
        overrides = _replay_overrides(args)
        if overrides:
            set_replay(model, **overrides)

    model.config.use_cache = False
    if not for_training:
        model.eval()
    return model, mem_cfg


def load_teacher(args, tokenizer=None):
    """The frozen base model with full attention and no memory (the distillation target)."""
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.tiny_random:
        vocab = len(tokenizer) if tokenizer is not None else 151669
        model = _tiny_model(vocab).to(dtype)
        model.config._attn_implementation = "sdpa"
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    set_attention_window(model, None, 0)
    model.to(args.device)
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def autocast(args):
    device_type = torch.device(args.device).type
    if device_type in ("cuda", "cpu") and args.dtype == "bfloat16":
        return torch.autocast(device_type, dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)
