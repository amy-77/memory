"""只训练冻结 Qwen3 上的记忆分支。注意力只看一个短窗口，窗口外的信息靠快权重记忆读写。

这个文件是自己写的训练循环，不调用官方 LaCT 的 flame 训练入口
（flame/flame/train.py）。快权重的“先读后写”规则在 lact_qwen/fast_weight.py，
对照的是官方 custom_models/lact_model/ttt_operation.py 里的 block_causal_lact_swiglu。

Stage 1（在长文本上学会使用记忆）:
    torchrun --nproc_per_node 8 scripts/train_memory.py \
        --model Qwen/Qwen3-1.7B --data text:hf:emozilla/pg19:train \
        --seq-len 16384 --window 2048 --sinks 4 --memory new --mem-chunk 2048 \
        --steps 2000 --lr 1e-3 --grad-ckpt --out checkpoints/stage1

--data text:<dir> 也可以读 datasets.save_to_disk 存下来的目录，例如 Clean-Long-Data-Collections。
默认每条样本只包含一篇文档；长文档分窗，窗口之间重置记忆。--text-packing legacy 仅用于旧实验续训。

合成对话（按轮写入，损失只算在答案上）:
    python scripts/train_memory.py --data synth:data/generated/synth_train.jsonl \
        --window 2048 --sinks 4 --memory checkpoints/stage1/final --steps 1000 --out checkpoints/synth

加 --replay oracle_key 或 --replay gmm，会在写入时带上重放。
"""

import argparse
from contextlib import nullcontext
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist

# 仓库根目录加入 import 路径，这样直接 python scripts/train_memory.py 也能找到 lact_qwen。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.dialog import encode_history, merge_spans, split_long_spans
from lact_qwen.training_loss import MemoryTrainingLoss
from lact_qwen.memory_branch import LaCTMemoryBranch
from lact_qwen.loading import add_model_args, autocast, load_model, load_teacher, load_tokenizer
from lact_qwen.modeling import CONFIG_NAME, freeze_base, memory_parameters, save_memory
from lact_qwen.checkpointing import OPTIMIZER_NAME, RUNTIME_NAME
from lact_qwen.training_data import optimizer_batch
from lact_qwen.teacher_cache import TeacherCache, file_sha256
from lact_qwen.training_state import (capture_runtime, restore_rng, replay_iterator,
                                      save_rank_checkpoint, raise_rank_errors)

from lact_qwen.synth import read_jsonl
from lact_qwen.text_data import document_sequences, packed_sequences
from lact_qwen.post_data import post_batches, supervised_token_scale
from lact_qwen.numerics import clip_grad_norm_fp64


def text_batches(spec, tok, args, rank, world, seed):
    """Default: independent documents, right-padded without padding loss.

    legacy mode intentionally retains cross-document packing for old checkpoints.
    Memory state starts at zero for every batch row, including long-document splits.
    """
    sequence_fn = document_sequences if args.text_packing == "document" else packed_sequences
    stream = sequence_fn(
        spec,
        tok,
        args.seq_len,
        rank=rank,
        world=world,
        shuffle_seed=seed if args.shuffle_buffer > 0 else None,
        shuffle_buffer=args.shuffle_buffer,
        min_doc_tokens=args.min_doc_tokens,
    )
    while True:
        rows = list(itertools.islice(stream, args.batch_size))
        if not rows:
            return
        x = torch.full((len(rows), max(map(len, rows))), tok.pad_token_id, dtype=torch.long)
        labels = torch.full_like(x, -100)
        mask = torch.zeros_like(x)
        for i, row in enumerate(rows):
            x[i, :len(row)] = torch.tensor(row)
            labels[i, :len(row)] = x[i, :len(row)]
            mask[i, :len(row)] = 1
        yield {"input_ids": x, "labels": labels, "attention_mask": mask,
               "segments": None, "update_flags": None}


def synth_batches(path, tok, loss_on, max_segment, rank, seed, categories=None, min_write=0):
    """合成对话：历史按轮写入记忆，问题加答案接在后面，默认只对答案算损失。

    最后一段是“提问 + 回答”，update_flags 为 False，这段只读记忆、不把答案写回去。
    每张卡用不同的随机种子自己抽样，所以没有按 rank 切分数据。
    """
    dialogues = read_jsonl(path)
    if categories:
        for d in dialogues:
            d["queries"] = [q for q in d["queries"] if q["category"] in categories]
        dialogues = [d for d in dialogues if d["queries"]]
    rng = random.Random(seed + 1009 * rank)
    while True:
        dlg = rng.choice(dialogues)
        enc = encode_history(dlg, tok)
        ids = list(enc.history_ids)
        # loss_on=answers 时，历史 token 的标签是 -100，交叉熵会跳过它们。
        labels = list(ids) if loss_on == "all" else [-100] * len(ids)
        n_hist = len(ids)
        queries = list(dlg["queries"])
        rng.shuffle(queries)
        for q in queries:
            q_ids = tok(q["question"], add_special_tokens=False).input_ids
            a_ids = tok(" " + q["answer"], add_special_tokens=False).input_ids
            ids += q_ids + a_ids
            labels += [-100] * len(q_ids) + a_ids
        # 历史按轮（或合并后的写入块）切开，再接上整段问答。
        segments = split_long_spans(merge_spans(enc.turn_spans, min_write), max_segment) + [(n_hist, len(ids))]
        yield {
            "input_ids": torch.tensor([ids]),
            "labels": torch.tensor([labels]),
            "segments": segments,
            "update_flags": [True] * (len(segments) - 1) + [False],
        }


def lr_at(step, base, warmup, total, min_ratio=0.05, legacy=False):
    """Warm up, then cosine decay; last optimizer step reaches the chosen floor.

    legacy=True reproduces the historical hard-coded 0.1 floor and off-by-one
    denominator, for continuing existing experiments without changing their LR.
    """
    if step < warmup:
        return base * (step + 1) / warmup
    if legacy:
        min_ratio = 0.1
    progress = (step - warmup) / max(1, total - warmup - (not legacy))
    if not legacy and total - warmup == 1:
        progress = 1.0
    progress = min(1.0, max(0.0, progress))
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def wsd_lr_at(step, base, warmup, total, decay_steps, min_ratio):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    start = total - decay_steps
    if not decay_steps or step < start:
        return base
    progress = min(1.0, (step - start + 1) / decay_steps)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="text:<spec>, synth:<jsonl>, or post:<jsonl>")
    p.add_argument("--seq-len", type=int, default=16384, help="text data only")
    p.add_argument("--batch-size", type=int, default=1, help="per device, text data only")
    p.add_argument("--shuffle-buffer", type=int, default=1000, help="text data only; 0 keeps the source order")
    p.add_argument("--min-doc-tokens", type=int, default=0, help="text data only; skip shorter documents")
    p.add_argument("--text-packing", choices=["document", "legacy"], default="document",
                   help="one independent document per row; legacy joins unrelated documents")
    p.add_argument("--loss-on", choices=["answers", "all"], default="answers", help="synth data only")
    p.add_argument(
        "--query-categories",
        default=None,
        help="synth data only; comma-separated subset of retention,correction,recent to ask about",
    )
    p.add_argument("--max-segment", type=int, default=4096)
    p.add_argument(
        "--min-write-tokens",
        type=int,
        default=0,
        help="synth data only; join turns into write units of at least this many tokens (0 = one write per turn)",
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--adam-betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--lr-schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--decay-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--legacy-cosine", action="store_true", help="restore the old fixed 0.1 cosine schedule")
    p.add_argument("--loss-chunk-size", type=int, default=128, help="token count per checkpointed LM-head chunk")
    p.add_argument("--distill-top-k", type=int, default=0)
    p.add_argument("--distill-temperature", type=float, default=1.0)
    p.add_argument("--teacher-cache", default=None, help="offline Top-K cache for post: data; omits the online teacher")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="add this times KL(full-attention base model || window + memory) to the loss; 0 = off. "
        "The teacher is a second copy of the base model with full attention and no memory.",
    )
    p.add_argument(
        "--qkv-lr-scale",
        type=float,
        default=1.0,
        help="learning-rate multiplier for the memory q/k/v projection; below 1 keeps a projection "
        "copied with --mem-qkv-init attention close to its start",
    )
    p.add_argument("--ce-weight", type=float, default=1.0, help="weight of the next-token loss when distilling")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--save-at", type=int, nargs="*", default=[], help="Additional checkpoint steps; optimization is unchanged")
    p.add_argument("--keep-last", type=int, default=0, help="keep only the newest N step_* checkpoints; 0 keeps all")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None, help="a step_N directory written by this script")
    p.add_argument(
        "--stop-at",
        type=int,
        default=0,
        help="stop after this optimizer step (saving step_N, no final/) while keeping the LR schedule of --steps",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-file", default=None, help="atomic rank-zero progress JSON")
    p.add_argument("--save-initial", default=None, help="save synchronized step-zero memory before training")
    p.add_argument("--exact-text-resume", action="store_true",
                   help="replay legacy text checkpoints; v3 checkpoints always restore their data cursor")
    p.add_argument("--fail-on-nonfinite", action="store_true")
    p.add_argument("--stable-grad-norm", action="store_true", help="reduce gradient norms in FP64")
    p.add_argument("--save-on-nonfinite", action="store_true",
                   help="save finite pre-update weights and optimizer at the completed step before failing")
    p.add_argument("--batch-audit-dir", default=None, help="record first optimizer batch token hashes on each rank")
    # --model、--window、--memory、--mem-* 等模型结构参数定义在 lact_qwen/loading.py。
    add_model_args(p)
    args = p.parse_args()
    kind, spec = args.data.split(":", 1)
    if kind not in ("text", "synth", "post"):
        raise ValueError("--data must start with text:, synth:, or post:")
    if args.teacher_cache and (not args.data.startswith("post:") or args.distill_top_k <= 0 or args.distill_weight <= 0):
        raise ValueError("teacher caches require post: data, positive distillation weight and Top-K")
    if min(args.steps, args.grad_accum, args.batch_size, args.log_every, args.save_every,
           args.loss_chunk_size) < 1 or args.seq_len < 2:
        raise ValueError("step, batch, accumulation, interval and chunk sizes must be positive; seq_len >= 2")
    if not 0 <= args.warmup < args.steps or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("require 0 <= warmup < steps and 0 <= min_lr_ratio <= 1")
    if args.ce_weight < 0 or args.distill_weight < 0 or args.ce_weight + args.distill_weight == 0:
        raise ValueError("loss weights must be nonnegative with at least one positive")
    if args.decay_steps < 0 or args.decay_steps > args.steps - args.warmup:
        raise ValueError("decay_steps must fit after warmup")
    if args.data.startswith("post:"):
        if args.batch_size != 1:
            raise ValueError("one-pass post data requires batch=1")
        if args.attn_segment <= 0 or args.window != args.attn_segment or args.sinks != 0:
            raise ValueError("post-training requires current-segment-only attention and zero sinks")
        if args.ce_weight != 0 or args.distill_weight <= 0:
            raise ValueError("MemT post-training must use KL only")
    if args.memory == "none" and not args.resume:
        raise SystemExit("--memory must be 'new' or a checkpoint directory")

    # torchrun 会设置 RANK / WORLD_SIZE / LOCAL_RANK。单卡直接 python 启动时这些变量不存在，当成 1 卡。
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        if args.device.startswith("cuda"):
            torch.cuda.set_device(local)
            args.device = f"cuda:{local}"
        dist.init_process_group("nccl" if args.device.startswith("cuda") else "gloo")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % 2**32)

    # Hash single-file sources, including all post-training JSONL. Remote/HF sources
    # still require an immutable external dataset revision and unchanged shards.
    source_sha256, source_error = None, None
    try:
        if os.path.isfile(spec) and not spec.endswith(".arrow"):
            source_sha256 = file_sha256(spec)
    except Exception as exc:
        source_error = f"{type(exc).__name__}: {exc}"
    raise_rank_errors(source_error, "training source fingerprint")
    if world > 1:
        hashes = [None] * world
        dist.all_gather_object(hashes, source_sha256)
        if len(set(hashes)) != 1:
            raise ValueError("ranks must read the same source file before sharding")

    if args.resume:
        # 分支结构（头数、初始化方式等）以检查点为准，续训时不用再传一遍 --mem-*。
        args.memory = args.resume
    tok = load_tokenizer(args)
    model, mem_cfg = load_model(args, tok, for_training=True)
    # Qwen 主干冻结，优化器里只会放记忆分支的参数。
    freeze_base(model)
    if args.grad_ckpt:
        # use_reentrant=False 才能和 DDP 一起用。
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ctx = model.lact_ctx

    # 蒸馏老师是同一底座的第二份拷贝：全注意力、没有记忆、不求梯度。默认关闭。
    teacher = load_teacher(args, tok) if args.distill_weight > 0 and not args.teacher_cache else None
    params = memory_parameters(model)
    n_params = sum(p.numel() for p in params)
    if rank == 0:
        print(f"trainable memory parameters: {n_params / 1e6:.1f}M across {len(mem_cfg.layer_indices(model.config.num_hidden_layers))} layers")

    # DDP 包的是同一个 module，下面的 params 仍然指向原来的记忆参数。
    core = model
    model = MemoryTrainingLoss(core, chunk=args.loss_chunk_size, top_k=args.distill_top_k,
                               temperature=args.distill_temperature)
    if world > 1:
        # Short documents / exhausted ranks may not use a memory write at all.
        device_ids = [torch.cuda.current_device()] if args.device.startswith("cuda") else None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids,
                                                         find_unused_parameters=True)
    if args.save_initial and not args.resume:
        if rank == 0:
            if os.path.exists(args.save_initial):
                raise FileExistsError(args.save_initial)
            save_memory(core, args.save_initial, {"step": 0, "training_args": vars(args), "trainable_parameters": n_params})
        if world > 1:
            dist.barrier()

    # 二维及以上的权重做 weight decay；门控、学习率偏置这类一维参数不做。
    slow = []
    if args.qkv_lr_scale != 1.0:
        slow = [p for n, p in core.named_parameters() if n.endswith(".memory.qkv.weight")]
    slow_ids = {id(p) for p in slow}
    decay = [p for p in params if p.ndim >= 2 and id(p) not in slow_ids]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=tuple(args.adam_betas),
        eps=args.adam_eps,
    )

    if slow:
        opt.add_param_group({"params": slow, "weight_decay": args.weight_decay, "lr_scale": args.qkv_lr_scale})

    start = 0
    processed_tokens = 0
    processed_before_resume_known = True
    resume_runtime = None
    if args.resume:
        # Adam relocates moment tensors to each parameter's device itself. Loading
        # everything onto CUDA also moves non-capturable step counters off CPU.
        opt.load_state_dict(torch.load(os.path.join(args.resume, OPTIMIZER_NAME),
                                       map_location="cpu", weights_only=True))
        with open(os.path.join(args.resume, CONFIG_NAME)) as f:
            resume_meta = json.load(f)
        start = resume_meta["step"]
        old_args = resume_meta.get("training_args", {})
        if start >= args.steps or (args.stop_at and args.stop_at <= start):
            raise ValueError("resume target (--steps / --stop-at) must exceed the saved step")
        old_packing = old_args.get("text_packing", "legacy")
        if args.data.startswith("text:") and args.text_packing != old_packing:
            raise ValueError("--resume cannot change text packing; use --memory for a new training stage")
        if resume_meta.get("training_version", 1) < 2 and args.lr_schedule == "cosine" and not args.legacy_cosine:
            raise ValueError("old cosine checkpoints require --legacy-cosine, or --memory for a new stage")
        # Loading Adam state restores its hyperparameters too; reject silently conflicting CLI values.
        for key in ("model", "lr", "warmup", "steps", "lr_schedule", "decay_steps",
                    "adam_eps", "adam_betas", "weight_decay", "qkv_lr_scale", "clip",
                    "ce_weight", "distill_weight", "distill_top_k", "distill_temperature"):
            if key in old_args and old_args[key] != getattr(args, key):
                raise ValueError(f"resume requires unchanged {key}; use --memory for a new stage")
        if resume_meta.get("training_version", 1) >= 2:
            for key in ("legacy_cosine", "min_lr_ratio"):
                if key in old_args and old_args[key] != getattr(args, key):
                    raise ValueError(f"resume requires unchanged {key}")
        if resume_meta.get("training_version", 1) >= 3:
            if not os.path.isfile(os.path.join(args.resume, "COMPLETE")):
                raise ValueError("cannot resume an unpublished checkpoint")
            saved_runtime = torch.load(os.path.join(args.resume, RUNTIME_NAME),
                                       map_location="cpu", weights_only=True)
            if saved_runtime.get("format_version") != 1 or len(saved_runtime["ranks"]) != world:
                raise ValueError("checkpoint runtime requires an unchanged world_size and supported format")
            resume_runtime = saved_runtime["ranks"][rank]
            if resume_meta.get("source_sha256") != source_sha256:
                raise ValueError("training source SHA-256 changed since checkpoint; use --memory for a new stage")
            if resume_meta.get("memory") != mem_cfg.to_dict():
                raise ValueError("resume cannot change memory configuration; use --memory for a new stage")
        else:
            warnings.warn("legacy checkpoint has no RNG snapshot; stochastic continuation is not exact")
        if args.exact_text_resume or kind == "post" or resume_runtime is not None:
            for key in ("data", "seq_len", "batch_size", "grad_accum", "seed", "shuffle_buffer", "min_doc_tokens"):
                if key in old_args and old_args[key] != getattr(args, key):
                    raise ValueError(f"data cursor resume requires unchanged {key}")
            if resume_meta.get("world_size", world) != world:
                raise ValueError("data cursor resume requires unchanged world_size")
        if resume_runtime is not None:
            for key in ("dtype", "tokenizer", "attn_impl", "window", "sinks", "attn_segment",
                        "teacher_cache", "loss_on", "query_categories", "max_segment",
                        "min_write_tokens", "stable_grad_norm"):
                if key in old_args and old_args[key] != getattr(args, key):
                    raise ValueError(f"resume requires unchanged {key}; use --memory for a new stage")
        processed_tokens = resume_meta.get("processed_tokens", 0)
        processed_before_resume_known = resume_meta.get("processed_before_resume_known", "processed_tokens" in resume_meta)
        if rank == 0:
            print(f"resumed from {args.resume} at step {start}")

    # v3 captures the actual per-rank cursor (excluding synthetic padding batches).
    data_seed = (resume_runtime["data_seed"] if resume_runtime is not None else
                 args.seed if args.exact_text_resume or kind == "post" else args.seed + start)
    if kind == "text":
        batches = text_batches(spec, tok, args, rank, world, data_seed)
    elif kind == "synth":
        cats = set(args.query_categories.split(",")) if args.query_categories else None
        batches = synth_batches(spec, tok, args.loss_on, args.max_segment, rank, data_seed, cats, args.min_write_tokens)
    elif kind == "post":
        if mem_cfg.chunk_size != args.attn_segment:
            raise ValueError("memory write step and attention segment must match")
        teacher_cache = (TeacherCache(args.teacher_cache, spec, args.model, args.distill_top_k,
                                      args.distill_temperature) if args.teacher_cache else None)
        batches = post_batches(spec, args.attn_segment, args.seq_len, rank, world, teacher_cache)
    else:
        raise SystemExit("--data must start with text:, synth:, or post:")
    consumed = (resume_runtime["consumed_batches"] if resume_runtime is not None else
                start * args.grad_accum if kind == "post" or (kind == "text" and args.exact_text_resume) else 0)
    batches = replay_iterator(batches, consumed)
    if resume_runtime is not None:
        restore_rng(resume_runtime, args.device)

    branches = [m for m in core.modules() if isinstance(m, LaCTMemoryBranch)]
    t0, tokens, run_loss, run_kl = time.time(), 0, 0.0, 0.0
    run_steps = 0
    run_started = time.time()
    batch_audit = []
    stopped = False
    completed = start
    exhausted = False

    def checkpoint_metadata(completed_step):
        return {"step": completed_step, "data": args.data, "training_args": vars(args),
                "world_size": world, "processed_tokens": processed_tokens,
                "processed_before_resume_known": processed_before_resume_known,
                "source_sha256": source_sha256}

    def save_checkpoint(destination, completed_step, *, runtime=None, **extra):
        runtime = (capture_runtime(args.device, batches.consumed, data_seed)
                   if runtime is None else runtime)
        save_rank_checkpoint(core, opt, destination, {**checkpoint_metadata(completed_step), **extra}, runtime)
    # start 是已经完成的步数。检查点 step_160 表示 160 步已完成，这里从 160 继续，下一次保存是 step_161。
    for step in range(start, args.steps):
        before_batch = (capture_runtime(args.device, batches.consumed, data_seed)
                        if args.save_on_nonfinite else None)
        for g in opt.param_groups:
            rate = (wsd_lr_at(step, args.lr, args.warmup, args.steps, args.decay_steps, args.min_lr_ratio)
                    if args.lr_schedule == "wsd" else lr_at(step, args.lr, args.warmup, args.steps,
                                                            args.min_lr_ratio, args.legacy_cosine))
            g["lr"] = rate * g.get("lr_scale", 1.0)
        microbatches, counts = optimizer_batch(batches, args.grad_accum, args.device, tok.eos_token_id)
        if microbatches is None:
            if kind != "post":
                raise ValueError("no usable training text; check the source, filters and rank sharding")
            if completed == 0:
                raise ValueError("post-training source contains no supervised examples")
            exhausted = True
            break
        global_count = counts[0]
        step_metrics = torch.zeros(2, dtype=torch.float64, device=args.device)
        for micro in range(args.grad_accum):
            b = microbatches[micro]
            # 长文本清掉分段，记忆分支内部按固定 chunk 先读后写。
            # 对话则把每一轮的边界和“这段要不要写入”交给分支。
            if b["segments"] is None:
                ctx.clear_segments()
            else:
                ctx.set_segments(b["segments"], b["update_flags"])
            x = b["input_ids"].to(args.device)
            y = b["labels"].to(args.device)
            attention_mask = b.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
            if args.batch_audit_dir and step == start:
                raw = b["input_ids"].numpy().astype("<u4").tobytes()
                batch_audit.append({"rank": rank, "step": step + 1, "micro": micro,
                                    "sha256_u32": hashlib.sha256(raw).hexdigest()})
            t_hidden = None
            cached_topk = b.get("teacher_topk")
            if cached_topk is not None:
                cached_topk = tuple(v.to(args.device) for v in cached_topk)
            if teacher is not None and bool((y[:, 1:] != -100).any()):
                with torch.no_grad(), autocast(args):
                    t_hidden = teacher.model(input_ids=x, attention_mask=attention_mask).last_hidden_state
            sync_context = model.no_sync() if world > 1 and micro < args.grad_accum - 1 else nullcontext()
            # no_sync must enclose both forward and backward, including checkpoint recomputation.
            with sync_context:
                with autocast(args):
                    ce, kl = model(input_ids=x, labels=y, attention_mask=attention_mask,
                                   teacher_hidden=t_hidden,
                                   teacher_topk=cached_topk,
                                   teacher_head=teacher.lm_head if t_hidden is not None else None)
                    local_count = (y[:, 1:] != -100).sum()
                    scale = supervised_token_scale(local_count, global_count, world)
                    loss = (args.ce_weight * ce + args.distill_weight * kl) * scale
                loss.backward()
            step_metrics += torch.stack((ce.detach(), kl.detach())).double() * local_count
        if world > 1:
            dist.all_reduce(step_metrics)
        run_loss += (step_metrics[0] / global_count).item()
        run_kl += (step_metrics[1] / global_count).item()
        run_steps += 1
        clip_function = clip_grad_norm_fp64 if args.stable_grad_norm else torch.nn.utils.clip_grad_norm_
        try:
            grad_norm = clip_function(params, args.clip, error_if_nonfinite=args.fail_on_nonfinite)
        except RuntimeError as exc:
            if not args.save_on_nonfinite:
                raise
            weights_finite = torch.tensor(int(all(bool(torch.isfinite(parameter).all()) for parameter in params)),
                                          device=args.device)
            if world > 1:
                dist.all_reduce(weights_finite, op=dist.ReduceOp.MIN)
            report, report_error = None, None
            if rank == 0:
                bad = [name for name, parameter in core.named_parameters()
                       if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                report = dict(completed_step=step, failed_step=step + 1,
                              error=str(exc), bad_gradient_parameters=bad,
                              trainable_weights_finite=bool(weights_finite))
                try:
                    os.makedirs(args.out, exist_ok=True)
                    with open(os.path.join(args.out, f"nonfinite_step_{step + 1}.json"), "w") as f:
                        json.dump(report, f, indent=2)
                except Exception as report_exc:
                    report_error = str(report_exc)
            raise_rank_errors(report_error, "nonfinite diagnostic")
            if bool(weights_finite):
                ckpt = os.path.join(args.out, f"nonfinite_before_{step + 1}")
                save_checkpoint(ckpt, step, runtime=before_batch, nonfinite_diagnostic=report)
                if rank == 0:
                    print(f"NONFINITE_PREUPDATE_CHECKPOINT_SAVED {ckpt}", flush=True)
            raise
        opt.step()
        opt.zero_grad(set_to_none=True)
        completed = step + 1
        tokens += counts[1].item()
        processed_tokens += counts[1].item()

        if rank == 0 and (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            # counts contains real tokens summed across all ranks, excluding padding.
            gate = torch.stack([m.scale_stat().float() for m in branches]).mean().item()
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print(
                f"step {step + 1} loss {run_loss / run_steps:.4f} "
                + (f"kl {run_kl / run_steps:.4f} " if args.distill_weight > 0 else "")
                + f"lr {opt.param_groups[0]['lr']:.2e} "
                f"|gate| {gate:.4f} tok/s {tokens / dt:.0f} peak_mem {mem:.1f}G",
                flush=True,
            )
            if args.progress_file:
                record = dict(state="training", step=step + 1, target_steps=args.steps,
                    time=time.time(), pid=os.getpid(), world_size=world,
                    global_batch=world * args.batch_size * args.grad_accum,
                    processed_tokens=processed_tokens,
                    processed_before_resume_known=processed_before_resume_known,
                    ce=run_loss / run_steps, kl=run_kl / run_steps, updates_since_log=run_steps,
                    learning_rate=opt.param_groups[0]["lr"], rank0_grad_norm=float(grad_norm),
                    tokens_per_second=tokens / dt, rank0_peak_gib=mem,
                    elapsed_seconds=time.time()-run_started)
                with open(args.progress_file + ".tmp", "w") as f:
                    json.dump(record, f, indent=2, allow_nan=False)
                os.replace(args.progress_file + ".tmp", args.progress_file)
            t0, tokens, run_loss, run_kl = time.time(), 0, 0.0, 0.0
            run_steps = 0
        if args.batch_audit_dir and step == start:
            os.makedirs(args.batch_audit_dir, exist_ok=True)
            with open(os.path.join(args.batch_audit_dir, f"rank{rank}.json"), "w") as f:
                json.dump(batch_audit, f, indent=2)
        if (step + 1) % args.save_every == 0 or step + 1 in args.save_at:
            ckpt = os.path.join(args.out, f"step_{step + 1}")
            save_checkpoint(ckpt, step + 1)
            if rank == 0 and args.keep_last > 0:
                old = sorted(
                    (d for d in os.listdir(args.out) if d.startswith("step_") and d[5:].isdigit()),
                    key=lambda d: int(d.split("_")[1])
                )
                for d in old[: -args.keep_last]:
                    shutil.rmtree(os.path.join(args.out, d))
        if args.stop_at and step + 1 >= args.stop_at:
            # 停在不是 save_every 整数倍的步数时，这里补存一次，避免这个点没有检查点。
            if (step + 1) % args.save_every != 0 and step + 1 not in args.save_at:
                ckpt = os.path.join(args.out, f"step_{step + 1}")
                save_checkpoint(ckpt, step + 1)
            if rank == 0:
                print(f"stopped at step {step + 1} (--stop-at)")
            stopped = True
            break

    # --stop-at 提前退出时不写 final/，避免把一个中间步误当成训练结束。
    if not stopped:
        save_checkpoint(os.path.join(args.out, "final"), completed, data_exhausted=exhausted)
        if rank == 0:
            print(f"saved {os.path.join(args.out, 'final')}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
