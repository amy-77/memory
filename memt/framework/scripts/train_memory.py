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
加上 --min-doc-tokens 16384 后，窗口以外的上下文大多来自同一篇文档，而不是拼进来的别的文档。

合成对话（按轮写入，损失只算在答案上）:
    python scripts/train_memory.py --data synth:data/generated/synth_train.jsonl \
        --window 2048 --sinks 4 --memory checkpoints/stage1/final --steps 1000 --out checkpoints/synth

加 --replay oracle_key 或 --replay gmm，会在写入时带上重放。
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.distributed as dist

# 仓库根目录加入 import 路径，这样直接 python scripts/train_memory.py 也能找到 lact_qwen。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.dialog import encode_history, merge_spans, split_long_spans
from lact_qwen.distill import ce_and_kl
from lact_qwen.memory_branch import LaCTMemoryBranch
from lact_qwen.loading import add_model_args, autocast, load_model, load_teacher, load_tokenizer
from lact_qwen.modeling import CONFIG_NAME, freeze_base, load_memory_weights, memory_parameters, save_memory

OPTIMIZER_NAME = "optimizer.pt"
from lact_qwen.synth import read_jsonl
from lact_qwen.text_data import packed_sequences
from lact_qwen.post_data import post_batches, supervised_token_scale
from lact_qwen.numerics import clip_grad_norm_fp64


def text_batches(spec, tok, args, rank, world, seed):
    """长文本：把多篇文档拼成固定长度，标签就是输入本身（下一个 token 预测）。

    packed_sequences 按 rank 把数据流切开，每张卡看到的文档不同。
    segments=None 表示不按对话轮次切块，记忆分支会改用固定的 --mem-chunk。
    """
    stream = packed_sequences(
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
        x = torch.tensor([next(stream) for _ in range(args.batch_size)])
        yield {"input_ids": x, "labels": x.clone(), "segments": None, "update_flags": None}


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


def lr_at(step, base, warmup, total):
    """线性 warmup，然后余弦衰减到 0.1 * base。和 flame 默认的 cosine + lr_min=0.1 同一形式。

    step 用的是全局步数，不是本次进程从 0 重新数。续训时学习率仍按完整的 --steps 走。
    """
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


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
    p.add_argument("--data", required=True, help="text:<spec> or synth:<jsonl>")
    p.add_argument("--seq-len", type=int, default=16384, help="text data only")
    p.add_argument("--batch-size", type=int, default=1, help="per device, text data only")
    p.add_argument("--shuffle-buffer", type=int, default=1000, help="text data only; 0 keeps the source order")
    p.add_argument("--min-doc-tokens", type=int, default=0, help="text data only; skip shorter documents")
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
    p.add_argument("--distill-top-k", type=int, default=0)
    p.add_argument("--distill-temperature", type=float, default=1.0)
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
    p.add_argument("--exact-text-resume", action="store_true", help="replay the original text iterator to the saved optimizer step")
    p.add_argument("--fail-on-nonfinite", action="store_true")
    p.add_argument("--stable-grad-norm", action="store_true", help="reduce gradient norms in FP64")
    p.add_argument("--save-on-nonfinite", action="store_true",
                   help="save finite pre-update weights and optimizer at the completed step before failing")
    p.add_argument("--batch-audit-dir", default=None, help="record first optimizer batch token hashes on each rank")
    # --model、--window、--memory、--mem-* 等模型结构参数定义在 lact_qwen/loading.py。
    add_model_args(p)
    args = p.parse_args()
    if args.ce_weight < 0 or args.distill_weight < 0 or args.ce_weight + args.distill_weight == 0:
        raise ValueError("loss weights must be nonnegative with at least one positive")
    if args.decay_steps < 0 or args.decay_steps > args.steps - args.warmup:
        raise ValueError("decay_steps must fit after warmup")
    if args.data.startswith("post:"):
        if args.batch_size != 1 or args.resume:
            raise ValueError("one-pass post data currently requires batch=1 and no --resume")
        if args.attn_segment <= 0 or args.window != args.attn_segment or args.sinks != 0:
            raise ValueError("post-training requires current-segment-only attention and zero sinks")
        if args.ce_weight != 0 or args.distill_weight <= 0:
            raise ValueError("MemT post-training must use KL only")
    if args.memory == "none":
        raise SystemExit("--memory must be 'new' or a checkpoint directory")

    # torchrun 会设置 RANK / WORLD_SIZE / LOCAL_RANK。单卡直接 python 启动时这些变量不存在，当成 1 卡。
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        args.device = f"cuda:{local}"
    torch.manual_seed(args.seed + rank)

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
    teacher = load_teacher(args, tok) if args.distill_weight > 0 else None
    params = memory_parameters(model)
    n_params = sum(p.numel() for p in params)
    if rank == 0:
        print(f"trainable memory parameters: {n_params / 1e6:.1f}M across {len(mem_cfg.layer_indices(model.config.num_hidden_layers))} layers")

    # DDP 包的是同一个 module，下面的 params 仍然指向原来的记忆参数。
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
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
    if args.resume:
        load_memory_weights(core, args.resume)
        opt.load_state_dict(torch.load(os.path.join(args.resume, OPTIMIZER_NAME), map_location=args.device))
        with open(os.path.join(args.resume, CONFIG_NAME)) as f:
            start = json.load(f)["step"]
        if rank == 0:
            print(f"resumed from {args.resume} at step {start}")

    # 续训恢复不了“数据流已经读到哪一条”，所以换一个种子，重新抽数据，而不是把旧数据再按原顺序读一遍。
    data_seed = args.seed if args.exact_text_resume else args.seed + start
    kind, spec = args.data.split(":", 1)
    if kind == "text":
        batches = text_batches(spec, tok, args, rank, world, data_seed)
        if args.exact_text_resume and start:
            batches = itertools.islice(batches, start * args.grad_accum, None)
    elif kind == "synth":
        cats = set(args.query_categories.split(",")) if args.query_categories else None
        batches = synth_batches(spec, tok, args.loss_on, args.max_segment, rank, data_seed, cats, args.min_write_tokens)
    elif kind == "post":
        if mem_cfg.chunk_size != args.attn_segment:
            raise ValueError("memory write step and attention segment must match")
        batches = post_batches(spec, args.attn_segment, args.seq_len, rank, world)
    else:
        raise SystemExit("--data must start with text: or synth:")

    branches = [m for m in core.modules() if isinstance(m, LaCTMemoryBranch)]
    t0, tokens, run_loss, run_kl = time.time(), 0, 0.0, 0.0
    run_started = time.time()
    batch_audit = []
    stopped = False
    # start 是已经完成的步数。检查点 step_160 表示 160 步已完成，这里从 160 继续，下一次保存是 step_161。
    for step in range(start, args.steps):
        for g in opt.param_groups:
            rate = (wsd_lr_at(step, args.lr, args.warmup, args.steps, args.decay_steps, args.min_lr_ratio)
                    if args.lr_schedule == "wsd" else lr_at(step, args.lr, args.warmup, args.steps))
            g["lr"] = rate * g.get("lr_scale", 1.0)
        post_microbatches = None
        if kind == "post":
            # Normalize over the complete optimizer batch, including accumulation and ranks.
            post_microbatches = [next(batches) for _ in range(args.grad_accum)]
            global_count = torch.tensor(sum(int((b["labels"][:, 1:] != -100).sum())
                for b in post_microbatches), dtype=torch.float32, device=args.device)
            if world > 1:
                dist.all_reduce(global_count)
        for micro in range(args.grad_accum):
            b = post_microbatches[micro] if post_microbatches is not None else next(batches)
            # 长文本清掉分段，记忆分支内部按固定 chunk 先读后写。
            # 对话则把每一轮的边界和“这段要不要写入”交给分支。
            if b["segments"] is None:
                ctx.clear_segments()
            else:
                ctx.set_segments(b["segments"], b["update_flags"])
            x = b["input_ids"].to(args.device)
            y = b["labels"].to(args.device)
            if args.batch_audit_dir and step == start:
                raw = b["input_ids"].numpy().astype("<u4").tobytes()
                batch_audit.append({"rank": rank, "step": step + 1, "micro": micro,
                                    "sha256_u32": hashlib.sha256(raw).hexdigest()})
            with autocast(args):
                logits = model(input_ids=x).logits
            if teacher is None:
                # 标准的下一个 token 损失：用第 t 个位置的 logits 预测第 t+1 个 token。
                loss = args.ce_weight * torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100
                )
            else:
                with torch.no_grad(), autocast(args):
                    t_hidden = teacher.model(input_ids=x).last_hidden_state
                ce, kl = ce_and_kl(logits, t_hidden, teacher.lm_head, y,
                                   top_k=args.distill_top_k, temperature=args.distill_temperature)
                run_kl += kl.item() / args.grad_accum
                # 日志里的 loss 始终是下一个 token 的损失，这样开不开蒸馏都能直接比。
                loss = args.ce_weight * ce + args.distill_weight * kl
                run_ce = ce.item()
            if kind == "post":
                local_count = (y[:, 1:] != -100).sum().float()
                loss = loss * supervised_token_scale(local_count, global_count, world) * args.grad_accum
            # 除以累积步数，相当于对这几次微批次取平均，而不是把梯度放大 grad_accum 倍。
            (loss / args.grad_accum).backward()
            run_loss += (loss.item() if teacher is None else run_ce) / args.grad_accum
            tokens += x.numel()
        clip_function = clip_grad_norm_fp64 if args.stable_grad_norm else torch.nn.utils.clip_grad_norm_
        try:
            grad_norm = clip_function(params, args.clip, error_if_nonfinite=args.fail_on_nonfinite)
        except RuntimeError as exc:
            if not args.save_on_nonfinite:
                raise
            if rank == 0:
                bad = [name for name, parameter in core.named_parameters()
                       if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                weights_finite = all(bool(torch.isfinite(parameter).all()) for parameter in params)
                report = dict(completed_step=step, failed_step=step + 1,
                              error=str(exc), bad_gradient_parameters=bad,
                              trainable_weights_finite=weights_finite)
                os.makedirs(args.out, exist_ok=True)
                with open(os.path.join(args.out, f"nonfinite_step_{step + 1}.json"), "w") as f:
                    json.dump(report, f, indent=2)
                if weights_finite:
                    ckpt = os.path.join(args.out, f"step_{step}")
                    save_memory(core, ckpt, {"step": step, "data": args.data,
                                            "training_args": vars(args), "nonfinite_diagnostic": report})
                    torch.save(opt.state_dict(), os.path.join(ckpt, OPTIMIZER_NAME))
                    print(f"NONFINITE_PREUPDATE_CHECKPOINT_SAVED {ckpt}", flush=True)
            if world > 1:
                dist.barrier()
            raise
        opt.step()
        opt.zero_grad(set_to_none=True)

        if rank == 0 and (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            # |gate| 是记忆输出缩放的平均绝对值。文本模式下每张卡的 token 数相同，所以乘 world 就是总吞吐。
            gate = torch.stack([m.scale_stat().float() for m in branches]).mean().item()
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print(
                f"step {step + 1} loss {run_loss / args.log_every:.4f} "
                + (f"kl {run_kl / args.log_every:.4f} " if teacher is not None else "")
                + f"lr {opt.param_groups[0]['lr']:.2e} "
                f"|gate| {gate:.4f} tok/s {tokens * world / dt:.0f} peak_mem {mem:.1f}G",
                flush=True,
            )
            if args.progress_file:
                record = dict(state="training", step=step + 1, target_steps=args.steps,
                    time=time.time(), pid=os.getpid(), world_size=world,
                    global_batch=world * args.batch_size * args.grad_accum,
                    processed_tokens=(step + 1) * world * args.batch_size * args.grad_accum * args.seq_len,
                    rank0_ce=run_loss / args.log_every, rank0_kl=run_kl / args.log_every,
                    learning_rate=opt.param_groups[0]["lr"], rank0_grad_norm=float(grad_norm),
                    tokens_per_second=tokens * world / dt, rank0_peak_gib=mem,
                    elapsed_seconds=time.time()-run_started)
                with open(args.progress_file + ".tmp", "w") as f:
                    json.dump(record, f, indent=2, allow_nan=False)
                os.replace(args.progress_file + ".tmp", args.progress_file)
            t0, tokens, run_loss, run_kl = time.time(), 0, 0.0, 0.0
        if args.batch_audit_dir and step == start:
            os.makedirs(args.batch_audit_dir, exist_ok=True)
            with open(os.path.join(args.batch_audit_dir, f"rank{rank}.json"), "w") as f:
                json.dump(batch_audit, f, indent=2)
        if rank == 0 and ((step + 1) % args.save_every == 0 or step + 1 in args.save_at):
            ckpt = os.path.join(args.out, f"step_{step + 1}")
            save_memory(core, ckpt, {"step": step + 1, "data": args.data, "training_args": vars(args)})
            torch.save(opt.state_dict(), os.path.join(ckpt, OPTIMIZER_NAME))
            if args.keep_last > 0:
                old = sorted(
                    (d for d in os.listdir(args.out) if d.startswith("step_")), key=lambda d: int(d.split("_")[1])
                )
                for d in old[: -args.keep_last]:
                    shutil.rmtree(os.path.join(args.out, d))
        if args.stop_at and step + 1 >= args.stop_at:
            # 停在不是 save_every 整数倍的步数时，这里补存一次，避免这个点没有检查点。
            if rank == 0 and (step + 1) % args.save_every != 0 and step + 1 not in args.save_at:
                ckpt = os.path.join(args.out, f"step_{step + 1}")
                save_memory(core, ckpt, {"step": step + 1, "data": args.data, "training_args": vars(args)})
                torch.save(opt.state_dict(), os.path.join(ckpt, OPTIMIZER_NAME))
            if rank == 0:
                print(f"stopped at step {step + 1} (--stop-at)")
            stopped = True
            break

    # --stop-at 提前退出时不写 final/，避免把一个中间步误当成训练结束。
    if rank == 0 and not stopped:
        save_memory(core, os.path.join(args.out, "final"), {"step": args.steps, "data": args.data})
        print(f"saved {os.path.join(args.out, 'final')}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
