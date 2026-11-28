"""LiveMem 公开 SFT 权重上的 oracle 写入诊断：不训练，只做前向。

不改 LiveMem 的任何源码（固定在 71a6a8b，启动时会检查 models/ 下两个文件和该提交一致）：
- 选择写入：用官方 forward 自带的 write_mask 参数。官方实现把 g、b、w 同乘掩码
  （models/memory_gdn2.py:128），掩码为 0 的 token 既不写、不擦，也不衰减。
- 去掉衰减：前向前把 36 层的 A_log 临时改成 -30，前向后恢复。每个 token 的对数衰减是
  -exp(A_log)·softplus(...)，exp(-30)≈1e-13，衰减近似为 0。只改内存里的参数，不改文件。
- 题目集合：直接调用 H200 实验的 experiment1.py 的 prepare（原样拷贝为 h200_experiment1.py），
  得到同一套"证据已全部被挤出 32K 窗口"的题目、同一套淘汰时序和答案标签。

子命令：
  turns      在 H200 的 eligible.json 上补充证据轮次（has_answer）的 token 区间和匹配的随机轮次
  run        逐题逐组前向，记录答案 token 上的平均交叉熵
  decay      用真实 hidden state 测证据写入后到读答案时还剩多少（只算遗忘门）
  summarize  汇总成 summary.json 和 summary.md
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h200_experiment1 as h200  # noqa: E402  H200 原脚本，未改

# 每组实验：(掩码类型, 是否去掉衰减, 是否按 32K 窗口淘汰注意力)
ARMS = {
    "write_all": ("all", False, True),               # 官方默认：每个 token 都写
    "write_nothing": ("h200_nothing", False, True),  # H200 同名组：历史不写，system 和问答照写
    "session_oracle": ("h200_oracle", False, True),  # H200 的 oracle：只写证据会话，问答照写
    "memory_off": ("zeros", False, True),            # 全部不写，状态始终为 0，记忆支路输出为 0
    "turn_oracle": ("turn_oracle", False, True),     # 只写 has_answer 轮次，问答只读
    "turn_random": ("turn_random", False, True),     # 同样 token 数的随机非证据轮次，问答只读；turn_oracle 的对照
    "turn_oracle_nodecay": ("turn_oracle", True, True),   # 同上，再去掉衰减
    "turn_random_nodecay": ("turn_random", True, True),   # 同样 token 数的随机非证据轮次，去掉衰减
    "full_attention": ("all", False, False),         # 不淘汰，注意力看得到全部历史，作上界
}
TURN_ARMS = {a for a, (kind, _, _) in ARMS.items() if kind.startswith("turn_")}
NO_DECAY_A_LOG = -30.0


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def check_livemem_source(repo):
    """确认模型代码就是官方 71a6a8b，没有被改过。"""
    files = ["models/memory_gdn2.py", "models/modeling_memory_qwen3.py"]
    head = subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", str(repo), "diff", "--quiet",
                           "HEAD", "--", *files])
    if not head.startswith("71a6a8b") or diff.returncode != 0:
        raise RuntimeError(f"LiveMem 模型代码不是干净的 71a6a8b: head={head}, diff_rc={diff.returncode}")
    return {"commit": head, "files": {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in files}}


# ----------------------------------------------------------------------------- turns
def build_parts(row, fmt):
    """和 h200_experiment1.prepare 完全相同的拼接方式，另外返回每轮的字符串。"""
    dates = row.get("haystack_dates") or [None] * len(row["haystack_sessions"])
    turn_strs = [[f"{t.get('role', '')}: {t.get('content', '')}" for t in turns]
                 for turns in row["haystack_sessions"]]
    prefixes = [f"[Date: {d}]\n" if d else "" for d in dates]
    docs = [p + "\n".join(ts) for p, ts in zip(prefixes, turn_strs)]
    question = fmt.user_single("longmemeval_s", row["question"], reason=False)
    gold = fmt.gold_single(str(row["answer"]))
    parts = [fmt.system_segment(), "<|im_start|>user\n"]
    parts += [doc + "\n\n" for doc in docs]
    parts += [question + "<|im_end|>\n<|im_start|>assistant\n", gold + "<|im_end|>\n"]
    return parts, prefixes, turn_strs


def add_turns(args):
    import numpy as np
    from transformers import AutoTokenizer
    sys.path.insert(0, str(args.repo / "tools/data_process"))
    import format as fmt
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    rows = json.loads(args.data.read_text())
    metas = json.loads((args.run / "data/eligible.json").read_text())
    out, no_turn = [], []
    for meta in metas:
        row = rows[meta["row_index"]]
        parts, prefixes, turn_strs = build_parts(row, fmt)
        enc = tok("".join(parts), add_special_tokens=False, return_offsets_mapping=True)
        ids = np.asarray(enc.input_ids, dtype=np.int32)
        # 必须和 H200 prepare 存下的 token 序列逐个相同
        assert hashlib.sha256(ids.tobytes()).hexdigest() == meta["input_ids_sha256"], meta["id"]
        offs = np.asarray(enc.offset_mapping)
        char0 = np.cumsum([0] + [len(p) for p in parts])

        # 每个会话里每一轮的 token 区间：字符区间有重叠的 token 都算进去
        turn_spans = {}
        for i, strs in enumerate(turn_strs):
            pos = int(char0[2 + i]) + len(prefixes[i])
            ss, se = meta["session_spans"][i]
            for j, s in enumerate(strs):
                idx = np.flatnonzero((offs[:, 1] > pos) & (offs[:, 0] < pos + len(s)))
                if len(idx):
                    a, b = int(idx[0]), int(idx[-1]) + 1
                    assert ss <= a < b <= se, (meta["id"], i, j)
                    turn_spans[(i, j)] = (a, b)
                pos += len(s) + 1

        evidence = []
        for i in meta["evidence"]:
            for j, t in enumerate(row["haystack_sessions"][i]):
                if t.get("has_answer") and (i, j) in turn_spans:
                    evidence.append(turn_spans[(i, j)])
        if not evidence:
            no_turn.append(meta["id"])
            meta = dict(meta, turn_spans=None, turn_random_spans=None, turn_tokens=0)
            out.append(meta)
            continue
        budget = sum(b - a for a, b in evidence)

        # 随机对照：从非证据会话的轮次里随机抽，凑够同样的 token 数，最后一轮截到正好相等
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}:turns:{meta['id']}".encode()).digest()[:8], "little")
        cands = [v for (i, j), v in turn_spans.items() if i not in set(meta["evidence"])]
        random.Random(seed).shuffle(cands)
        rand, left = [], budget
        for a, b in cands:
            if left <= 0:
                break
            take = min(b - a, left)
            rand.append((a, a + take))
            left -= take
        assert left == 0, meta["id"]
        meta = dict(meta, turn_spans=sorted(evidence), turn_random_spans=sorted(rand), turn_tokens=budget)
        out.append(meta)
    save(args.run / "data/eligible_turns.json", out)
    tokens = [m["turn_tokens"] for m in out if m["turn_tokens"]]
    info = {"questions": len(out), "with_evidence_turns": len(tokens), "without_evidence_turns": no_turn,
            "turn_tokens_median": float(np.median(tokens)), "turn_tokens_max": int(max(tokens)),
            "turn_tokens_p90": float(np.percentile(tokens, 90)),
            "session_oracle_tokens_median": float(np.median([m["oracle_history_tokens"] for m in out]))}
    save(args.run / "data/turns_info.json", info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


# ----------------------------------------------------------------------------- model
def build_mask(meta, kind):
    import numpy as np
    n = meta["length"]
    if kind == "all":
        return None
    if kind == "h200_nothing":
        return h200.make_mask(meta, "write_nothing", n)
    if kind == "h200_oracle":
        return h200.make_mask(meta, "oracle", n)
    mask = np.zeros(n, dtype=np.float32)
    if kind == "zeros":
        return mask
    spans = meta["turn_spans"] if kind == "turn_oracle" else meta["turn_random_spans"]
    for a, b in spans:
        mask[a:b] = 1
    return mask


def load_model(args, device):
    import torch
    from transformers.modeling_utils import PreTrainedModel
    sys.path.insert(0, str(args.repo))
    from models import MemoryQwen3ForCausalLM
    # 和 H200 脚本相同的加载方式：子类覆盖的 from_pretrained 不支持 output_loading_info
    model, info = PreTrainedModel.from_pretrained.__func__(
        MemoryQwen3ForCausalLM, args.checkpoint, dtype=torch.bfloat16,
        attn_implementation="flex_attention", local_files_only=True, output_loading_info=True)
    bad = {k: v for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if bad:
        raise RuntimeError(f"权重加载不匹配: {bad}")
    model = model.eval().to(device)
    mods = [m for m in model.modules() if type(m).__name__ == "MemoryGatedDeltaNet2"]
    assert len(mods) == 36
    return model, mods


def load_arrays(args, meta):
    import numpy as np
    payload = (args.run / "data" / meta["array_file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == meta["array_sha256"]
    with np.load(io.BytesIO(payload)) as packed:
        return {k: packed[k] for k in packed.files}


def forward_loss(model, mods, arr, mask, nodecay, evict, device):
    """一次前向，返回答案 token 上的平均交叉熵（官方 forward 只在答案位置算 logits）。"""
    import torch
    keys = ["input_ids", "labels"] + (["chunk_id", "evict_step"] if evict else [])
    kw = {k: torch.tensor(arr[k].astype("int64"), device=device)[None] for k in keys}
    if mask is not None:
        kw["write_mask"] = torch.tensor(mask, device=device)[None]
    saved = None
    if nodecay:
        saved = [m.A_log.data.clone() for m in mods]
        for m in mods:
            m.A_log.data.fill_(NO_DECAY_A_LOG)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    try:
        with torch.inference_mode():
            out = model(**kw, use_cache=False)
        torch.cuda.synchronize(device)
    finally:
        if saved is not None:
            for m, s in zip(mods, saved):
                m.A_log.data.copy_(s)
    value = float(out.loss)
    assert out.logits is None and math.isfinite(value)
    return value, time.monotonic() - start, torch.cuda.max_memory_allocated(device)


def validate(model, mods, arr, meta, device, out_path):
    """第一题上的三项检查：全 1 掩码等于不传掩码；全 0 掩码时记忆支路输出为 0；A_log 能恢复。"""
    import numpy as np
    import torch
    base = forward_loss(model, mods, arr, None, False, True, device)[0]
    ones = forward_loss(model, mods, arr, np.ones(meta["length"], np.float32), False, True, device)[0]
    side_max = []
    hooks = [m.register_forward_hook(lambda mod, a, o: side_max.append(float(o[0].abs().max()))) for m in mods]
    try:
        forward_loss(model, mods, arr, np.zeros(meta["length"], np.float32), False, True, device)
    finally:
        for h in hooks:
            h.remove()
    before = [m.A_log.detach().clone() for m in mods]
    forward_loss(model, mods, arr, None, True, True, device)
    restored = all(torch.equal(b, m.A_log) for b, m in zip(before, mods))
    checks = {"id": meta["id"], "write_all_loss": base, "all_ones_loss": ones,
              "ones_abs_diff": abs(base - ones), "zero_mask_side_abs_max": max(side_max),
              "a_log_restored": restored, "a_log_dtype": str(mods[0].A_log.dtype)}
    checks["pass"] = checks["ones_abs_diff"] <= 1e-5 and checks["zero_mask_side_abs_max"] == 0.0 and restored
    save(out_path, checks)
    if not checks["pass"]:
        raise AssertionError(f"检查没通过: {checks}")
    return checks


def run(args):
    import torch
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    src = check_livemem_source(args.repo)
    model, mods = load_model(args, device)
    arms = args.arms.split(",") if args.arms else list(ARMS)
    metas = json.loads((args.run / "data/eligible_turns.json").read_text())[:args.limit or None]
    metas = metas[args.rank::args.world]
    res = args.run / "results"
    res.mkdir(exist_ok=True)
    save(args.run / f"logs/runtime_{args.rank}.json", {
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint": str(args.checkpoint), "livemem": src, "arms": arms})
    for n, meta in enumerate(metas):
        arr = load_arrays(args, meta)
        if n == 0 and not (args.run / f"logs/validation_{args.rank}.json").exists():
            print(json.dumps(validate(model, mods, arr, meta, device,
                                      args.run / f"logs/validation_{args.rank}.json")), flush=True)
        for arm in arms:
            path = res / f"{meta['row_index']:04d}_{arm}.json"
            if path.exists():
                continue
            if arm in TURN_ARMS and not meta["turn_spans"]:
                continue
            kind, nodecay, evict = ARMS[arm]
            value, sec, peak = forward_loss(model, mods, arr, build_mask(meta, kind), nodecay, evict, device)
            rec = {"id": meta["id"], "row_index": meta["row_index"], "question_type": meta["question_type"],
                   "arm": arm, "loss": value, "answer_tokens": meta["answer_tokens"], "length": meta["length"],
                   "turn_tokens": meta["turn_tokens"], "session_tokens": meta["oracle_history_tokens"],
                   "seconds": sec, "peak_allocated_bytes": peak}
            save(path, rec)
            print(json.dumps(rec), flush=True)
        save(args.run / f"logs/progress_{args.rank}.json",
             {"done": n + 1, "assigned": len(metas), "last": meta["id"]})


# ----------------------------------------------------------------------------- decay
def decay(args):
    """只看遗忘门：证据写进状态后，到读答案那一刻还剩多少（exp(Σg)），用真实 hidden state。

    读答案的位置 a = 第一个答案 token 的前一个位置（它的输出预测第一个答案 token）。
    token t 的写入到 a 时的保留比例 = exp(g_{t+1} + ... + g_a)，g 已乘掩码，按 128 维 key 通道分别算。
    擦除门 b 对旧内容的覆盖这里不算，所以这是保留比例的上限。
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    check_livemem_source(args.repo)
    model, mods = load_model(args, device)
    metas = [m for m in json.loads((args.run / "data/eligible_turns.json").read_text()) if m["turn_spans"]]
    metas = metas[:args.limit or None][args.rank::args.world]
    outdir = args.run / "decay"
    outdir.mkdir(exist_ok=True)
    state = {}

    def hook(mod, a, kw):
        h = a[0] if a else kw["hidden_states"]
        T = h.shape[1]
        H, dk = mod.A_log.shape[0], mod.head_k_dim
        with torch.no_grad():
            sp = F.softplus(mod.f_proj(h).float() + mod.dt_bias.float())[0]          # [T, H*dk]
            g = (sp.view(T, H, dk) * (-mod.A_log.float().exp())[None, :, None]).view(T, H * dk)
            wm = kw.get("write_mask")
            if wm is not None:
                g = g * wm[0].float()[:, None]
            r = state["read"]
            # suffix[t] = g_t + ... + g_r；token t 的写入到 r 时的保留 = exp(suffix[t+1])
            suffix = g[:r + 1].double().flip(0).cumsum(0).flip(0)
            ev = state["ev_tokens"]
            log_keep = suffix[ev + 1] if ev.numel() else suffix[:0]                      # [n_ev, C]
            last = log_keep[-1]
            hist = slice(state["hist"][0], state["hist"][1])
            g_hist = g[hist]
            # 历史区间里每个通道的平均每 token 衰减 → 半衰期（token）；只在写入的 token 上平均
            if wm is not None:
                w = wm[0, hist].float()
                mean_g = (g_hist.sum(0) / w.sum().clamp(min=1)).double()
            else:
                mean_g = g_hist.mean(0).double()
            half = (math.log(0.5) / mean_g.clamp(max=-1e-30)).float()
            thr = math.log(0.01)
            state["layers"].append({
                "last_frac_keep_gt_1pct": float((last > thr).double().mean()),
                "last_median_log10_keep": float(last.median() / math.log(10)),
                "last_max_log10_keep": float(last.max() / math.log(10)),
                "all_frac_keep_gt_1pct": float((log_keep > thr).double().mean()),
                "half_life_median_tokens": float(half.median()),
                "half_life_p90_tokens": float(half.quantile(0.9)),
            })

    hooks = [m.register_forward_pre_hook(hook, with_kwargs=True) for m in mods]
    try:
        for meta in metas:
            path = outdir / f"{meta['row_index']:04d}.json"
            if path.exists():
                continue
            arr = load_arrays(args, meta)
            read = int(np.flatnonzero(arr["labels"] != -100)[0]) - 1
            sess = [tuple(meta["session_spans"][i]) for i in meta["evidence"]]
            rec = {"id": meta["id"], "question_type": meta["question_type"], "read_pos": read,
                   "length": meta["length"], "arms": {}}
            for arm, kind, spans in [("write_all", "all", sess), ("session_oracle", "h200_oracle", sess),
                                     ("turn_oracle", "turn_oracle", meta["turn_spans"])]:
                ev = np.concatenate([np.arange(a, b) for a, b in spans])
                state.update(read=read, ev_tokens=torch.tensor(ev, device=device),
                             hist=(meta["sink_end"], meta["question_start"]), layers=[])
                loss = forward_loss(model, mods, arr, build_mask(meta, kind), False, True, device)[0]
                rec["arms"][arm] = {"loss": loss, "evidence_tokens": int(len(ev)),
                                    "tokens_after_last_evidence": int(read - ev[-1]), "layers": state["layers"]}
            save(path, rec)
            print(json.dumps({"id": meta["id"], **{a: {k: v for k, v in r.items() if k != "layers"}
                                                     for a, r in rec["arms"].items()}}), flush=True)
    finally:
        for h in hooks:
            h.remove()


# ----------------------------------------------------------------------------- summarize
def summarize(args):
    import numpy as np
    recs = collections.defaultdict(dict)
    for p in (args.run / "results").glob("*.json"):
        r = json.loads(p.read_text())
        recs[r["id"]][r["arm"]] = r
    rng = np.random.default_rng(0)

    def paired(arm, base, ids):
        d = np.array([recs[i][arm]["loss"] - recs[i][base]["loss"] for i in ids])
        boot = rng.choice(d, size=(10000, len(d))).mean(1)
        return {"n": len(d), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5))], "median_diff": float(np.median(d)),
                "frac_lower_loss": float((d < 0).mean())}

    out = {"arms": {}, "vs_write_all": {}, "turn_subset": {}, "by_type": {}}
    for arm in ARMS:
        ids = [i for i in recs if arm in recs[i]]
        if ids:
            out["arms"][arm] = {"n": len(ids), "mean_loss": float(np.mean([recs[i][arm]["loss"] for i in ids]))}
    full = [i for i in recs if all(a in recs[i] for a in ARMS if a not in TURN_ARMS)]
    for arm in ARMS:
        if arm != "write_all" and arm not in TURN_ARMS and full:
            out["vs_write_all"][arm] = paired(arm, "write_all", full)
    # 第一批（write_all 加四个按轮次写入的组）先跑完，不必等第二批就能比较
    p1 = ["write_all", *sorted(TURN_ARMS)]
    pids = [i for i in recs if all(a in recs[i] for a in p1)]
    if pids:
        out["pass1"] = {a: {"n": len(pids), "mean_loss": float(np.mean([recs[i][a]["loss"] for i in pids])),
                            "median_loss": float(np.median([recs[i][a]["loss"] for i in pids]))} for a in p1}
        out["pass1_pairs"] = {f"{a}-{b}": paired(a, b, pids) for a, b in [
            ("turn_oracle", "turn_random"), ("turn_oracle_nodecay", "turn_random_nodecay"),
            ("turn_oracle_nodecay", "turn_oracle"), ("turn_random_nodecay", "turn_random"),
            ("turn_oracle", "write_all"), ("turn_random", "write_all"), ("turn_oracle_nodecay", "write_all")]}
    tids = [i for i in recs if all(a in recs[i] for a in ARMS)]
    if tids:
        for arm in ARMS:
            out["turn_subset"][arm] = {"n": len(tids), "mean_loss": float(np.mean([recs[i][arm]["loss"] for i in tids]))}
        pairs = [("turn_oracle", "turn_random"), ("turn_oracle_nodecay", "turn_random_nodecay"),
                 ("turn_oracle_nodecay", "turn_oracle"), ("turn_random_nodecay", "turn_random"),
                 ("turn_oracle", "write_all"), ("session_oracle", "write_nothing"),
                 ("turn_oracle_nodecay", "write_all"), ("full_attention", "write_all"),
                 ("write_all", "memory_off")]
        out["turn_subset_pairs"] = {f"{a}-{b}": paired(a, b, tids) for a, b in pairs}
        types = collections.defaultdict(list)
        for i in tids:
            types[recs[i]["write_all"]["question_type"]].append(i)
        for t, ids in sorted(types.items()):
            out["by_type"][t] = {"n": len(ids), **{a: float(np.mean([recs[i][a]["loss"] for i in ids])) for a in ARMS}}
    dec = [json.loads(p.read_text()) for p in sorted((args.run / "decay").glob("*.json"))]
    if dec:
        out["decay"] = {"questions": len(dec)}
        for arm in dec[0]["arms"]:
            L = len(dec[0]["arms"][arm]["layers"])
            keys = dec[0]["arms"][arm]["layers"][0].keys()
            out["decay"][arm] = {
                "evidence_tokens_median": float(np.median([d["arms"][arm]["evidence_tokens"] for d in dec])),
                "tokens_after_last_evidence_median": float(np.median([d["arms"][arm]["tokens_after_last_evidence"] for d in dec])),
                "layers": [{k: float(np.median([d["arms"][arm]["layers"][l][k] for d in dec])) for k in keys}
                           for l in range(L)]}
    save(args.run / "summary.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "decay"}, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["turns", "run", "decay", "summarize"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--repo", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--arms", default="")
    args = p.parse_args()
    for name in ["run", "repo", "checkpoint", "data"]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    (args.run / "logs").mkdir(parents=True, exist_ok=True)
    {"turns": add_turns, "run": run, "decay": decay, "summarize": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
