"""Same measurement as diag_lm.py, for the MemT 4B memory: how much of the state written by earlier chunks is still
there after the memory writes one more 2048-token chunk, and how much the earlier chunks change the answer logits."""
import json, sys, torch
sys.argv = ["x", "--model", "/data/models/Qwen/Qwen3-4B", "--memory", "/mnt/airs-business/memory/memt4b_20261120/step_2160",
            "--window", "2048", "--sinks", "0", "--attn-segment", "2048", "--attn-impl", "sdpa", "--mode", "write",
            "--data", "x", "--out", "/dev/null"]
sys.path.insert(0, "/data/qyl/memt_locomo_20261122")
import eval_locomo_memt as E
from lact_qwen.loading import add_model_args, load_model, autocast
import argparse
p = argparse.ArgumentParser(); p.add_argument("--mode"); p.add_argument("--data"); p.add_argument("--out"); add_model_args(p)
args = p.parse_args(sys.argv[1:])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
model, cfg = load_model(args)
dlg = json.loads(open("/data/qyl/memt_locomo_20261122/data/locomo2_g4.jsonl").readline()); q = dlg["queries"][0]
hist, spans = E.encode(dlg, tok); qq, aa = E.qa_ids(q, tok)
seq = hist + qq; P = len(seq); C = 2048
bounds = [(max(0, e - C), e) for e in range(P + 64, 0, -C)][::-1]
fs = bounds[-1][0]; dev = "cuda"
ctx = model.lact_ctx
def run(chunks):
    for s, e in chunks:
        E.forward_chunk(model, torch.tensor([seq[s:e]], device=dev), s, update=True)
def last_logp():
    lg = E.forward_chunk(model, torch.tensor([seq[fs:P]], device=dev), fs, update=False)
    return lg[0, -1].float().log_softmax(-1)
with torch.no_grad(), autocast(args):
    E.begin_stream(model, C); run(bounds[:-2]); pre = {l: s.clone() for l, s in ctx.states.items()}
    run(bounds[-2:-1]); a = {l: s.clone() for l, s in ctx.states.items()}; full = last_logp()
    ctx.states = {}; run(bounds[-2:-1]); b = {l: s.clone() for l, s in ctx.states.items()}; only_last = last_logp()
    ctx.states = {}; empty = last_logp()
    E.end_stream(model)
    fr = [float((a[l] - b[l]).norm() / pre[l].norm()) for l in sorted(pre)]
    kl = lambda x, y: float((x.exp() * (x - y)).sum())
    print("fraction of the earlier-chunk state still present after one more 2048-token chunk, per layer (first 12):",
          [round(v, 3) for v in fr[:12]], "mean", round(sum(fr) / len(fr), 3))
    print("KL(all earlier chunks || only the last earlier chunk) at the answer position:", round(kl(full, only_last), 4),
          " top1 same:", int(full.argmax()) == int(only_last.argmax()))
    print("KL(all earlier chunks || empty memory):", round(kl(full, empty), 4))
