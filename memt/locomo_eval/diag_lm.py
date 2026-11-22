"""Is the GDN2 state carried across chunks, and how much of it survives one 2048-token chunk?"""
import json, sys, torch
sys.argv = ["x", "--mode", "write", "--out", "/dev/null"]
sys.path.insert(0, "/data/qyl/memt_locomo_20261122")
import eval_locomo_livemem as E
from transformers import AutoModelForCausalLM, AutoTokenizer
P = "/data/qyl/memt_locomo_20261122/livemem_models/LiveMem-RL"
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa").cuda().eval()
dlg = json.loads(open("/data/qyl/memt_locomo_20261122/data/locomo2_g4.jsonl").readline()); q = dlg["queries"][0]
raw, ids, spans, hist_end = E.build(dlg, q, tok)
C = 2048; Pn = len(ids)
bounds = [(max(0, e - C), e) for e in range(Pn + 64, 0, -C)][::-1]
fs = bounds[-1][0]
dev = "cuda"
def run(chunk_list, mc):
    for s, e in chunk_list:
        model(input_ids=torch.tensor([ids[s:e]], device=dev), position_ids=torch.arange(s, e, device=dev)[None], use_cache=True, mem_cache=mc, logits_to_keep=1)
def last_logits(mc, wm0=False):
    ids_f = ids[fs:]
    kw = {}
    if wm0:
        kw["write_mask"] = torch.zeros(1, len(ids_f), device=dev)
    out = model(input_ids=torch.tensor([ids_f], device=dev), position_ids=torch.arange(fs, fs + len(ids_f), device=dev)[None],
                use_cache=True, mem_cache=mc, logits_to_keep=1, **kw)
    return out.logits[0, -1].float().log_softmax(-1)
with torch.no_grad():
    mc = E.MemCache(); run(bounds[:-1], mc)
    st = [x["recurrent_state"] for x in mc.layers if x is not None]
    print("layers with state", len(st), "state norm (mean over layers)", sum(float(s.float().norm()) for s in st) / len(st))
    base = last_logits(E.MemCache())
    full = last_logits(mc.clone())
    kl = lambda a, b: float((a.exp() * (a - b)).sum())
    print("final chunk written (as in the run): KL(prefix-state || empty-state) =", kl(full, base),
          " top1 same:", int(full.argmax()) == int(base.argmax()))
    base0 = last_logits(E.MemCache(), wm0=True); full0 = last_logits(mc.clone(), wm0=True)
    print("final chunk read-only (write_mask 0): KL =", kl(full0, base0), " top1 same:", int(full0.argmax()) == int(base0.argmax()))
    # decay of the state over the final chunk: run the final chunk on a state, compare with zero-initialised run
    mc_a = mc.clone(); last_logits(mc_a)  # state after final chunk starting from prefix state
    mc_b = E.MemCache(); last_logits(mc_b)  # same chunk from empty state
    diff = [float((a["recurrent_state"] - b["recurrent_state"]).float().norm()) / max(1e-9, float(s.float().norm()))
            for a, b, s in zip([x for x in mc_a.layers if x], [x for x in mc_b.layers if x], st)]
    print("fraction of the prefix state still present after one 2048-token chunk, per layer (first 12):",
          [round(d, 4) for d in diff[:12]], "mean", sum(diff) / len(diff))
    # per-head decay rate: g = -exp(A_log)*softplus(f(x)+dt_bias); average log-decay per token on the final chunk
    m0 = model.model.layers[10].self_attn.mem
    print("A_log exp range (layer 10):", float(m0.A_log.float().exp().min()), float(m0.A_log.float().exp().max()))
