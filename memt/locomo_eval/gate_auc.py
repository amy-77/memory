"""Can the MemT memory's own gates tell evidence tokens from the rest, compared with the question-attention probe?

For a sample of LoCoMo questions (evidence beyond the 2048 window), the history is streamed exactly as in row C
(every earlier chunk written, chunk-local attention). While each chunk is written, the per-token gates of every
layer are recorded, averaged over heads and channels:
    write  = sigmoid(write_proj(x))            (omega)
    erase  = sqrt(sigmoid(erase_proj(x)))      (sqrt(beta), the factor on the key)
    keep   = -decay  (so that larger = remembers more; decay = softplus(...) * exp(scale), capped at 0.03)
    step   = erase * write (how strongly the token's value is put in)
The probe is the question -> history attention of the frozen Qwen3-4B (full attention, summed over layers, heads
and question tokens). Label: token inside an annotated evidence session. AUC per question (Mann-Whitney), over
the tokens of the earlier chunks only; reported as the mean over questions, for every layer and the layer mean.
"""
import argparse, json, math, sys, time
import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, "/data/qyl/memt_locomo_20261122")
import eval_locomo_memt as E   # noqa: E402  (installs the attention capture hook; CAP off by default)
from lact_qwen.loading import add_model_args, load_model, load_teacher, autocast  # noqa: E402
from lact_qwen import memory_branch as MB  # noqa: E402

REC = {"on": False, "gates": {}}
_orig_write = MB.LaCTMemoryBranch._matrix_write


def _write(self, state, x):
    if REC["on"]:
        h, d = self.cfg.num_heads, self.cfg.head_dim
        with torch.no_grad():
            el = self.erase_proj(x).float()
            er = torch.exp(0.5 * F.logsigmoid(el)) if self.cfg.stable_matrix_gates else torch.sigmoid(el).sqrt()
            wr = torch.sigmoid(self.write_proj(x).float())
            dc = F.softplus(self.forget_up(self.forget_down(x)).float())
            dc = (dc.view(*dc.shape[:2], h, d) * self.forget_scale.float().exp()[:, None]).clamp(max=0.03)
            g = REC["gates"].setdefault(self.layer_idx, {"write": [], "erase": [], "keep": [], "step": []})
            g["write"].append(wr.mean(-1)[0].cpu()); g["erase"].append(er.mean(-1)[0].cpu())
            g["keep"].append((-dc).mean((-1, -2))[0].cpu()); g["step"].append((er * wr).mean(-1)[0].cpu())
    return _orig_write(self, state, x)


MB.LaCTMemoryBranch._matrix_write = _write


def auc(scores, labels):
    pos = labels.sum().item(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    r = torch.empty(len(scores)); r[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    return float((r[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/locomo2_g4.jsonl")
    p.add_argument("--every", type=int, default=8, help="take every N-th beyond-window question")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--out", required=True)
    add_model_args(p)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model, cfg = load_model(args)
    scorer = load_teacher(args)
    dev = next(model.parameters()).device
    C = 2048
    rows = [json.loads(l) for l in open(args.data)]
    res, n_seen, t0 = [], 0, time.time()
    for ri, dlg in enumerate(rows):
        hist, spans = E.encode(dlg, tok)
        for qi, q in enumerate(dlg["queries"]):
            ft = q["fact_turn"]
            if len(hist) - spans[ft][1] < 2048:
                continue
            n_seen += 1
            if (n_seen - 1) % args.every or ((n_seen - 1) // args.every) % args.nshards != args.shard:
                continue
            qq, aa = E.qa_ids(q, tok)
            P = len(hist) + len(qq)
            bounds = [(max(0, e - C), e) for e in range(P + 64, 0, -C)][::-1]
            fs = bounds[-1][0]
            if fs == 0:
                continue
            lab = torch.zeros(fs, dtype=torch.bool)
            for t in (q.get("evidence_turns") or [ft]):
                a_, b_ = spans[t]; lab[a_:min(b_, fs)] = True
            if not lab.any():
                continue
            with torch.no_grad(), autocast(args):
                REC.update(on=True, gates={})
                E.begin_stream(model, C)
                for s, e in bounds[:-1]:
                    E.forward_chunk(model, torch.tensor([hist[s:e]], device=dev), s, update=True)
                E.end_stream(model)
                REC["on"] = False
                E.CAP.update(on=True, nq=len(qq), mass=None, layers=0)
                scorer(input_ids=torch.tensor([hist + qq], device=dev), logits_to_keep=1)
                E.CAP["on"] = False
                probe = E.CAP["mass"][:fs].float().cpu(); probe[:4] = 0
            rec = {"row": ri, "query_index": qi, "category": q["category"], "probe": auc(probe, lab), "layers": {}}
            for l, g in sorted(REC["gates"].items()):
                rec["layers"][l] = {k: auc(torch.cat(v)[:fs], lab) for k, v in g.items()}
            res.append(rec)
        print(f"row {ri+1}/{len(rows)} questions {len(res)} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(args.out, "w"))
    print("DONE", len(res), flush=True)


if __name__ == "__main__":
    main()
