"""Can the memory branch tell, from its keys alone, that a later turn talks about the SAME entity as an earlier one?

For every entity of a dialogue, every turn that mentions its id (e.g. "N-942") is an occurrence. A full forward
(every turn written, one segment per turn) stores, in every layer, the memory keys of the id's tokens. An
occurrence is represented by the mean key over its id tokens (per head; cosine averaged over heads).

Per layer we report the cosine between occurrences of
  same      the same entity in two different turns (a fact and its correction)
  diff_tpl  two different entities stated with the same sentence template (the hard negatives)
  diff      any two different entities
and the AUROC of "same" against "diff_tpl" (0.5 = cannot tell, 1.0 = perfectly separable), plus the share of
corrections whose earlier occurrence is the most similar of all earlier occurrences (nearest = same entity).
The same is reported for the keys of the 4 tokens before the id (template words, often identical text in hard negatives).
"""
import argparse, json, os, sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_select import encode_with_offsets
from lact_qwen.loading import add_model_args, autocast, load_model, load_tokenizer
from lact_qwen.synth import TEMPLATES, read_jsonl


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = torch.tensor(pos + neg); lab = torch.tensor([1] * len(pos) + [0] * len(neg))
    order = s.argsort(); r = torch.empty(len(s)); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    return float((r[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--out", default=None)
    add_model_args(p)
    args = p.parse_args()
    tok = load_tokenizer(args)
    model, _ = load_model(args, tok)
    ctx = model.lact_ctx
    tpl_of = {}
    for subj, *_ in TEMPLATES:
        tpl_of[subj.split("{id}")[0].strip()] = subj
    stats = defaultdict(lambda: defaultdict(list))
    nearest = defaultdict(lambda: [0, 0])
    for di, d in enumerate(read_jsonl(args.data)[: args.limit]):
        enc, offsets, texts = encode_with_offsets(d, tok)
        ents = {}
        for q in d["queries"]:
            eid = q["subject"].split()[-1]
            ents[eid] = q["subject"].rsplit(" ", 1)[0]
        occ = []  # (entity, template, turn, id token positions, template token positions)
        for eid, tpl in ents.items():
            for t, text in enumerate(texts):
                pos = text.find(" " + eid)
                if pos < 0:
                    continue
                a, b = pos + 1, pos + 1 + len(eid)
                ids = [enc.turn_spans[t][0] + j for j, (x, y) in enumerate(offsets[t]) if x < b and y > a]
                tps = list(range(max(enc.turn_spans[t][0], ids[0] - 4), ids[0])) if ids else []  # the 4 tokens before the id
                if ids and tps:
                    occ.append((eid, tpl, t, ids, tps))
        if len(occ) < 2:
            continue
        allpos = sorted({x for o in occ for x in o[3] + o[4]})
        where = {x: i for i, x in enumerate(allpos)}
        hist = torch.tensor([enc.history_ids])
        ctx.set_segments(enc.turn_spans, [True] * len(enc.turn_spans))
        ctx.capture_key_pos, ctx.key_store = torch.tensor(allpos), {}
        with torch.no_grad(), autocast(args):
            model(input_ids=hist.to(args.device), logits_to_keep=1)
        store, ctx.capture_key_pos = ctx.key_store, None
        for layer, kk in store.items():  # kk [H, n_pos, d]
            kk = torch.nn.functional.normalize(kk, dim=-1)
            rep = lambda ps: torch.nn.functional.normalize(kk[:, [where[x] for x in ps]].mean(dim=1), dim=-1)  # [H, d]
            vid = [rep(o[3]) for o in occ]; vtp = [rep(o[4]) for o in occ]
            for i in range(len(occ)):
                for j in range(i):
                    kind = "same" if occ[i][0] == occ[j][0] else ("diff_tpl" if occ[i][1] == occ[j][1] else "diff")
                    stats[layer]["id_" + kind].append(float((vid[i] * vid[j]).sum(-1).mean()))
                    stats[layer]["tpl_" + kind].append(float((vtp[i] * vtp[j]).sum(-1).mean()))
                # a later occurrence of an entity: is its own earlier occurrence the most similar earlier one?
                earlier = [j for j in range(len(occ)) if occ[j][2] < occ[i][2]]
                if any(occ[j][0] == occ[i][0] for j in earlier):
                    best = max(earlier, key=lambda j: float((vid[i] * vid[j]).sum(-1).mean()))
                    nearest[layer][0] += occ[best][0] == occ[i][0]; nearest[layer][1] += 1
        print(f"[{di + 1}] occurrences {len(occ)}", flush=True)
    res = {}
    print(f"\n{'layer':>5s} {'id same':>8s} {'id diff_tpl':>11s} {'id diff':>8s} {'AUROC id':>9s} {'nearest=same':>13s} {'tpl same':>9s} {'tpl diff_tpl':>12s} {'AUROC tpl':>9s}")
    for layer in sorted(stats):
        s = stats[layer]; m = lambda k: sum(s[k]) / max(len(s[k]), 1)
        a_id, a_tp = auroc(s["id_same"], s["id_diff_tpl"]), auroc(s["tpl_same"], s["tpl_diff_tpl"])
        nn_ = nearest[layer][0] / max(nearest[layer][1], 1)
        res[layer] = {"id_same": m("id_same"), "id_diff_tpl": m("id_diff_tpl"), "id_diff": m("id_diff"), "auroc_id": a_id,
                      "nearest_same": nn_, "n_same": len(s["id_same"]), "n_diff_tpl": len(s["id_diff_tpl"]), "auroc_tpl": a_tp}
        print(f"{layer:5d} {m('id_same'):8.3f} {m('id_diff_tpl'):11.3f} {m('id_diff'):8.3f} {a_id:9.3f} {nn_:13.3f} {m('tpl_same'):9.3f} {m('tpl_diff_tpl'):12.3f} {a_tp:9.3f}")
    n = next(iter(stats.values()))
    print(f"pairs: same {len(n['id_same'])}, diff_tpl {len(n['id_diff_tpl'])}, diff {len(n['id_diff'])}; corrections checked {next(iter(nearest.values()))[1]}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    print("KEY_IDENTITY_DONE")


if __name__ == "__main__":
    main()
