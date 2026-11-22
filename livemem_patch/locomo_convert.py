"""LoCoMo (LiveMem processed parquet) -> the dialogue jsonl used by scripts/eval_select.py.
One session = one "turn" (one memory write unit). Questions of category 5 (adversarial, no answer) are dropped.
fact_turn = the LAST session that holds evidence (so "beyond window" means every evidence session is beyond it).
A conversation's questions are split into groups of --group so one forward holds a few rows only."""
import json, sys
sys.path.insert(0, "tools/data_process")
from common import read_parquet
out, group = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
CAT = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
n_conv = n_q = 0
with open(out, "w") as fo:
    for r in read_parquet("dataset/processed/locomo/test.parquet"):
        docs = json.loads(r["memory_docs"]) if isinstance(r["memory_docs"], str) else r["memory_docs"]
        qa = json.loads(r["qa"]) if isinstance(r["qa"], str) else r["qa"]
        meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else r["meta"]
        qs = []
        for x in qa:
            c = int(x.get("category") or 0)
            if c not in CAT or not x.get("evidence_doc_idx") or not x.get("answer"):
                continue
            ans = str(x["answer"][0] if isinstance(x["answer"], list) else x["answer"]).strip()
            qs.append({"subject": "", "question": f"\n\nQuestion: {x['question'].strip()}\nAnswer:", "answer": ans,
                       "old_answer": None, "fact_turn": max(x["evidence_doc_idx"]), "evidence_turns": sorted(set(x["evidence_doc_idx"])), "category": CAT[c]})
        n_conv += 1; n_q += len(qs)
        for g in range(0, len(qs), group):
            fo.write(json.dumps({"id": f"{meta.get('orig_id', r['id'])}#g{g // group}", "turns": docs, "queries": qs[g : g + group]}) + "\n")
print(f"{n_conv} conversations, {n_q} questions (category 5 dropped) -> {out}")
