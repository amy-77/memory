"""Training-time selective writing for LiveMem SFT (idempotent patch of three files).

1. tools/data_process/tokenize_train.py: keep the evidence labels. New column `evidence_chunk_idx`: for
   single-question samples whose memory is chunked one document per chunk, the chunks that hold the
   supporting documents (empty when unknown).
2. train/sft/data.py: MixedArrowDataset returns a per-token `write_mask`. With probability `sel_prob`, a
   sample that has evidence labels writes only its evidence chunks plus a random share (uniform in
   [0, sel_distract]) of the other memory chunks; every other token of the memory gets 0. System prompt,
   question and answer always get 1. Samples without labels, and the remaining samples, get all ones.
3. train/sft/train_sft.py: read data.sel_prob / data.sel_distract from the config.
The model already accepts `write_mask` (it scales decay / erase / write of the GDN2 update).
"""
import os, shutil, sys
root = sys.argv[1]

def patch(rel, pairs, marker):
    p = os.path.join(root, rel); s = open(p).read()
    if marker in s:
        print("already patched", rel); return
    if not os.path.exists(p + ".orig"):
        shutil.copy(p, p + ".orig")
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:60], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched", rel)

patch("tools/data_process/tokenize_train.py", [
("""    memory_docs, user_text, assistant_text, cfg = task
    return build_sample(memory_docs, user_text, assistant_text, cfg)
""", """    memory_docs, user_text, assistant_text, cfg, evidence = task
    return build_sample(memory_docs, user_text, assistant_text, cfg, evidence)
"""),
("""def build_sample(memory_docs, user_text, assistant_text, cfg):
""", """def build_sample(memory_docs, user_text, assistant_text, cfg, evidence=None):
"""),
("""    chunks, losses = [], []

    if mode == "unit":
        for i in range(0, len(memory_docs), size):
""", """    chunks, losses = [], []
    doc_chunks = {}  # memory document index -> indices of the chunks that hold it (unit mode, size 1)

    if mode == "unit":
        for i in range(0, len(memory_docs), size):
            _c0 = len(chunks)
"""),
("""            else:
                chunks.append(ids)
                losses.append([0] * len(ids))
        chunks.append(qa_ids)                                   # Keep the query/assistant QA tail in its own chunk.
""", """            else:
                chunks.append(ids)
                losses.append([0] * len(ids))
            if size == 1:
                doc_chunks[i] = list(range(_c0, len(chunks)))
        chunks.append(qa_ids)                                   # Keep the query/assistant QA tail in its own chunk.
"""),
("""        "qa_chunk_idx": qa_chunk_idx,
        "chunk_limit": cfg["chunk_limit"],
""", """        "qa_chunk_idx": qa_chunk_idx,
        "evidence_chunk_idx": sorted({c for d in (evidence or []) for c in doc_chunks.get(int(d), [])}),
        "chunk_limit": cfg["chunk_limit"],
"""),
("""_INT32 = {"input_ids", "qa_chunk_idx"}
""", """_INT32 = {"input_ids", "qa_chunk_idx", "evidence_chunk_idx"}
"""),
("""            u, a = make_text(name, kind, row, reason)
            yield (row["memory_docs"], u, a, cfg)
""", """            u, a = make_text(name, kind, row, reason)
            ev = None
            if kind == "single":
                qa = row["qa"]
                qa = json.loads(qa) if isinstance(qa, str) else qa
                if len(qa) == 1:
                    ev = list(qa[0].get("evidence_doc_idx") or [])
            yield (row["memory_docs"], u, a, cfg, ev)
"""),
], "evidence_chunk_idx")

patch("train/sft/data.py", [
("""    def __init__(self, table, n_sink: int = 0, system_ids: list[int] | None = None,
                 w0_prob: float = 0.0, w_seed: int = 0) -> None:
        self.t = table
""", """    def __init__(self, table, n_sink: int = 0, system_ids: list[int] | None = None,
                 w0_prob: float = 0.0, w_seed: int = 0,
                 sel_prob: float = -1.0, sel_distract: float = 0.3) -> None:
        self.t = table
        # Selective writing (sel_prob >= 0 turns the write_mask field on; see _get_one).
        self.sel_prob, self.sel_distract = sel_prob, sel_distract
        self.ev = table.column("evidence_chunk_idx") if "evidence_chunk_idx" in table.column_names else None
"""),
("""        chunk_id, evict_step = per_token_eviction(spans, chunk_limit, token_limit, self.n_sink)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "chunk_id": chunk_id,
            "evict_step": evict_step,
        }
""", """        chunk_id, evict_step = per_token_eviction(spans, chunk_limit, token_limit, self.n_sink)
        out = {
            "input_ids": input_ids,
            "labels": labels,
            "chunk_id": chunk_id,
            "evict_step": evict_step,
        }
        if self.sel_prob >= 0:
            # 1 = ordinary memory update; 0 = this token neither writes to nor decays the state.
            wm = torch.ones(input_ids.numel(), dtype=torch.float32)
            shift = 1 if self.system_ids is not None else 0  # the system prompt was inserted as chunk 0
            ev = set((self.ev[i].as_py() or [])) if self.ev is not None else set()
            rng = random.Random(f"{self.w_seed}:sel:{i}")
            if ev and rng.random() < self.sel_prob:
                qa = set(self.qa[i].as_py())
                p_keep = rng.uniform(0.0, self.sel_distract)
                for c in range(len(spans) - shift):  # c indexes the stored chunks (before the system shift)
                    if c in ev or c in qa or rng.random() < p_keep:
                        continue
                    s, e = spans[c + shift]
                    wm[s:e] = 0.0
            out["write_mask"] = wm
        return out
"""),
], "sel_prob")

patch("train/sft/train_sft.py", [
("""                                 w0_prob=dcfg.get("w0_prob", 0.0), w_seed=dcfg.get("seed", 0))
""", """                                 w0_prob=dcfg.get("w0_prob", 0.0), w_seed=dcfg.get("seed", 0),
                                 sel_prob=dcfg.get("sel_prob", -1.0), sel_distract=dcfg.get("sel_distract", 0.3))
"""),
], "sel_distract")
