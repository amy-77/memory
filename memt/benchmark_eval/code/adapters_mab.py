"""MemoryAgentBench (MAB) adapter for bench_memt.py, following MemT-1.0 appendix D.1 / D.3.1 / D.3.2.

Main set only (MemT D.1: "the main set is used, excluding the short-context ablation questions"):
  AR   ruler_qa1_197K, ruler_qa2_421K, eventqa_full (5 books), longmemeval_s* (5 histories)
  TTL  icl_banking77 / clinic150 / nlu / trec_coarse / trec_fine (one row each), recsys_redial_full
  LRU  infbench_sum_eng_shots2 (100 books, 1 question each), detective_qa (10 books)
  CR   factconsolidation_sh_262k, factconsolidation_mh_262k
  dropped: eventqa_65536 / eventqa_131072, factconsolidation 6k / 32k / 64k (MAB's context-length ablations)

One item per parquet row (MemT D.1: "source histories are isolated and written in complete chunk-N order without
source mixing"). As in MAB's long-context agent (agent.py _handle_long_context_agent), the context is cut into chunks
of about 4096 tokens at sentence boundaries, every chunk is wrapped in the source's memorize template, and the wrapped
chunks are joined by "\n": that text is the item's history. The full-context prompt is [system, user: history + "\n" +
query] (MAB: full_message = context + "\n" + message); the memory prompt is [system, user: query] (D.3.1: no rewording
needed for MAB). Templates: MAB utils/templates.py, long_context_agent queries verbatim; memorize templates with
{time_stamp} fixed to 1970-01-01 00:00:00 (D.3.1) and the repo's literal "\\n" after the time stamp written as a real
line break, as the report prints it.

Differences from MAB that cannot be avoided here (documented, small):
  - MAB counts chunk tokens with tiktoken (gpt-4o-mini) after nltk.sent_tokenize; neither is installed on our
    servers, so sentences are split at [.!?] + whitespace and a chunk holds at most 4096 * 4 characters. The chunk
    boundaries only decide where the memorize wrappers sit; every token of the context is written either way.
  - The InfBench "demo" field (2-shot) is loaded by MAB but never used in its prompts; it is ignored here as well.

Extra question fields (bench_memt.py ignores unknown fields): "plain_cue" (see below).
"""
import json
import os
import re

import pandas as pd

from adapters import HISTORY
from mab_official import BASE_TEMPLATES, SYSTEM_MESSAGE

TIME_STAMP = "1970-01-01 00:00:00"
CHUNK_CHARS = 4096 * 4
SPLITS = ["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution"]
CATEGORY = {"Accurate_Retrieval": "AR", "Test_Time_Learning": "TTL", "Long_Range_Understanding": "LRU",
            "Conflict_Resolution": "CR"}
KEEP = {"ruler_qa1_197K", "ruler_qa2_421K", "eventqa_full", "longmemeval_s*", "recsys_redial_full",
        "icl_banking77_5900shot_balance", "icl_clinic150_7050shot_balance", "icl_nlu_8296shot_balance",
        "icl_trec_coarse_6600shot_balance", "icl_trec_fine_6400shot_balance", "infbench_sum_eng_shots2",
        "detective_qa", "factconsolidation_sh_262k", "factconsolidation_mh_262k"}

# source family -> (MAB template key, max new tokens (D.3.2), plain-text stop strings, plain-text cue)
# plain_cue: what the plain-text prompt should end with. Queries that end with their own cue ("Answer:", "label:",
# "The event that happens next is:", "The recommendations are: \n") need none; render() in adapters.py appends
# "\nAnswer:" unless the text ends with "Answer:", so it should honour q.get("plain_cue") when the key is present.
FAMILY = {
    "ruler_qa":          ("ruler_qa", 50, ["\n"], ""),
    "eventqa":           ("eventqa", 40, ["\n"], ""),
    "longmemeval":       ("longmemeval", 50, ["\n"], ""),
    "icl":               ("in_context_learning", 20, ["\n"], ""),
    "recsys":            ("recsys_redial", 300, ["\n\n"], ""),            # the answer is a 20-line numbered list
    "infbench_sum":      ("infbench_sum", 1200, [], "\n"),                 # multi-paragraph summary: no stop string
    "detective_qa":      ("detective_qa", 2000, ["\n\n\n"], ""),
    "factconsolidation": ("factconsolidation", 10, ["\n"], ""),
}


def family_of(source):
    for prefix in ("ruler_qa", "eventqa", "longmemeval", "icl", "recsys", "infbench_sum", "detective_qa",
                   "factconsolidation"):
        if source.startswith(prefix):
            return prefix
    raise KeyError(source)


def memorize_template(template_key):
    t = BASE_TEMPLATES[template_key]["memorize"].replace("\\n", "\n")
    return t.replace("{time_stamp}", TIME_STAMP)


def query_template(template_key):
    return BASE_TEMPLATES[template_key]["query"]["long_context_agent"]


def sentence_chunks(text, max_chars=CHUNK_CHARS):
    """Greedy chunks of whole sentences, joined by " " like MAB's chunk_text_into_sentences."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    chunks, cur, size = [], [], 0
    for s in sentences:
        if cur and size + len(s) > max_chars:
            chunks.append(" ".join(cur))
            cur, size = [], 0
        cur.append(s)
        size += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _list(x):
    return [str(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def load_mab(root, tok=None):
    items = []
    for split in SPLITS:
        df = pd.read_parquet(os.path.join(root, f"MemoryAgentBench/data/{split}-00000-of-00001.parquet"))
        for row in range(len(df)):
            meta = df["metadata"].iloc[row]
            source = meta["source"]
            if source not in KEEP:
                continue
            fam = family_of(source)
            key, max_new, stop, cue = FAMILY[fam]
            wrap = memorize_template(key)
            history = "\n".join(wrap.replace("{context}", c) for c in sentence_chunks(df["context"].iloc[row]))
            questions = df["questions"].iloc[row]
            answers = df["answers"].iloc[row]
            pair_ids = meta.get("qa_pair_ids")
            qs = []
            for i, (question, answer) in enumerate(zip(questions, answers)):
                query = query_template(key).format(question=question)
                if fam == "longmemeval":
                    qid = str(meta["question_ids"][i])
                    extra = {"question_type": str(meta["question_types"][i]), "question_id": qid}
                else:
                    qid = f"{source}#{row}#{pair_ids[i] if pair_ids is not None else i}"
                    extra = {}
                if fam == "infbench_sum":
                    extra["keypoints"] = _list(meta["keypoints"])
                qs.append({
                    "qid": qid, "category": source, "question": str(question), "gold": _list(answer),
                    "max_new": max_new, "temperature": 0.0, "stop": stop, "plain_cue": cue,
                    "full": [{"role": "system", "content": SYSTEM_MESSAGE},
                             {"role": "user", "content": HISTORY + "\n" + query}],
                    "memory": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": query}],
                    "meta": {"source": source, "family": fam, "mab_category": CATEGORY[split], "row": row, **extra}})
            items.append({"id": f"{source}#{row}", "history": history, "questions": qs})
    return items


if __name__ == "__main__":
    import sys
    its = load_mab(sys.argv[1])
    print(json.dumps({it["id"]: len(it["questions"]) for it in its}, indent=0))
