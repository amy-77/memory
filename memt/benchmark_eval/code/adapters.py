"""Benchmark adapters: each turns a benchmark into histories with questions, following MemT-1.0 appendix D.3.1.

An item is {"id", "history" or "history_msgs", "questions"}:
  history       the history part of the full-context prompt, as text (mode memory writes exactly this text)
  history_msgs  instead of history, for benchmarks whose history is a list of chat messages (PersonaMem);
                rendered with the chat template (format chat) or as "Role: content" lines (format plain)
  questions     list of {"qid", "category", "question" (text shown to the judge), "gold", "max_new", "temperature",
                         "full": messages of the full-context prompt, with HISTORY where the history goes,
                         "memory": messages of the memory prompt (history removed, memory wording of D.3.1),
                         "prefix": True if the full-context prompt is the history followed by the memory prompt,
                         "stop": strings that end a plain-text answer, "meta": extra output fields}

Two prompt formats (bench_memt.py --format):
  chat   the report's format: chat template, thinking off
  plain  the same wording without the chat template, followed by "\nAnswer:"; the answer ends at a stop string.
         Our 4B model was trained on plain text only; with the chat template its memory readout breaks it.
All paths are relative to the data root (--data). Loaders take (root, tok).
"""
import ast
import csv
import json
import os

HISTORY = "<<HISTORY>>"
SHORT_STOP = ["\n", "Answer:", "Question:"]   # short answers end at the first newline or a repeated cue


def strip_lead(text):
    """Plain-text answers: drop leading blanks and a repeated "Answer:" cue before the stop strings are applied."""
    text = text.lstrip()
    while text.startswith("Answer:"):
        text = text[len("Answer:"):].lstrip()
    return text


def parse_list(text):
    """A list stored as JSON or as a Python literal (PersonaMem uses both)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def chat(tok, msgs, generation_prompt=True):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False)


def history_text(tok, item, fmt):
    if "history" in item:
        return item["history"]
    if fmt == "chat":
        return chat(tok, item["history_msgs"], generation_prompt=False)
    return "\n".join(m["content"] if m["content"].startswith(("User:", "Assistant:")) else
                     f"{m['role'].capitalize()}: {m['content']}" for m in item["history_msgs"])


def render(tok, q, history, fmt):
    """The prompt text: memory wording when history is None, else the full-context prompt with this history."""
    if history is None or q.get("prefix"):
        msgs = q["memory"]
    else:
        msgs = [{**m, "content": m["content"].replace(HISTORY, history)} for m in q["full"]]
    if fmt == "chat":
        text = chat(tok, msgs)
        return history + text if history is not None and q.get("prefix") else text
    text = "\n\n".join(m["content"] for m in msgs)
    if history is not None and q.get("prefix"):
        text = history + "\n\n" + text
    if "plain_cue" in q:                       # the query ends with its own cue (MemoryAgentBench: "label:", ...)
        return text + q["plain_cue"]
    return text if text.rstrip().endswith("Answer:") else text + "\nAnswer:"


# ------------------------------------------------------------------------------------------------ LoCoMo
def load_locomo(root, tok):
    """10 conversations, the 1540 questions outside the adversarial category (5), max 512 new tokens (D.3.2)."""
    data = json.load(open(os.path.join(root, "locomo/locomo10.json")))
    items = []
    for sample in data:
        conv = sample["conversation"]
        n_sessions = max(int(k.split("_")[1]) for k in conv if k.startswith("session_") and k.endswith("date_time"))
        blocks = []
        for k in range(1, n_sessions + 1):
            if not conv.get(f"session_{k}"):
                continue
            lines = [f"{m['speaker']}: {m['text']}" + (f" (description of attached image: {m['blip_caption']})"
                                                      if m.get("blip_caption") else "") for m in conv[f"session_{k}"]]
            blocks.append(f"### session_{k}\nDate: {conv[f'session_{k}_date_time']}\n" + "\n".join(lines))
        questions = []
        for n, qa in enumerate(sample["qa"]):
            if qa["category"] == 5:
                continue
            tail = f"Question: {qa['question']}\nAnswer briefly. Do not explain your reasoning."
            questions.append({
                "qid": f"{sample['sample_id']}#{n}", "category": str(qa["category"]), "question": qa["question"],
                "gold": str(qa["answer"]), "max_new": 512, "temperature": 0.0, "stop": SHORT_STOP,
                "full": [{"role": "user", "content": "Below is a conversation between two people. Answer the question "
                                                     f"using only this conversation.\n\n{HISTORY}\n\n{tail}"}],
                "memory": [{"role": "user", "content": "Answer the question using only the conversation stored in your "
                                                       f"memory.\n\n{tail}"}],
                "meta": {"evidence": qa.get("evidence", [])}})
        items.append({"id": sample["sample_id"], "history": "\n\n".join(blocks), "questions": questions})
    return items


# ------------------------------------------------------------------------------------------------ LongMemEval-S
def load_longmemeval_s(root, tok):
    """500 questions, each with its own history of ~50 sessions; the current date only in the query (D.1)."""
    data = json.load(open(os.path.join(root, "longmemeval-cleaned/longmemeval_s_cleaned.json")))
    items = []
    for r in data:
        blocks = []
        for n, (date, turns) in enumerate(zip(r["haystack_dates"], r["haystack_sessions"]), 1):
            body = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
            blocks.append(f"### Session {n}:\nSession Date: {date}\nSession Content:\n{body}")
        tail = f"Current Date: {r['question_date']}\nQuestion: {r['question']}\nAnswer:"
        q = {"qid": r["question_id"], "category": r["question_type"], "question": r["question"], "gold": str(r["answer"]),
             "max_new": 512, "temperature": 0.0, "stop": SHORT_STOP,
             "full": [{"role": "user", "content": "I will give you several history chats between you and a user. Please "
                                                  f"answer the question based on the relevant chat history.\n\n{HISTORY}\n\n{tail}"}],
             "memory": [{"role": "user", "content": "Answer the question based on the relevant chat history stored in "
                                                    f"your memory.\n\n{tail}"}],
             "meta": {"abstention": r["question_id"].endswith("_abs")}}
        items.append({"id": r["question_id"], "history": "History Chats:\n" + "\n\n".join(blocks), "questions": [q]})
    return items


# ------------------------------------------------------------------------------------------------ PersonaMem-32K
def load_personamem_32k(root, tok):
    """589 multiple-choice questions over 37 shared chat histories. Every (history, cut-off) pair is its own history;
    questions with the same cut-off share it (D.1). Roles and order kept; system messages after the first become user
    messages prefixed with [System message] (D.3.1). Max 2048 new tokens (D.3.2)."""
    contexts = {}
    for line in open(os.path.join(root, "PersonaMem/shared_contexts_32k.jsonl")):
        contexts.update(json.loads(line))
    groups = {}
    for r in csv.DictReader(open(os.path.join(root, "PersonaMem/questions_32k.csv"))):
        groups.setdefault((r["shared_context_id"], int(r["end_index_in_shared_context"])), []).append(r)
    items = []
    for (sid, end), rows in groups.items():
        msgs = []
        for n, m in enumerate(contexts[sid][:end]):
            if m["role"] == "system" and n > 0:
                m = {"role": "user", "content": "[System message] " + m["content"]}
            msgs.append({"role": m["role"], "content": m["content"]})
        questions = []
        for r in rows:
            options = "\n".join(parse_list(r["all_options"]))
            text = (f"{r['user_question_or_message']}\n\nFind the most appropriate model response and give your final "
                    f"answer (a), (b), (c), or (d) after the special token <final_answer>.\n\n{options}")
            questions.append({"qid": r["question_id"], "category": r["question_type"], "question": r["user_question_or_message"],
                              "gold": r["correct_answer"], "max_new": 2048, "temperature": 0.0, "prefix": True,
                              "stop": SHORT_STOP, "memory": [{"role": "user", "content": text}],
                              "meta": {"distance_to_ref_in_tokens": int(r["distance_to_ref_in_tokens"] or 0)}})
        items.append({"id": f"{sid[:12]}@{end}", "history_msgs": msgs, "questions": questions})
    return items


# ------------------------------------------------------------------------------------------------ BEAM-1M
BEAM_SYSTEM = ("You are a helpful assistant answering questions about a long-running conversation. Use only the "
               "conversation information available to you, respect later updates, and do not invent missing details.")
BEAM_MAX_NEW = {"abstention": 512, "information_extraction": 512, "knowledge_update": 512, "temporal_reasoning": 512,
                "contradiction_resolution": 1024, "instruction_following": 1024, "multi_session_reasoning": 1024,
                "preference_following": 1024, "event_ordering": 2048, "summarization": 4096}       # D.3.2
BEAM_SHORT = {"abstention", "information_extraction", "knowledge_update", "temporal_reasoning"}


def beam_stop(ability):
    """Plain-text stop strings: short abilities and event lists end at a blank line, long answers at two."""
    return (["\n\n", "Question:", "Answer:"] if ability in BEAM_SHORT or ability == "event_ordering"
            else ["\n\n\n", "Question:", "Answer:"])


def load_beam_1m(root, tok):
    """35 conversations of ~1M tokens (10 sessions each), 700 probing questions, 2 per ability per conversation.
    Each record of the history: "[{session} | {time anchor}]\\n{Role}: {content}" (D.3.1); answers are scored with the
    question's rubric (OmniMemEval BEAM judge)."""
    import ast
    import pandas as pd
    df = pd.read_parquet(os.path.join(root, "BEAM/data/1M-00000-of-00001.parquet"))
    items = []
    for _, row in df.iterrows():
        records, anchor = [], "January-01-2024"
        for s, session in enumerate(row["chat"], 1):
            for m in session:
                anchor = m.get("time_anchor") or anchor
                records.append(f"[{s} | {anchor}]\n{m['role'].capitalize()}: {m['content']}")
        pq = row["probing_questions"]
        pq = ast.literal_eval(pq) if isinstance(pq, str) else pq
        questions = []
        for ability, qs in pq.items():
            for n, q in enumerate(qs):
                gold = q.get("answer") or q.get("ideal_response") or q.get("ideal_answer") or q.get("ideal_summary") \
                    or q.get("expected_compliance", "")
                order = "Put one event on each line in chronological order.\n" if ability == "event_ordering" else ""
                tail = f"Only provide the answer without explaining your reasoning.\n{order}\nQuestion: {q['question']}\n\nAnswer:"
                stop = beam_stop(ability)
                rubric = list(q.get("rubric") or [])
                questions.append({
                    "qid": f"beam1m-{row['conversation_id']}-{ability}-{n}", "category": ability, "question": q["question"],
                    "gold": str(gold), "max_new": BEAM_MAX_NEW[ability], "temperature": 0.0, "stop": stop,
                    "full": [{"role": "system", "content": BEAM_SYSTEM}, {"role": "user", "content": f"{HISTORY}\n\n{tail}"}],
                    "memory": [{"role": "system", "content": BEAM_SYSTEM}, {"role": "user", "content": tail}],
                    "meta": {"rubric": [str(x) for x in rubric], "difficulty": q.get("difficulty", "")}})
        items.append({"id": f"beam1m-{row['conversation_id']}", "history": "\n\n".join(records), "questions": questions})
    return items


def load_mab(root, tok):
    """MemoryAgentBench main set: see adapters_mab.py."""
    import adapters_mab
    return adapters_mab.load_mab(root, tok)


BENCHMARKS = {"locomo": load_locomo, "longmemeval_s": load_longmemeval_s, "personamem_32k": load_personamem_32k,
              "beam_1m": load_beam_1m, "mab": load_mab}
