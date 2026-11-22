"""Turn a synthetic dialogue into token ids, write segments, and scored query rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .synth import turn_text


@dataclass
class EncodedDialogue:
    history_ids: list[int]
    turn_spans: list[tuple[int, int]]


def encode_history(dialogue: dict, tokenizer, max_turn_tokens: Optional[int] = None) -> EncodedDialogue:
    ids: list[int] = []
    spans = []
    for i, body in enumerate(dialogue["turns"]):
        t = tokenizer(turn_text(i, body), add_special_tokens=False).input_ids
        if max_turn_tokens:
            t = t[:max_turn_tokens]
        spans.append((len(ids), len(ids) + len(t)))
        ids.extend(t)
    return EncodedDialogue(ids, spans)


def split_long_spans(spans: list[tuple[int, int]], max_len: int) -> list[tuple[int, int]]:
    """Cut turns longer than max_len into consecutive pieces so one write never exceeds it."""
    out = []
    for s, e in spans:
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return out


def merge_spans(spans: list[tuple[int, int]], min_len: int) -> list[tuple[int, int]]:
    """Join consecutive turns into write units of at least min_len tokens (0 keeps one write per turn).

    With Muon every write has the same size whatever its token count, so many short writes
    overwrite earlier ones faster than a few long ones. The last unit may be shorter.
    """
    if not min_len:
        return list(spans)
    out, start = [], None
    for s, e in spans:
        if start is None:
            start = s
        if e - start >= min_len:
            out.append((start, e))
            start = None
    if start is not None:
        out.append((start, spans[-1][1]))
    return out


@dataclass
class QueryBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    segments: list[tuple[int, int]]
    update_flags: list[bool]
    rows: list[dict]


def build_query_batch(
    enc: EncodedDialogue, queries: list[dict], tokenizer, pad_id: int, max_segment: int, min_write: int = 0
) -> QueryBatch:
    """One row per (query, candidate answer). Corrections also get a row for the old value."""
    hist = enc.history_ids
    n_hist = len(hist)
    rows = []
    for qi, q in enumerate(queries):
        q_ids = tokenizer(q["question"], add_special_tokens=False).input_ids
        candidates = [("answer", q["answer"])]
        if q.get("old_answer"):
            candidates.append(("old_answer", q["old_answer"]))
        for which, ans in candidates:
            a_ids = tokenizer(" " + ans, add_special_tokens=False).input_ids
            # Qwen splits " 71281" into [" ", "7", ...]; a bare-space token is format, not content.
            lead = 0
            while lead < len(a_ids) - 1 and not tokenizer.decode(a_ids[lead : lead + 1]).strip():
                lead += 1
            rows.append({"query_index": qi, "which": which, "q_ids": q_ids + a_ids[:lead], "a_ids": a_ids[lead:]})

    tail = max(len(r["q_ids"]) + len(r["a_ids"]) for r in rows)
    total = n_hist + tail
    input_ids = torch.full((len(rows), total), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), total), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        seq = hist + r["q_ids"] + r["a_ids"]
        input_ids[i, : len(seq)] = torch.tensor(seq)
        a0 = n_hist + len(r["q_ids"])
        labels[i, a0 : a0 + len(r["a_ids"])] = torch.tensor(r["a_ids"])
        r["answer_span"] = (a0, a0 + len(r["a_ids"]))

    segments = split_long_spans(merge_spans(enc.turn_spans, min_write), max_segment) + [(n_hist, total)]
    flags = [True] * (len(segments) - 1) + [False]
    return QueryBatch(input_ids, labels, segments, flags, rows)


@torch.no_grad()
def score_rows(logits: torch.Tensor, batch: QueryBatch, offset: int = 0) -> list[dict]:
    """Exact match under teacher forcing and summed log-prob of each candidate answer.

    ``logits[:, j]`` holds the prediction made at sequence position ``offset + j``.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, r in enumerate(batch.rows):
        a0, a1 = r["answer_span"]
        target = batch.input_ids[i, a0:a1].to(logits.device)
        pred_logp = logp[i, a0 - 1 - offset : a1 - 1 - offset]
        greedy = pred_logp.argmax(dim=-1)
        out.append(
            {
                "query_index": r["query_index"],
                "which": r["which"],
                "exact": bool((greedy == target).all()),
                "logprob": float(pred_logp.gather(-1, target[:, None]).sum()),
            }
        )
    return out
