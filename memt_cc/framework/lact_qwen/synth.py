"""Synthetic multi-turn memory task: write facts, correct some, then ask.

Every dialogue is a list of turns. Each turn is a block of distractor text that may
carry one fact sentence. Facts are (entity -> 5-digit value) pairs with random ids so
the base model cannot guess them. After the dialogue three kinds of questions are
asked:

retention   the fact was written once, never changed, and not in the last turns
correction  the fact was later overwritten; the answer is the newest value and the
            superseded value is kept as ``old_answer``
recent      the fact was written in the last ``recent_turns`` turns
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Iterator, Optional

TEMPLATES = [
    ("access code for locker {id}", "The access code for locker {id} is {val}.", "Update: the access code for locker {id} has been changed to {val}."),
    ("budget for order {id}", "The budget for order {id} is {val} dollars.", "Correction: the budget for order {id} is now {val} dollars."),
    ("tracking number of package {id}", "The tracking number of package {id} is {val}.", "Correction: package {id} was re-shipped, its tracking number is now {val}."),
    ("extension of agent {id}", "The phone extension of agent {id} is {val}.", "Update: agent {id} moved desks, the extension is now {val}."),
    ("room for meeting {id}", "Meeting {id} is booked in room {val}.", "Change: meeting {id} has been moved to room {val}."),
]

# Common words that Qwen encodes as one token after a space. With value_kind="word" a fact's
# value is a single token, the easiest thing a memory can be asked to return.
VALUE_WORDS = (
    "red blue green yellow black white orange purple brown pink gold silver "
    "north south east west spring summer winter autumn "
    "apple bread water stone river mountain forest garden window door table chair "
    "horse tiger eagle rabbit monkey dragon "
    "music paper glass metal cotton sugar coffee butter "
    "Monday Friday Sunday London Paris Tokyo Berlin Madrid"
).split()

# Templates whose value reads naturally as one word.
WORD_TEMPLATES = [
    ("code word for locker {id}", "The code word for locker {id} is {val}.", "Update: the code word for locker {id} has been changed to {val}."),
    ("password for account {id}", "The password for account {id} is {val}.", "Correction: the password for account {id} is now {val}."),
    ("codename of project {id}", "The codename of project {id} is {val}.", "Change: the codename of project {id} is now {val}."),
    ("label on package {id}", "The label on package {id} says {val}.", "Correction: package {id} was relabelled, its label now says {val}."),
    ("call sign of agent {id}", "The call sign of agent {id} is {val}.", "Update: agent {id} has a new call sign, it is now {val}."),
]

QUESTION = "\n\nQuestion: What is the {subject}?\nAnswer:"

REPEAT_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)


@dataclass
class SynthConfig:
    num_turns: int = 24
    num_facts: int = 12
    correction_frac: float = 0.3
    recent_turns: int = 2
    words_per_turn: int = 300
    queries_per_category: int = 4
    filler: str = "repeat"
    filler_file: Optional[str] = None
    value_kind: str = "digits"  # "digits": random 5-digit number, "word": one of VALUE_WORDS
    seed: int = 0


_FILLER_CACHE: dict[str, list[str]] = {}


def _filler_words(path: str) -> list[str]:
    if path not in _FILLER_CACHE:
        with open(path, encoding="utf-8") as f:
            _FILLER_CACHE[path] = f.read().split()
    return _FILLER_CACHE[path]


class Filler:
    def __init__(self, cfg: SynthConfig, rng: random.Random):
        self.rng = rng
        self.words: list[str]
        if cfg.filler == "repeat":
            self.words = REPEAT_FILLER.split()
            self.random_offset = False
        elif cfg.filler == "file":
            if not cfg.filler_file:
                raise ValueError("filler='file' needs filler_file")
            self.words = _filler_words(cfg.filler_file)
            if len(self.words) < 1000:
                raise ValueError("filler_file is too short")
            self.random_offset = True
        else:
            raise ValueError("filler must be 'repeat' or 'file'")

    def take(self, n: int) -> list[str]:
        start = self.rng.randrange(len(self.words)) if self.random_offset else 0
        return [self.words[(start + i) % len(self.words)] for i in range(n)]


def _new_id(rng: random.Random, used: set[str]) -> str:
    while True:
        ident = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randint(100, 999)}"
        if ident not in used:
            used.add(ident)
            return ident


def _new_value(rng: random.Random, avoid: set[str], kind: str = "digits") -> str:
    while True:
        val = rng.choice(VALUE_WORDS) if kind == "word" else str(rng.randint(10000, 99999))
        if val not in avoid:
            avoid.add(val)
            return val


def _embed(filler_words: list[str], sentence: str, rng: random.Random) -> str:
    boundaries = [0] + [i + 1 for i, w in enumerate(filler_words) if w.endswith((".", "!", "?"))]
    cut = rng.choice(boundaries)
    words = filler_words[:cut] + [sentence] + filler_words[cut:]
    return " ".join(words)


def generate_dialogue(cfg: SynthConfig, index: int) -> dict:
    rng = random.Random(cfg.seed * 1_000_003 + index)
    filler = Filler(cfg, rng)
    if cfg.num_turns <= cfg.recent_turns + 2:
        raise ValueError("num_turns must leave room before the recent turns")

    used_ids: set[str] = set()
    values: set[str] = set()
    templates = WORD_TEMPLATES if cfg.value_kind == "word" else TEMPLATES
    early_turns = list(range(cfg.num_turns - cfg.recent_turns))
    recent_turns = list(range(cfg.num_turns - cfg.recent_turns, cfg.num_turns))

    num_recent = min(cfg.queries_per_category, len(recent_turns))
    num_early = cfg.num_facts - num_recent
    num_corrected = max(1, round(num_early * cfg.correction_frac))
    if num_early - num_corrected < 1:
        raise ValueError("not enough early facts for both retention and correction")

    events: dict[int, tuple] = {}
    facts = []

    free_early = early_turns[:-1]
    rng.shuffle(free_early)
    write_turns = sorted(free_early[:num_early])
    if len(write_turns) < num_early:
        raise ValueError("num_turns too small for num_facts")
    for t in write_turns:
        tmpl = rng.choice(templates)
        fid = _new_id(rng, used_ids)
        val = _new_value(rng, values, cfg.value_kind)
        facts.append({"subject": tmpl[0].format(id=fid), "tmpl": tmpl, "id": fid, "history": [(t, val)]})
        events[t] = ("write", len(facts) - 1)

    corrected = rng.sample(range(len(facts)), num_corrected)
    taken = set(events)
    for fi in corrected:
        first = facts[fi]["history"][0][0]
        options = [t for t in early_turns if t > first and t not in taken]
        if not options:
            continue
        t = rng.choice(options)
        taken.add(t)
        val = _new_value(rng, values, cfg.value_kind)
        facts[fi]["history"].append((t, val))
        events[t] = ("correct", fi)

    for t in recent_turns[:num_recent]:
        tmpl = rng.choice(templates)
        fid = _new_id(rng, used_ids)
        val = _new_value(rng, values, cfg.value_kind)
        facts.append({"subject": tmpl[0].format(id=fid), "tmpl": tmpl, "id": fid, "history": [(t, val)]})
        events[t] = ("write", len(facts) - 1)

    turns = []
    for t in range(cfg.num_turns):
        words = filler.take(cfg.words_per_turn)
        if t in events:
            kind, fi = events[t]
            f = facts[fi]
            val = dict(f["history"])[t]
            sentence = (f["tmpl"][1] if kind == "write" else f["tmpl"][2]).format(id=f["id"], val=val)
            turns.append(_embed(words, sentence, rng))
        else:
            turns.append(" ".join(words))

    recent_set = set(recent_turns)
    pools = {"retention": [], "correction": [], "recent": []}
    for f in facts:
        hist = f["history"]
        last_turn, answer = hist[-1]
        q = {
            "subject": f["subject"],
            "question": QUESTION.format(subject=f["subject"]),
            "answer": answer,
            "old_answer": hist[-2][1] if len(hist) > 1 else None,
            "fact_turn": last_turn,
        }
        if len(hist) > 1:
            pools["correction"].append({**q, "category": "correction"})
        elif last_turn in recent_set:
            pools["recent"].append({**q, "category": "recent"})
        else:
            pools["retention"].append({**q, "category": "retention"})

    queries = []
    for cat, pool in pools.items():
        rng.shuffle(pool)
        queries.extend(pool[: cfg.queries_per_category])

    return {"id": f"synth-{cfg.seed}-{index}", "config": asdict(cfg), "turns": turns, "queries": queries}


def generate(cfg: SynthConfig, n: int) -> Iterator[dict]:
    for i in range(n):
        yield generate_dialogue(cfg, i)


def write_jsonl(path: str, rows) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


TURN_HEADER = "\n\n[Turn {i}]\n"


def turn_text(i: int, body: str) -> str:
    return TURN_HEADER.format(i=i + 1) + body
