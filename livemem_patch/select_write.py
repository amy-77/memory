"""Selective writing for the GDN2 memory at serve time (not part of the LiveMem release).

A request may come with a per-token write multiplier in [0, 1] for its prompt. In every layer the
multiplier scales the three gates of the token's memory update (decay, erase, write) exactly like the
training module's `write_mask`: 0 leaves the state untouched by that token (it neither writes nor makes
the state forget), 1 is the ordinary update. Attention and everything else are unchanged, and with no
multipliers the server behaves exactly as released.

The multipliers are prepared offline (tools/build_select_masks.py) and found by the SHA-1 of the prompt's
token ids: $MEM_SELECT_DIR/<sha1>.npy, float16 [prompt_len]. A prompt without a file is served unchanged;
hits and misses are appended to $MEM_SELECT_DIR/_server.log so a silent fallback cannot go unnoticed.
Tokens generated after the prompt always use 1.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

SELECT_DIR = os.environ.get("MEM_SELECT_DIR") or None
MASKS: dict[str, torch.Tensor | None] = {}
STATE = {"w": None, "hit": 0, "miss": 0, "applied_tokens": 0, "steps": 0}


def enabled() -> bool:
    return SELECT_DIR is not None


def key_of(token_ids) -> str:
    return hashlib.sha1(np.asarray(list(token_ids), dtype=np.int32).tobytes()).hexdigest()


def _log(msg: str) -> None:
    try:
        with open(os.path.join(SELECT_DIR, "_server.log"), "a") as f:
            f.write(f"pid={os.getpid()} {msg}\n")
    except OSError:
        pass


def register(req_id: str, prompt_token_ids) -> None:
    if not enabled() or req_id in MASKS or not prompt_token_ids:
        return
    ids = list(prompt_token_ids)
    path = os.path.join(SELECT_DIR, key_of(ids) + ".npy")
    if os.path.exists(path):
        m = torch.from_numpy(np.load(path).astype(np.float32))
        if m.numel() != len(ids):
            _log(f"BAD_LENGTH {req_id} mask={m.numel()} prompt={len(ids)}")
            m = None
        else:
            STATE["hit"] += 1
            _log(f"hit {req_id} len={len(ids)} mean={float(m.mean()):.4f} zeros={int((m == 0).sum())}")
    else:
        m = None
        STATE["miss"] += 1
        _log(f"MISS {req_id} len={len(ids)}")
    MASKS[req_id] = m


def pop(req_id: str) -> None:
    MASKS.pop(req_id, None)


def build_step(cm, req_ids) -> None:
    """Called once per model step with the batch layout: row r holds the tokens at positions
    [seq_len[r] - query_len[r], seq_len[r]) of request req_ids[r]."""
    STATE["w"] = None
    if not enabled() or req_ids is None:
        return
    qsl = cm.query_start_loc_cpu.tolist()
    seq = cm.seq_lens.tolist() if hasattr(cm.seq_lens, "tolist") else list(cm.seq_lens)
    w = None
    for r in range(min(cm.num_reqs, len(req_ids))):
        m = MASKS.get(req_ids[r])
        if m is None:
            continue
        end = int(seq[r])
        start = end - (qsl[r + 1] - qsl[r])
        lo, hi = min(start, m.numel()), min(end, m.numel())
        if hi <= lo:
            continue
        if w is None:
            w = torch.ones(qsl[-1], dtype=torch.float32)
        w[qsl[r] : qsl[r] + (hi - lo)] = m[lo:hi]
        STATE["applied_tokens"] += hi - lo
    STATE["w"] = w
    STATE["steps"] += 1


def current(n: int, device) -> torch.Tensor | None:
    """Multipliers [n, 1] for the step being computed, or None for the ordinary update."""
    w = STATE["w"]
    if w is None:
        return None
    if w.shape[0] != n:
        # must never happen silently: a step whose multipliers are dropped is written like the baseline
        STATE["mismatch"] = STATE.get("mismatch", 0) + 1
        if STATE["mismatch"] <= 5:
            _log(f"SHAPE_MISMATCH step_tokens={n} multipliers={w.shape[0]}")
        return None
    STATE["used"] = STATE.get("used", 0) + 1
    if STATE["used"] % 3600 == 1:
        _log(f"applied calls={STATE['used']} mismatched={STATE.get('mismatch', 0)} steps={STATE['steps']} "
             f"tokens_with_multiplier={STATE['applied_tokens']} zeros_in_step={int((w == 0).sum())}/{n}")
    return w.to(device)[:, None]
