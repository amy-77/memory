"""Apply the selective-write hook to the LiveMem vLLM plugin (idempotent)."""
import os, shutil, sys
root = sys.argv[1]
pk = os.path.join(root, "models", "vllm", "memory_qwen3")
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "select_write.py"), os.path.join(pk, "select_write.py"))

def patch(fname, old, new, marker):
    p = os.path.join(pk, fname); s = open(p).read()
    if marker in s:
        print("already patched", fname); return
    if not os.path.exists(p + ".orig"):
        shutil.copy(p, p + ".orig")
    assert s.count(old) == 1, (fname, s.count(old))
    open(p, "w").write(s.replace(old, new)); print("patched", fname)

patch("gdn2_side.py",
"""    w = mem.w_proj(hidden).sigmoid()

    q = q.view(1, n, Hk, Dk)
""",
"""    w = mem.w_proj(hidden).sigmoid()

    # selective writing (select_write.py): per-token multiplier on decay / erase / write
    from . import select_write as _sw
    _wm = _sw.current(n, g.device) if is_prefill else None
    if _wm is not None:
        g = g * _wm.to(g.dtype)
        b = b * _wm.to(b.dtype)
        w = w * _wm.to(w.dtype)

    q = q.view(1, n, Hk, Dk)
""", "select_write as _sw")

patch("evict_registry.py",
"""def register_request(req_id: str, prompt_token_ids) -> None:
    if req_id in EVICT_REG or not prompt_token_ids:
        return
""",
"""def register_request(req_id: str, prompt_token_ids) -> None:
    from . import select_write as _sw
    _sw.register(req_id, prompt_token_ids)
    if req_id in EVICT_REG or not prompt_token_ids:
        return
""", "_sw.register")

patch("evict_registry.py",
"""def pop_request(req_id: str) -> None:
    EVICT_REG.pop(req_id, None)
""",
"""def pop_request(req_id: str) -> None:
    EVICT_REG.pop(req_id, None)
    from . import select_write as _sw
    _sw.pop(req_id)
""", "_sw.pop")

patch("flashinfer_evict.py",
"""        from .evict_registry import CFG, EVICT_REG, POLICY_PAGE_SIZE
        _EVICT["n_build"] += 1
""",
"""        from .evict_registry import CFG, EVICT_REG, POLICY_PAGE_SIZE
        from . import select_write as _sw
        _sw.build_step(common_attn_metadata, self.current_req_ids)
        _EVICT["n_build"] += 1
""", "_sw.build_step")
