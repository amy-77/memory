# memory

Selective writing and replay for fixed-size external memories attached to a frozen LLM.

- `docs/memory_selective_write_replay_20261122.md` — current design, experiment results and where things stand (Chinese).
- `lact_memory/` — memory branch on frozen Qwen3 (LaCT fast-weight core and matrix core), with
  - `lact_qwen/select.py`: token write weights (query-conditioned, SnapKV-style hard selection, input-level matching),
  - `lact_qwen/replay.py`: replay banks, including Robins-style pseudo-rehearsal with non-random probes (`pseudo`) and a score-kept KV bank (`scored_kv`),
  - `scripts/eval_select.py`, `scripts/diag_scores.py`, `scripts/diag_lme.py`, `tests/test_select.py`.
- `livemem_patch/` — patches and tools for [LiveMem](https://github.com/cafeii/LiveMem) at commit `71a6a8b` (LiveMem itself is not included):
  - `select_write.py` + `patch_plugin.py`: per-token write multipliers in the vLLM serving plugin,
  - `build_select_masks.py`: offline multipliers per prompt (oracle / BM25 / random / last-k sessions),
  - `patch_train.py`: training-time selective writing for the SFT stage,
  - `diag_qk_livemem.py`, `diag_attn_livemem.py`: can a score locate the evidence in LongMemEval?
  - run scripts used on our servers (paths inside them are machine-specific).

Status: research code in progress; results in `docs/` are preliminary.
