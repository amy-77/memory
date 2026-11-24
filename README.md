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

## Update 2026-11-24

- `memt/benchmark_eval/`: the 4B MemT-style model, base Qwen3-4B and the official MemT-1.0-Nano on the five memory benchmarks of the MemT-1.0 report's Table 2 (LoCoMo, LongMemEval-S, PersonaMem-32K, MemoryAgentBench, BEAM-1M): results, protocol, code-vs-official comparison and the evaluation code (Chinese).

## Update 2026-11-22

- `docs/LACT_SELECTIVE_WRITE.md`: turn-level selective writing on the LaCT memory (results, the two scores, commands, how to run on another checkpoint).
- `lact_memory/scripts/audit_scores.py`, `--select turn_attn|turn_qk|turn_oracle|turn_random` in `lact_memory/scripts/eval_select.py`.
- `livemem_patch/`: `score_attn.py` (attention relevance of every session), `attn:K:A` mask policy (`patch_masks.py`), reference runs (`eval_trunc.sh`), evidence-position breakdown (`bucket.py`), needle test (`needle_*.py/sh`).
