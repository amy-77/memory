# Selective writing on the LaCT memory: how to reproduce and how to run it on another checkpoint

Status 2026-11-22. Everything here is inference-only (no retraining of the memory branch).

## What was measured

| item | value |
|---|---|
| base model | Qwen3-0.6B, frozen, attention window 2048 tokens, 4 sink tokens |
| memory | LaCT core (SwiGLU fast weights, 4 heads x 512, Muon, one write per dialogue turn), all 28 layers |
| memory training | 1200 steps on synthetic 16-turn dialogues (`v8f_train`, 10 000 dialogues, each seen once), loss on the answers |
| test data | synthetic dialogues from `scripts/make_synth.py`: 32 and 64 turns, 8 facts per dialogue, ~200 words of filler per turn, 4 questions per category; first 40 dialogues = 160 retention questions |
| metric | exact match of the answer for facts stated outside the attention window (`retention/beyond_window`) |

The task is synthetic (key = value facts hidden in filler text). It shows the mechanism; it is not a benchmark result.

Only the best-scored K turns are written; the other turns are skipped entirely.

| policy | 32 turns, K=4 | 32 turns, K=8 | 64 turns, K=4 | 64 turns, K=8 |
|---|---|---|---|---|
| write every turn | 0.415 | | 0.218 | |
| random turns | 0.126 | 0.170 | 0.041 | 0.082 |
| memory q.k, layer 19 | 0.719 | 0.748 | 0.612 | 0.510 |
| attention mass (frozen base model) | 0.689 | 0.696 | 0.633 | 0.531 |
| oracle (the turn that states the fact) | 0.763 | 0.741 | 0.728 | 0.680 |

## The two scores

* `turn_attn`: the question is placed after each turn and the frozen base model is run on `turn + question`.
  Score of the turn = attention MASS the question tokens put on the turn's tokens (mean over heads and question
  tokens, summed over the turn's tokens, summed over the upper two thirds of the layers). Do not use the best few
  tokens: per-token attention is normalised within each forward and ranks turns at chance.
* `turn_qk`: a first pass writes every turn; the read vector at the LAST question token (the one the model uses
  for the first answer token) is compared (cosine, mean over heads) with every history token's key at ONE layer;
  score of a turn = mean of its 5 best tokens. On the 0.6B model only layer 19 localises the fact
  (top-1 0.83 / 0.76 on 16 / 32 turns; every other layer is near chance; averaging the layers gives 0.19).
  The layer must be found per model with `scripts/audit_scores.py`.

## Commands

```bash
# test data (seed 7 = the sets used above)
for t in 16 32 64; do python scripts/make_synth.py --out data/rc_t${t}_f8.jsonl --n 100 --num-turns $t --num-facts 8 \
    --words-per-turn 200 --queries-per-category 4 --seed 7; done

M="--model Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"   # your base model / checkpoint

# 1) which layer localises the fact (prints top-1 per score; per-layer numbers are in the json)
python scripts/audit_scores.py $M --data data/rc_t16_f8.jsonl --limit 12 --out out/audit_t16.json

# 2) selective writing
python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 40 --select none
python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 40 --select turn_attn   --turn-k 4
python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 40 --select turn_qk     --turn-k 4 --turn-layer 19
python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 40 --select turn_oracle --turn-k 4
python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 40 --select turn_random --turn-k 4
```
`turnsel.sh` runs the whole grid. Each run prints "fact turn written in X/Y rows" (how often the selection kept
the right turn) next to the accuracy, so the quality of the score and the use the memory makes of it are separate.

## Running it on another LaCT checkpoint (e.g. 760M / 4B branches)

1. `--model`, `--window`, `--sinks` and the `--mem-*` layout must match the checkpoint (the layout is read from
   `memory_config.json` in the checkpoint directory when present).
2. The test asks for an exact answer after a dialogue. A branch that was only pre-trained on text continuation
   has never seen this question format; check `--select none` against `--no-write` (`scripts/eval_synth.py`)
   first. If writing does not beat not writing, selection has nothing to improve; compare answer log-probabilities
   (`logprob` column) instead of exact match, or fine-tune on `v8f_train` first (`run_dialog.sh`).
3. `turn_attn` needs no change. For `turn_qk` run the audit and pass the best layer as `--turn-layer`.
4. The write unit is one dialogue turn (one segment per turn, `--max-segment 4096`). With a fixed chunk size
   instead of turns, selection would have to be per chunk; this is not implemented.

## Known gaps

* The positive control of the audit (score = tokens equal to the answer) reaches only 0.54-0.57 top-1 instead
  of ~1.0; not explained yet. Relative numbers are more reliable than absolute ones.
* `turn_qk` needs two passes over the history. The question is assumed to be known before writing.
* No replay is combined with turn selection yet (writing 8 turns is worse than writing 4: interference).
