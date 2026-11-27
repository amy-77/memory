# Selective Write Experiments: Plan and Code Changes

2026-11-28

## Purpose and scope

This plan tests whether writing only important tokens into the memory state improves recall of evidence that has left the attention window. It lists every code location I expect to change, so they can be checked against the code on the H200 machine.

- **Target checkpoint:** the fully trained checkpoint on the H200 machine. The partially trained checkpoint we used before is not a fair test.
- **Code references:** line numbers refer to the official LiveMem repository (commit 71a6a8b), because that is the code I could read. The section "Porting to the H200 checkpoint" says what to look for in other code.
- **Nothing here has been run on a real model.** The only evidence so far is two toy simulations with random vectors.
- **Order of work:** Experiment 1 first. It needs about 150 lines of new code and no training. Experiments 2 to 5 only make sense if Experiment 1 shows a gain.

## Background

Each memory head keeps a 128 x 128 matrix S and updates it once per token in three steps.

```
S   <- Diag(exp(g_t)) S                  forget: g_t <= 0
rho  = w_t * v_t - S^T (b_t * k_t)       erase gate b_t, write gate w_t
S   <- S + k_t rho^T                     write
r_t  = S^T q_t                           read
```

**Skipping a token means setting all three gates to zero.** Then the state does not change at that token. Setting only the write gate to zero is wrong: the token would still erase the old value at its key and still decay the rest. The existing `write_mask` already multiplies g, b and w by the same mask (`models/memory_gdn2.py:128`).

**Capacity.** A linear map from 128 dimensions is fixed by its outputs on 128 independent keys, so one head stores at most 128 key-value pairs exactly. Past that point recall degrades gradually. A 115K-token history is far past it, so the question is how the limited capacity is shared out.

**Toy simulation.** 2,048 random pairs pass through one 128 x 128 memory. 64 of them are evidence, placed in the first half. The score is the cosine between the read-out and the true value of the evidence pairs.

| Write rule | How capacity is shared | Evidence cosine |
| --- | --- | --- |
| Delta rule, write everything (step 1.0 / 0.3 / 0.1) | Favors recent tokens | 0.00 / 0.03 / 0.13 |
| Least squares, write everything | Equal share for all tokens | 0.24 |
| Delta rule, write evidence only | By importance | 0.84 |
| Delta rule, evidence only, skipped tokens still decay (0.999 per token) | By importance | 0.76 |
| Weighted least squares (evidence 1, others 0.01) | By importance, soft | 0.98 |
| Least squares, write evidence only | By importance | 1.00 |

Two things follow, both still to be confirmed on a real model. How capacity is shared matters much more than the update rule. And one delta step per token leaves capacity unused even after selection (0.84 against 1.00).

## Experiment 1: oracle write diagnostic

This experiment measures the upper bound of selective writing: how much better the model answers when only the evidence sessions are written into memory. It uses labels, so it is a diagnostic and not a method.

**Data.** LongMemEval-s. Each question has about 50 sessions (about 115K tokens), and the evidence sessions are labeled (`answer_session_ids`). LiveMem's preprocessing already stores their positions as `evidence_doc_idx` (`tools/data_process/preprocess/longmemeval.py:42`).

**Setup shared by all arms.**

- Window 32K tokens, chunk size 1,024, system prompt kept as the sink. This matches the paper's evaluation.
- The system prompt, the question and the answer always have mask 1.
- Only questions whose evidence sessions are fully outside the window when the question arrives are counted. Abstention questions (`_abs`) are dropped.
- Metric: mean loss on the gold answer tokens from one teacher-forced forward pass. No generation and no judge model.

| Arm | write_mask on the history | What it answers |
| --- | --- | --- |
| 1. Write all | None (current behavior) | Baseline |
| 2. Oracle | 1 on evidence sessions, 0 elsewhere | Upper bound of selective writing |
| 3. Random | 1 on random non-evidence sessions with a similar token count | Whether writing less helps by itself |
| 4. Write nothing | 0 everywhere | Lower bound; should be close to plain truncation |
| 5. Oracle, decay kept | Same as arm 2, but only b and w are zeroed | Whether the gain comes from less interference or from less decay |

**How to read the result.**

- Arm 2 clearly better than arm 1 and arm 3: writing and interference are the bottleneck. Go on to Experiment 2.
- Arm 2 about equal to arm 1: the bottleneck is state capacity or the read path. Selective writing is the wrong lever; stop here.
- Arm 4 not close to truncation: the script or the mask has a bug. Fix this before reading anything else.

**Cost.** About 115K tokens per question, batch size 1, no gradients. I estimate 100 questions and 5 arms at about 2 hours on one 80 GB GPU. This is not measured.

**Script outline.**

```python
for row in lme_rows:                       # preprocessed LongMemEval
    ids, sess_spans = build_ids(system, row["memory_docs"], question, gold_answer)
    labels = answer_only(ids)              # -100 everywhere except the answer
    chunk_id, evict_step = per_token_eviction(
        chunk_spans, chunk_limit=10**9, token_limit=32768, n_sink=1)
    if not evidence_fully_evicted(...):
        continue
    for arm in ARMS:
        write_mask = make_mask(arm, sess_spans, row["evidence_doc_idx"])
        loss = model(input_ids=ids, labels=labels, chunk_id=chunk_id,
                     evict_step=evict_step, write_mask=write_mask).loss
        log(row["question_type"], arm, loss)
```

## Experiment 2: per-chunk selective write without labels

This experiment replaces the oracle with a score that the model can compute while reading. In each memory chunk, only the top-scoring share of tokens is written. Run it only if Experiment 1 shows a gain.

**Rule for every scorer.** A score may use the current chunk and the memory state. It may not use later chunks, the question, or attention over the full history. Chunks that hold the question or the answer are never filtered.

**Step A: no training.** Same data and metric as Experiment 1. Write ratios 50% and 25%.

| Scorer | How the score is computed |
| --- | --- |
| Random | Uniform random; the control |
| Key norm | Norm of the memory key of each token |
| Chunk-end attention | Attention that the last 32 tokens of the chunk pay to each token in the chunk, in the same layer |
| Prediction error | Size of v - S^T k: how badly the current state already predicts this token's value |

- If 25% with any scorer loses little against write-all, the writes are redundant and a learned scorer is worth building.
- If every scorer is about equal to random, these scores carry no useful signal.

**Step B: learnable scorer.** A small linear head inside the memory module gives one score per token.

1. Train with a soft mask: sigmoid(score) multiplies g, b and w.
2. Add a penalty that keeps the mean mask near the target ratio.
3. Tighten to hard top-K per chunk, or use hard top-K forward with the sigmoid gradient backward.
4. Control arm: continue training the write-all model for the same number of steps. Without it, a gain could come from extra training alone.

**Why this is more than the existing write gate.** GDN2's write gate w is already learned, but it is soft and per channel. The new parts are a hard per-chunk budget, and no decay on skipped tokens.

## Experiments 3 to 5 (later, only if 1 and 2 work)

These three need changes to the kernel or the read path, so they come last.

**Experiment 3: exact solve for the selected tokens.** In the toy simulation, one delta step per token stores 64 selected pairs at cosine 0.84, and a least-squares solve stores them at 1.00. The test: after selection, run several delta passes over the selected tokens of a chunk, or keep an extra 128 x 128 matrix per head (the sum of k k^T) and solve. Cost: more state per head and a small solve per chunk.

**Experiment 4: keep keys of evicted tokens, drop values.** Today the model reads evicted history only as S^T q, with no softmax over that history. For a linear memory, the sum of a_i (S^T k_i) equals S^T applied to the sum of a_i k_i. So if the keys of selected evicted tokens are kept, the model can address them with a real softmax and rebuild the values from S. Keys alone are half of the KV cache; keeping 10% of them is about 5%. Risk: the rebuilt values are lossy, so this only helps together with Experiment 3. I have not checked whether this exact design has been published.

**Experiment 5: mapping knowledge against single facts.** Prediction: without softmax over evicted history, the state should hold knowledge shared by many examples (for example label mappings in in-context classification) better than one isolated fact. Test with a synthetic task that has the same token count in both conditions. The paper's numbers support this only weakly: the gain of state over truncation is between -0.01 and +0.04 on every dataset it reports.

## Code changes

Experiment 1 needs one new script and three added lines; everything else is for Experiment 2 or later. Line numbers are from the LiveMem repository at commit 71a6a8b.

| # | File and line | Change | Needed for |
| --- | --- | --- | --- |
| 1 | New script, about 150 lines | Build token ids and per-session spans, build the eviction schedule, build the mask per arm, run the forward pass, log answer loss by question type | Exp. 1 |
| 2 | `models/memory_gdn2.py:128` | Add a flag so that the mask can skip g (decay kept) while still zeroing b and w. See the snippet below | Exp. 1, arm 5 |
| 3 | `models/modeling_memory_qwen3.py:210` | No change for Exp. 1: an explicit `write_mask` argument already takes priority. Confirm this on the H200 code | Exp. 1 |
| 4 | `models/configuration_memory_qwen3.py:54` | Only designs X and Y are accepted. Add a third design, a write ratio and a scorer name | Exp. 2 |
| 5 | `models/modeling_memory_qwen3.py:166` (`_set_mem_control`) | Also pass `chunk_id`, `seq_ids` and a flag for question/answer chunks to every layer | Exp. 2 |
| 6 | `models/modeling_memory_qwen3.py:111` (`MemoryAttention.forward`) | After `o_main`, score the tokens, take top-K inside each chunk, build this layer's mask, then call `self.mem` | Exp. 2 |
| 7 | `models/memory_gdn2.py:129` | To let each head pick different tokens, widen the mask from [B, T] to [B, T, H]; one `view` call. Put the learnable score head in this module so its name contains `.self_attn.mem.` and it is trained automatically | Exp. 2 |
| 8 | `train/sft/data.py:233` (`_get_one`) | Return a per-token flag for memory chunks; the table already has `qa_chunk_idx` | Exp. 2, step B |
| 9 | `train/sft/collator.py`, `train/sft/loop.py:168` | Pack the new field and pass it to the model. Inside a pack, `chunk_id` restarts at 0 for every sample, so a chunk is identified by (`seq_ids`, `chunk_id`) | Exp. 2, step B |
| 10 | `models/vllm/memory_qwen3/gdn2_side.py` | The file states that it does not support `write_mask`. Add the same scoring, top-K and masking | Benchmark accuracy |
| 11 | `models/vllm/memory_qwen3/evict_registry.py:232` | `clamp_to_chunk_boundary` is on only in stride mode. Turn it on for evaluation too, so that each prefill slice is exactly one chunk and top-K picks the same tokens as in training | Benchmark accuracy |
| 12 | `third_party/flash-linear-attention/fla/ops/gdn2` | Kernel change for the exact solve or the extra k k^T state | Exp. 3 |
| 13 | `models/vllm/memory_qwen3/evict_manager.py:96` | Evicted KV blocks are freed here. Keep the selected keys instead of freeing everything | Exp. 4 |

**Snippet for row 2.**

```python
# models/memory_gdn2.py, around line 128
if write_mask is not None:
    wm = write_mask.to(g.dtype).view(write_mask.shape[0], write_mask.shape[1], 1, 1)
    if not getattr(self, "skip_keeps_decay", False):   # new: arm 5 keeps the decay
        g = g * wm
    b = b * wm.to(b.dtype)
    w = w * wm.to(w.dtype)
```

## Porting to the H200 checkpoint

I have not seen the code that produced the H200 checkpoint, so this section lists what to find there, not where it is. If that checkpoint was trained with the LiveMem repository, the table above applies as written.

| What to find | Why it matters |
| --- | --- |
| The line where the per-token decay, erase and write strengths are computed, just before the state update kernel is called | The mask must multiply all of them. If the code has one shared strength instead of separate b and w, multiply that one and the decay |
| Whether a per-token write mask already exists, and whether it also stops the decay | If it only zeroes the write, a skipped token still erases and decays; arm 2 would then test the wrong thing |
| Whether the state update runs once per token or once per chunk | With per-chunk updates, a per-token mask has to be applied to the per-token contributions before they are summed |
| How state is reset between samples in a batch | The mask must not interfere with the reset; test with batch size 1 first |
| How the attention window is enforced in the evaluation forward pass (mask or real KV eviction) | Experiment 1 needs the same window as the benchmark setting, in a single forward pass with labels |
| Whether the forward pass accepts labels and returns answer-only loss at 115K tokens without building full-vocabulary logits | Otherwise the run goes out of memory; LiveMem solves it at `models/modeling_memory_qwen3.py:413-436` |

## Checklist and open questions

**Before trusting any number from Experiment 1.**

- [ ] Arm 1 with `write_mask=None` gives the same loss as a mask of all ones.
- [ ] Arm 4 (write nothing) is close to truncation to 32K with a zero state.
- [ ] Tokenizing the sessions one by one and joining the ids gives the same ids as tokenizing the whole prompt.
- [ ] The prompt format matches the one used by the benchmark evaluation of that checkpoint.
- [ ] For each counted question, every evidence token has `evict_step` no larger than the question's `chunk_id`.
- [ ] The random arm writes about the same number of tokens as the oracle arm.
- [ ] Results are reported per question type, not only as one mean.

**Open questions.**

- Which code base and memory design produced the H200 checkpoint? The porting table depends on it.
- Was that checkpoint trained with a mask of zeros anywhere? A model that never saw skipped tokens may react badly to them, which would make the oracle look worse than it is.
- Is answer loss a good enough proxy for benchmark accuracy? If arm 2 wins on loss, the next step is rows 10 and 11 of the code table, to measure real accuracy.
- LongMemEval gold answers are short. If the loss differences are noisy at 100 questions, run all eligible questions.
- Two details of the paper's serving setup differ from its text and may affect comparisons: prefill slices are clamped to chunk boundaries only in stride mode, and a first full prefill is not evicted (`flashinfer_evict.py:123`). I read this from the code and have not run it.
