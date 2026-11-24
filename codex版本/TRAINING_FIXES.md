# Qwen3-4B MemT-style training: corrected implementation

This is a training-code correction, not a new trained checkpoint or a claim that
Qwen3-4B reproduces the official MemT results. The matrix update, 36 memory layers,
32 heads × 128 dimensions, and checkpoint parameter names are unchanged.

## Bugs and changes

| Problem | Corrected behavior |
|---|---|
| `sort -t_ -k2` sorted an underscore-containing full path instead of the checkpoint step | Parse `step_<number>` numerically; require weights, optimizer and matching metadata; resolve `MOVED.txt` |
| Cosine decay ignored `--min-lr-ratio` and used 0.1 | Use the configured floor and reach it at the last update; `--legacy-cosine` preserves the original schedule |
| Packing joined unrelated documents without resetting memory or attention | Default `--text-packing document`: independent examples, right padding with attention/label masks; split long documents with a one-token overlap so boundary targets are retained |
| Unequal post-training shard lengths caused `StopIteration` on different ranks | Synchronize exhaustion, use masked dummy microbatches, and normalize over all valid target tokens across ranks and accumulation; record the actual final step |
| DDP reduced gradients after every microbatch | `no_sync()` wraps forward and backward except the last microbatch; unused write parameters from short/dummy examples are handled |
| Student `[batch, length, vocabulary]` logits and FP32 softmax could exhaust memory | Compute CE/KL from hidden states in checkpointed token chunks; backward recomputes each head chunk instead of retaining every chunk's logits |
| Checkpoint readers could see partial saves | Stage weights, optimizer and metadata in a hidden directory, then atomically publish with `COMPLETE` |
| A partial archive file was skipped forever because it already existed | Verify SHA-256 and replace mismatched files; write the move pointer before removing local large files; retain archived optimizer states |
| Launchers executed an old external checkout | Resolve this repository's `framework/` relative to the launcher and accept paths through environment variables |
| CPU BF16 base weights conflicted with FP32 memory projections | Enable CPU BF16 autocast as well as CUDA BF16 autocast |
| Post-training could not resume, and checkpoints omitted RNG state | v3 checkpoints save each rank's actual data cursor and Python/NumPy/Torch CPU/CUDA RNG states, alongside weights and optimizer |
| Resuming CUDA Adam moved CPU step counters onto the GPU | Deserialize on CPU; Adam relocates moment tensors to their parameters while retaining non-capturable counters on CPU |
| A resumed partial logging interval divided loss by the full logging interval | Average over the updates actually observed since restart/last log |
| An empty post-training source produced a successful step-zero final checkpoint | Fail with an explicit no-supervised-examples error |

Losses and token counters exclude padding. Variable document lengths mean that
4768 updates no longer imply exactly 5 billion processed tokens. Check
`progress.json.processed_tokens`; if resuming old checkpoints without this count,
`processed_before_resume_known=false` means it only counts tokens since migration.

The chunked loss is mathematically equivalent to the original CE/KL objective;
floating-point summation order can change. It trades LM-head recomputation for
lower activation memory. It does not reduce the ~1.95B trainable memory parameters,
their optimizer states, or the cost of the full-attention teacher.
An optional offline Top-K cache removes the teacher model from the student training
process; an optional KDA write backend is separately validated. See [REFERENCES.md](REFERENCES.md).

## Start a new corrected CPT run

Install the CUDA-compatible PyTorch build in your training environment, then:

```bash
python -m pip install -r codex版本/requirements.txt
export MODEL_PATH=/path/to/Qwen3-4B
export DATA_SPEC='/path/to/ldc/train/data-*.arrow'
export RUN_DIR=/path/to/memt4b_corrected
export NPROC_PER_NODE=8
bash codex版本/train_run/train_8gpu.sh
```

`RUN_DIR` must be a new directory for a new experiment. The launcher saves the
initial memory, logs to `train.log`, and writes checkpoints under `ckpt/`.
Run it again to select the newest complete checkpoint and resume its text stream.
The supervisor entry `run_full.sh` now delegates to this same launcher; it does
not run the old external evaluator, repeatedly retry crashes, or delete checkpoints.
Extra arguments override the preset, for example `--stop-at 10` for a GPU smoke test.

The preset retains the previous 4B architecture, LR 1e-3, QKV multiplier 0.1,
warmup 256, Adam betas (0.9, 0.95), and weight decay 0.1. New runs use Adam epsilon
1e-8 instead of the archived 1e-15 and honor the cosine floor 0.05. These are starting
settings, not experimentally established optimal hyperparameters. Use
`--loss-chunk-size 64` or `256` to tune the memory/recomputation tradeoff.
`flex_attention` is intended for CUDA; the CPU tests use `eager`.

## Existing checkpoints

There are two distinct options:

1. **Continue the original experiment.** Keep the original model/data paths,
   rank count, sequence length, batch/accumulation, seed and optimizer settings.
   Set `LEGACY_RESUME=1` and `RESUME=/path/to/step_2160` with the launcher. This
   enables `--text-packing legacy --legacy-cosine --adam-eps 1e-15`. The new
   checkpoint selection and chunked loss work with the old weights and optimizer.
   Historical cross-document mixing remains deliberately enabled in this mode.
2. **Use old weights to initialize a corrected experiment.** Choose a new
   `RUN_DIR` and set `INIT_MEMORY=/path/to/step_2160`; do not set `RESUME` or
   `LEGACY_RESUME`. This starts a fresh optimizer/schedule and isolated-document
   data. It cannot retroactively fix the training history of the old weights.

For a clean comparison, train the corrected recipe from a new initialization.
New **v3** checkpoints include `training_state.pt`. All ranks contribute their
actual consumed-example cursor and Python/NumPy/Torch CPU/CUDA RNG states before
rank zero atomically publishes the checkpoint. Publication errors are reported
to every rank. The selector rejects a v3 checkpoint missing its runtime state;
the archiver preserves that state and the format marker.

V3 `--resume` automatically replays the saved iterator position, then restores RNG
after model/teacher loading and data replay. It requires unchanged device type, world size,
data/sharding, model/tokenizer, attention, loss and optimizer settings. Single-file
sources (including post-training JSONL and text JSONL/TXT) are SHA-256 checked.
HF streams and Arrow datasets still require an externally immutable revision;
replacing model/tokenizer weights under an unchanged path is also unsupported.
This is replay-based seeking, not an indexed token cache; replaying a large CPT
dataset can take time. No already-consumed example is trained again.

For older checkpoints, `--exact-text-resume` retains the legacy text-replay option.
Missing RNG snapshots produce a warning: old stochastic training history cannot
be reconstructed exactly. Old checkpoints without world size require keeping it
manually. CPU tests cover dropout and uneven two-rank post-training; equal results
in a fixed environment do not guarantee bitwise equality across GPU types, library
versions or reduction algorithms.

## Post-training

Post-training still requires genuine frozen-teacher trajectories in JSONL:
`id`, `history_ids`, `query_ids`, `response_ids`, `teacher_generated: true`.
History consists of complete storage steps, and query + response fits in one
current step. It uses response-only KL, with no write from the response segment.
For example (replace paths and choose `--steps`/schedule for your dataset):

```bash
torchrun --standalone --nproc_per_node 8 codex版本/framework/scripts/train_memory.py \
  --model /path/to/Qwen3-4B --memory /path/to/cpt/final \
  --data post:/path/to/teacher_trajectories.jsonl --out /path/to/post_run \
  --seq-len 32768 --batch-size 1 --grad-accum 4 \
  --window 2048 --attn-segment 2048 --sinks 0 --attn-impl flex_attention \
  --ce-weight 0 --distill-weight 1 --distill-top-k 64 --distill-temperature 1 \
  --lr 1e-4 --lr-schedule wsd --steps 2000 --warmup 500 --decay-steps 1000 \
  --min-lr-ratio .05 --adam-betas .9 .999 --adam-eps 1e-8 --weight-decay .01 \
  --grad-ckpt --loss-chunk-size 128 --fail-on-nonfinite
```

The numerical schedule above is an example, not an inferred size of your data.
The one-pass iterator stops at exhaustion or the requested step limit. If it
ends early, the scheduler does not retroactively compress its decay phase.
`final/memory_config.json` records `step` and `data_exhausted`. To resume mid-pass,
rerun the same command with `--resume /path/to/post_run/step_500` (replace the step
with the checkpoint you actually saved), retaining the original total `--steps`,
world size, accumulation and teacher/cache configuration. Remove `--stop-at` if
it was used for the interruption. V3 restores the actual per-rank cursor, including
uneven final batches; synthetic padding does not advance the source cursor.
Changing the JSONL source is rejected. Using `--memory` instead starts a new stage.
The current repository does not supply the official post-training dataset or
its entire generation pipeline, so fixing the trainer does not complete that stage.
`cache_teacher.py` caches the distributions for existing teacher-generated responses;
`--teacher-cache` uses them with the same KL objective and verifies data/token alignment.

## Archive and validation

```bash
# One pass; retains the local files. ARCHIVE_DIR is a mounted filesystem, not an s3:// URI.
RUN_DIR=/path/to/run ARCHIVE_DIR=/mounted/archive bash codex版本/train_run/mover.sh
# Optional: --watch polls; --move removes verified local weights/optimizer and leaves MOVED.txt.

OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python -m pytest codex版本/framework/tests -q
```

The new archiver accepts only checkpoints with a `COMPLETE` marker. The selector
also accepts legacy checkpoints with complete metadata/weights/optimizer files;
corruption inside a legacy tensor file is detected when loading, not by its name.
Do not run an old mover concurrently with the corrected trainer. The archived
`eval_ckpts.sh` and `locomo_eval/` retain their historical server paths and are not
called automatically by the corrected launchers.

Validation covers full-logit versus chunked CE/Top-K KL values and gradients,
memory gradients with activation checkpointing, masked padding, document
isolation, numeric checkpoint selection, archive repair, uninterrupted versus
resumed training, and two CPU/Gloo ranks with uneven finite post-training data
(including a rank with no remaining examples). All use local tiny Qwen3 models;
they require no model download. Full 4B, 8-GPU training, full-model peak-memory
measurements, and LoCoMo quality after retraining have not been run in this fix.
CUDA write-kernel and tiny-model checks are separate from full 4B training; see
[VALIDATION.md](VALIDATION.md) for measured results and limitations. CPU tests skip
these CUDA checks.

## Reproduction limits

This is a Qwen3-4B adaptation of the [MemT technical report](https://arxiv.org/html/2610.26825v2),
not its original Nano model/training budget. The archived run completed only
part of CPT and no task-specific post-training. The current code also retains
its numerical `MAX_DECAY` cap, which is an implementation choice. Reducing the
number of memory layers, changing optimization, or adding the second training
stage are separate experiments requiring new evaluations. Do not relabel the
historical LoCoMo scores as results from this corrected version.
