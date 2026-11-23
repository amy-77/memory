#!/usr/bin/bash
# Gated rehearsal. Always-on rehearsal with question probes protects the stored answer (K2_attc: +28 when the answer
# is in the earlier written chunk) but also resists a newly written answer (-25 when it is in the later chunk).
# Gate: rehearse a write only if the incoming chunk scores lower (BM25) than the best chunk already written.
#   C_write_R16g : every chunk written, gated rehearsal          (vs C_write 447, C_write_R16 471)
#   K2_bm25_R16  : BM25 K=2, rehearsal always on the second write (the gated K=2 result follows exactly from this
#                  run and K2_bm25, see replay_split.py)
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
W=/data/qyl/memt_locomo_20261122; cd $W; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp PYTHONDONTWRITEBYTECODE=1
until grep -q K1_DONE logs/k1_run.log; do sleep 60; done
CK=/mnt/airs-business/memory/memt4b_20261120/step_2160
BASE="--model /data/models/Qwen/Qwen3-4B --data data/locomo2_g4.jsonl --attn-impl sdpa --memory $CK --window 2048 --sinks 0 --attn-segment 2048"
SCORES="out/strat/s*.json,out/strat2/s*.json"
run() { n=$1; shift
  for s in 0 1 2 3 4 5 6 7; do
    [ -f out/full/$n.s$s.json ] && continue
    CUDA_VISIBLE_DEVICES=$s python eval_locomo_memt.py $BASE "$@" --shard $s --nshards 8 --out out/full/$n.s$s.json > logs/rp_${n}_s$s.log 2>&1 &
  done; }
run C_write_R16g --mode write --replay-k 16 --replay-alpha 1.0 --replay-gate bm25 --scores "$SCORES"
run K2_bm25_R16 --mode scored --scores "$SCORES" --strategy bm25 --k 2 --replay-k 16 --replay-alpha 1.0
wait
grep -hE "Traceback|Error" logs/rp_C_write_R16g_s*.log logs/rp_K2_bm25_R16_s*.log | head -3
for n in C_write_R16g K2_bm25_R16; do echo "$n $(grep -h '^replay' logs/rp_${n}_s0.log)"; done
python summary_memt.py out/full > /dev/null
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$W/pyshim /usr/bin/python3 judge.py out/full --tp 1 --seed-from out/full > logs/judge_rp3.log 2>&1
python3 subset_acc.py out/full
python3 replay_split.py
python3 replay_by_dist.py
echo RP3_DONE
