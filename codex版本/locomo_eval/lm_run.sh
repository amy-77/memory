#!/usr/bin/bash
# LiveMem-RL on LoCoMo, aligned with the MemT runs (2048 chunks, no sinks). Starts when the MemT generation runs are
# done (FULL_DONE); 5 modes x 7 shards, the 5 modes of one shard share a GPU (batch-1 decoding leaves it mostly idle).
W=/data/qyl/memt_locomo_20261122; cd $W
export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=/data/qyl/livemem_env/cache/triton
PY=/data/qyl/livemem_env/venv/bin/python
until grep -q FULL_DONE logs/full.log; do sleep 60; done
mkdir -p out/lm_full
G=(0 1 2 3 4 5 7); NS=${#G[@]}
declare -A NAME=( [full]=A_full [last]=B_last [write]=C_write [evidence]=F_evidence [random]=G_random )
for s in $(seq 0 $((NS-1))); do
  for m in full last write evidence random; do
    n=${NAME[$m]}; [ -f out/lm_full/$n.s$s.json ] && continue
    CUDA_VISIBLE_DEVICES=${G[$s]} $PY eval_locomo_livemem.py --mode $m --shard $s --nshards $NS \
      --out out/lm_full/$n.s$s.json > logs/lm_full_${n}_s$s.log 2>&1 &
  done
done
wait
grep -hE "Traceback|Error" logs/lm_full_*.log | head -5
python3 - <<'PY'
import json, glob, os
D = "out/lm_full"
for n in sorted({os.path.basename(f).split(".s")[0] for f in glob.glob(D + "/*.s*.json")}):
    recs = []
    for f in sorted(glob.glob(f"{D}/{n}.s*.json")):
        recs += json.load(open(f))["records"]
    recs.sort(key=lambda r: (r["row"], r["query_index"]))
    json.dump({"records": recs}, open(f"{D}/{n}.json", "w")); print(n, len(recs))
PY
echo LM_GEN_DONE
until grep -q ALL_DONE logs/finish3.log; do sleep 60; done      # the MemT judge uses GPU 6 first
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$W/pyshim /usr/bin/python3 judge.py out/lm_full --tp 1 > logs/judge_lm_full.log 2>&1
python3 subset_acc.py out/lm_full
echo LM_ALL_DONE
