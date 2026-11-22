#!/usr/bin/bash
# Evaluate every checkpoint on the object store that has no results yet: held-out long documents,
# 4 modes (write / nowrite / reset / wrong = memory of another document). Needs free GPUs.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1
W=/data/qyl/memt4b_20261120; DST=/mnt/airs-business/memory/memt4b_20261120
R=/data/qyl/lact-insight/overleaf_6af80380/memory_research_20261119/frameworks/memt
M=/data/models/Qwen/Qwen3-4B
mkdir -p $W/eval; cd $R
jobs_list=()
for ck in $(ls -d $DST/step_* $DST/final 2>/dev/null | sort -t_ -k3 -n); do
  [ -f $ck/memory_weights.pt ] || continue
  n=$(basename $ck)
  for mode in write nowrite reset wrong; do
    [ -f $W/eval/${n}_$mode.json ] || jobs_list+=("$ck $n $mode")
  done
done
echo "$(date) ${#jobs_list[@]} evaluation jobs" >> $W/eval/eval.log
i=0
for j in "${jobs_list[@]}"; do
  set -- $j
  CUDA_VISIBLE_DEVICES=$((i % 8)) timeout 60m python scripts/eval_streaming.py --model $M --memory $1 \
    --tokens $W/val48_32k.npy --num-docs 48 --mode $3 --window 2048 --sinks 0 --attn-segment 2048 \
    --attn-impl sdpa --out $W/eval/${2}_$3.json > $W/eval/${2}_$3.log 2>&1 &
  i=$((i + 1))
  [ $((i % 8)) = 0 ] && wait
done
wait
python - <<PY >> $W/eval/summary.txt 2>&1
import json, glob, os, re, time
W = "$W/eval"; print("==", time.strftime("%m-%d %H:%M"))
steps = sorted({re.match(r"(.*)_(write|nowrite|reset|wrong)\.json", os.path.basename(f)).group(1) for f in glob.glob(W + "/*.json")},
               key=lambda s: int(s.split("_")[1]) if "_" in s else 10**9)
for s in steps:
    row = []
    for m in ["write", "nowrite", "reset", "wrong"]:
        p = f"{W}/{s}_{m}.json"
        if not os.path.exists(p): row.append(f"{m} -"); continue
        d = json.load(open(p)); docs = d["documents"] if isinstance(d, dict) and "documents" in d else d.get("docs", d) if isinstance(d, dict) else d
        try:
            last = sum(x["metrics"]["last2k"] for x in docs) / len(docs); bey = sum(x["metrics"]["beyond2k"] for x in docs) / len(docs)
            row.append(f"{m} last2k {last:.4f} beyond2k {bey:.4f}")
        except Exception as e:
            row.append(f"{m} (keys {list(d.keys())[:6] if isinstance(d, dict) else type(d)})")
    print(s, " | ".join(row))
PY
echo "$(date) evaluation round done" >> $W/eval/eval.log
