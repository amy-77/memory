#!/usr/bin/bash
# One worker of the job queue: bash worker.sh <work dir> <env.sh> <gpu> <worker name>
# Jobs are lines "name bench mode format shard nshards" in <work dir>/jobs.txt. A worker claims a job by creating
# <work dir>/claims/<name> (mkdir is atomic), runs it on its GPU and leaves <work dir>/done/<name> (or failed/<name>).
# Results: <work dir>/out/<bench>/<format>_<mode>.s<shard>of<nshards>.jsonl (the runner resumes a partial file).
W=$1; ENV=$2; GPU=$3; NAME=$4
source $ENV >/dev/null 2>&1
cd $W/bench
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=$W/tmp PYTHONDONTWRITEBYTECODE=1 MEMT_FRAMEWORK=$W/framework
MEM="--model $W/models/Qwen3-4B --attn-impl sdpa --memory $W/models/memt4b_step2160/checkpoint --window 2048 --sinks 0 --attn-segment 2048"
BASE="--model $W/models/Qwen3-4B --attn-impl sdpa --memory none --window 0"
mkdir -p $W/claims $W/done $W/failed $W/logs/jobs
while true; do
  job=""
  while read -r name bench mode format shard nshards; do
    [ -z "$name" ] && continue
    [ -e $W/done/$name ] && continue
    [ -e $W/failed/$name ] && continue          # retried only after the failed/ marker is removed by hand
    mkdir $W/claims/$name 2>/dev/null || continue
    job="$name $bench $mode $format $shard $nshards"; break
  done < $W/jobs.txt
  [ -z "$job" ] && { echo "$(date) $NAME: no job left"; exit 0; }
  read -r name bench mode format shard nshards <<< "$job"
  [ "$mode" = full ] && M=$BASE || M=$MEM
  mkdir -p $W/out/$bench
  echo "$(date) $NAME gpu $GPU start $job"
  python bench_memt.py --bench $bench --data $W/data --mode $mode --format $format --shard $shard --nshards $nshards \
    --out $W/out/$bench/${format}_${mode}.s${shard}of${nshards}.jsonl $M > $W/logs/jobs/$name.log 2>&1
  if grep -q "^DONE" $W/logs/jobs/$name.log; then touch $W/done/$name; rm -rf $W/failed/$name
  else echo "$(date) $NAME FAILED $name: $(grep -E 'Error|error' $W/logs/jobs/$name.log | tail -1)"; touch $W/failed/$name; fi
  rmdir $W/claims/$name 2>/dev/null
done
