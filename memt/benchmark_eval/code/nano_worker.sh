#!/usr/bin/bash
# Job-queue worker for MemT-1.0-Nano (HF): bash nano_worker.sh <work dir> <env.sh> <gpu> <name>
# Jobs: lines "name bench mode shard nshards" in <work dir>/nano_jobs.txt; claims in nano_claims/, done in nano_done/.
# Results: <work dir>/out_nano/<bench>/chat_<mode>.s<shard>of<nshards>.jsonl
W=$1; ENV=$2; GPU=$3; NAME=$4
source $ENV >/dev/null 2>&1
cd $W/bench
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=$W/tmp PYTHONDONTWRITEBYTECODE=1 HF_MODULES_CACHE=$W/tmp/hf_modules
mkdir -p $W/nano_claims $W/nano_done $W/nano_failed $W/logs/nano_jobs
while true; do
  job=""
  while read -r name bench mode shard nshards; do
    [ -z "$name" ] && continue
    [ -e $W/nano_done/$name ] && continue
    [ -e $W/nano_failed/$name ] && continue
    mkdir $W/nano_claims/$name 2>/dev/null || continue
    job="$name $bench $mode $shard $nshards"; break
  done < $W/nano_jobs.txt
  [ -z "$job" ] && { echo "$(date) $NAME: no job left"; exit 0; }
  read -r name bench mode shard nshards <<< "$job"
  mkdir -p $W/out_nano/$bench
  echo "$(date) $NAME gpu $GPU start $job"
  python bench_nano_hf.py --bench $bench --data $W/data --model $W/models/MemT-1.0-Nano --mode $mode --shard $shard \
    --nshards $nshards --out $W/out_nano/$bench/chat_${mode}.s${shard}of${nshards}.jsonl > $W/logs/nano_jobs/$name.log 2>&1
  if grep -q "^DONE" $W/logs/nano_jobs/$name.log; then touch $W/nano_done/$name
  else echo "$(date) $NAME FAILED $name: $(grep -E 'Error' $W/logs/nano_jobs/$name.log | tail -1)"; touch $W/nano_failed/$name; fi
  rmdir $W/nano_claims/$name 2>/dev/null
done
