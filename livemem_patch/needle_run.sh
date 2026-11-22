#!/usr/bin/bash
# bash needle_run.sh <label> <model dir> ; four state-8k servers (one per write policy) on GPUs 0 and 6 next to training
source /data0/qyl/livemem_repro_20261121/env.sh
cd $W/LiveMem; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python HF_HUB_OFFLINE=1 TOKENIZER_PATH=$W/models/Qwen3-4B-Instruct-2507
L=$1; M=$2; N=$W/needle; P=9700
[ -f $N/prompts.jsonl ] || python $W/sel/needle_build.py --out $N
pids=(); i=0; S=""
for pol in all only without none; do
  g=$([ $i -lt 2 ] && echo 0 || echo 6)
  rm -f $W/logs/needle_${L}_$pol.log
  MEM_SELECT_DIR=$N/$pol CKPT=$M MODE=state WINDOW_SIZE=8192 GPUS=$g PORT=$((P + i)) UTIL=0.14 MAXLEN=65536 \
    nohup bash scripts/serve_livemem.sh > $W/logs/needle_${L}_$pol.log 2>&1 &
  pids+=($!); S="$S $pol=http://127.0.0.1:$((P + i))/v1"; i=$((i + 1)); sleep 45
done
for t in $(seq 1 60); do n=$(cat $W/logs/needle_${L}_*.log | grep -c "Application startup complete"); [ "$n" -ge 4 ] && break; sleep 10; done
echo "$L servers up: $n/4"
grep -hE "ValueError" $W/logs/needle_${L}_*.log | cut -c1-240
[ "$n" -ge 4 ] && python $W/sel/needle_client.py $N/result_$L.json $N/prompts.jsonl $S
for p in $(ps -eo pid,cmd | grep "vllm.entrypoints" | grep -E "port 970[0-3]( |$)" | grep -v grep | awk '{print $1}'); do kill $p; done
kill "${pids[@]}" 2>/dev/null; sleep 25
for pol in all only without none; do echo "$pol: hits $(grep -c " hit " $N/$pol/_server.log) miss $(grep -cE "MISS|BAD|SHAPE" $N/$pol/_server.log)"; done
echo "NEEDLE_DONE $L"
