#!/usr/bin/bash
# Controls for the needle test: (a) needle INSIDE the 8K window (must be answered: validates the test itself),
# (b) an easier fact (a dog's name) at several distances.
source /data/qyl/store_179/livemem_repro/env.sh
cd $W/LiveMem; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python HF_HUB_OFFLINE=1 NEEDLE_SRC=longmemeval_s
B=/data/qyl/needle_20261122; M=$W/models/LiveMem-SFT-vllm
NEEDLE_KIND=digits python $B/needle_build.py --out $B/ctl_digits --lengths 6000 2>&1 | tail -1
NEEDLE_KIND=word python $B/needle_build.py --out $B/word --lengths 6000,12000,20000,45000 2>&1 | tail -1
pids=(); i=0
for spec in ctl_digits/all word/all word/only word/none; do
  MEM_SELECT_DIR=$B/$spec CKPT=$M MODE=state WINDOW_SIZE=8192 GPUS=$((4 + i)) PORT=$((9710 + i)) UTIL=0.3 MAXLEN=65536 \
    nohup bash scripts/serve_livemem.sh > $B/logs/srv2_$i.log 2>&1 &
  pids+=($!); i=$((i + 1)); sleep 5
done
for t in $(seq 1 90); do n=$(cat $B/logs/srv2_*.log | grep -c "Application startup complete"); [ "$n" -ge 4 ] && break; sleep 10; done
echo "servers up: $n/4"
echo "--- digits, needle inside the window"; python $B/needle_client.py $B/result_ctl.json $B/ctl_digits/prompts.jsonl all=http://127.0.0.1:9710/v1
echo "--- dog name"; python $B/needle_client.py $B/result_word.json $B/word/prompts.jsonl all=http://127.0.0.1:9711/v1 only=http://127.0.0.1:9712/v1 none=http://127.0.0.1:9713/v1
for p in $(ps -eo pid,cmd | grep "vllm.entrypoints" | grep -E "port 971[0-3]( |$)" | grep -v grep | awk '{print $1}'); do kill $p; done
kill "${pids[@]}" 2>/dev/null
echo NEEDLE2_DONE
