#!/usr/bin/bash
# h20-1: needle test on the released LiveMem-SFT weights. Four state-8k servers (one per write policy) on GPUs 4-7.
source /data/qyl/store_179/livemem_repro/env.sh
cd $W/LiveMem; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python HF_HUB_OFFLINE=1 NEEDLE_SRC=longmemeval_s
N=/data/qyl/needle_20261122; mkdir -p $N $N/logs
# the release uses model_type "livemem"; the serving plugin registers "memory_qwen3"
M=$W/models/LiveMem-SFT-vllm
if [ ! -f $M/config.json ]; then mkdir -p $M; for f in $W/models/LiveMem-SFT/*; do ln -sf $f $M/; done; rm $M/config.json
  python - <<PY
import json
c = json.load(open("$W/models/LiveMem-SFT/config.json")); c["model_type"] = "memory_qwen3"; c["architectures"] = ["MemoryQwen3ForCausalLM"]; c.pop("auto_map", None)
json.dump(c, open("$M/config.json", "w"), indent=1)
PY
fi
P=$W/LiveMem/models/vllm/memory_qwen3
python $N/patch_plugin.py $W/LiveMem
grep -c "select_write" $P/gdn2_side.py $P/evict_registry.py $P/flashinfer_evict.py
python -c "import memory_qwen3, vllm; print('plugin', memory_qwen3.__file__, vllm.__version__)" 2>&1 | tail -1
[ -f $N/prompts.jsonl ] || python $N/needle_build.py --out $N 2>&1 | tail -2
pids=(); i=0; S=""
for pol in all only without none; do
  MEM_SELECT_DIR=$N/$pol CKPT=$M MODE=state WINDOW_SIZE=8192 GPUS=$((4 + i)) PORT=$((9700 + i)) UTIL=0.3 MAXLEN=65536 \
    nohup bash scripts/serve_livemem.sh > $N/logs/srv_$pol.log 2>&1 &
  pids+=($!); S="$S $pol=http://127.0.0.1:$((9700 + i))/v1"; i=$((i + 1)); sleep 5
done
for t in $(seq 1 90); do n=$(cat $N/logs/srv_*.log | grep -c "Application startup complete"); [ "$n" -ge 4 ] && break; sleep 10; done
echo "servers up: $n/4"; grep -hE "ValueError" $N/logs/srv_*.log | cut -c1-240 | head -3
[ "$n" -ge 4 ] && python $N/needle_client.py $N/result_sft.json $N/prompts.jsonl $S
for p in $(ps -eo pid,cmd | grep "vllm.entrypoints" | grep -E "port 970[0-3]( |$)" | grep -v grep | awk '{print $1}'); do kill $p; done
kill "${pids[@]}" 2>/dev/null; sleep 20
for pol in all only without none; do echo "$pol: hits $(grep -c ' hit ' $N/$pol/_server.log 2>/dev/null) problems $(grep -cE 'MISS|BAD|SHAPE' $N/$pol/_server.log 2>/dev/null)"; done
echo NEEDLE_H201_DONE
