#!/usr/bin/bash
# usage: bash setup_data.sh <work dir> [--nano]
# Downloads the base Qwen3-4B, the 4B MemT-style memory checkpoint and the benchmark data into <work dir>
# (layout expected by bench_memt.py / bench_nano_hf.py: <work dir>/models/..., <work dir>/data/...).
# --nano also downloads MemTensor/MemT-1.0-Nano (66 GB) and the judge Qwen/Qwen3.6-35B-A3B (67 GB).
# LoCoMo is not on the Hub: put locomo10.json (github.com/snap-research/locomo, data/locomo10.json) into
# <work dir>/data/locomo/locomo10.json by hand. Set HF_ENDPOINT to a mirror if huggingface.co is slow.
W=$1; NANO=$2
mkdir -p $W/models $W/data/locomo $W/tmp
python - "$W" "$NANO" <<'PY'
import sys
from huggingface_hub import snapshot_download, hf_hub_download
W, nano = sys.argv[1], sys.argv[2] == "--nano"
snapshot_download("Qwen/Qwen3-4B", local_dir=W + "/models/Qwen3-4B",
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
snapshot_download("Yanlin77/memt-style-qwen3-4b-step2160", local_dir=W + "/models/memt4b_step2160")
for rid, files in [("xiaowu0162/longmemeval-cleaned", ["longmemeval_s_cleaned.json"]),
                   ("bowen-upenn/PersonaMem", ["questions_32k.csv", "shared_contexts_32k.jsonl"]),
                   ("ai-hyz/MemoryAgentBench", ["data/Accurate_Retrieval-00000-of-00001.parquet",
                                                "data/Conflict_Resolution-00000-of-00001.parquet",
                                                "data/Long_Range_Understanding-00000-of-00001.parquet",
                                                "data/Test_Time_Learning-00000-of-00001.parquet", "entity2id.json"]),
                   ("Mohammadta/BEAM", ["data/1M-00000-of-00001.parquet"])]:
    for f in files:
        hf_hub_download(rid, f, repo_type="dataset", local_dir=W + "/data/" + rid.split("/")[1])
    print("ok", rid, flush=True)
if nano:
    snapshot_download("MemTensor/MemT-1.0-Nano", local_dir=W + "/models/MemT-1.0-Nano")
    snapshot_download("Qwen/Qwen3.6-35B-A3B", local_dir=W + "/models/Qwen3.6-35B-A3B")
print("SETUP_DONE", flush=True)
PY
