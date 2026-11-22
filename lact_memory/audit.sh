source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=${1:-0} python scripts/audit_scores.py --model Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final \
  --data data/rc_t16_f8.jsonl --limit 12 --out out/audit_t16.json > logs/audit_t16.log 2>&1
CUDA_VISIBLE_DEVICES=${1:-0} python scripts/audit_scores.py --model Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final \
  --data data/rc_t32_f8.jsonl --limit 12 --out out/audit_t32.json > logs/audit_t32.log 2>&1
echo AUDIT_ALL_DONE >> logs/audit_t32.log
