#!/usr/bin/bash
# First look at selection + Robins rehearsal on the newest intermediate checkpoint of the 16-turn model
# (training keeps running on the same GPUs). 50 dialogues with 32 turns.
source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
python -m pytest -q tests/test_select.py -x 2>&1 | tail -3
unset CUDA_VISIBLE_DEVICES
S=$(ls -d ckpt/dlg16_new/step_* | sort -t_ -k3 -n | tail -1); rm -rf ckpt/early_snapshot; cp -r $S ckpt/early_snapshot; echo "checkpoint $S"
CKPT=$D/ckpt/early_snapshot OUTDIR=$D/out/early bash batch_select.sh data/rc_t32_f8.jsonl e32 50 "0 1 2 3 0 1 2 3" '^(none|oracle|qalone_t05|hard20|hard10|robins_s1|robins_s5|robins_s20|robins_full_s5|soft_robins_s5|hard20_robins_s5|hard20_robins_s20|hard20_robins_both_s5|hard20_robins_bank_s5|replay_only|hard20_replay)$'
echo EARLY_ROBINS_DONE
