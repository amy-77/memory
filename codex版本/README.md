# Codex 版本：Qwen3-4B 的 MemT 风格训练

这是独立于仓库原有 `memt/` 和 CC 版本的实现，基于原仓库提交
`6f8a640e3451a4b483cc7038d246b29f14a48c49` 修复和扩展。
它是按公开资料实现的 Qwen3-4B 适配，不是官方 MemT 训练代码或完整复现结果。

- [修复与训练说明](TRAINING_FIXES.md)：问题、启动命令、旧 checkpoint 兼容、测试范围。
- [参考实现与取舍](REFERENCES.md)：MemT、LaCT、FLA/KDA、LiveMem 等工作的适用范围。
- `framework/scripts/train_memory.py`：冻结 Qwen3、训练记忆分支的 CPT / 回答位置 KL 蒸馏。
- `framework/scripts/cache_teacher.py`：离线缓存完整上下文教师的 Top-K 分布。
- `framework/scripts/check_kda_backend.py`：可选 CUDA KDA 算子的状态、跨块梯度及性能检查。
- `train_run/train_8gpu.sh`：参数化的 4B 启动入口；所有路径相对于本目录解析。

从仓库根目录启动一个新实验：

```bash
python -m pip install -r codex版本/requirements.txt
MODEL_PATH=/path/to/Qwen3-4B \
DATA_SPEC='/path/to/train/data-*.arrow' \
RUN_DIR=/path/to/memt_codex_run \
NPROC_PER_NODE=8 bash codex版本/train_run/train_8gpu.sh
```

先用 `--stop-at 10` 做 GPU 小规模检查，再决定完整训练预算。
默认保留原有的 36 层记忆结构，使用 PyTorch 写入实现。
可选 FLA/KDA 必须先按 [参考说明](REFERENCES.md) 安装并验证，再显式启用。

当前修复包括：正确选择完整 checkpoint、真正生效的学习率下限、文档隔离、
按有效 token 归一化、多卡有限数据结束同步、梯度累积期间减少通信，以及
按 token 分块并重计算词表损失。权重格式兼容旧模型；改变数据策略应新开实验。

`locomo_eval/`、`train_run/step_2160_memory_config.json` 和
`heldout_eval_summary.txt` 是保留的历史参考，不代表此版本重训后的成绩。
