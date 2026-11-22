# 验证记录（2026-11-23）

本页记录代码修复的测试结果，不是 Qwen3-4B 完整训练或下游任务成绩。

## CPU 训练回归

环境：macOS、Python 3.12、PyTorch 2.11.0、Transformers 5.20.2。

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest codex版本/framework/tests -q -p no:cacheprovider
# 40 passed, 3 skipped, 16 warnings in 27.29s
```

三个跳过项需要 CUDA，其中两个还需要可选 FLA 依赖。测试使用本地创建的微型 Qwen3，
无需下载预训练模型。关键覆盖如下：

| 检查 | 结果 |
|---|---|
| 完整词表损失 vs 分块、重计算损失 | CE、KL 和梯度一致；包含不同温度、Top-K 和完整教师分布 |
| 连续训练 vs 保存后恢复 | 确定性文本训练的最终权重逐位一致，步数和有效 token 计数一致 |
| 双进程 CPU/Gloo 的不等长数据 | 真实启动两个 worker；有限数据尾部和空 rank 不挂起，梯度符合全局有效 token 平均 |
| 在线教师 vs 离线 Top-K 教师 | 缓存生成 CLI、回答位置对齐及最终训练权重一致 |
| 文档边界与 padding | 不混入其他文档，padding 不参与损失；长文档分段保留边界目标 |
| Checkpoint 与归档 | 按数字选取完整版本，拒绝未完成保存，修复部分复制的归档 |
| 激活重计算 | 微型模型的记忆参数获得正确梯度 |

Python 语法、三个训练/归档入口的 Bash 语法和 Git 空白检查也已通过。

## CUDA 写入算子

可选 `fla_kda` 与默认 `torch` 后端使用相同写入递推；验收包括状态和跨存储块的
初始状态、key、value、衰减门梯度。默认后端仍为 `torch`。

环境：NVIDIA H20-3e，Python 3.10，PyTorch 2.14.1+cu130，Triton 3.8.0，
Transformers 5.20.2；FLA 固定提交见
[参考说明](REFERENCES.md)。初次使用 Triton 默认 TF32 的检查未通过：
状态相对 RMS 误差约 `1.13e-3`，key 梯度约 `1.33e-3`，高于预先设定的 `1e-3`
阈值。仅设置 `TRITON_F32_DEFAULT=ieee` 仍未通过更严格的逐元素检查，因为
FLA 的矩阵逆合并显式指定了 TF32。最终适配器还在首次编译前将固定版本的
`SOLVE_TRIL_DOT_PRECISION` 设为 IEEE，未放宽任何验收阈值，未修改安装目录文件。

IEEE 模式在 H20 上通过了写入等价性检查。使用 4 个 head、每 head 128 维、
2048 个 token，拆成两次写入并通过状态边界反传；FP32 输入和状态，随机种子 42。
使用仓库中的 H20 配置固定一个反向内核的原生 tile 候选，其余算子正常调优。
未经限制的首次调优先后触发两次 600 秒超时，因此提供这个配置以减少启动开销。

| 检查项 | 最大绝对误差 | 相对 RMS 误差 |
|---|---:|---:|
| 最终状态 | 1.42e-7 | 4.43e-7 |
| 初始状态梯度 | 2.49e-21 | 3.84e-10 |
| key / 擦除门组合 `u` 的梯度 | 3.58e-6 | 5.44e-7 |
| value / 写入门组合 `z` 的梯度 | 1.07e-6 | 4.69e-7 |
| 衰减门梯度 | 8.94e-7 | 1.12e-6 |

初始状态经过长序列后梯度很小，RMS 分母采用 `1e-12` 下限；另有短序列 CUDA
测试检查未充分衰减的初始状态梯度。上述检查的阈值为相对 RMS 不超过 `1e-3`。

```bash
export TRITON_F32_DEFAULT=ieee
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$PWD/codex版本/framework/configs/fla-h20-ieee"
python codex版本/framework/scripts/check_kda_backend.py --length 2048 --heads 4 --dim 128
python -m pytest codex版本/framework/tests/test_kda_cuda.py -q -p no:cacheprovider
# 3 passed, 2 warnings in 27.43s
```

三个 CUDA 回归均通过：256-token 的状态及跨块梯度逐元素对比、BF16 微型 Qwen3
加激活重计算时的损失/记忆梯度对比，以及默认 PyTorch 记忆配合分段
`flex_attention` 与 eager attention 的对比（包含不同长度样本、padding 和梯度）。
两条 warning 来自 Transformers 的 mask 编译参数弃用提示。

本次写入 forward + backward 的 5 次计时均值为 PyTorch **18.46 ms**、KDA
**71.71 ms**，增量峰值分配分别约 **34.50 MiB**、**61.13 MiB**。
GPU 在测量期间出现其他训练负载，因此这些数字不是可靠的独占性能比较；
即便如此，本次也没有证据证明 KDA 在该 FP32 配置下更快或更省显存，不能把
它作为默认加速方案。上述内存仅是写入检查的增量，不是完整模型峰值显存。

## 这些检查不能证明什么

- 没有完成 Qwen3-4B 的 8 卡 CPT / 后训练，也没有产生新的训练权重。
- 没有测得完整 4B 训练吞吐、峰值显存或重训后的 LoCoMo / LongMemEval 成绩。
- Kernel 的误差和速度不等于最终任务精度和整模型速度。
- 官方完整训练数据和配方未公开，因此不能宣称达到官方 MemT-1.0 的效果。
- 历史评测文件仅保留作参考，不是本版本结果。

实际训练应先用新输出目录运行 `--stop-at 10`，核对有效 token 数、梯度、显存和
checkpoint，再按预算安排完整实验及无记忆/错误记忆等对照。
