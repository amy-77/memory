# 参考实现与可复现边界

## 哪些可以借用

| 参考 | 可以借鉴 | 不能直接当作等价替换 |
|---|---|---|
| [MemT-1.0 论文](https://arxiv.org/html/2610.26825v2)、[官方仓库](https://github.com/MemTensor/MemT-1.0) | 对称稳定的 GDN-2 更新、存储步开始前的状态读出、冻结骨干、CPT 和回答位置 Top-64 KL | 未公开的完整训练数据、训练配比和训练系统；不能声称恢复了官方成绩 |
| [LaCT / lact_llm](https://github.com/a1600012888/LaCT/tree/main/lact_llm) | flame/torchtitan 训练工程、激活重计算、分片优化器、融合算子的组织方式 | LaCT 的 SwiGLU 快权重内循环、Muon/momentum 更新不是 MemT 的矩阵 delta 更新 |
| [FLA KDA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda) | 支持逐 key 通道衰减和状态梯度的并行 delta 算子；本版本提供适配器 | 普通 GDN 的每 head 标量遗忘不能直接代替逐通道遗忘；KDA 的 token 内读出也不是 MemT 存储步读出 |
| [LiveMem 论文](https://arxiv.org/html/2608.02515v2)、[训练代码](https://github.com/cafeii/LiveMem/tree/master/train) | 预分词、长文档 QA → 混合任务课程、证据离开窗口后的监督、回答位置 mask、训练/推理可见性一致 | 其公开后训练是回答 CE 的 SFT 加 GRPO，不能称为现成的 MemT 教师 KL 蒸馏实现 |
| [CLAM](https://arxiv.org/html/2611.20759v1) | 记忆依赖任务的合成方法，参数记忆和原文检索互补的评估 | 超网络生成低秩 adapter 和 episodic buffer 改变架构，应作为独立基线 |
| [TMEM](https://arxiv.org/html/2606.04536v1) | 经验抽取、在线适配的消融，参数记忆是否影响后续行为的评估 | rollout 内 LoRA 更新和 RL 数据抽取策略不是当前的冻结骨干 + 矩阵状态机制 |
| [Locas](https://arxiv.org/html/2602.05085v1) | 从预训练权重/激活初始化记忆、检查通用能力遗忘 | 局部 FFN/GLU 记忆及其写入方式与 MemT 矩阵状态不同 |

当前检查的源码版本：

```
MemTensor/MemT-1.0:              4b08b5d54072b80221f22c912a0d560e02847295
a1600012888/LaCT:               a648340f9798f173227a0626fde66a6e9b65879a
cafeii/LiveMem:                 71a6a8b05f359d98f9a2e302d1a8727a4a184d1b
fla-org/flash-linear-attention:  0d42d7ee9ffe88f42f7e2b08d81cd0f5b067f052
```

这些项目不是都公开了全部训练条件。MemT 当前目录重点是模型/推理发布；LiveMem
提供训练和数据处理入口，但数据访问、许可证和人工生成环节仍需分别核实。
本版本没有复制第三方 kernel 文件；可选 FLA 作为固定提交的依赖使用。

## KDA 与当前写入公式的对应

当前实现先计算 `u = sqrt(erase) * normalize(k)`、`z = write * v` 和逐通道
`D = diag(exp(lambda))`，然后执行：

```
S_new = D @ S + u @ (z - u.T @ D @ S).T
```

FLA KDA 的递推是 `S_new = D @ S + beta * k @ (v - k.T @ D @ S).T`。
所以设置 `k=u, v=z, beta=1, g=lambda`，状态更新在数学上等价。
这避免把通道擦除门压缩成标量，也避免除以接近零的擦除门。

- `use_qk_l2norm_in_kernel=False`，否则二次归一化会破坏 `u` 中的擦除门。
- `output_final_state=True`，保留前后存储块之间的梯度，不使用 `detach()`。
- 不采用 KDA 的每 token 输出；当前块仍读取此前的状态，再统一写入该块。
- 保持原代码的 FP32 状态和输入、遗忘上限，不擅自改用 BF16 状态或改变公式。
- 默认后端为 `torch`；`fla_kda` 是显式选择，失败时不会静默换用不同算法。

```bash
python -m pip install -r codex版本/requirements-kda.txt
export TRITON_F32_DEFAULT=ieee
# H20 上可选：固定一个原生候选配置，避免 FP32 反向内核的长时间自动调优。
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$PWD/codex版本/framework/configs/fla-h20-ieee"
python codex版本/framework/scripts/check_kda_backend.py --length 2048 --heads 32 --dim 128
# 在当前 CUDA / PyTorch / FLA 环境验证通过后，给训练命令追加：
# --mem-matrix-backend fla_kda
```

该检查覆盖最终状态及初始状态、key、value、遗忘门的梯度，且跨两个写入调用反传。
时间是写入算子 forward + backward，不代表整个 Qwen 模型的训练吞吐。
该 H20 配置只固定一个反向内核的 tile/warp/stage 参数，其余算子照常选择配置；
这是减少首次编译成本的候选参数，不是最优速度的声明，也不应直接外推到其他 GPU。
数学等价不保证任意 dtype、长度下误差相同，长序列和完整模型仍需实测。
H20 上首次检查发现 Triton 默认 TF32 导致状态相对 RMS 误差约 0.00113，超过
0.001 的验收阈值。因此适配器要求进程启动前显式设置 IEEE 精度，不能仅凭输入
张量标注为 FP32 就假定乘法按完整 FP32 执行。
这一环境变量不能覆盖 FLA 显式指定的 TF32 运算。固定版本中的矩阵逆合并仍然
触发逐元素误差，因此适配器在首次编译前将该版本的
`SOLVE_TRIL_DOT_PRECISION` 也设置为 IEEE；它影响当前进程内的 KDA 精度策略，
不修改安装目录或递推公式。使用了 FLA 内部常量，故必须保持依赖固定并在升级时
重新验证；找不到预期常量会报错。仅设置环境变量不足以通过完整检查。
本次 H20 的完整 IEEE 检查通过，最大相对 RMS 误差约 `1.1e-6`；不过没有观察到
提速，故保留 PyTorch 为默认实现。误差、计时条件与限制见 [验证记录](VALIDATION.md)。

## 高效后训练：保留 MemT 目标，借用数据工程

本版本继续采用完整上下文冻结教师的回答位置 Top-K KL，新增可选离线缓存。
缓存保存 FP32 Top-K 对数概率和 token ID，不再量化；学生分母仍是完整词表，
只有教师在 Top-K 支持集内归一化。数据哈希、token 对齐、模型标识、K 和温度必须匹配。

```bash
python codex版本/framework/scripts/cache_teacher.py \
  --model /path/to/Qwen3-4B --data /path/to/teacher_trajectories.jsonl \
  --out /path/to/top64_cache --segment 2048 --max-length 32768 \
  --top-k 64 --temperature 1 --dtype bfloat16
# 给 TRAINING_FIXES.md 中的 post: 训练命令追加：
# --teacher-cache /path/to/top64_cache
```

学生训练因此不需要在线加载第二个 4B 教师。教师计算移至预处理，成本没有消失；
缓存也需要磁盘空间。同一模型标识下替换本地模型权重或 tokenizer 时必须重建缓存。
脚本缓存已有教师轨迹的分布，不负责生成问答，也不会把参考答案标记为教师生成。

数据方面可以借鉴 LiveMem 的训练 split、预分词和长程 QA 混合流程，但要整理成
当前 `history_ids/query_ids/response_ids` 格式，并用冻结教师生成回答。保持存储步
边界、只对回答算损失，保证支持证据在学生当前窗口之外。不要把 LoCoMo、LongMemEval
的测试问题或答案加入训练。多问题、更新、冲突和无法回答的样本应保留来源和划分记录。

## 仍需实验的部分

FSDP / ZeRO 和预分词索引是下一层工程优化，本版未移植 flame 的整个训练器。
减少记忆层数、加入 MoS、使用 LaCT 更新或 CLAM/Locas 记忆、添加 GRPO 都会改变实验，
应分别比较。至少同时报告完整上下文、只看窗口、正常记忆、不写记忆、错误记忆等控制，
以及训练吞吐、峰值显存、长程任务分数和通用能力变化。

开源代码可以帮助搭建可验证的 MemT 风格系统；未公开数据和配方仍是复现边界。
“保持精度”需要最终任务评测，不能仅凭 loss 下降或 kernel 微基准宣布。
