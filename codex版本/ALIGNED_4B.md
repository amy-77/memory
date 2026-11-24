# 对齐 MemT-1.0 的 4B 重训(H200)

在 Codex 版本(训练脚本的修复见 `TRAINING_FIXES.md`)的基础上,把记忆的写入和读出对齐到 MemT-1.0。
原 `train_8gpu.sh` 不变;新的启动脚本是 `train_run/train_aligned.sh`。

## 和原 4B 训练相比改了什么

记忆结构不变:36 层、32 头 × 128、门控 delta 规则、注入在 `o_proj` 之前。对齐只改下面五处,都是 `MemoryConfig` 字段,
对应启动脚本里的 `--mem-*` 参数。默认值保持旧行为,所以旧检查点照常加载。

| 字段(参数) | 旧 4B | 对齐后 | 理由 |
|---|---|---|---|
| `decay_cap`(`--mem-decay-cap none`) | 每 token 衰减最多 0.03 | 无上限 | MemT 没有上限;旧上限只是为了不让下面的 1/P 写法溢出 |
| `write_impl`(`--mem-write-impl auto`) | `scaled` | `auto` | 同一个逐 token 递推,换成数值稳定的分块算法(MemT 报告第 4 节 "chunked WY")。衰减小的块仍走快的旧公式,大的块走稳定公式;没有截断 |
| `out_norm_eps`(`--mem-out-norm-eps 1e-6`) | 1e-5 | 1e-6 | 与 MemT(backbone 的 rms_norm_eps)一致 |
| `qk_norm`(`--mem-qk-norm memt`) | x/(‖x‖+1e-5) | x·rsqrt(Σx²+1e-6) | MemT 的归一化 |
| `memt_rounding`(`--mem-memt-rounding`) | 门控全 FP32 | 投影相关量舍入到 BF16 | 与 MemT 一致:投影计算用 BF16;状态、衰减、归一化用 FP32 |

训练配方与 `train_8gpu.sh` 相同:
- 数据:LDC(Clean-Long-Data-Collections,LaCT 用的数据集),每条 32K,每篇文档独立(不再跨文档拼接)。
- 批量:每步 32 条。
- 学习率:1e-3,warmup 256,余弦衰减到 0.05 倍。
- 损失:只有交叉熵。
- 步数:4768。

## 启动

环境:`pip install -r codex版本/requirements.txt`(先装好与 CUDA 匹配的 PyTorch;另外需要 `flex_attention`,即 torch ≥ 2.5)。

单机 8 卡(梯度累积 4):

```bash
export MODEL_PATH=/path/to/Qwen3-4B
export DATA_SPEC='/path/to/Clean-Long-Data-Collections/train/data-*.arrow'
export RUN_DIR=/path/to/memt4b_aligned        # 新目录
bash codex版本/train_run/train_aligned.sh
```

两机 16 卡(梯度累积 2,每步仍是 32 条)。`RUN_DIR` 必须放在两台机器都能访问的共享存储上,两台机器同时启动:

```bash
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=<节点0地址> MASTER_PORT=29577 bash codex版本/train_run/train_aligned.sh
# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=<节点0地址> MASTER_PORT=29577 bash codex版本/train_run/train_aligned.sh
```

其他环境变量:
- `STEPS`(默认 4768)、`SAVE_EVERY`(默认 240)、`GLOBAL_BATCH`(默认 32)。
- `MEMT_ALIGNED=0`:换回旧的记忆设置,用于 A/B 对比。
- 中断后重新运行同一条命令即可续训:会自动找最新的完整检查点。
- 先用 `--stop-at 10` 做小规模检查,再正式开跑。

## 检查要点

- `train.log`(多机时为 `train.node<N>.log`)中 loss 在前几十步应明显下降,不能出现 nan。
- `progress.json` 中的 `processed_tokens`:文档独立后有 padding,每步实际 token 数少于 100 万。
- 对齐配置下没有衰减上限。若训练中某些通道学到很大的衰减,`auto` 会在这些块改用稳定公式,步时会略增,结果不受影响。

## 已做的验证(H20,2026-11-25)

CPU 测试:84 项通过,4 项跳过(需要 CUDA/KDA)。其中 `framework/tests/test_memt_aligned_write.py` 共 38 项,覆盖:
- 分块写入与逐 token 定义一致(FP64,衰减 0.001–20)。
- 快公式在阈值边界上的 FP32 精度。
- gradcheck。
- 预先计算的 `auto` 计划。
- 命令行参数,以及检查点的保存与加载。

真实训练对比的设置:单卡,每步 1 条 32K 序列,warmup 10 步(学习率很快升到 1e-3)。初始权重和数据流完全相同。

| 写入 | 速度(tok/s,中位数) | 显存峰值 | 第 1–40 步平均 loss |
|---|---|---|---|
| `scaled` + 上限 0.03 | 1476–1496 | 55.3 GB | 1.4491 |
| `stable`(无上限) | 423 | 52.7 GB | 1.4513 |
| `auto`(无上限,正式用) | 1448 | 54.8 GB | 1.4492 |

**精度:**
- `stable` 与 `scaled` 只差写入算法。逐步 loss 差值:第 1 步 1e-4,中位数 8e-4,符号正负交替,没有累积。
  - 唯一较大的差值出现在第 21 步(+0.086)。这一步的 loss 冲到 4.0,是一条异常难的序列;之后又回到 4e-3 以内。
  - 作为对照,loss 本身在步与步之间的标准差是 0.61。
  - 结论:分块计算只带来 BF16 舍入量级的差异,不影响训练。
- `auto` 前 19 步与 `scaled` 逐位相同(此时它全部走快公式)。之后只有不超过 0.003 的差异,推测是有些块的衰减超过了阈值,改走稳定公式;对应地,`scaled` 那边这时开始被 0.03 的上限截断。

**速度:**
- `stable` 每层的写入要计算三遍(前向、层级 checkpoint 重算、组级重算),整步慢 3.5 倍。
- `auto` 每层每条序列只做一次 GPU→CPU 判断。最初实现是每写一个 2K 段判断一次,慢了 17%,已修复。现在比 `scaled` 只慢约 2%。
