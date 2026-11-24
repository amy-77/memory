# 4B MemT 风格记忆:训练代码与 LoCoMo 评测

冻结的 Qwen3-4B 加每层一个矩阵记忆(门控 delta 规则,按 MemT-1.0 技术报告的公式实现)。
这里的代码就是 h20-1 上实际训练用的那一份,没有改动。

## 目录

| 路径 | 内容 |
|---|---|
| `framework/` | 训练和推理代码,来自 Overleaf 归档 `memory_research_20261119/frameworks/memt`(提交 `6f595b5`,见 `SOURCE_COMMIT.txt`) |
| `framework/lact_qwen/matrix_memory.py` | 矩阵记忆的写入公式(`delta_write`,按块求解;`delta_write_reference` 是逐 token 的定义,测试用) |
| `framework/lact_qwen/memory_branch.py` | 记忆分支:读(`_read_matrix`,先读本块之前的状态,再写入本块)、写(`_matrix_write`,三个门)、遗忘门初始化 |
| `framework/lact_qwen/modeling.py` | 把记忆接到冻结的 Qwen3 上;注意力只看当前块(`attention_segment`) |
| `framework/lact_qwen/streaming.py` | 逐块推理接口(`begin_stream` / `forward_chunk` / `end_stream`),LoCoMo 评测用它 |
| `framework/scripts/train_memory.py` | 训练入口 |
| `framework/tests/` | 单元测试(包括和 MemT 公式对齐的 `test_memt_alignment.py`) |
| `train_run/memt.json` | 归档里的 MemT 配置(0.6B 用 16 头;4B 启动时改成 32 头) |
| `train_run/train_8gpu.sh` | 第一段训练(初始化记忆并训到第 480 步) |
| `train_run/run_full.sh` | 后续分段续训的守护脚本(每段之间评测新检查点) |
| `train_run/eval_ckpts.sh`、`mover.sh` | 检查点评测、把检查点搬到对象存储 |
| `train_run/step_2160_memory_config.json` | LoCoMo 评测所用检查点的完整配置(含全部训练参数) |
| `train_run/heldout_eval_summary.txt` | 各检查点在留出长文档上的损失 |
| `locomo_eval/` | LoCoMo 评测脚本(MemT 和 LiveMem),见下文 |
| `benchmark_eval/` | MemT-1.0 报告表 2 的 5 个记忆 benchmark 上的复现评测(4B、原版 Qwen3-4B、官方 MemT-1.0-Nano;结果、协议、代码),见 `benchmark_eval/README.md` |

## 训练配方

| 项目 | 设置 |
|---|---|
| 底座 | Qwen3-4B,全部冻结 |
| 记忆 | 36 层每层一个;32 头 × 128 维的矩阵状态;注入在 `o_proj` 之前;q/k/v 由注意力权重初始化;输出门(低秩 128,软上限 2.0) |
| 写入公式 | S ← D·S + u(ω⊙v − uᵀD·S)ᵀ,u = √β·k;β 擦除门、ω 写入门、D = diag(e^λ) 遗忘门,都由当前 token 算出 |
| 遗忘门 | 初始化为每个通道的半衰期在 2K–32K token 之间(对数均匀);之后可学,每个 token 的衰减上限 0.03 |
| 注意力 | 只看当前 2048 token 的块,无 sink;更早的内容只能通过记忆读到 |
| 写入粒度 | 每块先读后写:块内 token 读的是本块之前的状态;块内按 256 token 一段求解 |
| 数据 | LDC(Long-Data-Collections)通用长文本 12 个分片;每条 32K token;只用不短于 16384 token 的文档 |
| 损失 | 只有下一个 token 的交叉熵(`--ce-weight 1 --distill-weight 0`),没有蒸馏 |
| 优化 | AdamW,lr 1e-3(q/k/v 投影 ×0.1),warmup 256 步,余弦衰减到 0.05 倍,weight decay 0.1,β (0.9, 0.95),eps 1e-15,梯度裁剪 1 |
| 批量 | 8 卡 × 梯度累积 4 × 1 条 × 32K = 每步 32 条约 100 万 token |
| 步数 | 计划 4768 步;第 2289 步时按要求停止;LoCoMo 用 `step_2160`(约 23 亿 token) |
| 可训练参数 | 约 19.5 亿(只有记忆分支) |

和官方 MemT-1.0 的差别:官方第一阶段 200 亿 token 交叉熵,第二阶段 140 亿 token 记忆任务数据加完整上下文教师的蒸馏;这里只做了第一阶段的约 1/9,第二阶段没做;实现是按报告公式自己写的,不是官方代码。

启动命令见 `train_run/train_8gpu.sh` 最后的 `torch.distributed.run` 一行;续训用 `run_full.sh`(`--resume <检查点> --exact-text-resume`)。

### 留出长文档上的损失(48 篇 32K 文档,每 token 交叉熵,越低越好)

`heldout_eval_summary.txt` 里四种方式:write = 正常写入记忆;nowrite = 不写(只有当前块的注意力);reset = 每块清空记忆;wrong = 用另一篇文档的记忆。例如第 1440 步:write 1.02,nowrite 1.39,wrong 1.32。

## LoCoMo 评测(`locomo_eval/`)

| 文件 | 内容 |
|---|---|
| `eval_locomo_memt.py` | MemT:历史从末尾往前按 2048 切块,注意力只看最后一块;按模式决定哪些块写入记忆;生成答案并给出标准答案的对数概率 |
| `eval_locomo_livemem.py` | 官方 LiveMem-RL,用同样的切块和模式(PyTorch 实现,不用它的 vLLM 服务) |
| `judge.py` | 用 Qwen3.6-35B-A3B 判生成答案对错,另算 LoCoMo 词重叠 F1 |
| `subset_acc.py` | 各行答对题数,以及两两"只有一方答对"的题数 |
| `run.sh`、`finish3.sh`、`lm_run.sh` | h20-1 上的启动脚本 |
| `summary_memt.py` | 合并分片、对数概率汇总 |
| `diag_lm.py` | LiveMem 记忆在一个 2048 块之后还剩多少的诊断 |
| `fstats.py` | F 行每题写了几块 |
| `pyshim/sitecustomize.py` | 让 h20-1 上的 vLLM 0.20 跳过残缺的 flash_attn 包(裁判用) |

各行含义:A 完整注意力;B 只看最后 2048 token、不用记忆;C 全部块写入;W 写入另一段对话;F 只写和证据会话重叠的块(用了标注);G 随机写和 F 同样多的块;S 按问题对历史的注意力(全层全头全问题 token)选同样多的块。

MemT `step_2160` 在证据位于窗口外的 1298 题上的答对数(最终一次裁判):A 999,B 337,W 387,C 447,G 452,S 510,F 608。S、G、F 每题写入的块数相同(按证据块数定,所以 S 的预算也用到了标注;固定预算的版本见 docs/experiment_manual 的后续实验)。第一次裁判的数字(C 442、G 445、F 613 等)和这里差几题,来自裁判模型批处理的不确定性,模型的答案完全相同。LiveMem-RL 在同样设置下:A 554、B 115(这两行把记忆输出置零,模型脱离训练状态,不是公平基线),C 423、F 419、G 414。

数据 `data/locomo2_g4.jsonl` 由 `livemem_patch/locomo_convert.py` 从 LiveMem 处理过的 LoCoMo 生成(去掉第 5 类对抗题,1536 题)。

## 服务器上的位置(h20-1)

- 训练目录:`/data/qyl/memt4b_20261120/`;检查点:`/mnt/airs-business/memory/memt4b_20261120/step_*`
- LoCoMo 评测目录:`/data/qyl/memt_locomo_20261122/`
