# memt_cc:4B MemT 风格记忆的训练代码(Claude Code 修改版)

`memt/framework` 的一份拷贝加修改,用来和另一份修改版对比。模型结构和写入公式没有动(已对照 MemT-1.0 报告 2.2 节、附录 B/C 逐项核对,见下),改的是训练循环、数据和配方。

## 相对 `memt/framework` 的改动

| 改动 | 文件 | 原因 |
|---|---|---|
| 损失从最后一层隐状态分块计算,不再生成整段 logits | `lact_qwen/train_loss.py` | 32K × 15 万词表的 float32 logits 约 20 GB;做法同 LaCT 训练用的 fused linear cross entropy |
| 梯度累积时只在最后一个微批次同步梯度 | `scripts/train_memory.py`(`no_sync`) | 原来每个微批次都对全部记忆参数做一次 all-reduce |
| 一条样本只含一篇文档,记忆从零开始,2K 步从文档开头对齐 | `lact_qwen/token_store.py`、`text_data.py: document_sequences`、`--text-packing document` | 报告 3.4、4.1 节:记忆在轨迹边界清零。原来是多篇文档首尾相接再按 32K 切 |
| 预分词数据(`tok:` 数据源) | `scripts/pretokenize.py`、`lact_qwen/token_store.py` | 训练循环里不再分词;每张卡只读自己的窗口;续训按下标直接跳转,不用重放数据流 |
| 序列最后一个 2K 步不再写入 | `lact_qwen/memory_branch.py: _read_matrix` | 写完没有人读;32K 序列省 1/16 的写入计算。逐块推理(session)不受影响 |
| 检查点先写临时目录再改名 | `scripts/train_memory.py: save_checkpoint` | 进程在保存中途被杀时不会留下半个检查点 |
| 余弦下限可配 | `--cosine-floor`(默认 0.1,即原行为) | 原来写死 0.1,而 `--min-lr-ratio` 只对 wsd 生效 |
| 按报告配方的启动脚本 | `train_run/train_cpt_cc.sh` | 见下表 |

不同长度的窗口用固定分母 `grad_accum × (seq_len − 1)` 归一,所以每个 token 的权重相同,与它所在的窗口长短、哪张卡、第几个微批次无关。

## 配方:原训练 vs 报告(Nano)vs 这里

| 项目 | `memt/train_run` | MemT-1.0 报告 | `train_cpt_cc.sh` |
|---|---|---|---|
| 学习率 | 1e-3(q/k/v ×0.1),余弦 | 1e-4,warmup 600 步后恒定 | 同报告 |
| AdamW | β2 0.95,eps 1e-15,wd 0.1 | β2 0.999,eps 1e-8,wd 0.01 | 同报告 |
| 记忆层 | 全部 36 层 | 每 4 层一个 | `every:4`(3, 7, …, 35,共 9 层) |
| 遗忘门 / 输出门的秩 | 128 | 256 | 256 |
| 记忆头 | 32 × 128 | 16 × 256 | 32 × 128(与底座注意力头一一对应,才能直接拷 q/k/v) |
| 每条长度 | 32K | 最长 64K | 32K(`SEQ=65536` 可改) |
| 每步条数 | 32 | 64 | 卡数 × `ACCUM` |
| 文档边界 | 不清零 | 清零 | 清零 |

与报告一致、未改动的部分:写入公式(对称化的 u = √β⊙k)、三个门按通道、k/q 的 L2 归一、读出过 RMSNorm 再乘软上限 2 的输出门、块内读本块之前的状态、初始化(擦除门拷给写入门、遗忘半衰期 2K–32K 对数均匀、输出门扩展 N(0, 0.001²))。

## 用法

```bash
# 1) 预分词(一次)
python framework/scripts/pretokenize.py --data "/path/ldc/train/data-*.arrow" \
    --tokenizer /path/Qwen3-4B --out /path/ldc_tok --min-doc-tokens 4096 --workers 48
# 2) 第一阶段训练;重跑同一条命令即从最新检查点续训
MODEL=/path/Qwen3-4B TOK=/path/ldc_tok W=/path/run GPUS=5,6,7 bash train_run/train_cpt_cc.sh
# 测试
cd framework && python -m pytest -q tests
```

## 参考了什么、没用什么

- **LaCT / flame**:数据预分片、fused 损失、整段反传不截断,都借鉴了;LaCT 的优化器配方(lr 1e-3、eps 1e-15)是原训练用的,这里换成了 MemT 报告的。
- **flash-linear-attention 的 `chunk_gated_delta_rule`**:没有直接用。它的 β 是每头每 token 一个标量、衰减是每头一个标量,而 MemT 的擦除门、遗忘门都按通道;按通道衰减只在内核层有一个未公开的 `gk` 参数。`matrix_memory.py: delta_write` 是同一种分块求解(三角方程),纯 PyTorch。
- **LiveMem**:它的训练是对数据集标准答案算交叉熵,没有教师蒸馏,所以"后训练蒸馏"不能照搬;能借的是它的公开数据清单和切块方式(MuSiQue、2Wiki、NarrativeQA、QASPER、LongAlign、LongAlpaca、LongMIT 等)以及 64K 打包、`no_sync`。
- **MemT 第二阶段**(历史 + 问题 + 教师回答,只用 KL,Top-64):入口沿用原有的 `--data post:`,分块损失已支持 Top-K 蒸馏;**教师回答的数据生成脚本还没有写**,这是和报告差距最大的一块。

## 已知未做

- 第二阶段的数据构造。
- `post:` 数据仍要求历史长度是 2048 的整数倍。
- 多卡下的 `tok/s` 日志按"各卡 token 数相同"估算,文档模式下是近似值。
