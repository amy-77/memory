# MemT-1.0 论文 benchmark 上的复现评测(2026-11-24)

用 MemT-1.0 技术报告(arXiv 2610.26825)表 2 的 5 个记忆 benchmark,按报告附录 D 的协议,评测:

- 我们训练的 4B MemT 风格记忆模型(冻结 Qwen3-4B + 每层矩阵记忆,`step_2160`,只做了第一阶段训练,历史全部写入);
- 原版 Qwen3-4B(不加记忆)作对照;
- 官方 MemT-1.0-Nano(Qwen3.6-35B-A3B 底座,36B 总参数、每 token 激活 3B),用来检验评测流程本身。

所有 benchmark 的全部题目都跑完并判分。判分模型是本地的 Qwen3.6-35B-A3B(报告用 gpt-4.1-mini),
所以绝对分数只能和报告大致对比;同一列内、同一判分模型下的比较是严格的。

## 结论

1. **评测流程可信。** 官方 Nano 在同一套流程下的分数和报告相差都在 6 分以内(见下表),用 LLM 判分的几项我们都偏低
   3–6 分(本地判分模型略严),只比选项字母的 PersonaMem 高 3.4。清空记忆后 Nano 的 LoCoMo 从 51.7 掉到 2.7,
   说明流程确实在测记忆。
2. **我们的 4B 记忆只有一点作用。** LoCoMo 写入历史比空记忆高 5.3 分,也比写入别的历史高;LongMemEval 和 BEAM
   写入后反而比空记忆低;PersonaMem 的提升在写入无关历史时同样出现(51.4 对 51.8),来自格式而不是内容。
3. **原因在训练,不在代码。** 记忆模块的公式和官方 `modeling_memt.py` 逐条一致(见下文"和官方实现的对比");
   差距在于只做了第一阶段(约 23 亿 token 的续写交叉熵,计划步数的 45%),没有第二阶段(对话格式的记忆任务数据 +
   以读全文的底座为老师的 KL 蒸馏)。报告图 7:第一阶段 400 步(约 16 亿 token)后再做 200 步后训练,Nano 的
   LoCoMo 就到 44。4B 的回答经常很长、不停重复,套 chat 模板直接 0 分,都是缺第二阶段的表现。

## 主要结果

| Benchmark | 4B 写入历史 | 4B 空记忆 | 4B 写入别的历史 | Qwen3-4B 无记忆 + 全注意力 | Nano(我们测) | Nano 清空记忆 | Nano(报告) | Flash(报告) | QW 全文(报告) |
|---|---|---|---|---|---|---|---|---|---|
| LoCoMo(1540 题) | 24.1 | 18.8 | 16.2 | 66.7(完整上下文) | 51.7 | 2.7 | 57.5 | 65.1 | 83.8 |
| LongMemEval-S(500) | 9.2 | 12.2 | 8.2 | 27.2(截到最后 32K) | 29.0 | 7.4 | 32.6 | 37.2 | 76.8 |
| PersonaMem-32K(589) | 51.8 | 22.4 | 51.4 | 48.9(完整上下文) | 55.7 | 44.8 | 52.3 | 61.6 | 65.2 |
| MemoryAgentBench(2071) | 8.5 | 7.0 | 7.1 | 21.6(截到最后 32K) | 12.4 | – | 15.5 | 17.4 | 35.9 |
| BEAM-1M(700) | 17.7 | 19.4 | 15.1 | 18.4(截到最后 32K) | 29.8 | – | 35.8 | 37.0 | 43.7 |

各列的含义(4B 的三列都是纯文本 prompt,见"协议"):

- **4B 写入历史**:清空记忆,按时间顺序把整段历史以 2048 token 为一块写入记忆(写入时注意力只看当前块,和训练一致),
  然后只用问题作答;作答时注意力窗口里只有问题,历史只能从记忆里读。
- **4B 空记忆**:不写任何东西,只用问题作答。这就是"只有局部窗口"的基线:窗口里只有问题,看不到任何历史。
  "写入历史"减去这一列就是记忆带来的增益。
- **4B 写入别的历史**:写入同一 benchmark 里另一段无关的历史再作答,用来区分"读到了内容"和"被格式带着走"。
- **Qwen3-4B 无记忆 + 全注意力**:原版 Qwen3-4B,历史和问题放在同一个 prompt 里,全注意力。Qwen3-4B 原生上下文是
  32K(`max_position_embeddings` 40960,未开 YaRN),所以历史超过 32K 时只保留最后的部分。LoCoMo(最长 24.8K)和
  PersonaMem(最长 30.1K)没有任何截断,是真正的完整上下文;LongMemEval、MemoryAgentBench、BEAM 全部被截断,
  不代表 4B 读完整历史的能力。
- **Nano**:官方 MemT-1.0-Nano,历史作为一条 user 消息套 chat 模板,用 `commit_memory(document=True)` 写入,
  每题用 `generate(memory_state=...)` 作答。"清空记忆"只给问题。
- **报告**列抄自 MemT-1.0 报告表 2(判分模型 gpt-4.1-mini)。

chat 模板下的 4B(报告原样的对话格式):LoCoMo 写入历史 0.0(n=382,输出论文开头一类的文本,跑到 382 题后停止)、
空记忆 21.2;LongMemEval 写入历史 0.0、空记忆 8.0。

## 各 benchmark 的历史长度(Qwen3 tokenizer)

| Benchmark | 历史段数 | 题数 | 中位数 | 最短 ~ 最长 | 32K 窗口内放得下 |
|---|---|---|---|---|---|
| LoCoMo | 10 | 1540 | 22.7K | 13.0K ~ 24.8K | 全部 |
| PersonaMem-32K | 222 | 589 | 25.1K | 15.7K ~ 30.1K | 全部 |
| LongMemEval-S | 500 | 500 | 106.7K | 99.8K ~ 111.2K | 无 |
| MemoryAgentBench | 130 | 2071 | 145.6K | 67.4K ~ 1644K | 无 |
| BEAM-1M | 35 | 700 | 1043K | 839K ~ 1557K | 无 |

## 分题型

完整的分题型分数在 `results/per_category.txt`(每行:benchmark、格式、模式、题数、总分、各题型分数和题数)。要点:

- LoCoMo(写入 / 空记忆 / 无记忆全文 / Nano):单跳 29.8 / 23.9 / 86.6 / 63.6,多跳 24.1 / 16.7 / 61.7 / 58.2,
  时间 6.5 / 3.1 / 26.8 / 16.5,开放域 32.3 / 33.3 / 40.6 / 45.8。
- MemoryAgentBench(写入 / 空记忆 / 别的历史 / 无记忆 32K / Nano):精确检索 AR 16.8 / 11.7 / 14.7 / 8.4 / 34.3,
  冲突消解 CR 6.5 / 7.5 / 10.5 / 18.0 / 10.0,长程理解 LRU 0.0 / 0.0 / 0.0 / 0.2 / 2.9,
  测试时学习 TTL 10.6 / 8.9 / 3.4 / 59.9 / 2.5。
- BEAM(写入 / 空记忆):拒答 27.9 / 10.0 上升,信息抽取 7.0 / 10.9、知识更新 2.9 / 7.1、摘要 7.9 / 15.2 下降;
  记忆主要让模型更常说"没提到",而不是找到细节。
- LongMemEval(写入 / 空记忆):单会话-用户 2.9 / 15.7,知识更新 11.5 / 12.8,只有时间推理 15.8 / 14.3 略高。

**判分模型不是低分的原因。** 不用任何判分模型、只算答案和标准答案的词级 F1(LoCoMo):4B 写入 7.2、空记忆 3.1、
写入别的历史 4.8、Qwen3-4B 全文 29.7、Nano 26.7,排序和判分结果一致。4B 写入历史的答案中位数 40 词,36% 超过 100 词
(常见 "(description of attached image: ...)" 循环)。PersonaMem 只比选项字母,MemoryAgentBench 多数题型是规则打分,
4B 在这两个上同样很低。

## 协议与判分

| 项目 | 设置 |
|---|---|
| 子集 | 报告附录 D.1:LoCoMo 去掉第 5 类(1540 题)、LongMemEval-S cleaned、PersonaMem 32K、MemoryAgentBench 主集(去掉 eventqa 65K/131K 和 factconsolidation 6K/32K/64K 消融源,130 段 2071 题)、BEAM 1M |
| prompt | 报告附录 D.3.1 的全文模板和记忆版改写(例如 LoCoMo "Answer the question using only the conversation stored in your memory.") |
| 4B 的格式 | 纯文本(不套 chat 模板):4B 只训练过原始长文本,套 chat 模板后写入记忆会输出论文开头一类的文本 |
| 解码 | 报告 D.3.2:贪心,关闭 thinking,最大新 token 数 LoCoMo/LongMemEval 512、PersonaMem 2048,BEAM 和 MAB 按任务;纯文本答案在 "\n"、"Answer:"、"Question:" 处截断(判分前用同一规则重新截断) |
| 4B 作答 | 当前 2048 token 块内用 KV 缓存解码,块满了先写入记忆再开新块;已验证和逐 token 重算结果一致(只有 bf16 打平时的个别差异) |
| LoCoMo、LongMemEval 判分 | OmniMemEval `JUDGE_PROMPT`(报告所用),本地 Qwen3.6-35B-A3B(vLLM 0.31) |
| BEAM 判分 | OmniMemEval:每个评分要点 0 / 0.5 / 1 取平均;事件排序用 Kendall tau-b × 覆盖率 |
| MemoryAgentBench | 官方指标:SubEM、EM、Recall@5;longmemeval_s* 和 infbench_sum 用 LLM 判分;按 AR / CR / LRU / TTL 四类平均 |
| PersonaMem | 选项字母匹配 |

## 和官方实现的对比(为什么 4B 低)

记忆模块(`../framework/lact_qwen/matrix_memory.py`、`memory_branch.py`)和官方 `modeling_memt.py` 的 `MemTMemory`
对照:

| 环节 | 官方 MemT | 我们的 4B |
|---|---|---|
| 写入 | S ← D·S + u(ω·v − uᵀD·S),u = √sigmoid(擦除) × 归一化 k,ω = sigmoid(写入) | 同 |
| 遗忘 | 低秩 softplus,初始半衰期 2K–32K(对数均匀) | 同;另加每 token 衰减上限 0.03 |
| 读取 | 归一化 q · S / √d,RMSNorm × 软上限 2.0 的 silu 门 | 同 |
| 注入位置 | attention 输出上、o_proj 之前 | 同(`pre_o_proj`) |
| 读写顺序 | 2K 一块,先读块前状态,块结束后写入 | 同 |
| 层 | 40 层中的 10 个全注意力层,16 头 × 256 | Qwen3-4B 全部 36 层,32 头 × 128 |
| 可训练参数 | 4.51 亿 | 19.5 亿 |
| q/k/v 初始化 | 小随机(xavier,gain 2^-2.5) | 从 attention 的 q/k/v 拷贝 |

| 训练 | MemT-1.0-Nano | 我们的 4B |
|---|---|---|
| 第一阶段 | 200 亿 token 通用语料,交叉熵 | LDC 长文本,交叉熵,2160 步 × 约 100 万 token ≈ 23 亿 token(计划 4768 步) |
| 第二阶段 | 140 亿 token 记忆任务数据(context-query-answer,对话格式),top-64 KL 蒸馏,老师是读全文的底座 | 无 |
| 学习率 | 1e-4(WSD) | 1e-3(余弦) |

## 待办

- 原版 Qwen3-4B 开 YaRN 到 128K,重跑 LongMemEval(可完整放下)和 MemoryAgentBench(部分放下),得到真正的
  "完整上下文 + 全注意力"基线;BEAM 的 100 万 token 仍放不下。
- 在 4B 上补一个短的第二阶段(KL 蒸馏,框架里已有 `distill.py`、`post_data.py`),只测 LoCoMo:
  明显上升到 40 左右说明代码没问题、只是训练没复现完;仍在 25 左右再查代码或容量。
- 用 gpt-4.1-mini 重判全部答案,和报告严格对齐(答案文件都保留着)。

## 代码与复现

| 文件 | 作用 |
|---|---|
| `code/setup_data.sh` | 下载 Qwen3-4B、4B 记忆检查点(HF `Yanlin77/memt-style-qwen3-4b-step2160`)、各 benchmark 数据;`--nano` 另下 Nano 和判分模型 |
| `code/adapters.py`、`code/adapters_mab.py` | 各 benchmark 的读取和 prompt(全文版和记忆版) |
| `code/bench_memt.py` | 4B 记忆模型和原版 Qwen3-4B 的作答(模式 memory / question / wrong / full,格式 plain / chat) |
| `code/bench_nano_hf.py` | MemT-1.0-Nano 的作答(HF transformers 5.20 + 检查点自带的 `modeling_memt.py`,含两处兼容补丁) |
| `code/judge_bench.py`、`code/score_mab.py`、`code/mab_official.py` | 判分和打分 |
| `code/report.py`、`code/run_judge.sh` | 汇总成 `scores.json` 和对照报告表 |
| `code/worker.sh`、`code/nano_worker.sh`、`code/nano_shard.sh` | 多卡任务队列 |
| `code/test_cache.py`、`code/diag_protocol.py`、`code/check_mab.py`、`code/nano_smoke.py` | KV 缓存一致性检查、prompt 格式诊断、MAB 打分检查、Nano 冒烟测试 |
| `results/` | 最终 `scores.json`、`report.md`、`per_category.txt` |

```bash
W=/path/to/work_dir            # models/、data/、out/ 都放在这里
bash code/setup_data.sh $W     # 另外手动放 data/locomo/locomo10.json
export MEMT_FRAMEWORK=$(pwd)/../framework
MEM="--model $W/models/Qwen3-4B --attn-impl sdpa --memory $W/models/memt4b_step2160/checkpoint --window 2048 --sinks 0 --attn-segment 2048"
BASE="--model $W/models/Qwen3-4B --attn-impl sdpa --memory none --window 0"
cd code
python bench_memt.py --bench locomo --data $W/data --mode memory --format plain $MEM --out $W/out/locomo/plain_memory.s0of1.jsonl
python bench_memt.py --bench locomo --data $W/data --mode full   --format plain $BASE --out $W/out/locomo/plain_full.s0of1.jsonl
python bench_nano_hf.py --bench locomo --data $W/data --model $W/models/MemT-1.0-Nano --mode memory --out $W/out_nano/locomo/chat_memory.s0of1.jsonl
# Nano 的答案拷成 $W/out/locomo/nano_memory.s0of1.jsonl 后一起判分:
PY=/path/to/vllm-0.31/python JUDGE_GPUS=0,1 bash run_judge.sh $W $W/out
```

`--bench` 可选 `locomo`、`longmemeval_s`、`personamem_32k`、`mab`、`beam_1m`;`--shard i --nshards n` 按历史分片,
结果文件可断点续跑。所用记忆框架即本仓库 `memt/framework`(提交 `6f595b5`)。第三方代码见 `THIRD_PARTY_NOTICES.md`。
