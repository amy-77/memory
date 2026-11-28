# LiveMem oracle 写入诊断（公开 SFT 权重，不训练）

2026-11-28。在 h20-1 上用公开的 `LiveMem-SFT` 权重，在 LongMemEval-s 上测"只把证据写进记忆"能不能降低答案的 loss，并测证据写进去以后到读答案时还剩多少。

## 改了哪些代码

**LiveMem 源码一行没改。** 脚本启动时检查 `models/memory_gdn2.py` 和 `models/modeling_memory_qwen3.py` 与官方提交 71a6a8b 完全一致，不一致就退出。

| 文件 | 来源 | 作用 |
|---|---|---|
| `h200_experiment1.py` | H200 实验的 `oracle/code/experiment1.py` 原样拷贝，SHA-256 `fbdb6787…0571`，与 H200 原件一致 | 只用它的 `prepare`：选题、拼接、分词、淘汰时序、答案标签，以及 `make_mask` |
| `oracle_diag.py` | 新写 | 补充证据轮次区间、跑各组前向、测衰减、汇总 |
| `run_h20_1.sh` | 新写 | h20-1 上的路径、环境和两张卡的启动方式 |

两种干预都不需要改模型代码：

- **只写一部分 token**：用官方 forward 自带的 `write_mask` 参数。官方实现把遗忘门 g、擦除门 b、写入门 w 同乘这个掩码（`models/memory_gdn2.py:128`），掩码为 0 的 token 不写、不擦、也不衰减，状态保持不变。
- **去掉衰减**：前向前把 36 层的 `A_log` 临时设为 -30，前向后恢复（脚本会检查恢复后和原值逐位相同）。每个 token 的对数衰减是 `-exp(A_log)·softplus(f_proj(h)+dt_bias)`，exp(-30)≈1e-13，衰减近似为 0。

H200 实验对模型代码的唯一改动是 `memory_gdn2.py:130` 加的 `skip_keeps_decay` 开关（被跳过的 token 仍然衰减），只用于它的第 5 组。这里没有用这一组，所以不需要这个改动。

## 各组

所有组都用同一套题：证据会话在提问时已全部被挤出 32K 窗口，去掉弃权题。窗口 32K，块长 1,024，system 作为不淘汰的首块。指标是一次 teacher forcing 前向里金标准答案 token 的平均交叉熵（nats），不生成、不用评判模型。

| 组 | 写入掩码 | 衰减 | 注意力 | 回答什么 |
|---|---|---|---|---|
| `write_all` | 不传（官方默认，全写） | 有 | 32K 窗口 | 基线 |
| `write_nothing` | 历史为 0，system 和问答为 1（H200 同名组） | 有 | 32K 窗口 | 与 H200 对齐 |
| `session_oracle` | 证据会话为 1，system 和问答为 1（H200 的 oracle） | 有 | 32K 窗口 | 与 H200 对齐 |
| `memory_off` | 全 0：状态始终为 0，记忆支路输出为 0 | 无关 | 32K 窗口 | 没有记忆时的 loss |
| `turn_oracle` | 只有 `has_answer` 的轮次为 1，问答只读 | 有 | 32K 窗口 | 写入量压到容量附近时有没有收益 |
| `turn_oracle_nodecay` | 同上 | 去掉 | 32K 窗口 | 衰减是不是瓶颈 |
| `turn_random_nodecay` | 同样 token 数的随机非证据轮次 | 去掉 | 32K 窗口 | 上一组的对照：收益来自"选对了"还是"写得少" |
| `full_attention` | 不传（全写） | 有 | 不淘汰，看得到全部历史 | 证据可见时的上界 |

`decay` 子命令在 `write_all`、`session_oracle`、`turn_oracle` 三种掩码下，用真实 hidden state 计算每层每个 key 通道的 exp(Σg)：证据 token 写入后，到读第一个答案 token 时还剩多少。只算遗忘门，不算擦除门对旧内容的覆盖，所以是保留比例的上限。

## 运行

```sh
bash run_h20_1.sh prepare      # H200 prepare + 证据轮次区间
bash run_h20_1.sh pilot        # 每张卡 1 题，跑三项检查并量时间
bash run_h20_1.sh run          # 全部题目，两张卡，已有结果自动跳过
bash run_h20_1.sh decay        # 前 24 题的衰减测量
bash run_h20_1.sh summarize    # summary.json
```

第一题上的三项检查（`logs/validation_*.json`）：全 1 掩码和不传掩码的 loss 相差不超过 1e-5；全 0 掩码时 36 层记忆支路输出的最大绝对值为 0；去衰减后 `A_log` 逐位恢复。

## 结果

### 题目（`data/preparation.json`、`data/turns_info.json`）

500 题中 212 题入选，与 H200 相同：258 题因证据没有全部被挤出窗口被排除，30 题是弃权题。212 题都有 `has_answer` 轮次。写入的 token 数中位数：证据轮次 122.5（p90 251，最多 811），证据会话 5,008。

### 衰减（16 题，`decay/`）

真实 hidden state 下，各层遗忘门的半衰期中位数为 6–12 个 token，和用初始化参数估计的 9.9 基本一致：遗忘门几乎不随输入变化。

| 掩码 | 证据 token（中位数） | 最后一个证据 token 保留 >1% 的通道 | 全部证据 token 平均保留 >1% 的比例 |
|---|---|---|---|
| `write_all` | 4,877 | 36 层都是 0%；保留最多的通道也只剩 1e-8（第 35 层） | 0% |
| `session_oracle` | 4,877 | 41–50%（问题 token 也写入，会衰减证据） | 2–3% |
| `turn_oracle` | 141 | 100%（问答只读，最后一个证据之后不再衰减） | 48–56% |

全写时，证据到答案之间隔着中位数 5.8 万个 token，证据在记忆里一点都不剩。

### 损失

三项检查都已通过（两张卡各一题）：全 1 掩码与不传掩码的 loss 差为 0；全 0 掩码时记忆支路输出为 0；`A_log` 逐位恢复。

#### 第一批：5 组 × 212 题

金标准答案 token 的平均交叉熵（nats）：

| 组 | 平均 | 中位数 |
|---|---|---|
| `write_all` | 2.454 | 2.257 |
| `turn_oracle` | 3.088 | 2.616 |
| `turn_random` | 3.832 | 2.873 |
| `turn_oracle_nodecay` | 2.909 | 2.592 |
| `turn_random_nodecay` | 3.124 | 2.859 |

逐题配对差（`summary.json` 的 `pass1_pairs`；前者减后者，负数表示前者 loss 更低；95% 区间为 10,000 次 bootstrap）：

| 比较 | 平均差 | 95% 区间 | 中位数差 | 前者更低的题 |
|---|---|---|---|---|
| `turn_oracle` − `turn_random` | −0.744 | [−1.087, −0.461] | −0.181 | 149/212 |
| `turn_oracle_nodecay` − `turn_random_nodecay` | −0.215 | [−0.346, −0.082] | −0.120 | 134/212 |
| `turn_oracle_nodecay` − `turn_oracle` | −0.179 | [−0.333, −0.038] | −0.021 | 111/212 |
| `turn_random_nodecay` − `turn_random` | −0.708 | [−1.074, −0.404] | −0.102 | 122/212 |
| `turn_oracle` − `write_all` | +0.634 | [+0.408, +0.898] | +0.075 | 87/212 |
| `turn_random` − `write_all` | +1.379 | [+0.999, +1.814] | +0.319 | 60/212 |
| `turn_oracle_nodecay` − `write_all` | +0.456 | [+0.302, +0.626] | +0.196 | 75/212 |

按题型（平均 loss；最后两列是 oracle − 随机，有衰减 / 去衰减）：

| 题型 | 题数 | `write_all` | `turn_oracle` | `turn_random` | oracle − 随机 | 去衰减后 oracle − 随机 |
|---|---|---|---|---|---|---|
| knowledge-update | 23 | 2.168 | 3.099 | 4.217 | −1.118 | −0.048 |
| multi-session | 40 | 2.098 | 2.829 | 3.934 | −1.105 | −0.103 |
| single-session-assistant | 34 | 2.934 | 2.924 | 3.020 | −0.096 | +0.181 |
| single-session-preference | 17 | 2.832 | 2.698 | 2.849 | −0.151 | −0.137 |
| single-session-user | 38 | 2.251 | 4.263 | 4.920 | −0.657 | −0.396 |
| temporal-reasoning | 60 | 2.549 | 2.715 | 3.667 | −0.952 | −0.484 |

读法：

1. **写入量相同时，写证据比写随机轮次好**：平均低 0.744，212 题里 149 题更低。平均差被少数随机组 loss 很高的题拉大，中位数差只有 −0.181。所以记忆支路确实能把证据带到 5.8 万 token 之后，前提是证据之后不再写别的内容。
2. **只写证据比全写差**：平均高 0.634。这两组不只差在写了什么：`turn_*` 各组问答只读，读答案时状态里没有问题本身；全写时状态主要是答案前最后约 10 个 token，也就是问题。上面说过 SFT 的收益主要是适配作用，这种作用看来要求状态里有问题。第二批的 `write_nothing`（只写问答）可以检验这一点。
3. **去掉衰减没有让模型更好地用上证据**：oracle 组只改善 0.179，随机组反而改善 0.708，oracle 相对随机的优势从 0.744 缩小到 0.215。去衰减后，所有写入的证据 token 都叠加在状态里，但模型训练时状态里只有最近约 10 个 token（遗忘门在训练中没有更新，见下），读出部分没有学过从这样的状态里取信息。因此对这个模型，单独去掉衰减不能当作"记忆容量足够时"的上界。

遗忘门没学的原因：`train/sft/train_sft.py` 把全部参数（包括 fla 原本保留为 fp32 的 `A_log`、`dt_bias`）转成 bf16，AdamW 没有 fp32 主权重；学习率 5e-5 小于参数附近 bf16 间隔的一半，更新被舍入掉。h20-1 复现的 400 步里 `dt_bias` 变化正好为 0。检查脚本：`check_bf16_frozen.py`。

#### 第二批：`write_nothing`、`session_oracle`、`memory_off`、`full_attention`（进行中）

公开 SFT 权重的 398 个主干张量（注意力、FFN、归一化、词嵌入）与原版 Qwen3-4B-Instruct-2507 逐位相同，所以 `memory_off` 就是原版模型只看 system 加最近 32K。

| 题目 | 答案 | `memory_off`（原版） | `write_nothing`（只写问答） | `write_all` |
|---|---|---|---|---|
| 031748ae | 25 个 token 的句子 | 2.52 | 0.98 | 0.89 |
| 3a704032 | "3"（1 个 token） | 11.13 | 1.65 | 1.77 |

全写时证据在记忆里已经不剩（见上节），只写问答又和全写差不多，所以 SFT 相对原版的大幅降低主要不是来自记住了窗口外的内容，而是记忆支路作为训练过的适配器改变了冻结主干的作答方式（格式、直接给出答案）。这个作用要求状态不为 0，所以 `memory_off` 同时去掉了记忆内容和适配，不能当"没有记忆"的对照。衡量记忆内容的价值，应比较 `write_all` 与 `write_nothing`，以及写入量相同的 oracle 与随机两组。
