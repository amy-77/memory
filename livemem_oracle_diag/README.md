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

### 损失（进行中）

第一项检查已通过（两张卡各一题）：全 1 掩码与不传掩码的 loss 差为 0；全 0 掩码时记忆支路输出为 0；`A_log` 逐位恢复。

`memory_off` 的 loss 远高于其他组（前两题 2.52、11.13，全写是 0.89、1.77）：模型离开记忆支路就不能正常工作，所以它不能当"没有记忆"的下界，要看证据的作用，应该比较写入量相同的 oracle 与随机两组。
