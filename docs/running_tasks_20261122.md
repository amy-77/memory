# 正在进行的任务清单(2026-11-22 16:00 记录)

电脑关机 / 换网络后按这张表逐项检查。**服务器上的任务都是脱离终端运行的,关电脑不影响;只有第 1 项依赖这台 Mac。**

## 一、依赖 Mac 的任务(关机会中断,需要手动恢复)

### 1. 把 0.6B LaCT 模型权重上传到 Hugging Face

- 仓库:https://huggingface.co/Yanlin77/lact-qwen3-0.6b-dialog-memory (已公开;代码、README、说明文档已上传)
- **还没上传的**:`checkpoint/memory_weights.pt`(1059M)、`checkpoint/memory_config.json`、`data/rc_t{16,32,64}_f8.jsonl`
- 路线:9-5 → Mac(断点续传)→ Hugging Face。9-5 连不上 huggingface.co。
- 本地目录:`~/lact_memory_baselines/hf_upload/`(已从临时目录搬到这里,重启不会丢)
- 记录时进度:权重已拉 560M / 1059M
- **恢复方法**(两步):
  1. 续传:`cd ~/lact_memory_baselines/hf_upload && nohup bash resume_pull.sh > pull.log 2>&1 &`
     结束标志:`pull.log` 最后一行是 `PULL_OK_MD5_MATCH`(若是 `MD5_MISMATCH` 不要上传)
  2. 上传:`HF_TOKEN=<令牌> python3 upload.py`(令牌不写在任何文件里,需要重新提供)
- 检查:`ls -la repo/checkpoint/memory_weights.pt`(应为 1058938005 字节);`tail -3 pull.log`
- 完成后:到 Hugging Face 的 Settings → Access Tokens 作废今天用的令牌

## 二、h20-1 上的任务(`ssh h20-1-67`,目录 `/data/qyl/sw_20261122`)

### 2. 重训 0.6B 对话模型(因为 9-5 的权重传不过来)

- 记录时:第 1190 步 / 1200 步,即将结束
- 检查:`tail -5 logs/h201_train.log`,出现 `H201_TRAIN_DONE` 即完成;其上方有 16 轮测试集的写入 / 不写入召回(预期约 0.98 / 0.00)
- 权重:`ckpt/dlg16_new/final/`

### 3. 选择性写入 + Replay 的 42 组对比(第 2 项完成后自动启动,8 张卡并行)

- 脚本:`h201_grid.sh`;任务表:`logs/grid/jobs.txt`
- 检查:`cat logs/h201_grid.log`(每完成一组打印一行:事实所在轮的选中率、题数、准确率);`GRID_DONE` 表示全部结束
- 单组日志:`logs/grid/<数据集>_<名字>.log`;结果:`out/grid/*.json`
- 内容:基线(全部写入、注意力选 4/8/16 轮、上界、第 19 层内积);排练的 6 种变体(问题单独前向的探针、按写入强度入库的 key 库、**按问题相关度入库的 key 库(每轮留 32 或 8 个)**、**上下文里的问题探针**、两者合用),各在选 8 轮和选 16 轮下;32 轮和 64 轮对话各一套
- 若中断:重新执行 `(setsid nohup bash h201_grid.sh > logs/h201_grid.log 2>&1 < /dev/null &)`,已有结果的组会跳过

### 4. 藏针测试(已完成,仅留结果)

- 目录:`/data/qyl/needle_20261122/`;结果:`logs/run.log`、`logs/run2.log`、`result_*.json`
- 结论:官方 LiveMem-SFT,事实在窗口内 30/30;在窗口外(总长 2 万、4.5 万 token)0–1/30,且"全部写入"与"记忆为空"无差别

### 5. 4B MemT 训练(已按要求停止)

- 停在第 2289 步;最近检查点 `/mnt/airs-business/memory/memt4b_20261120/step_2160`
- 定时检查已删除。续训:在 `/data/qyl/memt4b_20261120` 的工作目录下用 `run_full.sh`

## 三、h20-9-5 上的任务(`ssh h20-9-5-bw`)

### 6. LiveMem 对照微调(不带选择性写入),`/data0/qyl/livemem_repro_20261121`

- 卡:0、1、3、6;记录时第 93 步 / 120 步,约 17:00 结束
- 检查:`grep -E "step +[0-9]+/" logs/ft_ctrl.log | tail -1`;`logs/phase2.log` 出现 `FT_ctrl rc=0`
- 之后 4 号卡自动导出并评测 `ctrl120`

### 7. LiveMem 检查点评测队列

- 4 号卡(`logs/phase2.log`):第 120 步检查点(全部写入、上界已完成)→ 参照 → BM25 / 注意力前 10 / 前 5 → 第 80 步 → 对照微调
- 5 号卡(`logs/q5b.log`):第 40 步(已完成)→ 官方 SFT 记忆模式(进行中)→ 官方 RL
- 结果:`LiveMem/results/eval_sub/<模型>-<策略>/<配置>/<数据集>/summary.json`(看 `item_acc`)
- 分桶分析:`python sel/bucket.py`(需先 `source env.sh`)
- **状态:藏针测试后已判断 LiveMem 的记忆读不出具体事实,这条线不再投入;这些任务只是跑完留档,可以随时停掉腾卡**

### 8. LaCT 选择 + Replay(旧做法),`/data0/qyl/select_write_20261121`,7 号卡

- `logs/combo.log`(旧探针的 11 组,32 轮已出)、`logs/combo2.log`(新做法的 10 组,排在后面)
- 这批和第 3 项重复,h20-1 上更快;h20-1 出结果后可停

### 9. 裁判模型

- 7 号卡,Qwen3.6-35B,端口 8790;第 7 项结束前不要停

## 四、今天已得到的主要结果

- LaCT(0.6B,合成对话,窗口外准确率):32 轮 全部写入 0.415 / 注意力选 4 轮 0.689 / 第 19 层内积选 4 轮 0.719 / 上界 0.763;64 轮 0.218 / 0.633 / 0.612 / 0.728
- 排练(旧探针,32 轮):选 8 轮 0.696 → 0.719 → 0.726;选 16 轮 0.563 → 0.585。幅度小,在噪声范围内
- 之前"内积选择不可用"是取错了层和 query:只有第 19 层、回答位置的读取向量有定位信号
- LongMemEval 上注意力总量的证据召回(前 10 个会话)0.84,BM25 0.93
- LiveMem:LongMemEval 子集 完整上下文 0.425 / 截断 0.292 / 记忆全部写入 0.283 / 只写证据 0.275;证据全在窗口外的 36 题,记忆模式 2 题、截断 3 题、完整上下文 15 题

## 五、待你决定

- 9-5 上 LiveMem 的任务(第 6、7 项)是否现在停掉
- H200 上的 4B 外挂模型是哪个分支训练的;760M 原生模型的训练代码仓库
- 是否训练一个 32 轮对话的 0.6B 模型,用来区分"减少干扰"和"回到训练范围"两种解释(约 2 小时)

## 六、代码位置

- GitHub:https://github.com/amy-77/memory (提交 55ac12d;今天下午新增的 key 库入库方式 `--bank-score relevance`、`--probe-ctx` 还没提交,在 Hugging Face 仓库的 `code/` 里有)
- Mac 工作副本在临时目录 `<scratchpad>/insight/`,**重启可能被清空**;同样的代码在 h20-1 `/data/qyl/sw_20261122/` 和 9-5 `/data0/qyl/select_write_20261121/` 各有一份
