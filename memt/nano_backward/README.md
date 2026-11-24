# MemT-1.0-Nano 记忆分支的手写 backward

官方 `MemTensor/MemT-1.0-Nano` 的 `modeling_memt.py`(Apache-2.0)只支持推理。这里给记忆分支补上训练所需的 backward。
来源和许可证见 `THIRD_PARTY_NOTICES.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `memt_nano/modeling_memt_orig.py` | 官方文件,未改动(测试的参照) |
| `memt_nano/modeling_memt.py` | 官方文件加下面的改动 |
| `memt_nano/memt_autograd.py` | 三处算子:手写的 forward / backward |
| `memt_nano/compat.py` | 让官方文件(针对 transformers 5.12.1)在 5.20 上运行;GPU 上 backbone 的 GDN 层用 FLA 的 kernel |
| `memt_nano/configuration_memt.py` | 官方文件,未改动 |
| `tests/test_backward.py` | 精度检查:与"对官方 forward 直接用 autograd 求导"逐项对比 |
| `tests/bench_memory_layer.py` | 一个记忆层的前向、反向耗时和显存(只依赖 torch,不需要 transformers) |
| `tests/bench_block_backward.py` | 只测写入算子:手写反向与 autograd、逐 token 循环的速度和精度 |

## 实现了什么

| 任务 | 官方代码位置 | 实现 |
|---|---|---|
| 1. 带软上限的门控 RMSNorm 的 backward | 44–52 行(`MemTRMSNormGated`,上限在 51 行) | `GatedRMSNormFn`、`gated_rmsnorm_backward`:RMS 归一化、SiLU、tanh 上限的导数全部手写 |
| 2. full attention 中记忆部分的 backward | 257–266 行(`MemTAttention.forward`) | `MemoryResidualFn`:"注意力输出 + 记忆读出"和"更新后的状态"合成一个节点;读/写状态的梯度传回它们原来所在的设备 |
| 3. 记忆的 backward | 92–132 行(`MemTMemory.forward`) | `MemTMemoryFn`、`memory_forward`、`memory_backward`:读、写两条路径的链式部分都手写;写入递推见下 |

softmax 注意力本身没有改:它走 PyTorch 内置的实现,前向和反向都是现成的。

### 写入算子

写入递推(每个 batch、每个头,S 为 K×V;λ = log d ≤ 0 是按通道的 log 衰减):

    S_t = D_t S_{t-1} + u_t (w_t − u_tᵀ D_t S_{t-1})ᵀ,   D_t = diag(exp λ_t)

训练时有两种算法,算的是同一个递推,由 `MemTMemory.writer` 选择:

- **`"block"`(默认)**:分块 WY 形式,对应 MemT 报告 4.2 节。每 `MemTMemory.write_block`(= 256)个 token 一块,块内 G = cumsum(λ):

        L[t,s] = Σ_k u_tk u_sk exp(G_tk − G_sk)  (s < t)
        (I + L) E = Z − (exp(G) ⊙ U) S_0                    一次三角求解
        S_C = exp(G_C) ⊙ S_0 + (exp(G_C − G) ⊙ U)ᵀ E

  - **L 怎么算(`wy_lower`)**:把块对半分,右半的 t 和左半的 s 之间取参考点 r(左半最后一个 token),
    exp(G_t − G_s) = exp(G_t − G_r) · exp(G_r − G_s),两个因子都不超过 1。于是这一层的 L 子块是一次普通矩阵乘;
    左右两半再各自对半分,共 log2(256) = 8 层,每层对所有块一次批量矩阵乘。
    衰减多大都不会溢出,不需要衰减上限,也没有"衰减小才能走快公式"的判断。
    (参考点的想法来自 FLA 的 KDA kernel;那里是固定切成 4 个 16-token 小块,这里改成逐层对半分。)
  - **只有"状态沿块往后传"这一步是顺序的**(2K 的 chunk 是 8 步)。L、exp 因子等都对所有块一起算。
  - **反向(`block_write_backward`)全部手写,不调用 autograd**。前向每块存一份块起始状态(FP32);
    反向从存档重算 L 和 E(报告里的 "reconstructing intermediate quantities during backward"),公式如下
    (G' = ∂L/∂S_C,Ũ = exp(G) ⊙ U,W = exp(G_C − G) ⊙ U):

        dE = W G',   dW = E G'ᵀ
        dR = (I + L)⁻ᵀ dE                      → ∂L/∂Z = dR      (转置的三角求解)
        dL = −dR Eᵀ(只用严格下三角),   dŨ = −dR S_0ᵀ
        ∂L/∂S_0 = exp(G_C) ⊙ G' − Ũᵀ dR
        X[t] = Σ_s dL[t,s] exp(G_t − G_s) ⊙ u_s,   Y[s] = Σ_t dL[t,s] exp(G_t − G_s) ⊙ u_t     (`wy_lower_backward`,与 L 同样的分层)
        ∂L/∂U = X + Y + exp(G) ⊙ dŨ + exp(G_C − G) ⊙ dW
        ∂L/∂G = U ⊙ (X − Y) + Ũ ⊙ dŨ − W ⊙ dW,最后一行再加 rowsum(G' ⊙ S_C)
        ∂L/∂λ_t = ∂L/∂G_t + … + ∂L/∂G_C

    只有"状态的梯度沿块往回传"是顺序的,其余梯度对所有块一起算。
- **`"token"`**:官方的逐 token 循环,加手写的逐 token backward,保留作参照。每一步的反向公式(G = ∂L/∂S_t,P_t = D_t S_{t-1}):

        g = Gᵀ u_t                           → ∂L/∂w_t = g
        ∂L/∂u_t = G (w_t − P_tᵀ u_t) − P_t g
        dP = G − u_t gᵀ
        ∂L/∂d_t = rowsum(dP ⊙ S_{t-1})
        ∂L/∂S_{t-1} = diag(d_t) dP

读出不受写入算法影响,始终与官方逐位一致。写入后的状态:`token` 逐位一致;`block` 与官方相差约 1e-7(FP32 舍入量级)。

### 精度约定

- 写入(状态、L、E)始终在 FP32 下算,即使外面开着 `torch.autocast`(矩阵乘不会被自动降到 BF16)。块内累加的 log 衰减用 FP64。
- 反向在 FP32 下算。唯一的例外是 `in_proj` 的两次大矩阵乘(∂in_proj = d_rowsᵀ x,∂hidden = d_rows · in_proj):
  默认按前向做这次线性变换时的精度算(BF16 模型就是 BF16,与 autograd 相同)。
  设 `memt_autograd.PROJECTION_BACKWARD_DTYPE = torch.float32` 可以改回 FP32,整层反向慢约 10 ms(H20)。

### `modeling_memt.py` 的改动

(`diff memt_nano/modeling_memt_orig.py memt_nano/modeling_memt.py`)

- **何时启用**:只有需要梯度时才走新代码。`torch.no_grad()` 下运行的仍是官方原代码,原 forward 原样保留为 `MemTMemory.forward_reference`。
- **`commit_memory_train(...)`**:即可以反传梯度的 `commit_memory`,状态跨 chunk 保留计算图。
  - 一个 QA 事件超过 2048 token 时,下一个 chunk 会沿用 backbone 的 KV/GDN cache。transformers 5.20 用原地 `copy_` 更新 GDN 的 conv/recurrent cache,autograd 会报错。
  - 所以在 `commit_memory_train` 运行期间,这两处改成不原地更新。
- **`modeling_memt.USE_CUSTOM_BACKWARD = False`**:让官方代码直接走 autograd,只给测试作参照。

## FLA(flash-linear-attention)

- **backbone 的 GDN 层**:装了 FLA 就用 FLA 的 kernel。安装(两个包,`--no-deps` 避免动 torch):

      pip install --no-deps fla-core flash-linear-attention      # 测试用的是 0.6.0,需要 triton >= 3.3

  在 import `modeling_memt` 之前调用 `memt_nano.compat.apply()`。它做两件事:补上官方文件需要、而 transformers 5.20 改了名字的几个函数;
  让 GDN 按张量所在设备选择实现(GPU 走 FLA,CPU 走纯 PyTorch。不这样做的话,transformers 5.20 在 CPU 上也会去调 FLA,报错)。
  FLA 的 kernel 由 Triton 编译,缓存默认在 `~/.triton/cache`;home 不可写或已满时设 `TRITON_CACHE_DIR`。

  GDN 的核心运算(16 头 × 128、2048 token、BF16,H20):

  | | 前向 | 前向 + 反向 | 显存峰值 |
  |---|---|---|---|
  | transformers 的纯 PyTorch 版 | 5.0 ms | 23.2 ms | 0.38 GiB |
  | FLA 的 kernel | 0.6 ms | 2.0 ms | 0.16 GiB |

  两者的输出相差 4.7e-3,即 BF16 的舍入量级(backbone 本来就是 BF16)。
- **记忆写入没有用 FLA 的 kernel,原因是精度**。FLA 的 KDA(`fla.ops.kda.chunk_kda`)与这里的写入是同一个递推
  (传 k = u、β = 1、v = w、g = λ、取 `final_state`;它的逐 token 参考实现与官方循环逐位一致)。
  但它的 kernel 在支持 TF32 的显卡上用 TF32 做块内点积,写入后的状态和梯度的相对误差约 1e-3,这里的实现约 1e-7:

  | 写入算子(Nano 尺寸,2048 token,H20) | 前向 | 前向 + 反向 | 状态误差 | 梯度误差 |
  |---|---|---|---|---|
  | 这里的 `block` | 4.3 ms | 12.2 ms | 2e-7 | ≤ 3e-6 |
  | FLA `chunk_kda`(TF32 点积,默认) | 1.6 ms | 6.5 ms | 1.1e-3 | ≤ 4e-3 |

  误差都是相对 FP64 真值。记忆状态要跨很多 chunk 累积,所以保持 FP32 的精度。
  `TRITON_F32_DEFAULT=ieee` 可以让 FLA 的点积改用完整 FP32,但 kernel 要全部重新编译(在 H20 上 40 分钟没有编完),没有采用。

## 测试

环境:torch 2.14、transformers 5.20.2、fla-core 0.6.0。

```
CUDA_VISIBLE_DEVICES="" python tests/test_backward.py                    # CPU 检查 1–5
python tests/test_backward.py --gpu-only --length 512 2048                # 只跑 6:GPU 上的端到端 + Nano 尺寸单层
python tests/bench_memory_layer.py --old-stable                           # 整层测速,不需要 transformers
python tests/bench_block_backward.py                                      # 只测写入算子
```

结果(2026-11-25,CPU 检查 269 项、GPU 检查全部通过,0 项失败):

1. **门控归一化**:软上限开/关。forward 与官方逐位一致;梯度相对误差 FP32 ≤ 1e-6,BF16 约 5e-3。
2. **写入递推**:
   - 逐 token 版本:FP64 `gradcheck` 通过。
   - `block` 版本:衰减尺度 0、0.001、0.5、5、50 下与逐 token 循环一致(FP64,1e-10);FP64 `gradcheck` 通过(块长 1、4、16、256)。
   - `block` 的手写反向:与"逐 token 循环走 autograd"和"同一前向走 autograd"两个参照都一致(FP64,≤ 1e-9,实测 1e-12)。
   - L 的分层矩阵乘与直接两两相乘的公式一致(FP64,1e-16)。
   - FP32 下,从零衰减到极大衰减(尺度 100):状态误差 ≤ 1e-6,四个梯度的误差 ≤ 7e-6(相对 FP64 逐 token 循环)。
3. **MemTMemory**:读/写、有无状态、有无 write_state、有无 bias、有无软上限,共 20 种组合,`block` 和 `token` 两种写法都测了。
   - 读出逐位一致;写入状态 `token` 逐位一致,`block` 相对误差 ≤ 1e-5(实测约 1e-7)。
   - 梯度相对误差 FP32 ≤ 1e-4(实测约 1e-6),BF16 ≤ 5e-2(实测约 5e-3,即 BF16 autograd 自身的舍入量级)。
   - FP32 权重在 `torch.autocast(bfloat16)` 下运行:读出逐位一致,写入状态仍是 FP32 精度(6e-8)。
4. **4 层小模型端到端**(2 层带记忆,backbone 冻结):多 chunk 写入(文档模式;文档 + QA 事件回放)后做 query。CPU 上和 GPU 上(backbone 的 GDN 层走 FLA)各跑一遍。
   - `token`:loss 逐位一致。`block`:loss 相对误差为 0(在打印精度下)。
   - 两种写法的 12 个记忆参数梯度相对误差 ≤ 4e-6。
   - 不求梯度时 `commit_memory` 与官方文件逐位一致。
5. **`in_proj` 反向的精度**(Nano 尺寸,GPU):默认做法与 FP32 做法相差 2.6e-3;BF16 autograd 与 FP32 做法相差 3.7e-3 到 4.1e-3。即默认做法比官方代码走 autograd 更接近 FP32 的结果。

## 速度

一个记忆层、一个 2048-token 的 chunk、BF16 权重、FP32 状态(`tests/bench_memory_layer.py`):

| | H20(独占) | | A100-40G(与别的任务共用,仅供参考) | |
|---|---|---|---|---|
| Nano 尺寸:hidden 2048、16 头 × 256 | 前向 + 反向 | 显存峰值 | 前向 + 反向 | 显存峰值 |
| `block`(默认) | 7.6 + 17.4 = **25 ms** | 2.0 GiB | 12.5 + 28.6 = 41 ms | 2.0 GiB |
| `block`,`in_proj` 反向用 FP32 | 7.6 + 28.4 = 36 ms | 2.0 GiB | 13.2 + 50.9 = 64 ms | 2.0 GiB |
| `token`(逐 token 循环 + 手写反向) | 101 + 519 = 620 ms | 2.0 GiB | 183 + 912 = 1094 ms | 2.0 GiB |
| 官方 forward 直接走 autograd | 720 ms | 16.9 GiB | 没测 | |

只算写入这一步(同样的尺寸):

| | H20 前向 / 反向 | A100 前向 / 反向 |
|---|---|---|
| 现在的分块 WY | 4.2 / 7.8 ms | 7.3 / 15.5 ms |
| 之前的稳定公式(块内直接算两两的 exp(G_t − G_s)) | 21.9 / 33–61 ms | 29.8 / 118.5 ms |

4B 框架的尺寸(hidden 2560、32 头 × 128)数字基本相同:H20 上 `block` 7.8 + 17.5 = 25 ms,`token` 548 ms。

几点说明:

- 现在的耗时主要是"运算次数",不是矩阵乘本身:整层反向约 500 次小运算,每次约 25 微秒的调度开销。所以它对显卡算力不敏感。
- A100 那台机器上每张卡都有别人的任务在跑,数字偏大,只能看相对关系(新旧写入公式约 6 倍,`block` 对 `token` 约 25 倍)。H200 上没有测过。
- 进一步提速要靠把多个小运算融合成一个 kernel(FLA 的做法)。FLA 的 KDA kernel 在同样的写入上快约 2 倍,但目前有上面说的精度问题。

## 还没做

- **真实的 Nano 权重(35B)上的反向**、训练循环和 loss、batch > 1、64K 的 pack。
- **显存**:跨多个 chunk 的 backbone 激活还没做梯度 checkpointing。
- **写入的融合 kernel**:满足 FP32 精度的版本(例如让 FLA 的 KDA 用完整 FP32 点积,或自己写)。
