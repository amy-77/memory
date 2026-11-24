# MemT-1.0-Nano 记忆分支的手写 backward

官方 `MemTensor/MemT-1.0-Nano` 的 `modeling_memt.py`(Apache-2.0)只支持推理。这里给记忆分支补上训练所需的 backward。
来源和许可证见 `THIRD_PARTY_NOTICES.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `memt_nano/modeling_memt_orig.py` | 官方文件,未改动(测试的参照) |
| `memt_nano/modeling_memt.py` | 官方文件加下面的改动 |
| `memt_nano/memt_autograd.py` | 手写的 forward / backward |
| `memt_nano/configuration_memt.py` | 官方文件,未改动 |
| `tests/test_backward.py` | 与"对官方 forward 直接用 autograd 求导"逐项对比 |

## 实现了什么

| 任务 | 官方代码位置 | 实现 |
|---|---|---|
| 1. 带软上限的门控 RMSNorm 的 backward | 44–52 行(`MemTRMSNormGated`,上限在 51 行) | `GatedRMSNormFn`、`gated_rmsnorm_backward`:RMS 归一化、SiLU、tanh 上限的导数全部手写 |
| 2. full attention 中记忆部分的 backward | 257–266 行(`MemTAttention.forward`) | `MemoryResidualFn`:"注意力输出 + 记忆读出"和"更新后的状态"合成一个节点;读/写状态的梯度传回它们原来所在的设备 |
| 3. 记忆的 backward | 92–132 行(`MemTMemory.forward`) | `MemTMemoryFn`、`memory_forward`、`memory_backward`:读、写两条路径的链式部分都手写;写入递推见下 |

写入递推(每个 batch、每个头,S 为 K×V;λ = log d ≤ 0 是按通道的 log 衰减):

    S_t = D_t S_{t-1} + u_t (w_t − u_tᵀ D_t S_{t-1})ᵀ,   D_t = diag(exp λ_t)

训练时有两种算法,算的是同一个递推,由 `MemTMemory.writer` 选择:

- **`"block"`(默认,2026-11-25 起)**:数值稳定的分块 WY 形式(`delta_block_stable`),对应 MemT 报告第 4 节训练时用的 "chunked WY"。
  - 每 64 个 token 做一次矩阵乘法加一次三角求解。块内 G = cumsum(λ),只出现衰减比 exp(G_t − G_s)(s ≤ t,值 ≤ 1),所以衰减多大都不会溢出,也不需要上限。

        L[t,s] = Σ_k u_tk u_sk exp(G_tk − G_sk)  (s < t)
        (I + L) E = Z − (exp(G) ⊙ U) S_0
        S_C = exp(G_C) ⊙ S_0 + (exp(G_C − G) ⊙ U)ᵀ E

  - 前向每 `MemTMemory.checkpoint_every`(= 256)个 token 存一份 FP32 状态。反向时从存档重算这一组,组内用 autograd 对 (状态, u, w, λ) 求导,再接回手写的链式部分。即报告里的 "reconstructing intermediate quantities during backward"。
- **`"token"`**:官方的逐 token 循环,加手写的逐 token backward,保留作参照。每一步的反向公式(G = ∂L/∂S_t,P_t = D_t S_{t-1}):

        g = Gᵀ u_t                           → ∂L/∂w_t = g
        ∂L/∂u_t = G (w_t − P_tᵀ u_t) − P_t g
        dP = G − u_t gᵀ
        ∂L/∂d_t = rowsum(dP ⊙ S_{t-1})
        ∂L/∂S_{t-1} = diag(d_t) dP

读出不受写入算法影响,始终与官方逐位一致。写入后的状态:`token` 逐位一致;`block` 与官方差在 FP32 舍入量级(相对 ~1e-5)。

`modeling_memt.py` 的改动(`diff memt_nano/modeling_memt_orig.py memt_nano/modeling_memt.py`):

- **何时启用**:只有需要梯度时才走新代码。`torch.no_grad()` 下运行的仍是官方原代码,原 forward 原样保留为 `MemTMemory.forward_reference`。
- **`commit_memory_train(...)`**:即可以反传梯度的 `commit_memory`,状态跨 chunk 保留计算图。
  - 一个 QA 事件超过 2048 token 时,下一个 chunk 会沿用 backbone 的 KV/GDN cache。transformers 5.20 用原地 `copy_` 更新 GDN 的 conv/recurrent cache,autograd 会报错。
  - 所以在 `commit_memory_train` 运行期间,这两处改成不原地更新。
- **`modeling_memt.USE_CUSTOM_BACKWARD = False`**:让官方代码直接走 autograd,只给测试作参照。

## 测试

环境:torch 2.14、transformers 5.20.2。测试文件里带了 transformers 5.12 → 5.20 的两个兼容补丁。

```
CUDA_VISIBLE_DEVICES="" python tests/test_backward.py              # CPU 检查 1–5
python tests/test_backward.py --gpu-only --length 512 2048          # 只跑 6:Nano 尺寸单层的速度和显存
```

结果(2026-11-25,171 项,0 项失败):

1. **门控归一化**:软上限开/关。forward 与官方逐位一致;梯度相对误差 FP32 ≤ 1e-6,BF16 约 5e-3。
2. **写入递推**:
   - 逐 token 版本:FP64 `gradcheck` 通过。
   - `block` 版本:在衰减尺度 0.001、0.5、5 下与逐 token 循环一致(FP64,1e-10);FP64 `gradcheck` 在 4 种块大小/存档间隔下通过。
3. **MemTMemory**:读/写、有无状态、有无 write_state、有无 bias、有无软上限,共 20 种组合,`block` 和 `token` 两种写法都测了。
   - 读出逐位一致;写入状态 `token` 逐位一致,`block` 相对误差 ≤ 1e-5。
   - 梯度相对误差 FP32 ≤ 3e-7,BF16 ≤ 6e-3。
4. **4 层小模型端到端**(2 层带记忆,backbone 冻结):多 chunk 写入(文档模式;文档 + QA 事件回放)后做 query。
   - `token`:loss 逐位一致。
   - `block`:loss 相对误差为 0(在打印精度下)。
   - 两种写法的 12 个记忆参数梯度相对误差都 ≤ 8e-7。
   - 不求梯度时 `commit_memory` 与官方文件逐位一致。
5. **Nano 尺寸单层**(hidden 2048、16 头 × 256、BF16,单卡,一个 2K chunk 的前向 + 反向):

   | 写入算法 | 2048 token | 512 token |
   |---|---|---|
   | `block` | 0.10 s,1.65 GiB | 0.03 s,0.97 GiB |
   | `token` | 0.56 s,1.74 GiB | 0.14 s,0.69 GiB |
   | 直接 autograd(参照) | 0.62 s,16.9 GiB | 0.14 s,4.2 GiB |

   与 autograd 相比,读出逐位一致,梯度差约 3e-3(BF16 噪声量级)。

## 还没做

- **速度**:`block` 写入是纯 PyTorch,每个记忆层每 2K chunk 约 0.1 s(前向 + 反向)。融合 kernel(如 FLA 的 KDA:k = u、β = 1、v = ω⊙v、g = λ)还可以再快。
- **显存**:跨多个 chunk 的 backbone 激活还没做梯度 checkpointing。
