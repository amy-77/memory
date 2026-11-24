# MemT-1.0-Nano 记忆分支的手写 backward(Step 1)

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
| 3. 记忆的 backward | 92–132 行(`MemTMemory.forward`) | `MemTMemoryFn`、`memory_forward`、`memory_backward`、`delta_rule_backward`:读、写两条路径都手写 |

写入递推(每个 batch、每个头,S 为 K×V):

    P_t = diag(d_t) S_{t-1}
    S_t = P_t + u_t (w_t − P_tᵀ u_t)ᵀ

已知 G = ∂L/∂S_t 时:

    g = Gᵀ u_t                           → ∂L/∂w_t = g
    ∂L/∂u_t = G (w_t − P_tᵀ u_t) − P_t g
    dP = G − u_t gᵀ
    ∂L/∂d_t = rowsum(dP ⊙ S_{t-1})
    ∂L/∂S_{t-1} = diag(d_t) dP

前向每 `MemTMemory.checkpoint_every`(= 64)个 token 存一份 FP32 状态。反向时一次只重算 64 个 token 的状态,再倒着走一遍。
所以显存按 L/64 份状态增长,而直接 autograd 要为每个 token 存一份状态。

`modeling_memt.py` 的改动(`diff memt_nano/modeling_memt_orig.py memt_nano/modeling_memt.py`):

- **何时启用**:只有需要梯度时才走新代码。`torch.no_grad()` 下运行的仍是官方原代码,原 forward 原样保留为 `MemTMemory.forward_reference`。
- **`commit_memory_train(...)`**:即可以反传梯度的 `commit_memory`,状态跨 chunk 保留计算图。
  - 一个 QA 事件超过 2048 token 时,下一个 chunk 会沿用 backbone 的 KV/GDN cache。transformers 5.20 用原地 `copy_` 更新 GDN 的 conv/recurrent cache,autograd 会报错。
  - 所以在 `commit_memory_train` 运行期间,这两处改成不原地更新。
- **`modeling_memt.USE_CUSTOM_BACKWARD = False`**:让官方代码直接走 autograd,只给测试作参照。

## 测试

环境:torch 2.14、transformers 5.20.2。测试文件里带了 transformers 5.12 → 5.20 的两个兼容补丁。

```
CUDA_VISIBLE_DEVICES="" python tests/test_backward.py          # CPU 检查 1–5
python tests/test_backward.py --gpu [--length 2048]             # 加 6:Nano 尺寸单层的速度和显存
```

结果(2026-11-24,0 项失败):

1. **门控归一化**:软上限开/关。forward 与官方逐位一致;梯度相对误差 FP32 ≤ 1e-6,BF16 约 5e-3。
2. **写入递推**:FP64 `gradcheck`,checkpoint 间隔 1/4/11/20 都通过。
3. **MemTMemory**:读/写、有无状态、有无 write_state、有无 bias、有无软上限,共 20 种组合。forward 逐位一致;梯度相对误差 FP32 ≤ 3e-7,BF16 ≤ 6e-3。
4. **4 层小模型端到端**(2 层带记忆,backbone 冻结):多 chunk 写入(文档模式;文档 + QA 事件回放)后做 query。
   - loss 与官方代码逐位一致。
   - 12 个记忆参数梯度相对误差 ≤ 6e-7。
   - 不求梯度时 `commit_memory` 与官方文件逐位一致。
5. **Nano 尺寸单层**(hidden 2048、16 头 × 256、BF16):
   - 512 token:梯度与 autograd 相差约 3e-3;0.17 s、0.68 GiB,autograd 为 0.19 s、4.2 GiB。
   - 2048 token:0.68 s、1.7 GiB。autograd 每层每个 token 约需 8 MB,一个 2K chunk 约 17 GiB。
   - 这组计时是在一张有其他任务的卡上测的,只能作参考。

## 还没做

- **速度**:写入递推及其反向是逐 token 的 PyTorch 循环,每个记忆层每 2K chunk 约 0.7 s。Nano 有 10 个记忆层,一个 chunk 约 7 s,正式训练需要融合或分块的 kernel。
- **显存**:跨多个 chunk 的 backbone 激活还没做梯度 checkpointing。
