"""手写的 MemT backward 与"官方 forward 直接走 autograd"的对照检查。

    python tests/test_backward.py                                # CPU 检查 1-5
    python tests/test_backward.py --gpu                          # 加检查 6:Nano 尺寸的记忆层,速度和显存峰值
    python tests/test_backward.py --gpu-only --length 512 2048   # 只跑检查 6

1. MemTRMSNormGated(软上限开/关,FP32/BF16):前向逐位相同,梯度与 autograd 对比
2. 写入递推本身:FP64 下的 torch.autograd.gradcheck;block 算法(分块 WY)与逐 token 循环一致;
   block 的手写反向与 autograd 对比(对同一前向、对逐 token 循环各一次);FP32 下从零衰减到极大衰减的误差
3. MemTMemory,读/写/state/write_state/bias/软上限的全部组合,FP32,两种写入算法:前向和梯度
4. BF16 下的 MemTMemory:读出逐位相同,梯度接近 BF16 的 autograd
   4b:FP32 权重在 torch.autocast(bfloat16) 下运行,写入仍保持 FP32
5. 小型 MemT 模型端到端(backbone 冻结):commit_memory_train 写入多个 chunk(文档模式,以及文档 + QA 事件回放),
   再做一次 query;loss 和记忆参数的梯度与官方代码走 autograd 的结果对比;不求梯度的 commit_memory
   输出与官方文件逐位相同
6. (--gpu)先在 GPU 上重跑检查 5(装了 FLA 时 backbone 的 GDN 层走 FLA 的 kernel);
   再测 Nano 尺寸的单个记忆层(hidden 2048、16 头 x 256、BF16):三种方式的时间和显存峰值"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memt_nano import compat  # noqa: E402

# modeling_memt.py 针对 transformers 5.12.1;compat.apply() 让它在 5.20 上运行,并在 GPU 上用 FLA 的 GDN kernel。
compat.apply()

from memt_nano import memt_autograd as ma  # noqa: E402
from memt_nano import modeling_memt as M  # noqa: E402
from memt_nano import modeling_memt_orig as O  # noqa: E402

FAILED = []


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def check_rel(name, a, b, tol):
    err = rel(a, b)
    check(name, err <= tol, f"rel {err:.2e} (tol {tol:.0e})")


# ---------------------------------------------------------------------------------------------------------------------
def test_gated_norm():
    print("== 1. MemTRMSNormGated")
    for softcap in (True, False):
        for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 5e-2)):
            cfg = {"output_gate_softcap_enabled": softcap, "output_gate_softcap_limit": 2.0}
            torch.manual_seed(1)
            ref = O.MemTRMSNormGated(16, 1e-6, cfg)
            new = M.MemTRMSNormGated(16, 1e-6, cfg)
            with torch.no_grad():
                ref.weight.normal_()
                new.weight.copy_(ref.weight)
            ref.to(dtype), new.to(dtype)
            value = (torch.randn(2, 5, 3, 16) * 3).to(dtype)
            gate = (torch.randn(2, 5, 3, 16) * 4).to(dtype)  # silu 最大约 12:上限 2 会生效
            probe = torch.randn(2, 5, 3, 16)
            outs, grads = [], []
            for module in (ref, new):
                v, g = value.clone().requires_grad_(), gate.clone().requires_grad_()
                out = module(v, g)
                (out.float() * probe).sum().backward()
                outs.append(out)
                grads.append((v.grad, g.grad, module.weight.grad))
            tag = f"softcap={softcap} {str(dtype)[6:]}"
            check(f"norm forward bitwise {tag}", torch.equal(outs[0], outs[1]))
            for name, a, b in zip(("value", "gate", "weight"), grads[1], grads[0]):
                check_rel(f"norm d{name} {tag}", a, b, tol)


# ---------------------------------------------------------------------------------------------------------------------
def test_delta_rule_gradcheck():
    print("== 2. write recurrence, gradcheck FP64")
    torch.manual_seed(2)
    batch, heads, kdim, vdim, length = 2, 2, 4, 3, 11
    for every in (1, 4, 11, 20):
        state = torch.randn(batch, heads, kdim, vdim, dtype=torch.float64)
        key = torch.randn(batch, length, heads, kdim, dtype=torch.float64)
        key = 0.9 * key / key.norm(dim=-1, keepdim=True)
        write = torch.randn(batch, length, heads, vdim, dtype=torch.float64)
        decay = 0.5 + 0.5 * torch.rand(batch, length, heads, kdim, dtype=torch.float64)
        inputs = tuple(t.requires_grad_() for t in (state, key, write, decay))
        ok = torch.autograd.gradcheck(
            lambda s, k, w, d: ma.DeltaRuleFn.apply(s, k, w, d, every), inputs, eps=1e-6, atol=1e-7
        )
        check(f"gradcheck ckpt_every={every}", ok)

    print("== 2b. block writer (chunked WY): equals the token loop for any decay, gradcheck FP64")
    for log_decay_scale in (0.0, 1e-3, 0.5, 5.0, 50.0):
        state = torch.randn(batch, heads, kdim, vdim, dtype=torch.float64)
        key = torch.randn(batch, 40, heads, kdim, dtype=torch.float64)
        key = torch.rand(batch, 40, heads, 1, dtype=torch.float64).sqrt() * key / key.norm(dim=-1, keepdim=True)
        write = torch.randn(batch, 40, heads, vdim, dtype=torch.float64)
        lam = -torch.rand(batch, 40, heads, kdim, dtype=torch.float64) * log_decay_scale
        ref = state
        for t in range(40):
            ref = ma.delta_rule_step(ref, key[:, t], write[:, t], lam[:, t].exp())
        for block in (1, 2, 8, 32, 64, 256):
            out, _ = ma.block_write_forward(state, key, write, lam, block)
            check(f"block == token loop, decay scale {log_decay_scale}, block {block}",
                  torch.allclose(out, ref, rtol=1e-10, atol=1e-10))
    state = torch.randn(batch, heads, kdim, vdim, dtype=torch.float64)
    key = torch.randn(batch, length, heads, kdim, dtype=torch.float64)
    key = 0.9 * key / key.norm(dim=-1, keepdim=True)
    write = torch.randn(batch, length, heads, vdim, dtype=torch.float64)
    for scale in (0.01, 1.0, 20.0):
        lam = -torch.rand(batch, length, heads, kdim, dtype=torch.float64) * scale
        inputs = tuple(t.detach().requires_grad_() for t in (state, key, write, lam))
        for block in (1, 4, 16, 256):
            ok = torch.autograd.gradcheck(
                lambda s, k, w, g: ma.BlockWriteFn.apply(s, k, w, g, block), inputs, eps=1e-6, atol=1e-7
            )
            check(f"gradcheck block writer, decay scale {scale}, block={block}", ok)

    print("== 2c. block writer pieces, FP64: L vs the pairwise formula; hand-written backward vs autograd")
    names = ("d state", "d key", "d write_value", "d lam")
    for count, kd, vd in ((1, 4, 3), (5, 4, 3), (37, 6, 5), (130, 6, 5)):
        s0 = torch.randn(2, 3, kd, vd, dtype=torch.float64)
        kk = torch.randn(2, count, 3, kd, dtype=torch.float64)
        kk = torch.rand(2, count, 3, 1, dtype=torch.float64).sqrt() * kk / kk.norm(dim=-1, keepdim=True)
        zz = torch.randn(2, count, 3, vd, dtype=torch.float64)
        gout = torch.randn(2, 3, kd, vd, dtype=torch.float64)
        for scale in (0.01, 3.0, 50.0):
            ll = -torch.rand(2, count, 3, kd, dtype=torch.float64) * scale
            inputs = [t.detach().requires_grad_() for t in (s0, kk, zz, ll)]
            # 参照 1:逐 token 循环走 autograd(与分块实现完全独立)
            loop = inputs[0]
            for t in range(count):
                loop = ma.delta_rule_step(loop, inputs[1][:, t], inputs[2][:, t], inputs[3][:, t].exp())
            ref_loop = torch.autograd.grad(loop, inputs, gout)
            for block in (1, 4, 64):
                # 参照 2:同一个分块前向走 autograd
                ref_same = torch.autograd.grad(ma.block_write_forward(*inputs, block)[0], inputs, gout)
                with torch.no_grad():
                    _, starts = ma.block_write_forward(s0, kk, zz, ll, block)
                    got = ma.block_write_backward(gout, starts, kk, zz, ll, block)
                worst_same = max(rel(a, b) for a, b in zip(got, ref_same))
                worst_loop = max(rel(a, b) for a, b in zip(got, ref_loop))
                check(f"block backward, L={count}, block={block}, decay scale {scale}",
                      worst_same <= 1e-9 and worst_loop <= 1e-9,
                      f"vs autograd of the same forward {worst_same:.1e}, of the token loop {worst_loop:.1e}")
    check("block backward runs without autograd", all(not t.requires_grad for t in got))
    # L 的分层矩阵乘算法 与 直接两两相乘的公式
    uu = torch.randn(3, 64, 8, dtype=torch.float64)
    gg = (-torch.rand(3, 64, 8, dtype=torch.float64) * 3).cumsum(1)
    ratio = (gg[:, :, None] - gg[:, None]).exp() * torch.ones(64, 64, dtype=torch.float64).tril(-1)[None, :, :, None]
    check_rel("wy_lower vs pairwise formula", ma.wy_lower(uu, gg), torch.einsum("ntk,nsk,ntsk->nts", uu, uu, ratio), 1e-12)

    print("== 2d. block writer in FP32 vs the FP64 token loop, weak to extreme decay (512 tokens, block 256)")
    for scale in (0.0, 0.03, 1.0, 10.0, 100.0):
        torch.manual_seed(5)
        s0 = torch.randn(1, 4, 32, 32, dtype=torch.float64)
        kk = torch.randn(1, 512, 4, 32, dtype=torch.float64)
        kk = torch.rand(1, 512, 4, 1, dtype=torch.float64).sqrt() * kk / kk.norm(dim=-1, keepdim=True)
        zz = torch.randn(1, 512, 4, 32, dtype=torch.float64)
        ll = -torch.rand(1, 512, 4, 32, dtype=torch.float64) * scale
        gout = torch.randn(1, 4, 32, 32, dtype=torch.float64)
        inputs = [t.detach().requires_grad_() for t in (s0, kk, zz, ll)]
        loop = inputs[0]
        for t in range(512):
            loop = ma.delta_rule_step(loop, inputs[1][:, t], inputs[2][:, t], inputs[3][:, t].exp())
        truth = torch.autograd.grad(loop, inputs, gout)
        f32 = [t.float() for t in (s0, kk, zz, ll)]
        out, starts = ma.block_write_forward(*f32, 256)
        got = ma.block_write_backward(gout.float(), starts, *f32[1:], 256)
        check_rel(f"FP32 written state, decay scale {scale}", out, loop.detach(), 1e-5)
        for name, a, b in zip(names, got, truth):
            if b.norm() > 1e-30:  # 衰减极大时 d state 的真值就是 0
                check_rel(f"FP32 {name}, decay scale {scale}", a, b, 1e-4)


# ---------------------------------------------------------------------------------------------------------------------
def memory_pair(hidden_size, heads, dim, bias, softcap, dtype, every, seed=3, writer="block", block=4):
    cfg = {"output_gate_bias": bias, "output_gate_softcap_enabled": softcap, "output_gate_softcap_limit": 2.0}
    torch.manual_seed(seed)
    ref = O.MemTMemory(cfg, hidden_size, heads, dim)
    with torch.no_grad():
        for p in ref.parameters():  # 离开接近零的初始值,让每条路径都有信号
            p.add_(torch.randn_like(p) * 0.3)
        ref.decay_bias.normal_()  # 衰减在约 0.3 到 1 之间,而不是约 0.9999
    new = M.MemTMemory(cfg, hidden_size, heads, dim)
    new.load_state_dict(ref.state_dict())
    new.checkpoint_every, new.writer, new.write_block = every, writer, block
    return ref.to(dtype), new.to(dtype)


def run_memory(module, x, state, write_state, commit, probes):
    x = x.clone().requires_grad_()
    state = None if state is None else state.clone().requires_grad_()
    write_state = None if write_state is None else write_state.clone().requires_grad_()
    read, updated = module(x, state, commit, write_state=write_state)
    loss = (read.float() * probes[0]).sum()
    if commit:
        loss = loss + (updated * probes[1]).sum()
    loss.backward()
    grads = {"hidden": x.grad}
    if state is not None:
        grads["state"] = state.grad
    if write_state is not None:
        grads["write_state"] = write_state.grad
    for name, p in module.named_parameters():
        grads[name] = p.grad
    return read, (updated if commit else None), grads


def test_memory(dtype, tol, cases, writer):
    print(f"== {3 if dtype == torch.float32 else 4}. MemTMemory {str(dtype)[6:]}, writer={writer}")
    batch, length, hidden_size, heads, dim = 2, 29, 24, 2, 8
    for commit, has_state, has_write, bias, softcap in cases:
        ref, new = memory_pair(hidden_size, heads, dim, bias, softcap, dtype, every=8, writer=writer)
        torch.manual_seed(4)
        x = torch.randn(batch, length, hidden_size).to(dtype)
        state = torch.randn(batch, heads, dim, dim) * 0.5 if has_state else None
        write_state = torch.randn(batch, heads, dim, dim) * 0.5 if has_write else None
        probes = (torch.randn(batch, length, heads * dim), torch.randn(batch, heads, dim, dim))
        r_ref, u_ref, g_ref = run_memory(ref, x, state, write_state, commit, probes)
        r_new, u_new, g_new = run_memory(new, x, state, write_state, commit, probes)
        tag = f"commit={int(commit)} state={int(has_state)} write={int(has_write)} bias={int(bias)} cap={int(softcap)}"
        if writer == "token" or u_ref is None:
            same = torch.equal(r_ref, r_new) and (u_ref is None or torch.equal(u_ref, u_new))
            check(f"forward bitwise  {tag}", same)
        else:  # 读出逐位相同;block 写入与逐 token 循环相差 FP32 舍入量级
            check(f"read bitwise     {tag}", torch.equal(r_ref, r_new))
            check_rel(f"written state    {tag}", u_new, u_ref, 1e-5)
        worst = max((rel(g_new[k], g_ref[k]), k) for k in g_ref if g_ref[k] is not None)
        missing = [k for k in g_ref if (g_ref[k] is None) != (g_new[k] is None)]
        check(f"grads            {tag}", worst[0] <= tol and not missing,
              f"worst rel {worst[0]:.2e} ({worst[1]}) tol {tol:.0e} {missing or ''}")


def test_memory_autocast():
    print("== 4b. MemTMemory with FP32 weights under torch.autocast(bfloat16)")
    batch, length, hidden_size, heads, dim = 2, 29, 24, 2, 8
    for writer in ("block", "token"):
        for has_state in (False, True):
            ref, new = memory_pair(hidden_size, heads, dim, False, True, torch.float32, every=8, writer=writer)
            torch.manual_seed(4)
            x = torch.randn(batch, length, hidden_size)
            state = torch.randn(batch, heads, dim, dim) * 0.5 if has_state else None
            probes = (torch.randn(batch, length, heads * dim), torch.randn(batch, heads, dim, dim))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                r_ref, u_ref, g_ref = run_memory(ref, x, state, None, True, probes)
                r_new, u_new, g_new = run_memory(new, x, state, None, True, probes)
            tag = f"writer={writer} state={int(has_state)}"
            check(f"read bitwise     {tag}", torch.equal(r_ref, r_new) and r_new.dtype == r_ref.dtype)
            # 写入必须留在 FP32:如果里面的矩阵乘被 autocast 降到 BF16,这里的误差会到 1e-2
            if writer == "token":
                check(f"written state bitwise {tag}", torch.equal(u_ref, u_new))
            else:
                check_rel(f"written state    {tag}", u_new, u_ref, 1e-5)
            worst = max((rel(g_new[k], g_ref[k]), k) for k in g_ref if g_ref[k] is not None)
            dtypes = all(g_new[k].dtype == g_ref[k].dtype for k in g_ref if g_ref[k] is not None)
            check(f"grads            {tag}", worst[0] <= 5e-2 and dtypes, f"worst rel {worst[0]:.2e} ({worst[1]})")


# ---------------------------------------------------------------------------------------------------------------------
def tiny_config():
    text = dict(
        model_type="qwen3_5_moe_text", vocab_size=97, hidden_size=64, num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        full_attention_interval=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        attention_bias=False, attention_dropout=0.0, attn_output_gate=True, hidden_act="silu",
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_value_head_dim=16, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4,
        num_experts_per_tok=2, rms_norm_eps=1e-6, max_position_embeddings=4096, partial_rotary_factor=0.25,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        mtp_num_hidden_layers=0, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=None,
        router_aux_loss_coef=0.001, output_router_logits=False, initializer_range=0.02, dtype="float32",
    )
    memt = {
        "memory_update_mode": "token", "gdn_version": "stable_gdn2", "memory_use_short_conv": False,
        "memory_head_mode": "mha", "memory_read_scale": "k_inv_sqrt", "num_memory_heads": 2, "num_kv_heads": 2,
        "linear_attention_state_policy": "reset_per_chunk", "chunk_tokens": 2048, "tail_policy": "commit",
        "output_gate_type": "silu", "output_gate_bias": False, "output_gate_softcap_enabled": True,
        "output_gate_softcap_limit": 2.0, "stable_gdn2_sqrt_erase_gate": True, "backbone_gdn_override": False,
    }
    return M.MemTQwen3_5MoeConfig(backbone_configs={"model_type": "qwen3_5_moe", "text_config": text}, memt=memt)


def build(module, config, chunk, every):
    torch.manual_seed(5)
    model = module.MemTQwen3_5MoeForCausalLM(config).float().eval()
    for layer in model.model.language_model.layers:  # 5.20 的 decoder forward 会读取 block_type
        if not hasattr(layer, "block_type"):
            layer.block_type = layer.layer_type
    model.config.memt["chunk_tokens"] = chunk  # 短测试里也有多个写入 chunk
    for name, p in model.named_parameters():
        p.requires_grad_(".memt." in name)
    for layer in model.model.language_model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.memt.checkpoint_every = every
            layer.self_attn.memt.write_block = 4
            with torch.no_grad():
                layer.self_attn.memt.decay_bias.normal_()
                layer.self_attn.memt.g_down.weight.mul_(8)  # 放大读出,让 query 依赖记忆
    return model


def test_end_to_end(device="cpu"):
    backend = compat.gdn_backend() if device == "cuda" else "torch"
    print(f"== 5. tiny MemT model, end to end ({device}, backbone GDN: {backend})")
    config = tiny_config()
    model = build(M, config, chunk=16, every=5).to(device)
    torch.manual_seed(6)
    document = torch.randint(3, 97, (1, 50), device=device)
    events = torch.randint(3, 97, (1, 37), device=device)
    query = torch.randint(3, 97, (1, 21), device=device)

    def run(custom, replay_events):
        M.USE_CUSTOM_BACKWARD = custom
        model.zero_grad(set_to_none=True)
        state = model.commit_memory_train(document, document=True)
        if replay_events:
            state = model.commit_memory_train(events, state)
        logits = model(input_ids=query, memory_state=state, use_cache=False).logits
        loss = F.cross_entropy(logits[0, :-1].float(), query[0, 1:])
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}

    for writer in ("block", "token"):
        for layer in model.model.language_model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.memt.writer = writer
        for replay in (False, True):
            loss_ref, g_ref = run(False, replay)
            loss_new, g_new = run(True, replay)
            tag = f"{writer} writer, " + ("document + QA-event replay" if replay else "document")
            if writer == "token":
                check(f"loss bitwise ({tag})", torch.equal(loss_ref, loss_new), f"{loss_new.item():.6f}")
            else:  # block 写入与逐 token 循环相差 FP32 舍入量级
                check_rel(f"loss ({tag})", loss_new, loss_ref, 1e-6)
            nonzero = sum(int(g.abs().max() > 0) for g in g_ref.values())
            worst = max((rel(g_new[k], g_ref[k]), k) for k in g_ref)
            check(f"memory grads ({tag})", worst[0] <= 1e-4,
                  f"worst rel {worst[0]:.2e} ({worst[1]}), {nonzero}/{len(g_ref)} tensors nonzero")
    M.USE_CUSTOM_BACKWARD = True

    original = build(O, config, chunk=16, every=5).to(device)
    original.load_state_dict(model.state_dict())
    with torch.no_grad():
        s_new = model.commit_memory(events, model.commit_memory(document, document=True))
        s_old = original.commit_memory(events, original.commit_memory(document, document=True))
    check("no-grad commit_memory bitwise vs released file", all(torch.equal(s_new[k], s_old[k]) for k in s_old))
    for writer in ("block", "token"):
        for layer in model.model.language_model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.memt.writer = writer
        s_train = model.commit_memory_train(events, model.commit_memory_train(document, document=True))
        if writer == "token":
            check("commit_memory_train values bitwise vs released commit_memory (token writer)",
                  all(torch.equal(s_train[k].detach(), s_old[k]) for k in s_old))
        else:
            worst = max(rel(s_train[k].detach(), s_old[k]) for k in s_old)
            check("commit_memory_train values vs released commit_memory (block writer)", worst <= 1e-5,
                  f"worst rel {worst:.2e}")


# ---------------------------------------------------------------------------------------------------------------------
def test_gpu_nano_layer(length, with_autograd=True):
    print(f"== 6. Nano-size memory layer on GPU, {length} tokens, BF16")
    device = torch.device("cuda")
    ref, block = memory_pair(2048, 16, 256, False, True, torch.bfloat16, every=256, writer="block", block=256)
    _, token = memory_pair(2048, 16, 256, False, True, torch.bfloat16, every=64, writer="token")
    ref, block, token = ref.to(device), block.to(device), token.to(device)
    torch.manual_seed(7)
    x = torch.randn(1, length, 2048, device=device).to(torch.bfloat16)
    state = (torch.randn(1, 16, 256, 256, device=device) * 0.02)
    probes = (torch.randn(1, length, 4096, device=device), torch.randn(1, 16, 256, 256, device=device))
    runs = [("block", block), ("token", token)] + ([("autograd", ref)] if with_autograd else [])
    short = (x[:, :64], (probes[0][:, :64], probes[1]))
    for _, module in runs:  # 计时前先预热(CUDA 上下文、cuBLAS)
        run_memory(module, short[0], state, None, True, short[1])
        for p in module.parameters():
            p.grad = None
    results = {}
    for name, module in runs:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        start = time.time()
        try:
            out = run_memory(module, x, state, None, True, probes)
            torch.cuda.synchronize()
            results[name] = out
            print(f"   {name:8s} fwd+bwd {time.time() - start:6.2f} s, "
                  f"peak extra {(torch.cuda.max_memory_allocated() - base) / 2**30:6.2f} GiB")
        except torch.OutOfMemoryError as error:
            print(f"   {name:8s} out of memory: {str(error).splitlines()[0][:200]}")
        for p in module.parameters():
            p.grad = None
        torch.cuda.empty_cache()
    reference = "autograd" if "autograd" in results else "token"
    for name in ("block", "token"):
        if name == reference or name not in results or reference not in results:
            continue
        r_new, u_new, g_new = results[name]
        r_ref, u_ref, g_ref = results[reference]
        check(f"{name} vs {reference}: read bitwise (GPU)", torch.equal(r_new, r_ref))
        print(f"   {name} vs {reference}: written state rel diff {rel(u_new, u_ref):.2e}")
        for k in g_ref:
            print(f"   {name} vs {reference}: {k:18s} grad rel diff {rel(g_new[k], g_ref[k]):.2e}"
                  f"   |grad| {g_ref[k].float().norm():.3e}")
    # in_proj 反向的两次大矩阵乘:默认按前向的精度(BF16)算。与在 FP32 下算的结果比较,并与 BF16 autograd 比较
    ma.PROJECTION_BACKWARD_DTYPE = torch.float32
    torch.cuda.synchronize()
    start = time.time()
    g_fp32 = run_memory(block, x, state, None, True, probes)[2]
    torch.cuda.synchronize()
    print(f"   block with FP32 in_proj backward: fwd+bwd {time.time() - start:6.2f} s")
    ma.PROJECTION_BACKWARD_DTYPE = None
    for k in ("hidden", "in_proj.weight"):
        ours = rel(results["block"][2][k], g_fp32[k])
        line = f"   {k:15s} grad: default (forward precision) vs FP32 in_proj backward, rel diff {ours:.2e}"
        if "autograd" in results:
            theirs = rel(results["autograd"][2][k], g_fp32[k])
            line += f";  BF16 autograd vs FP32: {theirs:.2e}"
            check(f"{k}: default in_proj backward is closer to FP32 than BF16 autograd", ours <= theirs)
        print(line)
    for p in block.parameters():
        p.grad = None
    if with_autograd and "autograd" in results:
        # 谁更接近真实梯度:同一组权重和输入放到 FP64 下,用官方代码走 autograd,作为参照
        import copy
        exact = copy.deepcopy(ref).double()
        _, _, g_exact = run_memory(exact, x.double(), state.double(), None, True,
                                   (probes[0].double(), probes[1].double()))
        worst = {name: max(rel(results[name][2][k], g_exact[k]) for k in g_exact) for name in results}
        for name, value in worst.items():
            print(f"   {name:8s} vs FP64 gradients of the released code: worst rel diff {value:.2e}")
        check("block gradients are as close to FP64 as BF16 autograd", worst["block"] <= 1.1 * worst["autograd"])
        del exact, g_exact
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--length", type=int, nargs="+", default=[2048])
    parser.add_argument("--gpu-only", action="store_true", help="skip the CPU checks 1-5")
    parser.add_argument("--no-autograd", action="store_true", help="skip the plain-autograd reference (long inputs)")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.gpu_only:
        test_end_to_end("cuda")
        for length in args.length:
            test_gpu_nano_layer(length, not args.no_autograd)
        print(f"\n{len(FAILED)} failed" + (f": {FAILED}" if FAILED else ""))
        sys.exit(1 if FAILED else 0)
    test_gated_norm()
    test_delta_rule_gradcheck()
    import itertools
    # (commit, state, write_state, bias, softcap);不写入时必须有 state(否则读出恒为零),
    # 且用不到 write_state
    full = [c for c in itertools.product((False, True), repeat=5) if c[0] or (c[1] and not c[2])]
    for writer in ("block", "token"):
        test_memory(torch.float32, 1e-4, full, writer)
    test_memory(torch.bfloat16, 5e-2, [c for c in full if c[3] is False], "block")
    test_memory_autocast()
    test_end_to_end()
    if args.gpu:
        test_end_to_end("cuda")
        for length in args.length:
            test_gpu_nano_layer(length, not args.no_autograd)
    print(f"\n{len(FAILED)} failed" + (f": {FAILED}" if FAILED else ""))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
