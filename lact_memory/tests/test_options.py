"""Options added on 2026-11-15: LaCT-style fast-weight init, per-token output scale, RoPE on q/k,
merged write units, and that the defaults still load checkpoints written before them."""

import copy

import pytest
import torch

from conftest import tiny_qwen3
from lact_qwen import MemoryConfig, attach_memory, freeze_base, memory_parameters, set_attention_window
from lact_qwen.dialog import merge_spans
from lact_qwen.memory_branch import rotate_by_position


def cfg(**kw):
    base = dict(num_heads=2, head_dim=16, chunk_size=8, use_muon=True, use_momentum=True)
    base.update(kw)
    return MemoryConfig(**base)


def build(**kw):
    model = tiny_qwen3()
    set_attention_window(model, 6)
    attach_memory(model, cfg(**kw))
    return model


def test_old_config_dict_still_loads_with_same_parameter_names():
    old = {"num_heads": 2, "head_dim": 16, "chunk_size": 8, "gate_init": 0.0, "layers": "all", "replay": {"mode": "none"}}
    c = MemoryConfig.from_dict(old)
    assert (c.fw_init, c.out_scale, c.rope) == ("random", "gate", False)
    model = tiny_qwen3()
    attach_memory(model, c)
    names = {n.split(".memory.")[1] for n, _ in model.named_parameters() if ".memory." in n}
    assert {"w0", "w1", "w2", "gate", "qkv.weight", "o_proj.weight"} <= names
    assert not any(n.startswith(("w0_left", "scale_proj")) for n in names)


def test_lowrank_identity_init_matches_lact_shape_and_trains():
    model = build(fw_init="lowrank_identity", fw_rank=4, gate_init=0.1)
    branch = model.model.layers[0].memory
    w0, w1, w2 = branch.fast_weights()
    assert w0.shape == (2, 16, 16) and w1.shape == (2, 16, 16) and w2.shape == (2, 16, 16)
    # the identity part dominates the diagonal
    assert (torch.diagonal(w0, dim1=1, dim2=2).mean() - 0.5).abs() < 0.2
    freeze_base(model)
    ids = torch.randint(0, 97, (1, 40))
    model(ids).logits.sum().backward()
    assert branch.w0_left.grad is not None and branch.w0_left.grad.abs().sum() > 0
    assert branch.w2_right.grad.abs().sum() > 0


@pytest.mark.parametrize("norm", ["post", "pre"])
def test_token_scale_changes_output_and_all_memory_params_get_gradients(norm):
    base = tiny_qwen3()
    set_attention_window(base, 6)
    model = copy.deepcopy(base)
    attach_memory(model, cfg(out_scale="token", norm=norm))
    freeze_base(model)
    ids = torch.randint(0, 97, (2, 40))
    logits = model(ids).logits
    assert not torch.allclose(logits, base(ids).logits)
    logits.sum().backward()
    branch = model.model.layers[0].memory
    assert not hasattr(branch, "gate")
    assert float(branch.scale_stat()) > 0
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not missing, missing
    assert len(memory_parameters(model)) > 0


def test_rotation_keeps_norm_and_depends_on_relative_position():
    t = torch.randn(3, 10, 16)
    r = rotate_by_position(t, 0, 10000.0)
    torch.testing.assert_close(r.norm(dim=-1), t.norm(dim=-1), rtol=1e-5, atol=1e-5)
    # <rot(a, p), rot(b, p)> does not depend on p; shifting both by an offset changes nothing
    a, b = torch.randn(1, 1, 16), torch.randn(1, 1, 16)
    d0 = (rotate_by_position(a, 3, 10000.0) * rotate_by_position(b, 3, 10000.0)).sum()
    d1 = (rotate_by_position(a, 500, 10000.0) * rotate_by_position(b, 500, 10000.0)).sum()
    torch.testing.assert_close(d0, d1, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(d0, (a * b).sum(), rtol=1e-4, atol=1e-4)


def test_rope_branch_runs_and_stays_causal():
    model = build(rope=True, gate_init=0.1)
    ids = torch.randint(0, 97, (1, 48))
    other = ids.clone()
    other[0, 40:] = (other[0, 40:] + 1) % 97
    with torch.no_grad():
        a, b = model(ids).logits, model(other).logits
    torch.testing.assert_close(a[:, :40], b[:, :40])


def test_lora_starts_exact_then_trains_and_roundtrips(tmp_path):
    from lact_qwen.modeling import load_memory_config, load_memory_weights, save_memory

    base = tiny_qwen3()
    set_attention_window(base, 6)
    model = copy.deepcopy(base)
    attach_memory(model, cfg(lora_rank=2))
    ids = torch.randint(0, 97, (1, 40))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, base(ids).logits, rtol=1e-5, atol=1e-5)
    freeze_base(model)
    branch = model.model.layers[0].memory
    assert len(branch.lora_a) == 7
    with torch.no_grad():
        for layer in model.model.layers:
            for p in layer.memory.lora_b.values():
                p.normal_(std=0.1)
    out = model(ids).logits
    assert not torch.allclose(out, base(ids).logits)
    out.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.lora_a.values())
    assert all(p.grad is None for n, p in model.named_parameters() if '.memory.' not in n)
    save_memory(model, str(tmp_path))
    fresh = copy.deepcopy(base)
    attach_memory(fresh, load_memory_config(str(tmp_path))[0])
    load_memory_weights(fresh, str(tmp_path))
    with torch.no_grad():
        torch.testing.assert_close(fresh(ids).logits, out.detach(), rtol=1e-5, atol=1e-5)


def test_segment_attention_hides_earlier_blocks():
    model = tiny_qwen3()
    set_attention_window(model, 8, 0, segment=8)
    ids = torch.randint(0, 97, (1, 32))
    other = ids.clone()
    other[0, 3] = (other[0, 3] + 1) % 97
    with torch.no_grad():
        a, b = model(ids).logits, model(other).logits
    assert not torch.allclose(a[:, 3:8], b[:, 3:8])
    torch.testing.assert_close(a[:, 8:], b[:, 8:])
    # a plain window of the same length does let position 8 see position 3
    plain = tiny_qwen3()
    set_attention_window(plain, 8, 0)
    with torch.no_grad():
        assert not torch.allclose(plain(ids).logits[:, 8:10], plain(other).logits[:, 8:10])


def test_segment_attention_keeps_sinks_visible():
    model = tiny_qwen3()
    set_attention_window(model, 8, 2, segment=8)
    ids = torch.randint(0, 97, (1, 32))
    other = ids.clone()
    other[0, 1] = (other[0, 1] + 1) % 97
    with torch.no_grad():
        assert not torch.allclose(model(ids).logits[:, 20:], model(other).logits[:, 20:])


def test_attention_qkv_copy_and_pre_o_proj_injection(tmp_path):
    from lact_qwen.modeling import load_memory_config, load_memory_weights, save_memory

    base = tiny_qwen3()
    set_attention_window(base, 8)
    model = copy.deepcopy(base)
    # tiny model: 4 attention heads x 16, 2 KV heads -> o_proj input width 64
    c = cfg(num_heads=2, head_dim=32, qkv_init='attention', qkv_act='none', inject='pre_o_proj', delta_readout=True)
    attach_memory(model, c)
    layer = model.model.layers[0]
    branch, attn = layer.memory, layer.self_attn
    assert not hasattr(branch, 'o_proj')
    w = branch.qkv.weight
    torch.testing.assert_close(w[:64], attn.q_proj.weight)
    # KV head 0 serves attention heads 0 and 1, KV head 1 serves heads 2 and 3
    torch.testing.assert_close(w[64:80], attn.k_proj.weight[:16])
    torch.testing.assert_close(w[80:96], attn.k_proj.weight[:16])
    torch.testing.assert_close(w[96:112], attn.k_proj.weight[16:])
    torch.testing.assert_close(w[128:144], attn.v_proj.weight[:16])
    torch.testing.assert_close(w[176:192], attn.v_proj.weight[16:])

    ids = torch.randint(0, 97, (1, 40))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, base(ids).logits, rtol=1e-5, atol=1e-5)  # gate is 0
        for l in model.model.layers:
            l.memory.gate.fill_(0.5)
        out, ref = model(ids).logits, base(ids).logits
    # delta readout: nothing has been written during the first chunk (8 tokens)
    torch.testing.assert_close(out[:, :8], ref[:, :8], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out[:, 8:], ref[:, 8:])
    assert branch.pending is None

    # causal: a later token does not change earlier outputs
    other = ids.clone()
    other[0, 30:] = (other[0, 30:] + 1) % 97
    with torch.no_grad():
        torch.testing.assert_close(model(other).logits[:, :30], out[:, :30], rtol=1e-5, atol=1e-5)

    freeze_base(model)
    model(ids).logits.sum().backward()
    assert branch.qkv.weight.grad.abs().sum() > 0 and branch.w1.grad.abs().sum() > 0
    assert attn.o_proj.weight.grad is None

    save_memory(model, str(tmp_path))
    fresh = copy.deepcopy(base)
    attach_memory(fresh, load_memory_config(str(tmp_path))[0])
    load_memory_weights(fresh, str(tmp_path))
    with torch.no_grad():
        torch.testing.assert_close(fresh(ids).logits, out, rtol=1e-5, atol=1e-5)


def test_out_ref_attention_scales_with_the_attention_output():
    base = tiny_qwen3()
    set_attention_window(base, 8)
    kw = dict(num_heads=2, head_dim=32, inject='pre_o_proj', delta_readout=True, gate_init=0.5)
    plain, rel = copy.deepcopy(base), copy.deepcopy(base)
    attach_memory(plain, cfg(**kw))
    attach_memory(rel, cfg(out_ref='attention', **kw))
    rel.load_state_dict(plain.state_dict())
    seen = {}
    for name, m in (('plain', plain), ('rel', rel)):
        m.model.layers[0].self_attn.o_proj.register_forward_hook(
            lambda mod, args, out, name=name: seen.__setitem__(name, args[0].detach())
        )
    attn_in = {}
    base.model.layers[0].self_attn.o_proj.register_forward_hook(
        lambda mod, args, out: attn_in.__setitem__('x', args[0].detach())
    )
    ids = torch.randint(0, 97, (1, 40))
    with torch.no_grad():
        base(ids), plain(ids), rel(ids)
    x = attn_in['x']
    ref = x.view(1, 40, 2, 32).pow(2).mean(-1, keepdim=True).sqrt()
    add_plain = (seen['plain'] - x).view(1, 40, 2, 32)
    add_rel = (seen['rel'] - x).view(1, 40, 2, 32)
    assert add_plain[:, 8:].abs().sum() > 0
    torch.testing.assert_close(add_rel, add_plain * ref, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        attach_memory(tiny_qwen3(), cfg(out_ref='attention'))


def test_pre_o_proj_needs_matching_width():
    model = tiny_qwen3()
    with pytest.raises(ValueError):
        attach_memory(model, cfg(num_heads=2, head_dim=16, inject='pre_o_proj'))


def test_merge_spans():
    spans = [(0, 5), (5, 9), (9, 20), (20, 22), (22, 23)]
    assert merge_spans(spans, 0) == spans
    assert merge_spans(spans, 8) == [(0, 9), (9, 20), (20, 23)]
    assert merge_spans(spans, 100) == [(0, 23)]
    merged = merge_spans(spans, 6)
    assert merged[0][0] == 0 and merged[-1][1] == 23
    assert all(a[1] == b[0] for a, b in zip(merged, merged[1:]))
