"""core='matrix': every token written into a fixed-size state matrix (gated delta rule)."""

import copy

import torch

from conftest import tiny_qwen3
from lact_qwen import MemoryConfig, attach_memory, freeze_base, set_attention_window
from lact_qwen.matrix_memory import MAX_DECAY, delta_write, delta_write_reference
from lact_qwen.modeling import load_memory_config, load_memory_weights, save_memory


def test_block_write_equals_token_by_token():
    torch.manual_seed(0)
    b, length, dk, dv = 3, 70, 16, 12
    k = torch.nn.functional.normalize(torch.randn(b, length, dk), dim=-1)
    u = torch.rand(b, length, dk).sqrt() * k
    z = torch.randn(b, length, dv)
    lam = -torch.rand(b, length, dk) * MAX_DECAY
    start = torch.randn(b, dk, dv)
    ref = delta_write_reference(start, u, z, lam)
    for block in (1, 7, 32, 256):
        torch.testing.assert_close(delta_write(start, u, z, lam, block), ref, rtol=1e-4, atol=1e-5)


def test_write_stores_and_overwrites_a_key():
    # full erase and write gates, no decay: reading a stored key returns its value, and writing
    # the same key again replaces the value instead of adding to it
    dk = dv = 8
    key = torch.eye(dk)[None, :3]
    val = torch.randn(1, 3, dv)
    state = delta_write(torch.zeros(1, dk, dv), key, val, torch.zeros(1, 3, dk))
    torch.testing.assert_close(key @ state, val)
    new = torch.randn(1, 1, dv)
    state = delta_write(state, key[:, :1], new, torch.zeros(1, 1, dk))
    torch.testing.assert_close(key[:, :1] @ state, new)
    torch.testing.assert_close(key[:, 1:] @ state, val[:, 1:])


def build(base, **kw):
    cfg = dict(num_heads=2, head_dim=32, chunk_size=8, core="matrix", gate_init=0.5, write_block=5, forget_rank=4)
    cfg.update(kw)
    model = copy.deepcopy(base)
    attach_memory(model, MemoryConfig(**cfg))
    return model


def test_matrix_memory_in_model(tmp_path):
    base = tiny_qwen3()
    set_attention_window(base, 6)
    # 4 attention heads x 16 = 2 memory heads x 32
    model = build(base, qkv_init="attention", qkv_act="none", inject="pre_o_proj", out_ref="attention")
    branch = model.model.layers[0].memory
    names = {n for n, _ in branch.named_parameters()}
    assert {"qkv.weight", "erase_proj.weight", "write_proj.weight", "forget_up.bias", "forget_scale", "gate"} <= names
    assert not names & {"w0", "w1", "w2", "lr_proj.weight", "o_proj.weight"}

    ids = torch.randint(0, 97, (2, 40))
    with torch.no_grad():
        out, ref = model(ids).logits, base(ids).logits
    # the state is zero until the first chunk (8 tokens) has been written
    torch.testing.assert_close(out[:, :8], ref[:, :8], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out[:, 8:], ref[:, 8:])

    # without writes the model is the base model
    segments = [(s, min(s + 8, 40)) for s in range(0, 40, 8)]
    model.lact_ctx.set_segments(segments, [False] * len(segments))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, ref, rtol=1e-5, atol=1e-5)
    model.lact_ctx.clear_segments()

    # causal, and a chunk does not read its own tokens: changing tokens 16.. leaves 0..23 alone
    other = ids.clone()
    other[:, 30:] = (other[:, 30:] + 1) % 97
    with torch.no_grad():
        torch.testing.assert_close(model(other).logits[:, :30], out[:, :30], rtol=1e-5, atol=1e-5)

    freeze_base(model)
    model(ids).logits.sum().backward()
    for n, p in branch.named_parameters():
        # forget_up starts at zero, so nothing reaches forget_down on the first step
        assert p.grad is not None and (p.grad.abs().sum() > 0 or n == "forget_down.weight"), n
    assert model.model.layers[0].self_attn.o_proj.weight.grad is None

    save_memory(model, str(tmp_path))
    fresh = copy.deepcopy(base)
    attach_memory(fresh, load_memory_config(str(tmp_path))[0])
    load_memory_weights(fresh, str(tmp_path))
    with torch.no_grad():
        torch.testing.assert_close(fresh(ids).logits, out, rtol=1e-5, atol=1e-5)


def test_matrix_memory_turn_by_turn_matches_one_pass():
    base = tiny_qwen3()
    model = build(base, inject="post")
    with torch.no_grad():
        for layer in model.model.layers:
            layer.memory.o_proj.weight.normal_(std=0.2)
    ids = torch.randint(0, 97, (1, 24))
    turns = [(0, 8), (8, 16), (16, 24)]
    ctx = model.lact_ctx
    ctx.set_segments(turns)
    with torch.no_grad():
        full = model(ids).logits
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    ctx.start_session()
    pieces = []
    with torch.no_grad():
        for s, e in turns:
            ctx.set_segments([(0, e - s)])
            out = model(ids[:, s:e], past_key_values=cache, use_cache=True, position_ids=torch.arange(s, e)[None])
            pieces.append(out.logits)
    ctx.end_session()
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-4, atol=1e-4)
