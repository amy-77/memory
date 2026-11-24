import copy

import torch

from conftest import tiny_qwen3
from lact_qwen import MemoryConfig, ReplayConfig, attach_memory, freeze_base, memory_parameters, set_attention_window


def small_memory_cfg(**kw):
    base = dict(num_heads=2, head_dim=16, chunk_size=8, use_muon=True, use_momentum=True)
    base.update(kw)
    return MemoryConfig(**base)


def test_zero_gate_reproduces_base_model():
    base = tiny_qwen3()
    model = copy.deepcopy(base)
    set_attention_window(base, 6)
    set_attention_window(model, 6)
    attach_memory(model, small_memory_cfg())
    ids = torch.randint(0, 97, (2, 40))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, base(ids).logits)


def test_history_beyond_window_only_reaches_through_memory():
    model = tiny_qwen3()
    set_attention_window(model, 6)
    attach_memory(model, small_memory_cfg())
    ids = torch.randint(0, 97, (1, 48))
    ids2 = ids.clone()
    ids2[0, 0] = (ids[0, 0] + 1) % 97

    with torch.no_grad():
        a, b = model(ids).logits, model(ids2).logits
    torch.testing.assert_close(a[:, 20:], b[:, 20:])

    with torch.no_grad():
        for layer in model.model.layers:
            layer.memory.gate.fill_(1.0)
        a, b = model(ids).logits, model(ids2).logits
    assert (a[:, 20:] - b[:, 20:]).abs().max() > 1e-5


def test_turn_by_turn_session_matches_one_shot():
    model = tiny_qwen3()
    set_attention_window(model, 10)
    ctx = attach_memory(model, small_memory_cfg(use_muon=False))
    with torch.no_grad():
        for layer in model.model.layers:
            layer.memory.gate.fill_(1.0)

    ids = torch.randint(0, 97, (1, 30))
    turns = [(0, 7), (7, 19), (19, 30)]

    ctx.set_segments(turns)
    with torch.no_grad():
        full = model(ids).logits
    ctx.clear_segments()

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
    ctx.clear_segments()
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-4, atol=1e-4)


def test_attention_sinks_stay_visible_and_causal():
    model = tiny_qwen3()
    set_attention_window(model, 6, sinks=2)
    ids = torch.randint(0, 97, (1, 30))

    def logits(x):
        with torch.no_grad():
            return model(x, use_cache=False).logits

    def bump(pos):
        out = ids.clone()
        out[0, pos] = (ids[0, pos] + 1) % 97
        return out

    base = logits(ids)
    torch.testing.assert_close(logits(bump(2))[:, 20:], base[:, 20:])
    assert (logits(bump(1))[:, 20:] - base[:, 20:]).abs().max() > 1e-5
    torch.testing.assert_close(logits(bump(1))[:, :1], base[:, :1])


def test_attention_sinks_sdpa_matches_eager():
    eager = tiny_qwen3()
    sdpa = copy.deepcopy(eager)
    sdpa.config._attn_implementation = "sdpa"
    set_attention_window(eager, 6, sinks=2)
    set_attention_window(sdpa, 6, sinks=2)
    ids = torch.randint(0, 97, (2, 30))
    with torch.no_grad():
        torch.testing.assert_close(sdpa(ids, use_cache=False).logits, eager(ids, use_cache=False).logits)


def test_only_memory_parameters_train():
    model = tiny_qwen3()
    set_attention_window(model, 6)
    attach_memory(model, small_memory_cfg(replay=ReplayConfig(mode="gmm", num_samples=8, gmm_components=4)))
    freeze_base(model)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in memory_parameters(model)}

    model.train()
    ids = torch.randint(0, 97, (2, 40))
    loss = model(ids, labels=ids).loss
    loss.backward()
    grads = [p.grad for p in memory_parameters(model)]
    assert all(g is not None for g in grads)
    gate_grads = [p.grad for n, p in model.named_parameters() if n.endswith("memory.gate")]
    assert all(g.abs().sum() > 0 for g in gate_grads)
