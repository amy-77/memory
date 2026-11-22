import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import tiny_qwen3
from lact_qwen import MemoryConfig, attach_memory, freeze_base, set_attention_window
from lact_qwen.distill import ce_and_kl
from lact_qwen.post_data import encode_post_row, supervised_token_scale


def test_topk_teacher_only_normalization_and_response_mask():
    torch.manual_seed(7)
    student = torch.randn(1, 6, 11, requires_grad=True)
    teacher = torch.randn(1, 6, 11)
    labels = torch.tensor([[-100, -100, -100, 2, 5, 7]])
    _, loss = ce_and_kl(student, teacher, torch.nn.Identity(), labels, chunk=2, top_k=3, temperature=1.7)
    s = student[0, 2:5] / 1.7
    t, idx = (teacher[0, 2:5] / 1.7).topk(3, dim=-1)
    log_q = t.log_softmax(-1)
    expected = 1.7**2 * (log_q.exp() * (log_q - s.log_softmax(-1).gather(-1, idx))).sum(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert student.grad[0, :2].count_nonzero() == 0
    assert student.grad[0, 5].count_nonzero() == 0
    outside = torch.ones_like(student.grad[0, 2:5], dtype=torch.bool).scatter_(-1, idx, False)
    assert (student.grad[0, 2:5][outside] > 0).all()


def test_post_row_masks_history_query_and_keeps_storage_differentiable():
    row = dict(id="train-1", history_ids=list(range(16)), query_ids=[20, 21],
               response_ids=[22, 23], teacher_generated=True)
    b = encode_post_row(row, 8, 32)
    assert b["labels"][0, :18].eq(-100).all()
    assert b["labels"][0, 18:].tolist() == [22, 23]
    assert b["segments"] == [(0, 8), (8, 16), (16, 20)]
    assert b["update_flags"] == [True, True, False]
    short = encode_post_row(dict(row, history_ids=list(range(13))), 8, 32)
    assert short["segments"] == [(0, 8), (8, 13), (13, 17)] and short["update_flags"] == [True, True, False]
    bad = dict(row, query_ids=list(range(8)))
    with pytest.raises(ValueError):
        encode_post_row(bad, 8, 32)
    # Two DDP ranks with unequal response lengths must form one token mean.
    scales = [supervised_token_scale(torch.tensor(n), torch.tensor(8), 2) for n in (2, 6)]
    assert (scales[0] * 3 + scales[1] * 7) / 2 == (2 * 3 + 6 * 7) / 8


def test_segment_attention_no_history_and_streaming_state_only():
    torch.manual_seed(4)
    base = tiny_qwen3()
    set_attention_window(base, 8, sinks=0, segment=8)
    ids = torch.randint(0, 97, (1, 24))
    changed = ids.clone()
    changed[:, :8] = (changed[:, :8] + 13) % 97
    with torch.no_grad():
        # No sink or sliding-window access to an earlier segment.
        torch.testing.assert_close(base(ids).logits[:, 8:], base(changed).logits[:, 8:], atol=1e-6, rtol=1e-6)
    model = copy.deepcopy(base)
    attach_memory(model, MemoryConfig(num_heads=4, head_dim=16, core="matrix", chunk_size=8,
        qkv_init="attention", qkv_act="none", inject="pre_o_proj", out_ref="none",
        out_scale="memt", output_gate_rank=16, forget_rank=16, write_block=4))
    model.eval()
    with torch.no_grad():
        packed = model(ids).logits
        model.lact_ctx.start_session()
        chunks = []
        for s in range(0, 24, 8):
            # Only the current segment is submitted; there is no past_key_values object.
            chunks.append(model(ids[:, s:s + 8], use_cache=False,
                                position_ids=torch.arange(s, s + 8)[None]).logits)
        assert all(isinstance(v, torch.Tensor) for v in model.lact_ctx.states.values())
        model.lact_ctx.end_session()
        torch.testing.assert_close(torch.cat(chunks, 1), packed, atol=2e-5, rtol=2e-5)
    freeze_base(model)
    loss = F.cross_entropy(model(ids).logits[:, -4:-1].reshape(-1, 97), ids[:, -3:].reshape(-1))
    loss.backward()
    for layer in model.model.layers:
        branch = layer.memory
        for name in ("qkv.weight", "erase_proj.weight", "write_proj.weight", "output_gate_up.weight"):
            grad = dict(branch.named_parameters())[name].grad
            assert grad is not None and grad.abs().sum() > 0, name
        assert layer.self_attn.q_proj.weight.grad is None
