import torch

from conftest import tiny_qwen3
from lact_qwen import MemoryConfig, attach_memory, set_attention_window
from lact_qwen.streaming import begin_stream, forward_chunk, end_stream, state_bytes


def test_formal_evaluator_streaming_matches_packed_and_resets():
    torch.manual_seed(19)
    model = tiny_qwen3()
    set_attention_window(model, 8, sinks=0, segment=8)
    attach_memory(model, MemoryConfig(num_heads=4, head_dim=16, core="matrix", chunk_size=8,
        qkv_init="attention", qkv_act="none", inject="pre_o_proj", out_ref="none",
        out_scale="memt", output_gate_rank=16, forget_rank=16, write_block=4))
    model.eval()
    ids = torch.randint(0, 97, (1, 24))
    with torch.no_grad():
        packed = model(ids).logits
        begin_stream(model, 8)
        actual, sizes = [], []
        for s in range(0, 24, 8):
            actual.append(forward_chunk(model, ids[:, s:s+8], s))
            sizes.append(state_bytes(model.lact_ctx.states))
        torch.testing.assert_close(torch.cat(actual, 1), packed, atol=2e-5, rtol=2e-5)
        assert len(set(sizes)) == 1 and sizes[0] > 0
        before = {k: v.clone() for k, v in model.lact_ctx.states.items()}
        forward_chunk(model, ids[:, :8], 24, update=False)
        for k, v in before.items():
            torch.testing.assert_close(v, model.lact_ctx.states[k], rtol=0, atol=0)
        reset = forward_chunk(model, ids[:, :8], 0, update=False, reset=True)
        end_stream(model)
        begin_stream(model, 8)
        fresh = forward_chunk(model, ids[:, :8], 0, update=False)
        torch.testing.assert_close(reset, fresh, rtol=0, atol=0)
        end_stream(model)
    assert model.lact_ctx.states == {} and not model.lact_ctx.session
