"""Optional CUDA regression: original memory versus KDA, including model gradients."""
import pytest
import torch

from lact_qwen.kda_backend import kda_write
from lact_qwen.matrix_memory import delta_write
from lact_qwen.memory_branch import MemoryConfig
from lact_qwen.modeling import attach_memory, freeze_base, set_attention_window
from lact_qwen.training_loss import MemoryTrainingLoss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the optional FLA backend")


def test_cuda_kda_state_and_cross_step_gradient():
    pytest.importorskip("fla")
    torch.manual_seed(5)
    state = torch.randn(2, 128, 128, device="cuda") * .1
    u = torch.nn.functional.normalize(torch.randn(2, 256, 128, device="cuda"), dim=-1) * .7
    z = torch.randn_like(u) * .1
    lam = -torch.rand_like(u) * .03
    inputs = [x.requires_grad_() for x in (state, u, z, lam)]
    probe = torch.randn_like(state)
    def evaluate(write):
        s = write(state, u[:, :128], z[:, :128], lam[:, :128])
        result = write(s, u[:, 128:], z[:, 128:], lam[:, 128:])
        return result, torch.autograd.grad((result * probe).sum(), inputs)
    expected, expected_grads = evaluate(delta_write)
    actual, actual_grads = evaluate(kda_write)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-3)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-3)


def test_cuda_model_with_bf16_and_activation_checkpointing():
    pytest.importorskip("fla")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(5)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=128, attn_implementation="eager")).to("cuda", torch.bfloat16)
    set_attention_window(model, 128, 0, 128)
    cfg = MemoryConfig(core="matrix", num_heads=2, head_dim=128, chunk_size=128,
        write_block=64, qkv_init="attention", qkv_act="none", inject="pre_o_proj",
        out_scale="memt", forget_rank=8, output_gate_rank=8, stable_matrix_gates=True)
    attach_memory(model, cfg)
    freeze_base(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    wrapper = MemoryTrainingLoss(model, chunk=32)
    ids = torch.randint(1, 97, (1, 256), device="cuda")
    def evaluate(backend):
        cfg.matrix_backend = backend
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, _ = wrapper(ids, ids)
        ce.backward()
        return ce.detach(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
    expected_loss, expected_grads = evaluate("torch")
    actual_loss, actual_grads = evaluate("fla_kda")
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=1e-4)
    assert actual_grads.keys() == expected_grads.keys()
    for name in expected_grads:
        # Backbone BF16 arithmetic can amplify tiny kernel differences near rounding boundaries.
        torch.testing.assert_close(actual_grads[name], expected_grads[name], atol=5e-5, rtol=1e-2, msg=name)
