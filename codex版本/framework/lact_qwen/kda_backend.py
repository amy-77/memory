"""Optional FLA KDA adapter for the existing symmetric MemT write recurrence.

KDA: S = D*S + beta*k*(v - k.T*D*S). Set k=u, v=z, beta=1.
Do NOT normalize k again: u already contains the per-channel sqrt(erase) gate.
Ignore the tokenwise KDA read; MemT reads the state from BEFORE a storage step.
No upstream kernel source is copied here. See REFERENCES.md for the pinned FLA revision.
"""
import os

import torch


def kda_write(state, u, z, lam):
    if not state.is_cuda:
        raise ValueError("fla_kda requires CUDA; use the torch matrix backend on CPU")
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise ValueError("fla_kda requires TRITON_F32_DEFAULT=ieee before starting Python; default TF32 exceeded validation tolerance")
    if state.dtype != torch.float32 or u.dtype != torch.float32 or z.dtype != torch.float32:
        raise ValueError("the FLA adapter preserves FP32 state and write inputs")
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise ImportError("install the pinned optional FLA dependency in requirements-kda.txt") from exc
    # Existing layout combines batch and heads; each head is still independent.
    key = u.transpose(0, 1).unsqueeze(0).contiguous()
    value = z.transpose(0, 1).unsqueeze(0).contiguous()
    decay = lam.transpose(0, 1).unsqueeze(0).contiguous()
    query = torch.zeros_like(key)
    beta = torch.ones(key.shape[:-1], dtype=key.dtype, device=key.device)
    _, final_state = chunk_kda(
        q=query, k=key, v=value, g=decay, beta=beta, scale=1.0,
        initial_state=state.unsqueeze(0).contiguous(), output_final_state=True,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    )
    return final_state.squeeze(0)
