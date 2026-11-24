"""Optional FLA KDA adapter for the existing symmetric MemT write recurrence.

KDA: S = D*S + beta*k*(v - k.T*D*S). Set k=u, v=z, beta=1.
Do NOT normalize k again: u already contains the per-channel sqrt(erase) gate.
Ignore the tokenwise KDA read; MemT reads the state from BEFORE a storage step.
No upstream kernel source is copied here. See REFERENCES.md for the pinned FLA revision.
"""
from functools import lru_cache
import importlib
import os

import torch


@lru_cache(maxsize=1)
def _load_kda_fp32():
    try:
        from fla.ops.kda import chunk_kda
        import triton.language as tl
    except ImportError as exc:
        raise ImportError("install the pinned optional FLA dependency in requirements-kda.txt") from exc
    intra = importlib.import_module("fla.ops.kda.chunk_intra")
    precision = getattr(intra, "SOLVE_TRIL_DOT_PRECISION", None)
    if getattr(precision, "value", None) not in ("tf32", "ieee"):
        raise RuntimeError("unsupported FLA precision policy; reinstall the pinned requirements-kda.txt revision")
    # The pinned revision explicitly uses TF32 for the inverse merge even when
    # TRITON_F32_DEFAULT=ieee. That caused state/gradient regressions. Configure
    # this constant before the first KDA compilation; Triton includes it in the
    # JIT cache key. This changes precision for KDA in this process, not weights
    # or the recurrence, and never modifies installed third-party files.
    intra.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    return chunk_kda


def kda_write(state, u, z, lam):
    if not state.is_cuda:
        raise ValueError("fla_kda requires CUDA; use the torch matrix backend on CPU")
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise ValueError("fla_kda requires TRITON_F32_DEFAULT=ieee before starting Python; default TF32 exceeded validation tolerance")
    if any(x.dtype != torch.float32 for x in (state, u, z, lam)):
        raise ValueError("the FLA adapter preserves FP32 state and write inputs")
    chunk_kda = _load_kda_fp32()
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
        chunk_size=64,
    )
    return final_state.squeeze(0)
