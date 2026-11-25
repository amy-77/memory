"""让针对 transformers 5.12.1 写的 modeling_memt.py 在更新的 transformers(已测 5.20)上运行,
并让 backbone 的 GDN 层在 GPU 上用 FLA 的 kernel。必须在 import modeling_memt 之前调用 ``apply()``。

modeling_memt.py 在 import 时从 qwen3_5_moe 模块取这几个名字,并自己按张量所在的设备选择实现
(``_device_dispatch``:张量在 CUDA 上且有加速版就用加速版,否则用纯 PyTorch 版):

    torch_chunk_gated_delta_rule     / chunk_gated_delta_rule            纯 PyTorch 版 / FLA 版(没有时为 None)
    torch_recurrent_gated_delta_rule / fused_recurrent_gated_delta_rule
    torch_causal_conv1d_update       / causal_conv1d_update

transformers 5.20 的变化:
- 没有单独的 FLA 版名字了。``torch_*`` 这两个函数外面包了一层分发:只要装了 FLA 就调 FLA,
  连 CPU 上的张量也是,而 FLA 的 kernel 只能在 GPU 上跑(CPU 上会报 Triton 没有可用的 driver)。
- ``torch_causal_conv1d_update`` 改名为 ``causal_conv1d_update``。

这里恢复 5.12.1 的约定:``torch_*`` 按设备分发(CUDA 张量走 transformers 的分发层,也就是装了 FLA 就用 FLA;
CPU 张量走纯 PyTorch 版),单独的加速版名字设为 None。"""

import inspect


def _by_device(dispatching):
    """dispatching:transformers 5.20 的分发函数(装了 FLA 就调 FLA)。返回按设备选择实现的函数。"""
    plain = inspect.unwrap(dispatching)  # 最里层的纯 PyTorch 实现
    allowed = inspect.signature(plain).parameters

    def run(*args, **kwargs):
        tensor = args[0] if args else next(iter(kwargs.values()))
        if tensor.is_cuda:
            return dispatching(*args, **kwargs)
        return plain(*args, **{name: value for name, value in kwargs.items() if name in allowed})

    run.__name__ = getattr(plain, "__name__", "run")
    run.__wrapped__ = plain
    return run


def apply():
    """给 qwen3_5_moe 模块补上 modeling_memt.py 需要的名字。可以重复调用。返回该模块。"""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen

    if getattr(qwen, "_memt_compat_applied", False):
        return qwen
    if not hasattr(qwen, "torch_causal_conv1d_update"):
        qwen.torch_causal_conv1d_update = qwen.causal_conv1d_update
    for torch_name, fast_name in (
        ("torch_chunk_gated_delta_rule", "chunk_gated_delta_rule"),
        ("torch_recurrent_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    ):
        if hasattr(qwen, fast_name):  # 5.12.1:名字都在,不用改
            continue
        setattr(qwen, torch_name, _by_device(getattr(qwen, torch_name)))
        setattr(qwen, fast_name, None)
    qwen._memt_compat_applied = True
    return qwen


def gdn_backend():
    """backbone 的 GDN 层在 GPU 上实际会用哪种实现:"fla" 或 "torch"(给日志和测试用)。"""
    import torch

    try:
        import fla.ops.gated_delta_rule  # noqa: F401
    except Exception:  # noqa: BLE001
        return "torch"
    return "fla" if torch.cuda.is_available() else "torch"
