# Third-party code

## MemT-1.0-Nano (Apache-2.0)

`memt_nano/modeling_memt_orig.py`, `memt_nano/configuration_memt.py` and the base of `memt_nano/modeling_memt.py`
are the inference code of the Hugging Face model `MemTensor/MemT-1.0-Nano`, released under the Apache License 2.0
(http://www.apache.org/licenses/LICENSE-2.0). `modeling_memt_orig.py` and `configuration_memt.py` are unchanged.

`modeling_memt.py` is modified. The changes:

- Hand-written backward used when gradients are needed, in `MemTRMSNormGated.forward`, `MemTMemory.forward` and the
  memory part of `MemTAttention.forward`. The original memory forward is kept unchanged as
  `MemTMemory.forward_reference`.
- New `commit_memory_train`.
- An out-of-place update of the transformers GDN cache while `commit_memory_train` runs.

`memt_autograd.py` and `tests/` are new.

Attribution carried by the original code:

    Portions of the inference code are adapted from Hugging Face Transformers and retain their upstream attribution:
    Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved. Apache License 2.0.

`_conv_state_out_of_place` and `_recurrent_state_out_of_place` in `modeling_memt.py` follow
`LinearAttentionLayer.update_conv_state` / `update_recurrent_state` of Hugging Face Transformers 5.20
(`cache_utils.py`, Apache License 2.0). The change is that they assign the new state instead of using `copy_`.
