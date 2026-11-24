"""Inference that passes only the current chunk to Qwen, with no attention KV cache."""
from dataclasses import fields, is_dataclass
import torch


def state_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(state_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(state_bytes(v) for v in value)
    if is_dataclass(value):
        return sum(state_bytes(getattr(value, f.name)) for f in fields(value))
    if value is None or isinstance(value, (bool, str, float, int)):
        return 0
    raise TypeError('Unexpected persistent memory object: '+str(type(value)))


def begin_stream(model, chunk):
    cfg = model.lact_cfg
    if cfg.chunk_size != chunk or model.config.attention_segment != chunk:
        raise ValueError('attention block and memory chunk must match the input chunk')
    if getattr(model.config, 'attention_sinks', 0) != 0:
        raise ValueError('Strict streaming requires sinks=0')
    if cfg.replay.mode != 'none':
        raise ValueError('Strict streaming does not retain raw replay tokens')
    model.lact_ctx.start_session()


def forward_chunk(model, ids, offset, update=True, reset=False):
    ctx = model.lact_ctx
    if reset:
        ctx.states = {}
    if not ctx.session or ids.shape[1] > model.lact_cfg.chunk_size:
        raise ValueError('Call begin_stream and pass only one chunk')
    ctx.position_offset = offset
    ctx.set_segments([(0, ids.shape[1])], [update])
    result = model(input_ids=ids, position_ids=torch.arange(offset, offset+ids.shape[1], device=ids.device)[None],
                   past_key_values=None, use_cache=False)
    if getattr(result, 'past_key_values', None) is not None:
        raise AssertionError('Unexpected attention KV cache')
    return result.logits


def end_stream(model):
    model.lact_ctx.end_session()
    model.lact_ctx.clear_segments()
