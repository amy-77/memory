# vLLM 0.20 imports flash_attn.ops.triton.rotary whenever a package named flash_attn exists; on h20-1 flash_attn is
# only a namespace holding FA4 "cute" (no ops), so the import fails. Hide flash_attn from find_spec for the judge.
import importlib.util as _u
_orig = _u.find_spec
def find_spec(name, package=None):
    if name == "flash_attn" or name.startswith("flash_attn."):
        return None
    return _orig(name, package)
_u.find_spec = find_spec
