"""检查 bf16 训练时记忆支路的参数有没有被真正更新。

LiveMem 的 train_sft.py 把所有可训练参数（包括 fla 原本保留为 fp32 的 A_log、dt_bias）都转成 bf16，
AdamW 也没有 fp32 主权重。每步更新量约等于学习率；如果参数附近 bf16 相邻两个数的间隔的一半
比更新量还大，更新会被四舍五入掉，参数永远停在原值。

用法：
  python check_bf16_frozen.py 早的检查点 晚的检查点 [--lr 5e-5]
检查点可以是 trainable.pt（torch.save 的 {名字: 张量}），也可以是含 model.safetensors 的目录。
只看名字里带 ".self_attn.mem." 的张量，按参数类型（去掉层号）汇总。
"""
import argparse
import collections
import math
from pathlib import Path
import re

import torch


def load(path):
    path = Path(path)
    if path.is_dir():
        from safetensors import safe_open
        f = safe_open(str(path / "model.safetensors"), "pt")
        return {k: f.get_tensor(k) for k in f.keys() if ".self_attn.mem." in k}
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    return {k: v for k, v in sd.items() if ".self_attn.mem." in k}


def half_spacing(x):
    """bf16 下 |x| 附近相邻两个可表示数间隔的一半：|x| 在 [2^e, 2^(e+1)) 时间隔为 2^(e-7)。"""
    x = x.float().abs()
    e = torch.floor(torch.log2(x.clamp(min=1e-38)))
    return torch.where(x > 0, 2.0 ** (e - 8), torch.zeros_like(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("early")
    p.add_argument("late")
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()
    a, b = load(args.early), load(args.late)
    common = sorted(set(a) & set(b))
    assert common, "两个检查点没有共同的记忆支路张量"
    g = collections.defaultdict(lambda: {"diff2": 0.0, "norm2": 0.0, "n": 0, "changed": 0, "stuck": 0, "dtype": None})
    for k in common:
        name = re.sub(r"\.\d+\.", ".", k.split(".self_attn.mem.")[1])
        x, y = a[k], b[k]
        d = (y.float() - x.float())
        s = g[name]
        s["diff2"] += float((d ** 2).sum())
        s["norm2"] += float((x.float() ** 2).sum())
        s["n"] += x.numel()
        s["changed"] += int((d != 0).sum())
        # 一步更新量约等于 lr：间隔的一半大于 lr 的元素，单步更新会被舍掉
        s["stuck"] += int((half_spacing(y) > args.lr).sum())
        s["dtype"] = str(y.dtype)
    print(f"{'参数':<22}{'dtype':<16}{'相对变化':>10}{'变了的元素':>12}{'更新会被舍掉的元素':>20}")
    for name, s in sorted(g.items(), key=lambda t: t[1]["diff2"] / max(t[1]["norm2"], 1e-30)):
        rel = math.sqrt(s["diff2"] / max(s["norm2"], 1e-30))
        print(f"{name:<22}{s['dtype']:<16}{rel:>10.2e}{s['changed'] / s['n']:>12.1%}{s['stuck'] / s['n']:>20.1%}")
    for name in ("A_log", "dt_bias"):
        ks = [k for k in common if k.endswith("mem." + name)]
        if ks:
            v = torch.cat([b[k].float().flatten() for k in ks])
            v = v.exp() if name == "A_log" else torch.nn.functional.softplus(v)
            label = "A=exp(A_log)" if name == "A_log" else "softplus(dt_bias)"
            print(f"晚的检查点 {label} 范围 {float(v.min()):.5f} – {float(v.max()):.5f}"
                  f"（初始化：A 在 [1, 16]，dt 在 [0.001, 0.1]）")


if __name__ == "__main__":
    main()
