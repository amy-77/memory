# 溢出场景：2048 个 (k,v) 流过一个 128x128 线性记忆，其中 64 个是"证据"，散落在前半段
# 看不同"容量分配"方式下，证据还能读回多少（cos 越接近 1 越好）
import numpy as np
rng = np.random.default_rng(0); d, n, m = 128, 2048, 64
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
res = {}
for seed in range(5):
    K, V = unit(rng.standard_normal((n, d))), unit(rng.standard_normal((n, d)))
    ev = np.sort(rng.choice(n // 2, m, replace=False))           # 证据位置
    imp = np.zeros(n); imp[ev] = 1
    def cos(S): R = K[ev] @ S; return float(np.mean(np.sum(R * V[ev], 1) / (np.linalg.norm(R, axis=1) + 1e-9)))
    def delta(mask=None, decay=1.0, keep_decay=False):
        S = np.zeros((d, d))
        for t in range(n):
            a = 1.0 if mask is None else mask[t]
            if a > 0 or keep_decay: S *= decay                    # 跳过的 token 是否仍然衰减
            if a > 0: S += a * np.outer(K[t], V[t] - S.T @ K[t])
        return S
    def wls(wt, lam=1e-3):                                         # 加权岭回归（MesaNet 式精确解）
        H = (K * wt[:, None]).T @ K + lam * np.eye(d); G = (K * wt[:, None]).T @ V
        return np.linalg.solve(H, G)
    arms = {
        "delta 全写":                 delta(),
        "delta 全写 + 衰减0.999":      delta(decay=0.999),
        "delta 只写证据":              delta(imp),
        "delta 只写证据(跳过仍衰减)":   delta(imp, decay=0.999, keep_decay=True),
        "delta 只写证据(跳过不衰减)":   delta(imp, decay=0.999),
        "最小二乘 全写(均匀分配)":      wls(np.ones(n)),
        "最小二乘 证据权重1/其余0.01":  wls(np.where(imp > 0, 1.0, 0.01)),
        "最小二乘 只写证据":            wls(imp),
    }
    for k, S in arms.items(): res.setdefault(k, []).append(cos(S))
for k, v in res.items(): print(f"{k:<28s} {np.mean(v):.2f}")
