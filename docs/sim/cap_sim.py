# 线性联想记忆容量的小模拟：d_k=d_v=128，随机单位 key/value
import numpy as np
rng = np.random.default_rng(0)
d = 128
def unit(n): x = rng.standard_normal((n, d)); return x / np.linalg.norm(x, axis=1, keepdims=True)
def acc(S, K, V):
    R = K @ S                                   # 用 key 去读：r = S^T k
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
    return float((np.argmax(R @ V.T, 1) == np.arange(len(K))).mean())  # 读出最像哪个存过的 value
def cos(S, K, V):
    R = K @ S; return float(np.mean(np.sum(R * V, 1) / (np.linalg.norm(R, axis=1) + 1e-9)))
print(f"{'n':>5} | {'简单累加':>8} {'cos':>5} | {'delta':>6} {'cos':>5} | {'最优(最小二乘)':>8} {'cos':>5}")
for n in [16, 32, 64, 96, 128, 160, 256, 512, 1024]:
    K, V = unit(n), unit(n)
    S_h = K.T @ V                                # 线性 attention：S += k v^T
    S_d = np.zeros((d, d))
    for _ in range(1):
        for k, v in zip(K, V):                   # delta rule（beta=1，不衰减）
            S_d += np.outer(k, v - S_d.T @ k)
    S_ls = np.linalg.pinv(K) @ V                 # 一次性最优解
    print(f"{n:>5} | {acc(S_h,K,V):>8.3f} {cos(S_h,K,V):>5.2f} | {acc(S_d,K,V):>6.3f} {cos(S_d,K,V):>5.2f} | {acc(S_ls,K,V):>8.3f} {cos(S_ls,K,V):>5.2f}")
