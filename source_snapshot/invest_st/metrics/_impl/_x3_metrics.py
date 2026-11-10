# -*- coding: utf-8 -*-
"""X3 三轴指标（numpy 手写，绕开 sklearn/OpenMP 崩溃）：
- kmeans_np: 纯 numpy k-means（不碰 scipy.linalg/threadpoolctl）
- ari_np: adjusted Rand index（contingency 手写）
- batch_decoding_accuracy: 每 spot 到 proc_batch 中心最近邻解码准确率（描述性，注明泄漏限制）
"""
import numpy as np


def kmeans_np(X, k, iters=60, seed=0, tol=1e-6, n_init=3):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best = None
    best_inertia = np.inf
    for init in range(n_init):
        idx = rng.choice(n, k, replace=False)
        centers = X[idx].copy()
        labels = np.zeros(n, dtype=int)
        inertia = np.inf
        for it in range(iters):
            d = np.zeros((n, k))
            for c in range(k):
                diff = X - centers[c]
                d[:, c] = np.einsum('ij,ij->i', diff, diff)
            new_labels = d.argmin(1)
            new_inertia = float(d[np.arange(n), new_labels].sum())
            if abs(new_inertia - inertia) < tol:
                break
            inertia = new_inertia
            labels = new_labels
            for c in range(k):
                m = labels == c
                if m.sum() > 0:
                    centers[c] = X[m].mean(0)
        if inertia < best_inertia:
            best_inertia = inertia
            best = labels.copy()
    return best


def contingency(yt, yp):
    """手写 contingency 表（避免 sklearn）"""
    a = np.unique(yt)
    b = np.unique(yp)
    tab = np.zeros((len(a), len(b)), dtype=np.int64)
    ia = {v: i for i, v in enumerate(a)}
    ib = {v: i for i, v in enumerate(b)}
    for t, p in zip(yt, yp):
        tab[ia[t], ib[p]] += 1
    return tab


def ari_np(yt, yp):
    tab = contingency(yt, yp)
    n = int(tab.sum())
    a = tab.sum(1)
    b = tab.sum(0)

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_comb = comb2(tab).sum()
    sum_a = comb2(a).sum()
    sum_b = comb2(b).sum()
    expected = sum_a * sum_b / comb2(n) if n > 1 else 0.0
    max_idx = 0.5 * (sum_a + sum_b)
    if max_idx - expected == 0:
        return 0.0
    return float((sum_comb - expected) / (max_idx - expected))


def batch_decoding_accuracy(latent, proc_batch, donor_labels=None):
    """spot 级最近中心解码 proc_batch 准确率（描述性敏感性指标；X3-E05 修复）：
    leave-one-donor-out 估中心——每个 donor 的 spots 用『排除该 donor』的
    其余 spots 估算的 batch 中心解码，消除训练-测试复用泄漏。
    donor_labels=None 时退化为每个 spot 独立 LOO（中心由其余 spots 估计）。
    若某 donor 是唯一持有某 batch 的供体（LODO 中心缺失），返回 None。"""
    bs = np.unique(proc_batch)
    if len(bs) < 2:
        return 1.0  # 单标签平凡结果（调用方负责 NA 判定）
    if donor_labels is None:
        donor_labels = np.arange(latent.shape[0])
    preds = np.empty(latent.shape[0], dtype=bs.dtype)
    for d in np.unique(donor_labels):
        m = donor_labels == d
        train = ~m
        if train.sum() < len(bs):
            return None  # LODO 中心不可估计
        # 审计 P1：每个 LODO fold 中每个 batch 至少有一个训练供体的样本
        for b in bs:
            if (train & (proc_batch == b)).sum() == 0:
                return None  # 该 fold 缺某 batch 的训练样本
        centers = np.vstack([latent[train & (proc_batch == b)].mean(0) for b in bs])
        dm = np.zeros((m.sum(), len(bs)))
        for c in range(len(bs)):
            diff = latent[m] - centers[c]
            dm[:, c] = np.einsum('ij,ij->i', diff, diff)
        preds[m] = bs[dm.argmin(1)]
    return float((preds == proc_batch).mean())


def recovery_slope(injected, recovered, min_n=3):
    """注入-恢复效应斜率（预注册 disease secondary）：
    recovered（供体级恢复效应，如 INSPIRE full 8-marker 供体均值）
    对 injected（供体级注入效应，如 donor_theta）的 OLS 回归斜率。
    斜率=1 表示单位注入完全恢复；=0 表示无恢复；样本 < min_n 返回 None。
    只报告表观读出尺度斜率，不解释为生物学恢复效率。"""
    x = np.asarray(injected, dtype=float)
    y = np.asarray(recovered, dtype=float)
    if x.shape != y.shape or len(x) < min_n:
        return None
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 0:
        return None
    yc = y - y.mean()
    return float((xc * yc).sum() / denom)


def nmi_np(yt, yp):
    """normalized mutual information（numpy 手写，基于 contingency）"""
    tab = contingency(yt, yp).astype(float)
    n = tab.sum()
    if n == 0:
        return 0.0
    pxy = tab / n
    px = pxy.sum(1)
    py = pxy.sum(0)
    mi = (pxy * np.log((pxy + 1e-12) / (px[:, None] * py[None, :] + 1e-12))).sum()
    hx = -(px * np.log(px + 1e-12)).sum()
    hy = -(py * np.log(py + 1e-12)).sum()
    denom = 0.5 * (hx + hy)
    return float(mi / denom) if denom > 0 else 0.0


def spatial_ari(latent, layer_labels, k=8, seed=0):
    """kmeans(k) 聚类 vs 层标签 的 ARI"""
    km = kmeans_np(latent, k, seed=seed)
    return ari_np(layer_labels, km)


def ilisi_np(latent, batch_labels, k=90, block=500, donor_labels=None):
    """normalized iLISI（X3-E01 修复 + 预注册'供体分层'语义）：
    每 spot 的 kNN（排除自身，真正 k 个邻居）中 batch 逆 Simpson 指数
    -> 均值 -> 归一化 (x-1)/(n_batch-1)。
    供体分层（预注册 note）：donor_labels 提供时，kNN 排除同供体 spots
    （避免 spot 随机拆分泄漏/同供体假混合）。返回 None 表示单一 batch。"""
    bs = np.unique(batch_labels)
    if len(bs) < 2:
        return None
    n = latent.shape[0]
    bcode = np.array([{b: i for i, b in enumerate(bs)}[b] for b in batch_labels])
    lisi_sum = 0.0
    for s in range(0, n, block):
        e = min(s + block, n)
        Xb = latent[s:e]
        d = np.zeros((e - s, n), dtype=np.float32)
        for c in range(latent.shape[1]):
            diff = Xb[:, c][:, None] - latent[:, c][None, :]
            d += diff * diff
        for i in range(e - s):
            d[i, s + i] = np.inf                       # 排除自身
            if donor_labels is not None:               # 供体分层：排除同供体
                same_d = donor_labels == donor_labels[s + i]
                d[i, same_d] = np.inf
        knn = np.argpartition(d, k, axis=1)[:, :k]     # 真正 k 个邻居
        for i in range(e - s):
            nb = bcode[knn[i]]
            counts = np.bincount(nb, minlength=len(bs)).astype(float)
            counts = counts[counts > 0]
            p = counts / counts.sum()
            lisi_sum += 1.0 / (p @ p)
    lisi_mean = lisi_sum / n
    return float((lisi_mean - 1.0) / (len(bs) - 1.0))


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 10))
    y = rng.integers(0, 4, 2000)
    km = kmeans_np(X, 4)
    print('ARI(随机) ≈ 0:', round(ari_np(y, km), 3))
    # 可分数据 → ARI ≈ 1（簇间距 10）
    X2 = np.vstack([rng.normal(loc=i * 10, size=(500, 5)) for i in range(4)])
    y2 = np.repeat(np.arange(4), 500)
    km2 = kmeans_np(X2, 4)
    print('ARI(可分) ≈ 1:', round(ari_np(y2, km2), 3))
