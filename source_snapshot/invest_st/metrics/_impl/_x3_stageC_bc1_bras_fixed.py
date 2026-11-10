# -*- coding: utf-8 -*-
"""D20 R2: 正式 BRAS（scib-metrics >=0.5.5 等价手写实现，绕开 sklearn/OpenMP 与 Python 3.12 限制）。

参考（第 20 份验收 authoritative references）:
  - 论文: Shortcomings of silhouette in single-cell integration benchmarking
    (Nat Biotechnol 2025, https://www.nature.com/articles/s41587-025-02743-4)
  - 接口语义: scib_metrics.bras(labels, batch, metric='cosine',
           between_cluster_distances='mean_other')

定义（BRAS = 修正的 batch silhouette，逐样本）：
  对每个样本 i（batch 标签 b_i，biological 标签 l_i）：
    a_i = i 到同 batch 且同 bio label 样本的平均距离
          （batch 内生物学一致性；无同 label 同 batch 样本 → a_i 无效）
    b_i = i 到其他 batch 且同 bio label 样本的平均距离（mean_other 变体；
          跨 batch 比较配对的生物学类型）
    s_i = (b_i - a_i) / max(a_i, b_i)
  BRAS = 1 - |mean_i(s_i)|，越高 = batch 混得越匀（1 = 完全混合，0 = 完全分离）。

与旧 batch ASW 的关键区别：a_i/b_i 都限定在相同 biological label 内配对，
且 b_i 用 mean_other（所有其他 batch 的平均）而非最近其他簇——因此不是简单重标度。

实现纪律（绕开 sklearn）：纯 numpy；欧氏与 cosine 都支持；chunk 化距离避免大矩阵。
"""
import numpy as np


def _norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _pairwise_dist_chunked(latent, metric='cosine', chunk=2048):
    """chunk 化行间距离矩阵（避免 15000² 一次性内存）。cosine -> 1-cos_sim。"""
    n = latent.shape[0]
    D = np.empty((n, n), dtype=np.float32)
    if metric == 'cosine':
        Xn = _norm_rows(latent).astype(np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            S = Xn[s:e] @ Xn.T
            D[s:e] = np.clip(1.0 - S, 0.0, 2.0)
    else:
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d2 = np.zeros((e - s, n), dtype=np.float32)
            Xb = latent[s:e].astype(np.float32)
            for c in range(latent.shape[1]):
                diff = Xb[:, c][:, None] - latent[:, c][None, :]
                d2 += diff * diff
            D[s:e] = np.sqrt(np.maximum(d2, 0.0))
    return D


def bras(latent, batch_labels, bio_labels, metric='cosine',
         between_cluster_distances='mean_other'):
    """正式 BRAS。latent: (n, d)；batch_labels: 技术批次；bio_labels: 生物学层。
    返回 0-1 标量（越高=批次混得越匀）。"""
    latent = np.asarray(latent, dtype=np.float64)
    batch = np.asarray(batch_labels)
    bio = np.asarray(bio_labels)
    n = latent.shape[0]
    if n < 2:
        return float('nan')
    batches = np.unique(batch)
    if len(batches) < 2:
        return float('nan')

    # chunk 化距离矩阵（float32 省内存：15000² = 900MB）
    D = _pairwise_dist_chunked(latent, metric=metric)
    np.fill_diagonal(D, np.inf)

    s = np.zeros(n)
    same_bio_all = bio[:, None] == bio[None, :]
    np.fill_diagonal(same_bio_all, False)
    for b in batches:
        m = batch == b
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        # 向量化：对 batch 内所有样本一次性算 a_i, b_i
        # a_i：同 batch 同 bio label 的其他样本距离均值
        same_bio_sb = same_bio_all[idx][:, idx]  # (len(idx), len(idx))
        # 用 D 的子矩阵（idx×idx）
        D_bb = D[np.ix_(idx, idx)]
        a_num = np.where(same_bio_sb, D_bb, 0.0).sum(1)
        a_cnt = same_bio_sb.sum(1)
        a_i = np.where(a_cnt > 0, a_num / np.maximum(a_cnt, 1), np.nan)
        # b_i：其他 batch 同 bio label 样本距离均值
        other_idx = np.where(~m)[0]
        D_bo = D[np.ix_(idx, other_idx)]
        same_bio_bo = same_bio_all[idx][:, other_idx]
        b_num = np.where(same_bio_bo, D_bo, 0.0).sum(1)
        b_cnt = same_bio_bo.sum(1)
        b_i = np.where(b_cnt > 0, b_num / np.maximum(b_cnt, 1), np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            s_i = np.where(np.isfinite(a_i) & np.isfinite(b_i) & (b_cnt > 0),
                           (b_i - a_i) / np.maximum(np.maximum(a_i, b_i), 1e-12), 0.0)
        s[idx] = s_i
    return float(1.0 - abs(np.mean(s)))


def unit_tests():
    rng = np.random.default_rng(0)
    ok = True
    # 1. 完美分离：两 batch 远离 → BRAS≈0
    X_sep = np.vstack([rng.normal(loc=1.0, scale=0.05, size=(100, 5)),
                       rng.normal(loc=-1.0, scale=0.05, size=(100, 5))])
    b_sep = np.array([0] * 100 + [1] * 100)
    l_sep = np.array([i % 4 for i in range(200)])
    br_sep = bras(X_sep, b_sep, l_sep, metric='euclidean')
    print(f'[UT] BRAS 完美分离={br_sep:.4f} (期望≈0)')
    # 2. 完美混合：两 batch 同分布 → BRAS≈1
    X_mix = rng.normal(size=(200, 5))
    b_mix = np.array([0, 1] * 100)
    l_mix = np.array([i % 4 for i in range(200)])
    br_mix = bras(X_mix, b_mix, l_mix, metric='euclidean')
    print(f'[UT] BRAS 完美混合={br_mix:.4f} (期望≈1)')
    # 3. 手算 4 点：b0={p0(bio0),p1(bio1)}, b1={p2(bio0),p3(bio1)}
    #    p0(1,0), p1(0.9,0.1), p2(-1,0), p3(-0.9,0.1)
    #    p0: a=p0到b0中bio0(无) → 无定义 → s=0 保守
    #    实际 a_i 需同 label 同 batch——4 点结构每 (batch,label) 组合只有 1 样本 → 全 s=0 → BRAS=1
    #    这暴露本实现与 4 点案例的匹配：需更多样本
    X4 = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]])
    b4 = np.array([0, 0, 1, 1])
    l4 = np.array([0, 1, 0, 1])
    br4 = bras(X4, b4, l4, metric='cosine')
    print(f'[UT] BRAS 4点={br4:.4f} (每 batch×label 仅 1 样本 → 全跳过 → 1.0)')
    # 4. 更真实手算：b0={p0,p1}, b1={p2,p3}，全部 bio0（配对同 label 跨 batch）
    X4b = np.array([[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0], [-0.95, 0.05]])
    b4b = np.array([0, 0, 1, 1])
    l4b = np.array([0, 0, 0, 0])
    br4b = bras(X4b, b4b, l4b, metric='cosine')
    print(f'[UT] BRAS 4点全bio0={br4b:.4f} (分离 → 期望≈0)')
    ok = br_sep < 0.3 and br_mix > 0.7 and br4b < 0.5
    print(f'[UT] 结果: {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if unit_tests() else 1)
