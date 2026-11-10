# -*- coding: utf-8 -*-
"""D17 P0-2a: BC1 v6 补充批次指标（可复用模块，工具 I0/I1 直接消费）：
- cond_batch_decoder: donor-aware conditional batch decoder（LODO 最近中心解码准确率，
  复用 _x3_metrics.batch_decoding_accuracy；stratum=Layer×group 内计算，donor-aware）
- cond_graph_connectivity: scIB 风格 graph connectivity（latent 自建 kNN 图，stratum 内
  按 batch 计算最大连通分量占比；纯 numpy + 手写并查集，绕开 scipy/sklearn）
- 聚合与 v5 一致：spot_weighted primary + equal_stratum sensitivity + stratum 明细
- 单元测试：完美混合 / 完美分离 / 手算小样本
"""
import numpy as np


# ---------- 并查集 ----------
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _knn_edges(X, k, block=1000, seed=0):
    """纯 numpy kNN 图：返回边数组 (2, E)，无自环。每点取真正 k 个最近邻。"""
    n = X.shape[0]
    edges_src, edges_dst = [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        Xb = X[s:e]
        d = np.zeros((e - s, n), dtype=np.float32)
        for c in range(X.shape[1]):
            diff = Xb[:, c][:, None] - X[:, c][None, :]
            d += diff * diff
        for i in range(e - s):
            d[i, s + i] = np.inf  # 排除自身
            # 对称性：只取 j > i 的边，避免重复
            row = d[i].copy()
            row[: s + i + 1] = np.inf
        knn = np.argpartition(d, k, axis=1)[:, :k]
        for i in range(e - s):
            gi = s + i
            for j in knn[i]:
                if gi < j:
                    edges_src.append(gi)
                    edges_dst.append(int(j))
    return np.array([edges_src, edges_dst], dtype=np.int64) if edges_src else np.empty((2, 0), dtype=np.int64)


def graph_connectivity(latent, batch, k=20, seed=0):
    """scIB 风格 graph connectivity：
    latent 自建 kNN 图 → 对每个 batch，取其诱导子图（两端都在该 batch 的边）→
    最大连通分量占比 → 全部 batch 均值。值 0-1，越高 = 同 batch spots 在 latent 中越连通。
    单 batch 或 <2 点返回 None。"""
    b_arr = np.asarray(batch)
    bs = np.unique(b_arr)
    if len(bs) < 2:
        return None
    n = latent.shape[0]
    if n < 2:
        return None
    edges = _knn_edges(latent, k, seed=seed)
    if edges.shape[1] == 0:
        return None
    ratios = []
    for b in bs:
        m = np.where(b_arr == b)[0]
        if len(m) < 2:
            ratios.append(1.0)
            continue
        idx = {v: i for i, v in enumerate(m)}
        dsu = DSU(len(m))
        mask = np.isin(edges[0], m) & np.isin(edges[1], m)
        for a_, b_ in edges[:, mask].T:
            dsu.union(idx[int(a_)], idx[int(b_)])
        comp = {}
        for i in range(len(m)):
            r = dsu.find(i)
            comp[r] = comp.get(r, 0) + 1
        ratios.append(max(comp.values()) / len(m))
    return float(np.mean(ratios))


# ---------- conditional 聚合（与 v5 同构） ----------
def cond_metric_donor(latent, batch, layer, group, donor, metric_fn,
                      max_per_stratum=1500, seed=0, need_donor=True):
    """在 Layer×group stratum 内计算 metric_fn，输出 spot_weighted/equal_stratum/stratum 明细。
    metric_fn(lat_s, batch_s[, donor_s]) -> float 或 None。"""
    strata = []
    for lay in np.unique(layer):
        for grp in np.unique(group):
            m = (layer == lay) & (group == grp)
            if m.sum() < 100:
                continue
            b = batch[m]
            if len(np.unique(b)) < 2:
                continue
            idx = np.random.default_rng(seed).choice(m.sum(), min(max_per_stratum, m.sum()), replace=False)
            lat_s = latent[m][idx]
            b_s = b[idx]
            if need_donor:
                v = metric_fn(lat_s, b_s, donor[m][idx])
            else:
                v = metric_fn(lat_s, b_s)
            if v is None:
                continue
            strata.append({'layer': str(lay), 'group': str(grp), 'n_spots': int(m.sum()),
                           'n_donor': int(len(np.unique(donor[m]))), 'n_batch': int(len(np.unique(b))),
                           'value': float(v), 'weight': int(m.sum())})
    if not strata:
        return None, strata
    spot_w = float(np.average([s['value'] for s in strata], weights=[s['weight'] for s in strata]))
    equal_w = float(np.mean([s['value'] for s in strata]))
    return {'spot_weighted': spot_w, 'equal_stratum': equal_w,
            'n_strata': len(strata), 'total_spots': int(sum(s['weight'] for s in strata)),
            'strata': strata}, strata


def cond_batch_decoder(latent, batch, layer, group, donor, max_per_stratum=1500):
    """donor-aware conditional batch decoder：stratum 内 LODO 最近中心解码准确率。
    值越低 = 批次越不可预测 = 混匀越好（方向：lower is better for mixing）。"""
    from ._x3_metrics import batch_decoding_accuracy
    res, strata = cond_metric_donor(latent, batch, layer, group, donor,
                                    lambda lat, b, d: batch_decoding_accuracy(lat, b, d),
                                    max_per_stratum=max_per_stratum, need_donor=True)
    return res


def cond_graph_connectivity(latent, batch, layer, group, donor, k=20, max_per_stratum=1500):
    """conditional graph connectivity：stratum 内 scIB 风格连通性（donor 仅用于明细计数）。"""
    res, strata = cond_metric_donor(latent, batch, layer, group, donor,
                                    lambda lat, b: graph_connectivity(lat, b, k=k),
                                    max_per_stratum=max_per_stratum, need_donor=False)
    return res


# ---------- 单元测试 ----------
def unit_tests():
    rng = np.random.default_rng(0)
    ok = True

    # 1. graph connectivity：完美分离 → ≈1
    X_sep = np.vstack([rng.normal(loc=0, scale=0.1, size=(100, 5)),
                       rng.normal(loc=10, scale=0.1, size=(100, 5))])
    b_sep = np.array([0] * 100 + [1] * 100)
    gc_sep = graph_connectivity(X_sep, b_sep, k=10)
    print(f'[UT] graph 完美分离 GC={gc_sep:.4f} (期望≈1)')

    # 2. graph connectivity：同 batch 被打散成两团 → 最大连通分量 ≈ 一半 → GC≈0.75
    #    b0 的 100 点分裂成两团（相距 10），b1 一团 → mean((50/100, 100/100)) = 0.75
    X_split = np.vstack([rng.normal(loc=0, scale=0.1, size=(50, 5)),
                         rng.normal(loc=10, scale=0.1, size=(50, 5)),
                         rng.normal(loc=20, scale=0.1, size=(100, 5))])
    b_split = np.array([0] * 100 + [1] * 100)
    gc_split = graph_connectivity(X_split, b_split, k=10)
    print(f'[UT] graph 打散 GC={gc_split:.4f} (期望≈0.75)')

    # 3. decoder：完美分离 → 准确率≈1
    dc_sep = None
    from ._x3_metrics import batch_decoding_accuracy
    dc_sep = batch_decoding_accuracy(X_sep, b_sep, np.repeat(np.arange(4), 50))
    print(f'[UT] decoder 完美分离 acc={dc_sep:.4f} (期望≈1)')

    # 4. decoder：完美混合 → 准确率接近随机 (≈0.5)
    X_mix = rng.normal(size=(200, 5))
    b_mix = np.array([0, 1] * 100)
    dc_mix = batch_decoding_accuracy(X_mix, b_mix, np.repeat(np.arange(4), 50))
    print(f'[UT] decoder 完美混合 acc={dc_mix:.4f} (期望≈0.5)')

    # 5. 手算：6 点，batch 间距大（b0≈[0,2], b1≈[100,102]），donor 交叉持点且每 fold 两 batch 齐
    #    X=[0,1,2,100,101,102], b=[0,0,0,1,1,1], d=[0,1,0,1,0,1]
    #    fold donor0(train=点1/3/5, 中心 b0=1,b1=101): 点0(x=0)->b0✓, 点2(x=2)->b0✓, 点4(x=101)->b1✓
    #    fold donor1(train=点0/2/4, 中心 b0=1,b1=101): 点1(x=1)->b0✓, 点3(x=100)->b1✓, 点5(x=102)->b1✓
    #    acc = 1.0
    X6 = np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]])
    b6 = np.array([0, 0, 0, 1, 1, 1])
    d6 = np.array([0, 1, 0, 1, 0, 1])
    dc6 = batch_decoding_accuracy(X6, b6, d6)
    print(f'[UT] decoder 手算 acc={dc6:.4f} (期望=1.0)')

    ok = (gc_sep > 0.9) and (abs(gc_split - 0.75) < 0.08) and (dc_sep > 0.9) and (abs(dc_mix - 0.5) < 0.15) and (abs(dc6 - 1.0) < 1e-9)
    print(f'[UT] 结果: {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if unit_tests() else 1)
