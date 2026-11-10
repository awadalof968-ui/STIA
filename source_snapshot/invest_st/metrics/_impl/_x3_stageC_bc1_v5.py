# -*- coding: utf-8 -*-
"""BC1 v5 条件指标（包内自包含版本）。

从 scripts/_x3_stageC_bc1_v5.py 提取的纯函数子集（B2 标准欧氏 ASW、
B6 stratum 明细 + spot-weighted/equal-stratum 聚合），供 invest_st.metrics
包内使用；不依赖任何项目脚本或数据路径。
"""
import numpy as np

from ._x3_metrics import ilisi_np


def euclid_silhouette(X, labels):
    """标准欧氏距离 silhouette（手写，避免 sklearn/OpenMP 崩溃；B2）。"""
    n = len(X)
    if n < 2:
        return 0.0
    labs = np.unique(labels)
    d2 = np.zeros((n, n), dtype=np.float64)
    for c in range(X.shape[1]):
        diff = X[:, c][:, None] - X[:, c][None, :]
        d2 += diff * diff
    d = np.sqrt(np.maximum(d2, 0.0))
    s = np.zeros(n)
    for i in range(n):
        li = labels[i]
        same = labels == li
        if same.sum() <= 1:
            continue
        a_i = d[i, same].sum() / (same.sum() - 1)
        b_i = np.inf
        for l in labs:
            if l == li:
                continue
            m = labels == l
            b_i = min(b_i, d[i, m].mean())
        s[i] = (b_i - a_i) / max(b_i, a_i)
    return float(s.mean())


def cond_ilisi_donor(latent, batch, layer, group, donor, max_per_stratum=1500):
    strata = []
    for lay in np.unique(layer):
        for grp in np.unique(group):
            m = (layer == lay) & (group == grp)
            if m.sum() < 100:
                continue
            b = batch[m]
            if len(np.unique(b)) < 2:
                continue
            idx = np.random.default_rng(0).choice(m.sum(), min(max_per_stratum, m.sum()), replace=False)
            v = ilisi_np(latent[m][idx], b[idx], donor_labels=donor[m][idx])
            if v is not None:
                strata.append({'layer': str(lay), 'group': str(grp), 'n_spots': int(m.sum()),
                               'n_donor': int(len(np.unique(donor[m]))), 'n_batch': int(len(np.unique(b))),
                               'ilisi': float(v), 'weight': int(m.sum())})
    if not strata:
        return None, strata
    spot_w = float(np.average([s['ilisi'] for s in strata], weights=[s['weight'] for s in strata]))
    equal_w = float(np.mean([s['ilisi'] for s in strata]))
    return {'spot_weighted': spot_w, 'equal_stratum': equal_w,
            'n_strata': len(strata), 'total_spots': int(sum(s['weight'] for s in strata)),
            'strata': strata}, strata


def cond_asw(latent, batch, layer, group, max_per_stratum=1500):
    strata = []
    for lay in np.unique(layer):
        for grp in np.unique(group):
            m = (layer == lay) & (group == grp)
            if m.sum() < 100:
                continue
            b = batch[m]
            if len(np.unique(b)) < 2:
                continue
            idx = np.random.default_rng(0).choice(m.sum(), min(max_per_stratum, m.sum()), replace=False)
            s = euclid_silhouette(latent[m][idx], b[idx])
            strata.append({'layer': str(lay), 'group': str(grp), 'n_spots': int(m.sum()), 'asw': float(s)})
    if not strata:
        return None, strata
    spot_w = float(np.average([s['asw'] for s in strata], weights=[s['n_spots'] for s in strata]))
    equal_w = float(np.mean([s['asw'] for s in strata]))
    return {'spot_weighted': spot_w, 'equal_stratum': equal_w,
            'n_strata': len(strata), 'total_spots': int(sum(s['n_spots'] for s in strata)),
            'strata': strata}, strata
