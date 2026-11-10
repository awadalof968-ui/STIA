# -*- coding: utf-8 -*-
"""invest_st.metrics.osi — Order-Sensitivity Index（D22b，E11 定义对齐）。

OSI 定义（E11_MULTI_ORDER_INVARIANCE_V1 / E12 独立验收）：
  对每个 condition×seed 单元，在 ≥4 个冻结 label-blind 输入顺序下分别拟合/得到 latent，
  对每个顺序计算 donor-keyed 的成对距离矩阵（Euclidean，对称，去对角），
  Frobenius 归一化后，OSI = 全部顺序对之间的最大 Frobenius 漂移。
  冻结 margin = 0.05；OSI > margin → 该单元 FAIL（顺序不变性门拒绝）。

用途：供体级整合结论被接受前，必须先通过此门——防止"有利顺序特权"支撑
顺序无关 claim（E8/E11 实证：INSPIRE 90/90 超 margin，含疾病零对照）。

实现纪律：纯 numpy；donor key 只用于对齐，不用于构造顺序；label-blind。
"""
import numpy as np


def donor_distance_matrix(latent, donor_keys):
    """donor-keyed 成对欧氏距离矩阵（对称，去对角，Frobenius 归一化）。
    latent: (n, d)；donor_keys: (n,) 与 latent 行对齐。返回 (n_donors, n_donors)。"""
    lat = np.asarray(latent, dtype=np.float64)
    keys = np.asarray(donor_keys)
    donors = sorted(set(keys), key=str)
    means = np.vstack([lat[keys == d].mean(0) for d in donors])
    nd = len(donors)
    D = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        diff = means - means[i]
        D[i] = np.sqrt((diff * diff).sum(1))
    np.fill_diagonal(D, 0.0)
    # Frobenius 归一化（去对角，避免 0 对角线主导）
    off = D[~np.eye(nd, dtype=bool)]
    n = np.sqrt((off * off).sum())
    if n > 1e-12:
        D = D / n
    return D


def frobenius_drift(D1, D2):
    """两个归一化距离矩阵的 Frobenius 漂移（对应元素差平方和开方）。"""
    d = D1 - D2
    return float(np.sqrt((d * d).sum()))


def osi(latents_by_order, donor_keys_by_order, margin=0.05):
    """顺序不变性指数：≥2 个顺序的 latent 集。
    latents_by_order: {order_name: (n,d) latent}
    donor_keys_by_order: {order_name: (n,) donor keys}
    返回 {osi, margin, pass, drifts: {pair: drift}, n_orders}。"""
    orders = sorted(latents_by_order.keys())
    if len(orders) < 2:
        return {'osi': None, 'margin': margin, 'pass': None,
                'drifts': {}, 'n_orders': len(orders), 'note': 'need >=2 orders'}
    Ds = {}
    for o in orders:
        Ds[o] = donor_distance_matrix(latents_by_order[o], donor_keys_by_order[o])
    drifts = {}
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            a, b = orders[i], orders[j]
            drifts[f'{a}__{b}'] = frobenius_drift(Ds[a], Ds[b])
    osi_val = max(drifts.values()) if drifts else None
    return {'osi': osi_val, 'margin': margin,
            'pass': bool(osi_val is not None and osi_val <= margin),
            'drifts': drifts, 'n_orders': len(orders)}


def unit_tests():
    rng = np.random.default_rng(0)
    ok = True
    # 1. 顺序不变（相同数据不同行序）：OSI ≈ 0 → PASS
    X = rng.normal(size=(400, 16))
    keys = np.array([f'd{i % 10}' for i in range(400)])
    perm = rng.permutation(400)
    r = osi({'A': X, 'B': X[perm]}, {'A': keys, 'B': keys[perm]}, margin=0.05)
    print(f'[UT] 顺序不变 OSI={r["osi"]:.4f} pass={r["pass"]}（期望≈0 PASS）')
    ok &= r['pass']
    # 2. 顺序敏感（不同 latent 内容）：OSI 大 → FAIL
    X2 = X + rng.normal(loc=2.0, scale=0.5, size=(400, 16))
    r2 = osi({'A': X, 'B': X2}, {'A': keys, 'B': keys}, margin=0.05)
    print(f'[UT] 顺序敏感 OSI={r2["osi"]:.4f} pass={r2["pass"]}（期望 FAIL）')
    ok &= not r2['pass']
    # 3. 少顺序
    r3 = osi({'A': X}, {'A': keys}, margin=0.05)
    print(f'[UT] 单顺序 OSI={r3["osi"]}（期望 None 提示需≥2）')
    ok &= r3['osi'] is None
    print(f'[UT] 结果: {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if unit_tests() else 1)
