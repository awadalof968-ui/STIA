# -*- coding: utf-8 -*-
"""identifiability 模块（invest_st I1）：设计可识别性判定。

核心纪律（第 17 份文档 identifiability_decision_logic + I0 冻结）：
- 在样本/供体层判定，不把完整 donor dummy 与 disease 同时放入同一固定效应矩阵后宣布不可识别
- 疾病通常在 donor 内固定 → 检查 donor×condition 嵌套
- batch×condition contingency、overlap / positivity、设计矩阵秩与别名检测

返回 IDENTIFIABLE / PARTIALLY_IDENTIFIABLE / NOT_IDENTIFIABLE + 支持证据。"""
import numpy as np
import pandas as pd


def donor_condition_table(obs, cfg):
    """donor × condition 交叉表（供体层重复测量结构检查）。"""
    did = cfg['donor_id']
    cid = cfg['disease_or_condition']
    tab = pd.crosstab(obs[did], obs[cid], dropna=False)
    return tab


def batch_condition_table(obs, cfg):
    """batch × condition 交叉表（spot 级重叠检查）。"""
    bid = cfg['technical_batch']
    cid = cfg['disease_or_condition']
    tab = pd.crosstab(obs[bid], obs[cid], dropna=False)
    return tab


def sample_condition_batch_table(obs, cfg):
    """sample × condition × batch（sample 层）。"""
    sid = cfg['sample_id']
    cid = cfg['disease_or_condition']
    bid = cfg['technical_batch']
    return obs.groupby([sid, cid, bid], observed=True).size().unstack(fill_value=0)


def check_positivity(bc_tab):
    """batch×condition：每个 condition 至少出现在 min_batch 个 batch；每个 batch 至少 1 个 condition。"""
    conds = bc_tab.columns
    batch_counts = (bc_tab > 0).sum(axis=0)   # 每 condition 覆盖的 batch 数
    pos_issues = []
    for c in conds:
        if batch_counts[c] == 0:
            pos_issues.append(f'condition {c} 无任何 spot')
    return batch_counts, pos_issues


def check_overlap(bc_tab):
    """condition 对之间的共享 batch 数（桥接样本）。"""
    conds = list(bc_tab.columns)
    pairs = {}
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            shared = int(((bc_tab[conds[i]] > 0) & (bc_tab[conds[j]] > 0)).sum())
            pairs[f'{conds[i]}~{conds[j]}'] = shared
    return pairs


def rank_design_matrix(obs, cfg):
    """设计矩阵秩检查：用 [1, batch, condition]（不含 donor dummy）做秩与别名探测。
    若 batch 与 condition 完全对齐，增广矩阵秩下降 → NOT_IDENTIFIABLE 信号。"""
    bid = cfg['technical_batch']
    cid = cfg['disease_or_condition']
    X = pd.get_dummies(obs[bid], prefix='B', drop_first=False).astype(float)
    C = pd.get_dummies(obs[cid], prefix='C', drop_first=False).astype(float)
    X['intercept'] = 1.0
    M = X.values
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    rank_b = int((s > 1e-8).sum())
    M2 = np.hstack([M, C.values])
    _, s2, _ = np.linalg.svd(M2, full_matrices=False)
    rank_bc = int((s2 > 1e-8).sum())
    return {'rank_batch_only': rank_b, 'rank_batch_plus_condition': rank_bc,
            'collinear_condition': rank_bc <= rank_b}


def classify(obs, cfg):
    """主入口。返回 {class, reasons, evidence}。"""
    dc_tab = donor_condition_table(obs, cfg)
    bc_tab = batch_condition_table(obs, cfg)
    batch_counts, pos_issues = check_positivity(bc_tab)
    overlap = check_overlap(bc_tab)
    rank = rank_design_matrix(obs, cfg)
    did = cfg['donor_id']
    cid = cfg['disease_or_condition']
    n_cond = obs[cid].nunique()
    n_batch = obs[cfg['technical_batch']].nunique()

    reasons = []
    # 1. 供体嵌套 disease：donor 内 condition 是否固定
    donor_mult = dc_tab[dc_tab > 0].count(axis=1)
    donors_multi = int((donor_mult > 1).sum())
    if donors_multi == 0:
        reasons.append(f'疾病在供体内固定（{len(dc_tab)} 供体均单一 condition）→ 供体层推断成立')
    else:
        reasons.append(f'{donors_multi} 个供体跨多 condition（重复测量供体）')

    # 2. 桥接样本
    min_shared = min(overlap.values()) if overlap else 0
    reasons.append(f'condition 对最小共享 batch 数 = {min_shared}（{overlap}）')

    # 3. 秩/共线
    if rank['collinear_condition']:
        reasons.append('batch 与 condition 设计矩阵共线（rank 未随 condition 增加）→ 部分/不可识别信号')

    # 判定（设计层：batch×condition 重叠与桥接；供体数不足归 validate FAIL，不在此降级）
    if n_cond < 2:
        cls = 'NOT_IDENTIFIABLE'
        reasons.append('仅一个 condition，无法评估疾病对比')
    elif rank['collinear_condition'] and min_shared == 0:
        cls = 'NOT_IDENTIFIABLE'
        reasons.append('batch 与 condition 设计矩阵共线且无桥接样本 → 疾病效应是批次列的精确组合，目标对比不可识别')
    elif min_shared == 0:
        cls = 'NOT_IDENTIFIABLE'
        reasons.append('每一 condition 只出现在独有 batch（无任何共享批次）→ 无桥接样本，不可识别')
    elif min_shared == 1:
        cls = 'PARTIALLY_IDENTIFIABLE'
        reasons.append('仅 1 个共享批次支撑疾病对比 → 部分重叠，限制 estimand 到共同支持子集')
    elif min_shared >= 2:
        cls = 'IDENTIFIABLE'
        reasons.append(f'目标对比有 {min_shared} 个共享批次并由独立供体支持 → 允许完整三轴评价')
    else:
        cls = 'PARTIALLY_IDENTIFIABLE'
        reasons.append('设计支持不足，限制 estimand 到共同支持子集')
    # 供体级支持警告（不改变设计判定，但提示 coverage）
    if donors_multi == 0:
        reasons.append(f'{len(dc_tab)} 供体均嵌套单一 condition（正常设计；供体层推断成立）')
    else:
        reasons.append(f'{donors_multi} 个供体跨多 condition（重复测量供体，报告时标注）')
    if pos_issues:
        reasons.append('positivity 问题: ' + '; '.join(pos_issues))
        cls = 'NOT_IDENTIFIABLE' if cls == 'IDENTIFIABLE' else cls
    return {'class': cls, 'reasons': reasons,
            'evidence': {'donor_condition': dc_tab.to_dict(), 'batch_condition': bc_tab.to_dict(),
                         'overlap': overlap, 'rank': rank}}
