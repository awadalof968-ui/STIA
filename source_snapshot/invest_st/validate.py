# -*- coding: utf-8 -*-
"""validate 模块（invest_st I1）：元数据完整性 / spot_id 唯一 / latent↔metadata 键对齐 /
sample-donor 嵌套 / 最小 donor 与 batch 计数。

关键纪律（G2 gate）：latent 与 metadata 必须通过 spot_id 键连接，禁止按行截尾对齐。"""
import numpy as np
import pandas as pd


class ValidationError(Exception):
    pass


def validate_metadata(obs, cfg):
    """obs: DataFrame（spot × metadata）。返回 issues 列表。"""
    issues = []
    sid = cfg['spot_id']
    if sid not in obs.columns:
        issues.append(f'缺少 spot_id 列: {sid}')
        return issues
    if obs[sid].duplicated().any():
        n = int(obs[sid].duplicated().sum())
        issues.append(f'spot_id 不唯一: {n} 个重复')
    for col in ['sample_id', 'donor_id', 'disease_or_condition', 'technical_batch']:
        c = cfg[col]
        if c not in obs.columns:
            issues.append(f'缺少列 {col} 的映射 {c}')
        elif obs[c].isna().any():
            issues.append(f'列 {c} 有 {int(obs[c].isna().sum())} 个缺失值')
    return issues


def check_nesting(obs, cfg):
    """sample 嵌套在 donor 内；batch 与 sample 关系检查。返回 (ok, 说明列表)。"""
    notes = []
    ok = True
    sid, did, bid = cfg['sample_id'], cfg['donor_id'], cfg['technical_batch']
    grp = obs.groupby(sid, observed=True)[did].nunique()
    multi = grp[grp > 1]
    if len(multi):
        ok = False
        notes.append(f'{len(multi)} 个 sample 跨多个 donor: {list(multi.index[:5])}')
    else:
        notes.append('所有 sample 均嵌套于单一 donor ✓')
    bgrp = obs.groupby(sid, observed=True)[bid].nunique()
    multi_b = bgrp[bgrp > 1]
    if len(multi_b):
        notes.append(f'{len(multi_b)} 个 sample 含多个 batch（可能是同一切片多批次，需确认）')
    return ok, notes


def check_min_counts(obs, cfg):
    """每个 condition 的有效 donor 数与 batch 数（最低要求由 config 阈值决定）。"""
    res = {}
    cid = cfg['disease_or_condition']
    did, bid = cfg['donor_id'], cfg['technical_batch']
    for cond, sub in obs.groupby(cid, observed=True):
        res[str(cond)] = {
            'n_spots': int(len(sub)),
            'n_donors': int(sub[did].nunique()),
            'n_batches': int(sub[bid].nunique()),
        }
    return res


def align_latent_by_key(latent_df, metadata_df, cfg):
    """latent_df: DataFrame（index=spot_id, 列=latent 维度）。metadata_df 同 index。
    按 spot_id 对齐，报告未匹配数。禁止截尾。返回 (aligned_meta, issues)。"""
    issues = []
    common = latent_df.index.intersection(metadata_df.index)
    miss_lat = metadata_df.index.difference(latent_df.index)
    miss_meta = latent_df.index.difference(metadata_df.index)
    if len(miss_lat):
        issues.append(f'{len(miss_lat)} 个 metadata spot 在 latent 中缺失')
    if len(miss_meta):
        issues.append(f'{len(miss_meta)} 个 latent spot 在 metadata 中缺失')
    if len(common) == 0:
        raise ValidationError('latent 与 metadata 无共同 spot_id')
    return metadata_df.loc[common], issues


def run_validate(obs, cfg, latents=None):
    """总入口：返回 {'status': 'PASS'|'FAIL', 'issues': [...], 'nesting': [...], 'counts': {...}}"""
    issues = validate_metadata(obs, cfg)
    if issues:
        # 缺列/重复等基础问题未解决前，不做需要完整列的嵌套与计数检查
        return {'status': 'FAIL', 'issues': issues, 'nesting_ok': None,
                'nesting_notes': [], 'counts': {}}
    ok_nest, nest_notes = check_nesting(obs, cfg)
    if not ok_nest:
        issues.append('sample_id 跨多个 donor_id，违反 sample 嵌套约束')
    counts = check_min_counts(obs, cfg)
    for cond, c in counts.items():
        if c['n_donors'] < cfg['min_donor_per_condition']:
            issues.append(f"condition {cond} 供体数 {c['n_donors']} < 阈值 {cfg['min_donor_per_condition']}")
        if c['n_batches'] < cfg['min_batch_per_condition']:
            issues.append(f"condition {cond} 批次数 {c['n_batches']} < 阈值 {cfg['min_batch_per_condition']}")
    status = 'PASS' if not issues else 'FAIL'
    return {'status': status, 'issues': issues, 'nesting_ok': ok_nest,
            'nesting_notes': nest_notes, 'counts': counts}
