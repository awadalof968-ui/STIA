# -*- coding: utf-8 -*-
"""设计配置加载与 schema 检查（invest_st I1）。

config 为 YAML/JSON，定义 estimand、batch/condition/donor/sample 列名、
strata、seed、阈值与统计家族。缺失/未知字段给出明确错误。"""
from pathlib import Path

import yaml

REQUIRED_METADATA = ['spot_id', 'sample_id', 'donor_id', 'disease_or_condition', 'technical_batch']


class ConfigError(ValueError):
    pass


DEFAULTS = {
    'k_ilisi': 90,
    'k_graph': 20,
    'max_per_stratum': 1500,
    'min_stratum_spots': 100,
    'bootstrap_seed': 20260816,
    'bootstrap_n': 2000,
    'alpha': 0.05,
    'min_donor_per_condition': 3,
    'min_batch_per_condition': 2,
    'report_dir': 'audit_report',
    'osi_orders_required': 4,
    'osi_margin': 0.05,
}


def load_config(path):
    p = Path(path)
    if not p.exists():
        raise ConfigError(f'config 不存在: {p}')
    raw = yaml.safe_load(p.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ConfigError('config 顶层必须是 mapping')
    cfg = dict(DEFAULTS)
    cfg.update(raw)

    # 必需列映射
    for k in ['spot_id', 'sample_id', 'donor_id', 'disease_or_condition', 'technical_batch']:
        if not cfg.get(k):
            raise ConfigError(f'config 缺少必需列映射: {k}')
    # estimand
    if not cfg.get('estimand'):
        raise ConfigError('config 缺少 estimand（供体级疾病信号保留等）')
    unknown = set(cfg) - set(DEFAULTS) - {
        'spot_id', 'sample_id', 'donor_id', 'disease_or_condition', 'technical_batch',
        'estimand', 'geometry', 'embedding_dir', 'adata', 'stratum_cols', 'spatial_domain',
        'method_representations', 'thresholds'}
    if unknown:
        raise ConfigError(f'config 含未知字段: {sorted(unknown)}')
    return cfg


def load_adata_obs(adata_path, cfg):
    """读取 AnnData 的 obs 并校验必需列存在（不加载 X，节省内存）。
    spot_id 可用列名或 index 名（anndata 常把 spot id 存为 index）。"""
    import anndata as ad
    a = ad.read_h5ad(adata_path, backed='r')
    obs = a.obs
    sid = cfg['spot_id']
    if sid not in obs.columns and obs.index.name == sid:
        obs = obs.reset_index()  # index -> 列
    missing = [k for k in REQUIRED_METADATA if cfg.get(k) not in obs.columns]
    if missing:
        raise ConfigError(f'obs 缺少列: {missing}（config 映射: { {k: cfg[k] for k in REQUIRED_METADATA} }）')
    return obs
