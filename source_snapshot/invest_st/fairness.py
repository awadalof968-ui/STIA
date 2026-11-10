# -*- coding: utf-8 -*-
"""STIA M4: feature_preprocessing_representation_fairness。

对应计划: MULTIDIMENSIONAL_AUDIT_TOOL T1 (M4)
检查:
1. 特征面板公平性（方法间面板基因一致 + 哈希）
2. 预处理一致性（log1p-CPM 或等价）
3. 表示公平性（维度/类型可比；禁止直接比较不可比方向）
4. 泄漏检查（marker 是否在训练前选定/面板外）
"""
from typing import Dict, List, Optional
import hashlib
import json


def panel_fairness(panel_a: List[str], panel_b: List[str],
                   panel_a_name: str = 'method_A', panel_b_name: str = 'method_B',
                   expected_panel_size: Optional[int] = None) -> Dict:
    """比较两个方法的特征面板。"""
    set_a, set_b = set(panel_a), set(panel_b)
    common = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    issues = []
    if expected_panel_size is not None:
        if len(set_a) != expected_panel_size:
            issues.append(f'{panel_a_name} panel size {len(set_a)} != expected {expected_panel_size}')
        if len(set_b) != expected_panel_size:
            issues.append(f'{panel_b_name} panel size {len(set_b)} != expected {expected_panel_size}')
    if only_a or only_b:
        issues.append(f'panel mismatch: only_in_{panel_a_name}={len(only_a)}, only_in_{panel_b_name}={len(only_b)}')
    return {
        'panel_a': panel_a_name, 'panel_b': panel_b_name,
        'n_a': len(set_a), 'n_b': len(set_b), 'n_common': len(common),
        'only_in_a': sorted(only_a)[:20], 'only_in_b': sorted(only_b)[:20],
        'fair': not issues, 'issues': issues,
        'panel_a_sha256': hashlib.sha256(json.dumps(sorted(set_a), ensure_ascii=False).encode()).hexdigest(),
        'panel_b_sha256': hashlib.sha256(json.dumps(sorted(set_b), ensure_ascii=False).encode()).hexdigest(),
    }


def representation_fairness(dim_a: int, dim_b: int, type_a: str, type_b: str,
                            method_a: str = 'A', method_b: str = 'B') -> Dict:
    """检查表示维度/类型可比性。"""
    issues = []
    if type_a != type_b:
        issues.append(f'representation type mismatch: {type_a} vs {type_b}')
    # 维度差异本身不是不公平（latent dim 不同是正常的），但若做方向比较需 warning
    dim_ratio = max(dim_a, dim_b) / max(1, min(dim_a, dim_b))
    warning = dim_ratio > 2
    return {
        'method_a': method_a, 'method_b': method_b,
        'dim_a': dim_a, 'dim_b': dim_b, 'type_a': type_a, 'type_b': type_b,
        'fair': not issues, 'issues': issues,
        'dim_ratio_warning': warning,
        'direction_comparison_requires_probe': type_a != type_b or warning,
    }


def leakage_check(panel_genes: List[str], marker_genes: List[str],
                  marker_selected_after_training: bool = False) -> Dict:
    """检查 marker 泄漏：marker 是否在面板内 + 是否事后选择。"""
    panel_set = set(panel_genes)
    markers_in_panel = [g for g in marker_genes if g in panel_set]
    markers_out = [g for g in marker_genes if g not in panel_set]
    issues = []
    if marker_selected_after_training:
        issues.append('markers selected after outcome/training (post-hoc marker selection)')
    return {
        'n_markers': len(marker_genes),
        'markers_in_panel': len(markers_in_panel),
        'markers_out_of_panel': len(markers_out),
        'markers_out_list': markers_out[:20],
        'post_hoc_marker_selection': marker_selected_after_training,
        'clean': not issues and not markers_out, 'issues': issues,
    }
