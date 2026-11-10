# -*- coding: utf-8 -*-
"""STIA M5: evaluator_and_statistical_validity。

对应计划: MULTIDIMENSIONAL_AUDIT_TOOL T1 (M5)
检查:
1. 统计单位（donor 为推断单位，spot 不得当独立 n）
2. 伪重复检测（type-I error 膨胀）
3. 多重检验（Holm/FDR 校正）
4. CI 覆盖率和效应量报告
"""
from typing import Dict, List, Optional
import numpy as np


def unit_check(analysis_unit: str, n_donors: int, n_spots: int,
               donor_level_claim: bool) -> Dict:
    """统计单位检查。"""
    issues = []
    if donor_level_claim and analysis_unit != 'donor':
        issues.append(f'donor-level claim requires donor unit, got {analysis_unit}')
    if donor_level_claim and n_donors < 3:
        issues.append(f'only {n_donors} donors for donor-level inference (underpowered)')
    return {'analysis_unit': analysis_unit, 'n_donors': n_donors, 'n_spots': n_spots,
            'donor_level_claim': donor_level_claim, 'valid': not issues, 'issues': issues}


def holm_adjust(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Holm-Bonferroni 多重校正（从最小 p 开始，running max 单调）。"""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return {'n_tests': 0, 'significant': [], 'adjusted': []}
    order = np.argsort(p)          # 升序
    adj = np.full(n, np.nan)
    prev = 0.0
    for rank, idx in enumerate(order):
        # Holm: (n - rank) * p_rank，且单调不减
        step = (n - rank) * p[idx]
        prev = max(prev, step)
        adj[idx] = min(prev, 1.0)
    sig = [int(i) for i in range(n) if adj[i] < alpha]
    return {'n_tests': n, 'alpha': alpha, 'adjusted_p': [round(float(x), 4) for x in adj],
            'significant_indices': sig, 'n_significant': len(sig)}


def pseudoreplication_type1(type1_donor: float, type1_spot: float,
                            alpha: float = 0.05) -> Dict:
    """伪重复 type-I 膨胀检查。"""
    issues = []
    if type1_donor > alpha * 1.5:
        issues.append(f'donor-level type-I {type1_donor:.3f} > 1.5*alpha ({1.5*alpha:.3f})')
    if type1_spot > type1_donor * 3:
        issues.append(f'spot-level type-I {type1_spot:.3f} inflated vs donor {type1_donor:.3f} (pseudoreplication)')
    return {'type1_donor': type1_donor, 'type1_spot': type1_spot, 'alpha': alpha,
            'pseudoreplication_detected': bool(issues), 'issues': issues}


def ci_coverage(coverage: float, target: float = 0.95, tolerance: float = 0.03) -> Dict:
    """CI 覆盖率校准检查。"""
    ok = abs(coverage - target) <= tolerance
    return {'coverage': coverage, 'target': target, 'tolerance': tolerance,
            'calibrated': ok,
            'issue': None if ok else f'CI coverage {coverage:.3f} off target {target}'}


def power_report(power: float, effect: float, n: int, min_power: float = 0.8) -> Dict:
    """功效报告（不足以支持阴性结论时警告）。"""
    return {'power': power, 'effect': effect, 'n': n,
            'adequate': power >= min_power,
            'issue': None if power >= min_power else f'power {power:.2f} < {min_power} (inconclusive null)'}
