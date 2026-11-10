"""Identical optional task outputs for every augmented framework comparator.

This module never replaces a native workflow's decision. Pairing is by supplied
independent-unit identity; algorithm seeds are not silently treated as donors.
Weights and eligible axes must be fixed before evaluating held-out outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .inference_optimization_v1 import infer_axes
from .task_selection_v1 import select_task

SYSTEMS = ('B0', 'B1', 'B2', 'B3', 'B5', 'B6', 'STIA')
AXES = ('biology', 'batch', 'space')
SCHEMA = 'stia.common_task_postprocessing.v1'


def _contract(values, unit_ids, axes, ndim):
    x = np.asarray(values, dtype=float)
    if x.ndim != ndim or x.shape[0] < 3 or not np.isfinite(x).all():
        raise ValueError('At least three complete finite independent units required')
    axes = tuple(axes)
    if not axes or len(axes) != x.shape[-1] or len(set(axes)) != len(axes) or not set(axes) <= set(AXES):
        raise ValueError('Explicit unique eligible axes must align with values')
    if unit_ids is None or len(unit_ids) != len(x):
        raise ValueError('Aligned independent-unit identities required')
    if any(not isinstance(v, str) or not v.strip() for v in unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError('Independent-unit identities must be unique nonempty strings')
    return x, axes


def interpret_axes(paired_differences, unit_ids, axes=AXES, alpha=.05):
    """Global intersection-union tradeoff test; Holm directional localization."""
    x, axes = _contract(paired_differences, unit_ids, axes, 2)
    r = infer_axes(x, alpha=alpha)
    local = r['holm_directional']
    return {'policy': 'global_iut_with_holm_localization', 'axes': list(axes),
            'independent_units': len(x), 'mean_difference': r['mean'].tolist(),
            'standard_error': r['standard_error'].tolist(), 'alpha': alpha,
            'tradeoff': bool(r['tradeoff_iut']['tradeoff']),
            'tradeoff_p': float(r['tradeoff_iut_p']),
            'improved_axes': [a for a, flag in zip(axes, local['improved']) if flag],
            'deteriorated_axes': [a for a, flag in zip(axes, local['deteriorated']) if flag],
            'directional_family_size': r['directional_family_size']}


def recommend_method(training_scores, methods, unit_ids, axes=AXES, weights=None,
                     tolerances=None, reference=0, alpha=.05):
    """Common paired-t lower-bound protection with explicit reference fallback.

    Input: independent training unit x candidate method x eligible axis, with
    larger values better. Held-out outcomes are not an argument to this API.
    """
    x, axes = _contract(training_scores, unit_ids, axes, 3)
    if len(methods) != x.shape[1] or not methods or any(not isinstance(m, str) or not m.strip() for m in methods) or len(set(methods)) != len(methods):
        raise ValueError('Unique nonempty method identities required')
    if not isinstance(reference, int) or isinstance(reference, bool) or not 0 <= reference < len(methods):
        raise ValueError('Valid fixed reference index required')
    weights = np.ones(len(axes)) if weights is None else weights
    tolerances = np.zeros(len(axes)) if tolerances is None else tolerances
    differences = x - x[:, reference:reference+1]
    se = differences.std(axis=0, ddof=1) / np.sqrt(len(x))
    r = select_task(x.mean(0), weights, tolerances, 't_bound_constrained',
                    reference=reference, standard_errors=se, df=len(x)-1, alpha=alpha)
    return {'policy': 't_bound_constrained', 'selected_index': r['selected'],
            'selected_method': methods[r['selected']], 'reference_method': methods[reference],
            'reference_fallback': r['selected'] == reference, 'eligible': r['eligible'],
            'axes': list(axes), 'methods': list(methods), 'independent_training_units': len(x),
            'weights': r['weights'].tolist(), 'tolerances': r['tolerances'].tolist(),
            'estimated_weighted_gains': r['weighted_gain_estimates'].tolist(), 'alpha': alpha}


def augment_workflow(system, core_decision, paired_differences=None, training_scores=None,
                     unit_ids=None, axes=AXES, methods=None, weights=None, tolerances=None,
                     reference=0, alpha=.05):
    """Apply exactly the same postprocessor; preserve native rejection/abstention."""
    if system not in SYSTEMS:
        raise ValueError('Unknown comparator')
    if core_decision not in ('ACCEPT', 'ACCEPT_WITH_SCOPE', 'REJECT', 'ABSTAIN'):
        raise ValueError('Unknown native decision')
    result = {'schema_version': SCHEMA, 'system': system, 'core_decision': core_decision,
              'augmentation': 'identical_shared_postprocessor', 'axis_interpretation': None,
              'recommendation': None, 'status': 'BLOCKED_BY_CORE'}
    if core_decision in ('REJECT', 'ABSTAIN'):
        return result
    if paired_differences is None and training_scores is None:
        raise ValueError('At least one explicitly requested task required')
    result['status'] = 'TASK_OUTPUTS_AVAILABLE'
    if paired_differences is not None:
        result['axis_interpretation'] = interpret_axes(paired_differences, unit_ids, axes, alpha)
    if training_scores is not None:
        if methods is None:
            raise ValueError('Method identities required for recommendation')
        result['recommendation'] = recommend_method(training_scores, methods, unit_ids, axes,
                                                    weights, tolerances, reference, alpha)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8-sig'))
    result = augment_workflow(**data)
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
