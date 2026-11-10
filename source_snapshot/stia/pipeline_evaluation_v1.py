"""Composition diagnostics: native validity gating followed by common readouts.

This is not a replacement for the original axis F1 or numeric-loss endpoints.
The Cartesian product is software composition evidence, not independent cases.
Risk and coverage follow the selective-prediction convention; no distributional
risk guarantee or independently adjudicated biological truth is claimed here.
"""
from __future__ import annotations
import numpy as np


def _boolean(value, ndim, name):
    x = np.asarray(value)
    if x.ndim != ndim or x.dtype != np.bool_ or not x.size:
        raise ValueError(name+' must be a nonempty boolean array')
    return x


def evaluate_composition(valid, native_accept, truth_tradeoff, predicted_tradeoff,
                         truth_down, predicted_down, population_shortfall,
                         bootstrap_correct_retention, invalid_prevalence=None):
    """Evaluate every native receipt x frozen independent score query.

    ``invalid_prevalence`` is a declared scenario weight, not an estimated
    deployment prevalence. Within each validity group receipts are equally
    weighted; score queries are equally weighted. Population shortfall on an
    invalid or withheld query is unavailable, never silently set to zero.
    """
    valid = _boolean(valid, 1, 'valid'); accept = _boolean(native_accept, 1, 'native_accept')
    true = _boolean(truth_tradeoff, 1, 'truth_tradeoff')
    pred = _boolean(predicted_tradeoff, 1, 'predicted_tradeoff')
    down = _boolean(truth_down, 2, 'truth_down'); guessed_down = _boolean(predicted_down, 2, 'predicted_down')
    short = np.asarray(population_shortfall, float); retention = np.asarray(bootstrap_correct_retention, float)
    if accept.shape != valid.shape or not valid.any() or valid.all():
        raise ValueError('Aligned receipts with both valid and invalid evidence required')
    if true.shape != pred.shape or down.shape != guessed_down.shape or down.shape[0] != len(true) or down.shape[1] != 3:
        raise ValueError('Aligned three-axis query labels required')
    if short.shape != true.shape or retention.shape != true.shape:
        raise ValueError('Aligned query losses and resampling readouts required')
    if not np.isfinite(short).all() or np.any(short < 0) or not np.isfinite(retention).all() or np.any((retention < 0) | (retention > 1)):
        raise ValueError('Finite nonnegative loss and retention probabilities in [0,1] required')
    prior = float(np.mean(~valid)) if invalid_prevalence is None else invalid_prevalence
    if not isinstance(prior, (float, int)) or isinstance(prior, bool) or not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError('Declared invalid prevalence must be in (0,1)')
    w = np.where(valid, (1-prior)/int(valid.sum()), prior/int((~valid).sum()))
    true_class = np.where(valid[:, None], np.where(true[None, :], 1, 2), 0)
    pred_class = np.where(accept[:, None], np.where(pred[None, :], 1, 2), 0)
    weights = np.broadcast_to(w[:, None]/len(true), true_class.shape)
    confusion = np.bincount((3*true_class+pred_class).ravel(), weights=weights.ravel(), minlength=9).reshape(3, 3)
    f1 = []
    for j in range(3):
        denominator = confusion[j].sum()+confusion[:, j].sum()
        f1.append(float(2*confusion[j, j]/denominator) if denominator > 0 else None)
    valid_mass = float(w[valid & accept].sum()); invalid_mass = float(w[~valid & accept].sum())
    coverage = valid_mass+invalid_mass
    true_correct = float(np.mean(pred == true)); localized = float(np.mean(np.all(down == guessed_down, axis=1)))
    return {
        'engineering_receipts': len(valid), 'score_queries': len(true),
        'composition_pairs_not_independent_cases': len(valid)*len(true),
        'invalid_prevalence': float(prior),
        'valid_query_coverage': float(np.mean(accept[valid])),
        'invalid_output_rate': float(np.mean(accept[~valid])),
        'overall_query_coverage': coverage,
        'selective_tradeoff_risk': (invalid_mass+valid_mass*(1-true_correct))/coverage if coverage > 0 else None,
        'validity_aware_tradeoff_macro_f1': float(np.mean(f1)) if all(v is not None for v in f1) else None,
        'validity_aware_class_f1': dict(zip(('withhold_invalid', 'valid_tradeoff', 'valid_no_tradeoff'), f1)),
        'weighted_three_class_confusion': confusion.tolist(),
        'joint_axis_set_accuracy': float(w[~valid & ~accept].sum())+valid_mass*localized,
        'valid_selected_mean_max_axis_shortfall': float(short.mean()) if valid_mass > 0 else None,
        'safe_correct_output_yield': valid_mass*true_correct,
        'joint_correct_retention': float(w[~valid & ~accept].sum())+valid_mass*float(retention.mean()),
        'risk_scope': 'An invalid receipt makes its issued analysis unsupported; this is not the numeric correctness of a valid-input three-axis estimator',
        'attribution': 'Any between-system difference here arises from native acceptance gating, not the shared statistical estimator',
    }
