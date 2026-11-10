"""Optional evaluation helpers; the existing STIA decision engine is unchanged.

Student t intervals assume independent Gaussian measurement units. Simultaneous
intervals use Bonferroni across a prespecified family, not post-hoc selection.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t
from stia.tradeoff_v1 import compare_axes

AXES = ('biology', 'batch', 'space')


def _samples(values):
    x = np.asarray(values, dtype=float)
    if x.ndim < 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two complete finite independent units required')
    return x


def axis_intervals(samples, family_size=3, alpha=.05):
    x = _samples(samples)
    if not isinstance(family_size, int) or family_size < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid interval family or alpha')
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(len(x))
    width = t.ppf(1-alpha/(2*family_size), len(x)-1) * se
    return mean, mean-width, mean+width


def classify_tradeoff(samples, policy='simultaneous', keep_axes=AXES):
    x = _samples(samples)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError('Expected unit x three-axis paired differences')
    if policy not in ('point', 'marginal', 'simultaneous'):
        raise ValueError('Unknown detection policy')
    if not keep_axes or len(set(keep_axes)) != len(keep_axes) or not set(keep_axes) <= set(AXES):
        raise ValueError('Invalid retained axes')
    mean, lo, hi = axis_intervals(x, family_size=1 if policy == 'marginal' else 3)
    candidate = {a: float(mean[i]) for i, a in enumerate(AXES) if a in keep_axes}
    reference = {a: 0. for a in candidate}
    ci = None if policy == 'point' else {a: [float(lo[i]), float(hi[i])] for i, a in enumerate(AXES) if a in keep_axes}
    r = compare_axes(candidate, reference, ci)
    up = r['improved_axes'] if policy == 'point' else r['interval_supported_improvements']
    down = r['deteriorated_axes'] if policy == 'point' else r['interval_supported_deteriorations']
    return {'detected': bool(up and down), 'improved_axes': up, 'deteriorated_axes': down,
            'mean': mean, 'lower': lo, 'upper': hi, 'readout': r}


def wilson(successes, total, alpha=.05):
    if total == 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError('Invalid binomial counts')
    z = norm.ppf(1-alpha/2)
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0., float(center-half)), min(1., float(center+half))]


def binary_metrics(truth, predicted):
    a, b = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('Unaligned truth and predictions')
    tp, fp = int((a & b).sum()), int((~a & b).sum())
    fn, tn = int((a & ~b).sum()), int((~a & ~b).sum())
    out = {'n': len(a), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    for name, num, den in [('precision', tp, tp+fp), ('recall', tp, tp+fn),
                           ('false_positive_rate', fp, fp+tn), ('specificity', tn, fp+tn)]:
        out[name] = num/den if den else None
        out[name+'_wilson95'] = wilson(num, den)
    den = 2*tp+fp+fn
    out['f1'] = 2*tp/den if den else None
    return out


def localization_metrics(truth, predicted):
    a, b = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Unaligned three-axis localization arrays')
    per_axis = {name: binary_metrics(a[:, i], b[:, i]) for i, name in enumerate(AXES)}
    f1 = [x['f1'] for x in per_axis.values() if x['f1'] is not None]
    return {'per_axis': per_axis, 'macro_f1': float(np.mean(f1)) if f1 else None,
            'exact_set_agreement': float(np.mean(np.all(a == b, axis=1))) if len(a) else None}


def choose_candidate(samples, policy, minimum=.5):
    x = _samples(samples)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError('Expected unit x candidate x three-axis observations')
    means = x.mean(axis=0)
    if policy in AXES:
        return int(np.argmax(means[:, AXES.index(policy)]))
    if policy == 'mean':
        return int(np.argmax(means.mean(axis=1)))
    if policy == 'point':
        lower = means
    elif policy == 'simultaneous':
        _, lower, _ = axis_intervals(x, family_size=means.size)
    else:
        raise ValueError('Unknown selection policy')
    eligible = np.flatnonzero(np.all(lower >= minimum, axis=1))
    return int(eligible[np.argmax(means[eligible].mean(axis=1))]) if len(eligible) else None


def selection_metrics(task_has_feasible, selected, selected_is_feasible):
    truth = np.asarray(task_has_feasible, dtype=bool)
    valid = np.asarray(selected_is_feasible, dtype=bool)
    active = np.array([v is not None for v in selected], dtype=bool)
    if truth.ndim != 1 or truth.shape != valid.shape or truth.shape != active.shape:
        raise ValueError('Unaligned selection observations')
    if np.any(valid & (~active | ~truth)):
        raise ValueError('A valid selection must be active and feasible')
    counts = {
        'selection_coverage': (int(active.sum()), len(truth)),
        'unsafe_selection_rate': (int((active & ~valid).sum()), int(active.sum())),
        'feasible_task_success_rate': (int(valid.sum()), int(truth.sum())),
        'correct_infeasible_task_abstention_rate': (int((~truth & ~active).sum()), int((~truth).sum())),
    }
    result = {'n': len(truth)}
    for name, (num, den) in counts.items():
        result[name] = num/den if den else None
        result[name+'_counts'] = [num, den]
        result[name+'_wilson95'] = wilson(num, den)
    return result
