"""Paired multi-axis readout for STIA. Higher scores must mean better retention/mixing.

This reports relative measured changes, not biological truth or framework accuracy.
Callers are responsible for identity-matched inputs, axis eligibility, metric direction,
and sample-level uncertainty; unavailable axes are never coerced to zero.
"""
import math

def compare_axes(candidate, reference, difference_intervals=None):
    intervals=difference_intervals or {}
    axes=sorted(k for k in set(candidate)&set(reference) if candidate[k] is not None and reference[k] is not None)
    deltas={}
    for k in axes:
        a,b=float(candidate[k]),float(reference[k])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Nonfinite axis score: '+k)
        deltas[k]=a-b
    improved=[k for k in axes if deltas[k]>0]
    deteriorated=[k for k in axes if deltas[k]<0]
    supported_up=[];supported_down=[]
    for k in axes:
        if k not in intervals:continue
        bounds=intervals[k]
        if len(bounds)!=2 or not all(math.isfinite(float(x)) for x in bounds) or bounds[0]>bounds[1]:
            raise ValueError('Invalid paired-difference interval: '+k)
        if bounds[0]>0 and deltas[k]>0:supported_up.append(k)
        if bounds[1]<0 and deltas[k]<0:supported_down.append(k)
    status='INSUFFICIENT_COMPARABLE_AXES' if len(axes)<2 else 'NO_MIXED_DIRECTION'
    if len(axes)>=2 and improved and deteriorated:status='DESCRIPTIVE_TRADEOFF'
    if supported_up and supported_down:status='INTERVAL_SUPPORTED_TRADEOFF'
    return {'status':status,'compared_axes':axes,'differences':deltas,
            'improved_axes':improved,'deteriorated_axes':deteriorated,
            'unchanged_axes':[k for k in axes if deltas[k]==0],
            'interval_supported_improvements':supported_up,
            'interval_supported_deteriorations':supported_down,
            'scope':'Measured reference-relative changes only; unadjusted descriptive intervals, not a simultaneous hypothesis test or a clinically meaningful threshold.'}
