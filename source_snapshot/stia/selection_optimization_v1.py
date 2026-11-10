"""Exploratory score selection, separate from the production audit."""
import numpy as np

def choose(training_scores, scale, penalty, reference=0):
    x=np.asarray(training_scores,dtype=float)
    if x.ndim!=2 or not x.size or not np.isfinite(x).all():
        raise ValueError('Complete finite method-by-axis training scores required')
    if not 0<=reference<len(x) or not np.isfinite(penalty) or penalty<0:
        raise ValueError('Invalid reference or penalty')
    delta=x-x[reference]
    if scale=='training_range':
        ranges=np.ptp(x,axis=0)
        delta=np.divide(delta,ranges,out=np.zeros_like(delta),where=ranges>0)
    elif scale!='native':raise ValueError('Unknown scale')
    utility=delta.mean(1)-penalty*np.maximum(-delta,0).mean(1)
    tied=np.flatnonzero(np.abs(utility-utility.max())<=1e-12)
    return int(reference if reference in tied else tied[0])

def select_on_other_datasets(dataset_scores,target,candidates):
    source=sorted(d for d in dataset_scores if d!=target)
    if not source:raise ValueError('At least one other dataset required')
    scores={p:float(np.mean([dataset_scores[d][p] for d in source])) for p in candidates}
    if not all(np.isfinite(v) for v in scores.values()):raise ValueError('Nonfinite validation objective')
    best=max(scores.values())
    chosen=next(p for p in candidates if abs(scores[p]-best)<=1e-12)
    return {'chosen':chosen,'training_datasets':source,'source_dataset_mean_utilities':scores}

def outcome(delta,is_reference):
    d=np.asarray(delta,dtype=float)
    if d.ndim!=1 or not d.size or not np.isfinite(d).all():raise ValueError('Invalid held-out difference')
    no_loss=bool(np.all(d>=0))
    return {'mean_axis_gain':float(d.mean()),
        'mean_worst_axis_loss':float(max(0.,-d.min())),
        'all_axis_no_loss_rate':int(no_loss),
        'strict_gain_without_loss_rate':int(no_loss and np.any(d>0)),
        'PCA_fallback_rate':int(is_reference),
        'preference_utility':float(d.mean()-np.maximum(-d,0).mean())}
