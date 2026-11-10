"""Real-data reporting utilities, not biological truth or diagnostic accuracy."""
from __future__ import annotations
import numpy as np


def distance_ranking_auc(points, labels):
    z=np.asarray(points,dtype=float)
    y=np.asarray(labels,dtype=str)
    if z.ndim!=2 or len(z)!=len(y) or not np.isfinite(z).all():
        raise ValueError('Invalid paired donor centers/labels')
    classes,counts=np.unique(y,return_counts=True)
    if len(classes)!=2 or np.any(counts<2):
        raise ValueError('Two conditions with at least two donors each required')
    d=np.sqrt(np.sum((z[:,None,:]-z[None,:,:])**2,axis=-1))
    i,j=np.triu_indices(len(z),1)
    same=y[i]==y[j]
    within,between=d[i[same],j[same]],d[i[~same],j[~same]]
    delta=between[:,None]-within[None,:]
    return float((np.count_nonzero(delta>0)+.5*np.count_nonzero(delta==0))/delta.size)


def select_from_training(scores, policy, reference_index=0):
    x=np.asarray(scores,dtype=float)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Complete finite method x eligible-axis scores required')
    if not 0<=reference_index<len(x):raise ValueError('Invalid reference method')
    if policy.startswith('axis_'):
        try:axis=int(policy[5:])
        except ValueError:raise ValueError('Invalid axis policy')
        if not 0<=axis<x.shape[1]:raise ValueError('Axis absent')
        return int(np.argmax(x[:,axis]))
    if policy not in ('mean','guarded'):raise ValueError('Unknown selection policy')
    ranges=np.ptp(x,axis=0)
    normalized=np.divide(x-x.min(0),ranges,out=np.zeros_like(x),where=ranges>0)
    utility=normalized.mean(1)
    pool=np.arange(len(x)) if policy=='mean' else np.flatnonzero(np.all(x>=x[reference_index],axis=1))
    # Reference always qualifies exactly; ties follow frozen candidate order.
    return int(pool[np.argmax(utility[pool])])


def summarize_selection(records):
    if not records:return {'folds':0}
    d=np.asarray([r['delta'] for r in records],dtype=float)
    no_loss=np.all(d>=0,axis=1)
    gain=no_loss & np.any(d>0,axis=1)
    return {'folds':len(records),'all_axis_no_loss_rate':float(no_loss.mean()),
        'all_axis_no_loss_count':int(no_loss.sum()),
        'strict_gain_without_axis_loss_rate':float(gain.mean()),
        'strict_gain_without_axis_loss_count':int(gain.sum()),
        'pca_fallback_rate':sum(r['chosen']=='PCA' for r in records)/len(records),
        'mean_heldout_axis_differences':d.mean(0).tolist(),
        'boundary':'Held-out observed metric comparisons under fixed fitted embeddings; overlapping folds are not independent experiments.'}
