"""Experimental cohort-balanced utility regression; not the default STIA audit."""
from __future__ import annotations
import numpy as np
from .task_selection_v1 import preferences


def training_features(means, standard_errors, weights, tolerances, n_train, n_test, reference=0):
    x,w,tol=preferences(means,weights,tolerances,reference)
    se=np.asarray(standard_errors,float)
    if se.shape!=x.shape or not np.isfinite(se).all() or np.any(se<0) or np.any(se[reference]!=0):
        raise ValueError('Finite paired nonnegative SE, with zero reference SE, required')
    if not np.isfinite([n_train,n_test]).all() or n_train<2 or n_test<1:
        raise ValueError('Actual positive independent-unit counts required')
    d=x-x[reference];margin=d+tol
    std=np.clip(margin/np.maximum(se,1e-6),-10,10)
    columns=[d@w,margin.min(1),margin.max(1),np.maximum(-margin,0).max(1),
             np.abs(d).mean(1),np.sqrt(((se*w)**2).sum(1)),se.max(1),se.mean(1),
             std.min(1),std.mean(1)]
    constants=[x.shape[1]/3,np.log1p(n_train)/5,np.log1p(n_test)/5,
               np.log1p(len(x))/3,tol.mean(),tol.max()]
    return np.column_stack(columns+[np.full(len(x),c) for c in constants])


def cohort_weights(cohorts,profiles):
    c=np.asarray(cohorts);p=np.asarray(profiles)
    if c.ndim!=1 or p.shape!=c.shape or not len(c):raise ValueError('Aligned row identities required')
    out=np.zeros(len(c),float);groups=np.unique(c)
    for g in groups:
        tasks=np.unique(p[c==g])
        for task in tasks:
            mask=(c==g)&(p==task)
            out[mask]=1/(len(groups)*len(tasks)*mask.sum())
    return out


def fit_ridge(features,outcomes,weights,alpha):
    x=np.asarray(features,float);y=np.asarray(outcomes,float);w=np.asarray(weights,float)
    if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or not len(x):
        raise ValueError('Aligned feature matrix, outcomes and weights required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Finite observations and nonnegative positive-total weights required')
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('Positive ridge regularization required')
    w=w/w.sum();mean=w@x;scale=np.sqrt(w@((x-mean)**2));scale[scale<1e-12]=1
    z=(x-mean)/scale;design=np.column_stack([np.ones(len(x)),z])
    penalty=np.diag([0.]+[alpha]*x.shape[1])
    coef=np.linalg.solve(design.T@(w[:,None]*design)+penalty,design.T@(w*y))
    return {'mean':mean.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),'alpha':float(alpha)}


def predict_ridge(model,features):
    x=np.asarray(features,float);mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
    if x.ndim!=2 or x.shape[1]!=len(mean) or not np.isfinite(x).all():
        raise ValueError('Aligned finite features required')
    return np.column_stack([np.ones(len(x)),(x-mean)/scale])@np.asarray(model['coef'])


def choose_predictions(predictions,reference=0):
    p=np.array(predictions,dtype=float,copy=True)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or not 0<=reference<len(p):
        raise ValueError('Finite candidate utility predictions required')
    p[reference]=0.;best=p.max();ties=np.flatnonzero(np.abs(p-best)<=1e-12)
    return int(reference if reference in ties else ties[0])
