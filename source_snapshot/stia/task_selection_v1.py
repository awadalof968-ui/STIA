"""Optional explicit task preferences; no universal best method is assumed."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .real_axis_evaluation_v1 import select_from_training

POLICIES=('reference','weighted_mean','original_range_mean','original_hard_guard',
          'point_constrained','t_bound_constrained','shortfall_penalty_1',
          'shortfall_penalty_4','shortfall_penalty_16')

def preferences(means,weights,tolerances,reference):
    x=np.asarray(means,float);w=np.asarray(weights,float);tol=np.asarray(tolerances,float)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Complete finite method by eligible-axis means required')
    if w.shape!=(x.shape[1],) or tol.shape!=w.shape or not np.isfinite(w).all() or not np.isfinite(tol).all():
        raise ValueError('Prespecified weights and tolerances must align with axes')
    if np.any(w<0) or w.sum()<=0 or np.any(tol<0) or not 0<=reference<len(x):
        raise ValueError('Nonnegative tolerances/weights and valid reference required')
    return x,w/w.sum(),tol


def select_task(means,weights,tolerances,policy='point_constrained',reference=0,
                standard_errors=None,df=None,alpha=.05):
    x,w,tol=preferences(means,weights,tolerances,reference)
    if policy not in POLICIES:raise ValueError('Unknown task policy')
    d=x-x[reference];utility=d@w;lower=d.copy()
    if policy=='t_bound_constrained':
        se=np.asarray(standard_errors,float)
        if se.shape!=d.shape or not np.isfinite(se).all() or np.any(se<0) or df is None or not np.isfinite(df) or df<=0 or not 0<alpha<1:
            raise ValueError('Aligned paired SE, positive df and alpha required')
        if np.any(se[reference]!=0):raise ValueError('Self-reference SE must be zero')
        family=max(1,(len(x)-1)*x.shape[1])
        lower=d-t.ppf(1-alpha/family,df)*se
        # Zero observed variance does not certify a nonreference population mean.
        lower[se==0]=-np.inf;lower[reference]=0
    eligible=np.all(lower>=-tol,axis=1)
    if policy=='reference':chosen=reference
    elif policy in ('original_range_mean','original_hard_guard'):
        chosen=select_from_training(x,'mean' if policy=='original_range_mean' else 'guarded',reference)
    else:
        if policy.startswith('shortfall_penalty_'):
            utility=utility-float(policy.rsplit('_',1)[1])*np.maximum(-(d+tol),0).max(1)
        pool=np.flatnonzero(eligible) if policy in ('point_constrained','t_bound_constrained') else np.arange(len(x))
        if not len(pool):raise ValueError('Reference fallback was not feasible')
        best=utility[pool].max();tied=pool[np.abs(utility[pool]-best)<=1e-12]
        chosen=int(reference if reference in tied else tied[0])
    return {'selected':int(chosen),'eligible':eligible.tolist(),'lower':lower,
            'weighted_gain_estimates':d@w,'ranking_utility':utility,'weights':w,'tolerances':tol,
            'reference':int(reference),'policy':policy}


def evaluate_task(test_means,selected,weights,tolerances,reference=0,loss_weight=4.):
    x,w,tol=preferences(test_means,weights,tolerances,reference)
    if not 0<=selected<len(x) or not np.isfinite(loss_weight) or loss_weight<0:
        raise ValueError('Invalid selection or task loss weight')
    d=x-x[reference];gains=d@w;short=np.maximum(-(d+tol),0).max(1)
    feasible=short<=1e-12
    best=np.max(gains[feasible]) if np.any(feasible) else None
    return {'weighted_gain':float(gains[selected]),'maximum_shortfall':float(short[selected]),
            'violated':bool(short[selected]>1e-12),
            'task_utility':float(gains[selected]-loss_weight*short[selected]),
            'nonreference':bool(selected!=reference),'selected_feasible':bool(feasible[selected]),
            'nonreference_feasible_exists':bool(np.any(feasible & (np.arange(len(x))!=reference) & (gains>1e-12))),
            'selected_positive_feasible':bool(feasible[selected] and gains[selected]>1e-12),
            'feasible_oracle_regret':float(max(0,best-gains[selected])) if best is not None else None}


def tune_other_datasets(scores,target,candidates):
    sources=sorted(k for k in scores if k!=target)
    if not sources:raise ValueError('Other complete datasets with this task are required')
    utility={c:float(np.mean([scores[s][c] for s in sources])) for c in candidates}
    if not all(np.isfinite(v) for v in utility.values()):raise ValueError('Nonfinite tuning scores')
    best=max(utility.values());chosen=next(c for c in candidates if abs(utility[c]-best)<=1e-12)
    return {'selected':chosen,'sources':sources,'equal_dataset_validation_utilities':utility}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='JSON means, weights, tolerances, optional SE/df and identities')
    p.add_argument('--output',required=True);p.add_argument('--policy',choices=POLICIES,default='point_constrained')
    a=p.parse_args();data=json.loads(Path(a.input).read_text(encoding='utf-8-sig'))
    r=select_task(data['means'],data['weights'],data['tolerances'],a.policy,
                  reference=data.get('reference',0),standard_errors=data.get('standard_errors'),df=data.get('df'))
    r['schema_version']='stia.task_selection.v1'
    for key,expected in [('methods',len(data['means'])),('axes',len(data['weights']))]:
        names=data.get(key)
        if names is None or len(names)!=expected or len(set(names))!=expected:
            raise ValueError('Explicit unique method and axis identities required')
        r[key]=names
    r['input_sha256']=__import__('hashlib').sha256(Path(a.input).read_bytes()).hexdigest()
    def serialize(x):
        if isinstance(x,np.ndarray):return serialize(x.tolist())
        if isinstance(x,list):return [serialize(v) for v in x]
        if isinstance(x,dict):return {k:serialize(v) for k,v in x.items()}
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(serialize(r),f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()
