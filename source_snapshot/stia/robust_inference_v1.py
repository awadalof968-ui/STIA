"""Optional studentized paired-row bootstrap for population mean differences.

Resampling is asymptotically justified, not exact for arbitrary small-sample
asymmetric distributions. Never substitute this estimand with a median or rank.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .inference_optimization_v1 import holm_adjust, _jsonable


def infer_from_pvalues(p_up, p_down, alpha=.05):
    up,down=np.asarray(p_up,float),np.asarray(p_down,float)
    if up.shape!=down.shape or up.ndim<1 or up.shape[-1]<1:
        raise ValueError('Aligned nonempty directional p-value families required')
    if not 0<alpha<1 or not np.isfinite(up).all() or not np.isfinite(down).all() or np.any((up<0)|(up>1)|(down<0)|(down>1)):
        raise ValueError('Finite p-values in [0,1] and alpha in (0,1) required')
    m=up.shape[-1]
    h=holm_adjust(np.concatenate((up,down),axis=-1))
    bu,bd=up<alpha/(2*m),down<alpha/(2*m)
    hu,hd=h[...,:m]<alpha,h[...,m:]<alpha
    p=np.maximum(np.minimum(1,m*up.min(-1)),np.minimum(1,m*down.min(-1)))
    return {'bonferroni_up':bu,'bonferroni_down':bd,'bonferroni_tradeoff':bu.any(-1)&bd.any(-1),
            'holm_up':hu,'holm_down':hd,'holm_tradeoff':hu.any(-1)&hd.any(-1),
            'holm_adjusted_directional_p':h,'iut_p':p,'iut_tradeoff':p<alpha}


def resampled_statistics(centered, indices):
    x=np.asarray(centered,float);ix=np.asarray(indices)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all() or ix.ndim!=2 or ix.shape[1]!=len(x):
        raise ValueError('Complete centered rows and resampling index matrix required')
    if not np.issubdtype(ix.dtype,np.integer) or np.any((ix<0)|(ix>=len(x))):
        raise ValueError('Resampling indices out of range')
    y=x[ix];mu=y.mean(1);se=y.std(1,ddof=1)/np.sqrt(len(x))
    with np.errstate(divide='ignore',invalid='ignore'):
        out=mu/se
    return np.where((se==0)&(mu==0),0,out)


def paired_bootstrap(samples, replicates=1999, seed=20261011, alpha=.05):
    x=np.asarray(samples,float)
    if x.ndim!=2 or len(x)<4 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('At least four complete independent-unit rows required')
    if not isinstance(replicates,(int,np.integer)) or replicates<99:
        raise ValueError('At least 99 integer bootstrap replicates required')
    mu=x.mean(0);se=x.std(0,ddof=1)/np.sqrt(len(x))
    if np.any(se<=0):raise ValueError('Constant axes cannot establish population variance')
    observed=mu/se
    ix=np.random.default_rng(seed).integers(0,len(x),size=(replicates,len(x)))
    stars=resampled_statistics(x-mu,ix)
    # Both tails are counted separately, including ties; do not assume symmetry.
    up=(1+np.sum(stars>=observed,axis=0))/(replicates+1)
    down=(1+np.sum(stars<=observed,axis=0))/(replicates+1)
    result=infer_from_pvalues(up,down,alpha)
    result.update(mean=mu,standard_error=se,p_up=up,p_down=down,
                  replicates=int(replicates),seed=int(seed),alpha=alpha,
                  monte_carlo_minimum_p=1/(replicates+1),
                  infinite_resampled_statistics=int(np.isinf(stars).sum()),
                  interpretation='Mean differences; asymptotic bootstrap calibration, not finite-sample distribution-free.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--replicates',type=int,default=1999);p.add_argument('--seed',type=int,default=20261011)
    a=p.parse_args();r=paired_bootstrap(np.load(a.input,allow_pickle=False),a.replicates,a.seed)
    r['input_sha256']=__import__('hashlib').sha256(Path(a.input).read_bytes()).hexdigest()
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(_jsonable(r),f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()
