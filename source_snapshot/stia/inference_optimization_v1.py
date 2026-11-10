"""Optional paired-axis inference with explicit directional/global error families.

Holm (1979) supplies the multiple-testing procedure. The global tradeoff
intersection-union test tests a different, weaker claim than axis localization.
The existing STIA engine and descriptive score reports are unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def holm_adjust(pvalues):
    """Holm adjusted p-values; last dimension is one complete fixed family."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim < 1 or p.shape[-1] < 1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('A finite nonempty p-value family in [0,1] is required')
    order = np.argsort(p, axis=-1, kind='stable')
    ranked = np.take_along_axis(p, order, axis=-1)
    adjusted = np.minimum(1., np.maximum.accumulate(ranked*np.arange(p.shape[-1],0,-1), axis=-1))
    return np.take_along_axis(adjusted, np.argsort(order, axis=-1), axis=-1)


def infer_axis_statistics(mean, standard_error, degrees_freedom, alpha=.05):
    """Last dimension: prespecified axes; leading dimensions: distinct families.

Zero-SE axes are refused: constant observations do not establish a population
variance model. Missing axes must be removed according to the design beforehand.
"""
    mu, se = np.asarray(mean,float), np.asarray(standard_error,float)
    df = np.asarray(degrees_freedom,float)
    if mu.shape != se.shape or mu.ndim<1 or mu.shape[-1]<1:
        raise ValueError('Aligned nonempty axis mean and SE arrays are required')
    if not np.isfinite(mu).all() or not np.isfinite(se).all() or np.any(se<=0):
        raise ValueError('Finite means and strictly positive standard errors required')
    if not np.isfinite(df).all() or np.any(df<=0) or not 0<alpha<1:
        raise ValueError('Positive degrees of freedom and alpha in (0,1) required')
    try:
        df = np.broadcast_to(df, mu.shape[:-1])[...,None]
    except ValueError as exc:
        raise ValueError('Degrees of freedom must align with the families') from exc
    statistic = mu/se
    up, down = t.sf(statistic,df), t.cdf(statistic,df)
    elementary = np.concatenate((up,down),axis=-1)
    m=mu.shape[-1]
    bonf=np.minimum(1,elementary*(2*m));holm=holm_adjust(elementary)
    global_p=np.maximum(np.minimum(1,m*up.min(-1)),np.minimum(1,m*down.min(-1)))
    result={'mean':mu,'standard_error':se,'p_improvement':up,'p_deterioration':down,
            'holm_adjusted_directional_p':holm,'bonferroni_adjusted_directional_p':bonf,
            'tradeoff_iut_p':global_p,'alpha':alpha,'directional_family_size':2*m}
    masks={'point':(mu>0,mu<0),
           'marginal':(up<alpha/2,down<alpha/2),
           'bonferroni_directional':(bonf[...,:m]<alpha,bonf[...,m:]<alpha),
           'holm_directional':(holm[...,:m]<alpha,holm[...,m:]<alpha)}
    for name,(positive,negative) in masks.items():
        result[name]={'improved':positive,'deteriorated':negative,
                      'tradeoff':positive.any(-1)&negative.any(-1)}
    # Global IUT does not export extra axis-specific significant declarations.
    result['tradeoff_iut']={'tradeoff':global_p<alpha,
                            'directional_localization_policy':'holm_directional'}
    return result


def infer_axes(samples, alpha=.05):
    x=np.asarray(samples,float)
    if x.ndim!=2 or len(x)<2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Complete independent-unit by eligible-axis paired differences required')
    return infer_axis_statistics(x.mean(0),x.std(0,ddof=1)/np.sqrt(len(x)),len(x)-1,alpha)


def _jsonable(value):
    if isinstance(value,dict):return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='NPY independent-unit by paired-axis differences')
    parser.add_argument('--output',required=True,help='New JSON output path; existing files are refused')
    parser.add_argument('--alpha',type=float,default=.05)
    args=parser.parse_args()
    report=infer_axes(np.load(args.input,allow_pickle=False),args.alpha)
    with Path(args.output).open('x',encoding='utf-8') as file:
        json.dump(_jsonable(report),file,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__':main()
