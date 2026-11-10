"""Experimental mechanism-targeted STAligner triplet patch, not a proven repair.

Model objective and supplied row order are unchanged. Native trainer source is
cloned in memory, with a guarded exact substitution of its sampling block.
The installed package is never edited. Positive and pseudo-random negative
selection now depend on stable entity identity instead of traversal order.
"""
from __future__ import annotations
import hashlib
import inspect
import itertools
import json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def identity_triplet_indices(mapping,ids,groups,*,seed,update):
    ids=np.asarray(ids,dtype=str); groups=np.asarray(groups,dtype=str)
    if ids.ndim!=1 or len(ids)!=len(groups) or len(set(ids))!=len(ids) or len(ids)<2:
        raise ValueError('Unique row-aligned stable IDs required')
    lookup={v:i for i,v in enumerate(ids)}; pools={g:sorted(ids[groups==g].tolist()) for g in set(groups)}
    triples={}
    for neighbors in mapping.values():
        for anchor,positives in neighbors.items():
            anchor=str(anchor); positives=sorted(set(map(str,positives)))
            if anchor not in lookup or not positives or any(p not in lookup for p in positives): raise ValueError('Unknown or missing MNN identity')
            own=groups[lookup[anchor]]; other=set(groups[[lookup[p] for p in positives]])
            if len(other)!=1 or own in other: raise ValueError('MNN positives must belong to one different section')
            pair=tuple(sorted((str(own),str(next(iter(other)))))); key=(pair,anchor)
            token=digest([int(seed),int(update),list(pair),anchor]); pool=pools[own]
            negative=pool[int(token,16)%len(pool)]; value=(pair,anchor,positives[0],negative)
            if key in triples and triples[key]!=value: raise ValueError('Conflicting duplicate pair-anchor evidence')
            triples[key]=value
    ordered=[triples[k] for k in sorted(triples)]
    if not ordered: raise ValueError('No native MNN triplets; cannot label collapsed alignment as invariant')
    ix=np.array([[lookup[v] for v in row[1:]] for row in ordered],dtype=np.int64)
    return ix[:,0],ix[:,1],ix[:,2],{'update':int(update),'identity_triplets':ordered,'triplet_sha256':digest(ordered),'count':len(ordered)}


def train_identity_triplet_candidate(adata, *, precision='float32', trace=None, **kwargs):
    import importlib
    native=importlib.import_module('STAligner.train_STAligner')
    if precision not in ('float32','float64'): raise ValueError('Unsupported precision')
    trace=[] if trace is None else trace
    source=inspect.getsource(native.train_STAligner)
    start=source.index('            anchor_ind = []\n')
    stop=source.index('\n        model.train()',start)
    oldblock=source[start:stop]
    if 'np.random.randint(section_size)' not in oldblock or 'positive_spot = mnn_dict' not in oldblock:
        raise ValueError('Native source does not match reviewed sampling block')
    source=source[:start]+('            anchor_ind, positive_ind, negative_ind = _stia_triplets(mnn_dict, adata, random_seed, epoch)\n')+source[stop:]
    global_vars=dict(native.train_STAligner.__globals__)
    original_data=global_vars['Data']; original_model=global_vars['STAligner']; original_mnn=global_vars['create_dictionary_mnn']
    def data_ctor(*a,**kw):
        data=original_data(*a,**kw)
        if precision=='float64': data.x=data.x.double()
        return data
    def model_ctor(*a,**kw):
        model=original_model(*a,**kw); h=hashlib.sha256()
        for name,value in sorted(model.state_dict().items()):
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
        trace.append({'type':'initialization','native_float32_parameters_sha256':h.hexdigest()})
        return model.double() if precision=='float64' else model
    canonical=sorted(adata.obs.batch_name.astype(str).unique()); actual=list(adata.obs.batch_name.astype(str).unique())
    pos={v:i for i,v in enumerate(actual)}; schedule=[(pos[a],pos[b]) for a,b in itertools.combinations(canonical,2)]
    def dictionary(*a,**kw):
        obj=a[0]; kw['iter_comb']=schedule
        result=original_mnn(*a,**kw)
        edges=sorted({tuple(sorted((str(anchor),str(p)))) for mp in result.values() for anchor,neighbors in mp.items() for p in neighbors})
        trace.append({'type':'mnn','edge_sha256':digest(edges),'edges':edges})
        if not any(r['type']=='pretraining' for r in trace): trace.append({'type':'pretraining','latent':np.asarray(obj.obsm['STAGATE']).copy()})
        return result
    def triplets(mapping,obj,seed,update):
        a,p,n,record=identity_triplet_indices(mapping,obj.obs_names,obj.obs.batch_name,seed=seed,update=update)
        trace.append({'type':'triplets',**record}); return a,p,n
    global_vars.update(Data=data_ctor,STAligner=model_ctor,create_dictionary_mnn=dictionary,_stia_triplets=triplets,tqdm=lambda it,*a,**kw:it)
    exec(compile(source,'<stia_staligner_identity_triplet_candidate>','exec'),global_vars)
    # Legacy final z is intentionally retained for matched endpoint semantics.
    return global_vars['train_STAligner'](adata,**kwargs)
