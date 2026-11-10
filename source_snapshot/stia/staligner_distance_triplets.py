"""Experimental nearest-positive triplets: unchanged loss, new selection policy.
Never described as a validated repair before complete independent analysis.
"""
from __future__ import annotations
import hashlib, inspect, itertools, json
import numpy as np
from .staligner_identity_triplets import identity_triplet_indices, digest

def distance_triplet_indices(mapping,ids,groups,representation,*,seed,update):
    ids=np.asarray(ids,dtype=str); groups=np.asarray(groups,dtype=str)
    z=np.asarray(representation,dtype=np.float64)
    if z.ndim!=2 or z.shape[0]!=len(ids) or not np.isfinite(z).all():
        raise ValueError('Finite row-aligned mining representation required')
    _,_,_,record=identity_triplet_indices(mapping,ids,groups,seed=seed,update=update)
    lookup={v:i for i,v in enumerate(ids)}; neighbors={}
    for mp in mapping.values():
        for a,ps in mp.items():
            a=str(a); pair=tuple(sorted((str(groups[lookup[a]]),str(groups[lookup[str(ps[0])]]))))
            key=(pair,a); neighbors.setdefault(key,set()).update(map(str,ps))
    rows=[]
    for pair,a,_,n in record['identity_triplets']:
        candidates=neighbors[(tuple(pair),a)]
        def score(p):
            difference=z[lookup[a]]-z[lookup[p]]
            return float(np.dot(difference,difference)),p
        rows.append((pair,a,min(candidates,key=score),n))
    ix=np.asarray([[lookup[v] for v in row[1:]] for row in rows],dtype=np.int64)
    result={'update':int(update),'identity_triplets':rows,'triplet_sha256':digest(rows),'count':len(rows),
            'positive_policy':'minimum_squared_latent_distance_then_stable_ID',
            'negative_policy':'unchanged_previous_identity_hash_seed_update_pair_anchor'}
    return ix[:,0],ix[:,1],ix[:,2],result

def train_distance_triplet_candidate(adata, *, precision='float32', trace=None, **kwargs):
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
        a,p,n,record=distance_triplet_indices(mapping,obj.obs_names,obj.obs.batch_name,obj.obsm['STAGATE'],seed=seed,update=update)
        trace.append({'type':'mining_representation','update':int(update),'latent':np.asarray(obj.obsm['STAGATE']).copy()})
        trace.append({'type':'triplets',**record}); return a,p,n
    global_vars.update(Data=data_ctor,STAligner=model_ctor,create_dictionary_mnn=dictionary,_stia_triplets=triplets,tqdm=lambda it,*a,**kw:it)
    exec(compile(source,'<stia_staligner_identity_triplet_candidate>','exec'),global_vars)
    # Legacy final z is intentionally retained for matched endpoint semantics.
    return global_vars['train_STAligner'](adata,**kwargs)


