"""Computed numerical/reference-relative gates; never biological certification.

Calibrations carry raw, hash-checked neutral controls. Action limits are
recomputed here, not accepted from an inline claimed PASS or chosen cutoff.
All spatial label readouts are annotation proxies; donor AUC is descriptive.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples
from .io import resolve_input, sha256_file, load_table
from .disease_metric_v1 import donor_batch_separation_auc
from .diagnostics import EvidenceFault

NUMERIC_FLOOR = 100 * float(np.finfo(np.float32).eps)
AXES = ('conditional_batch_asw','annotation_neighbor_purity','disease_separation_auc')
IDENTITIES = ('spot_id','sample_id','donor_id','disease_or_condition','technical_batch')


def checked_file(root, spec):
    if not isinstance(spec, dict) or not spec.get('path') or not spec.get('sha256'):
        raise EvidenceFault('External evidence requires path and frozen sha256', 'custody', 'evidence_contract')
    path = resolve_input(Path(root), spec['path'])
    if sha256_file(path) != spec['sha256']:
        raise EvidenceFault(f'Evidence sha256 mismatch: {path.name}', 'custody', 'evidence_hash_mismatch')
    return path


def valid_array(values):
    z=np.asarray(values,dtype=np.float64)
    if z.ndim!=2 or min(z.shape)<2 or not np.isfinite(z).all():
        raise EvidenceFault('Embedding must be 2D, finite and have at least two rows/dimensions', 'quality', 'invalid_embedding')
    if np.linalg.norm(z-z.mean(axis=0))==0:
        raise EvidenceFault('Embedding is collapsed', 'quality', 'collapsed_embedding')
    return z


def aligned_array(root, spec, metadata, cfg=None):
    cfg=cfg or {}; sid=cfg.get('spot_id','spot_id')
    p=checked_file(root,spec)
    with np.load(p,allow_pickle=False) as a:
        z=valid_array(a[spec.get('latent_key','latent')]); ids=a[spec.get('spot_ids_key','spot_ids')].astype(str)
    target=metadata[sid].astype(str).to_numpy()
    if len(ids)!=len(z) or len(set(ids))!=len(ids) or len(set(target))!=len(target) or set(ids)!=set(target):
        raise EvidenceFault('Embedding spot identity is not a bijection with metadata', 'identity', 'embedding_bijection')
    ix={v:i for i,v in enumerate(ids)}
    return z[[ix[v] for v in target]]


def normalized_metadata(metadata,cfg):
    mapping={cfg.get(c,c):c for c in IDENTITIES}
    out=metadata.rename(columns=mapping).copy()
    if not set(IDENTITIES).issubset(out.columns) or out[list(IDENTITIES)].isna().any().any():
        raise EvidenceFault('Missing metadata identities', 'identity', 'metadata_identity')
    if out.spot_id.astype(str).duplicated().any():
        raise EvidenceFault('Metadata spot identity is not unique', 'identity', 'metadata_identity')
    return out


def neighbor_sets(values, ids, k=10):
    z=valid_array(values); order=np.argsort(np.asarray(ids,dtype=str),kind='stable')
    x=z[order]; distances=cdist(x,x,metric='sqeuclidean'); np.fill_diagonal(distances,np.inf)
    nearest=np.argsort(distances,axis=1,kind='stable')[:,:min(k,len(x)-1)]
    result=[None]*len(x)
    for i,row in enumerate(nearest): result[order[i]]=set(order[row].tolist())
    return result


def compare_arrays(a,b,groups,ids=None):
    from invest_st.metrics.osi import osi
    a=valid_array(a); b=valid_array(b)
    if len(a)!=len(b) or len(groups)!=len(a): raise ValueError('Pair comparison rows mismatch')
    geometry=float(osi({'reference':a,'candidate':b},{'reference':groups,'candidate':groups})['osi'])
    ids=np.arange(len(a)).astype(str) if ids is None else ids
    na=neighbor_sets(a,ids); nb=neighbor_sets(b,ids)
    mismatch=1-float(np.mean([len(x&y)/len(x|y) for x,y in zip(na,nb)]))
    if not np.isfinite(geometry): raise ValueError('Nonfinite geometry readout')
    return {'geometry':geometry,'neighbor_mismatch':mismatch}


def axis_scores(values, metadata, label_column, cfg=None):
    cfg=cfg or {}; z=valid_array(values); m=normalized_metadata(metadata,cfg).reset_index(drop=True)
    if len(z)!=len(m) or label_column not in m or m[label_column].isna().any():
        raise ValueError('Missing or mismatched annotation strata')
    m['_row']=np.arange(len(m)); labels=m[label_column].astype(str).to_numpy()
    # Technical mixing is evaluated only within condition x annotation strata.
    # This is a descriptive cell-weighted index, not cell-level inference.
    vals=[]
    for _,part in m.groupby(['disease_or_condition',label_column],sort=True):
        batch=part.technical_batch.astype(str).to_numpy(); ix=part['_row'].to_numpy(dtype=int)
        counts=pd.Series(batch).value_counts()
        if len(counts)>1 and counts.min()>=2 and len(counts)<len(ix):
            vals.extend((1-np.abs(silhouette_samples(z[ix],batch))).tolist())
    if not vals: raise EvidenceFault('No eligible condition x annotation batch-mixing stratum', 'design', 'ineligible_strata')
    neigh=neighbor_sets(z,m.spot_id.astype(str).to_numpy())
    purity=float(np.mean([np.mean(labels[list(row)]==labels[i]) for i,row in enumerate(neigh)]))
    disease=donor_batch_separation_auc(z,m)
    return {'conditional_batch_asw':float(np.mean(vals)), 'annotation_neighbor_purity':purity,
            'disease_separation_auc':disease['score']}


def calibrate_controls(root,cal):
    if cal.get('schema')!='stia.numeric_reference_calibration.v1': raise ValueError('Invalid calibration schema')
    m=load_table(checked_file(root,cal['metadata'])); cfg=cal.get('column_mapping',{})
    ref=aligned_array(root,cal['reference'],m,cfg); label=cal['label_column']
    base=axis_scores(ref,m,label,cfg); nm=normalized_metadata(m,cfg)
    controls=cal.get('controls',[]); names=[v.get('name') for v in controls]
    if len(controls)<4 or len(set(names))!=len(names): raise ValueError('At least four uniquely named numerical controls required')
    rows=[]
    for spec in controls:
        z=aligned_array(root,spec,m,cfg)
        transforms={'identity':ref,'rotation':-ref,'translation':ref+5,'scale':ref*2.5,
                    'float32':ref.astype(np.float32).astype(np.float64)}
        if spec['name'] not in transforms or not np.array_equal(z,transforms[spec['name']]):
            raise ValueError('Calibration control is not the declared predefined numerical neutral transform')
        rows.append({'name':spec['name'],'scores':axis_scores(z,m,label,cfg),
                     'robustness':compare_arrays(ref,z,nm.sample_id.to_numpy(),nm.spot_id.to_numpy())})
    # Absolute numerical envelope is conservative for both sign directions.
    rules={k:max(abs(row['scores'][k]-base[k]) for row in rows)+NUMERIC_FLOOR for k in AXES}
    rules.update({k:max(row['robustness'][k] for row in rows)+NUMERIC_FLOOR for k in ('geometry','neighbor_mismatch')})
    return {'rules':rules,'reference_scores':base,'controls':rows,'numeric_floor':NUMERIC_FLOOR,
            'scope':'Numerical neutral-control equivalence only; not a biological effect cutoff or significance test.'}


def verify_calibration(root,spec):
    if spec is None:
        raise FileNotFoundError('Frozen numerical calibration is unavailable')
    p=checked_file(root,spec)
    cal=json.loads(p.read_text(encoding='utf-8')); value=calibrate_controls(p.parent,cal)
    declared=cal.get('rules')
    if not isinstance(declared,dict) or set(declared)!=set(value['rules']) or any(
        not np.isfinite(float(declared[k])) or not np.isclose(float(declared[k]),v,rtol=0,atol=1e-12)
        for k,v in value['rules'].items()):
        raise EvidenceFault('Calibration rules differ from independently recomputed numerical controls', 'custody', 'calibration_rules_mismatch')
    value['artifact_sha256']=sha256_file(p)
    return value


def reference_guard(root,cfg,metadata,values):
    spec=cfg.get('three_axis_reference')
    if not isinstance(spec,dict): raise FileNotFoundError('Frozen three_axis_reference is required')
    if spec.get('scope')!='reference_relative_only': raise ValueError('Reference guard must declare reference_relative_only scope')
    p=checked_file(root,spec['metadata']); refmeta=load_table(p)
    m=normalized_metadata(metadata,cfg); rm=normalized_metadata(refmeta,cfg)
    if len(m)!=len(rm) or set(m.spot_id.astype(str))!=set(rm.spot_id.astype(str)):
        raise EvidenceFault('Frozen reference metadata spot identity mismatch', 'identity', 'reference_identity')
    rm=rm.set_index(rm.spot_id.astype(str)).loc[m.spot_id.astype(str)].reset_index(drop=True)
    label=str(cfg.get('layer_column','cell_type'))
    for col in (*IDENTITIES,label):
        if col not in m or col not in rm or not np.array_equal(m[col].astype(str).to_numpy(),rm[col].astype(str).to_numpy()):
            raise EvidenceFault(f'Frozen reference metadata differs at {col}', 'custody', 'reference_metadata_mismatch')
    # Load using original names for column-mapping compatibility.
    canonical=refmeta.set_index(refmeta[cfg.get('spot_id','spot_id')].astype(str)).loc[m.spot_id.astype(str)].reset_index(drop=True)
    reference=aligned_array(root,spec['embedding'],canonical,cfg)
    scores=axis_scores(values,metadata,label,cfg); base=axis_scores(reference,canonical,label,cfg)
    calibration=verify_calibration(root,cfg.get('calibration'))
    failed=[k for k in AXES if base[k]-scores[k]>calibration['rules'][k]]
    return {'status':'FAIL' if failed else 'PASS','scope':'reference_relative_only',
            'scores':scores,'reference_scores':base,'failed_metrics':failed,
            'rules':calibration['rules'],'calibration_sha256':calibration['artifact_sha256'],
            'reference_metadata_sha256':sha256_file(p),'reference_embedding_sha256':spec['embedding']['sha256'],
            'claim_boundary':'Preserves these measured indices relative to this reference within numerical tolerance. Not clinical truth, disease causality, human spatial gold, or biological equivalence.'}


def seed_guard(root,cfg,metadata):
    specs=cfg.get('robustness_replicates')
    if not isinstance(specs,list) or len(specs)<3: raise FileNotFoundError('At least three explicit seed replicates are required')
    names=[s.get('name') for s in specs]; seeds=[s.get('seed') for s in specs]
    if any(n is None for n in names) or any(s is None for s in seeds) or len(set(names))!=len(names) or len(set(seeds))!=len(seeds):
        raise EvidenceFault('Replicate names and seed identities must be unique and explicit', 'stability', 'replicate_contract')
    m=normalized_metadata(metadata,cfg); arrays=[aligned_array(root,s,metadata,cfg) for s in specs]
    calibration=verify_calibration(root,cfg.get('calibration')); rules=calibration['rules']
    group_col=str(cfg.get('osi_group_id',cfg.get('sample_id','sample_id')))
    group_col=next((col for col in IDENTITIES if cfg.get(col,col)==group_col),group_col)
    rows=[]
    for i,j in itertools.combinations(range(len(arrays)),2):
        vals=compare_arrays(arrays[i],arrays[j],m[group_col].to_numpy(),m.spot_id.to_numpy())
        rows.append({'left':names[i],'right':names[j],**vals,'failed':any(vals[k]>rules[k] for k in vals)})
    return {'status':'FAIL' if any(r['failed'] for r in rows) else 'PASS','pairs':rows,
            'calibration_sha256':calibration['artifact_sha256'],'rules':rules,
            'claim_boundary':'Repeat consistency of supplied identity-bound arrays; not proof that the declared seeds were really used during training.'}
