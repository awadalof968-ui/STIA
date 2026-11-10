"""Reversible group-list nuisance-order guard around a configured native fit.

This does not alter STAligner's objective. Callers must configure deterministic
native kernels and single-thread neighbor search separately. Group identities
must be stable physical sections, not fabricated technical batches.
"""
from __future__ import annotations
import numpy as np
from .order_repair import build_canonical_order_plan


def fit_group_order_guard(adata, *, rebuild_graph, fit, group_column='batch_name', embedding_key='STAligner'):
    """Fit a canonical copy, restoring the original caller row order exactly.

    Explicit graph rebuilding is mandatory because row-indexed graph arrays
    in ``uns`` do not automatically update when AnnData rows are permuted.
    The input object is never trained or modified in-place. ``fit`` is a
    configured native trainer callable accepting/returning AnnData.
    """
    if group_column not in adata.obs: raise ValueError('Stable section/group column required')
    ids=np.asarray(adata.obs_names,dtype=str); groups=adata.obs[group_column].astype(str).to_numpy()
    plan=build_canonical_order_plan(ids,groups,axis='group')
    canonical=adata[np.asarray(plan.forward_indices,dtype=int)].copy()
    expected=np.asarray(canonical.obs_names,dtype=str).copy()
    canonical=rebuild_graph(canonical)
    if not np.array_equal(np.asarray(canonical.obs_names,dtype=str),expected): raise ValueError('Graph builder changed row identities')
    fitted=fit(canonical)
    if not np.array_equal(np.asarray(fitted.obs_names,dtype=str),expected): raise ValueError('Native trainer changed row identities')
    z=np.asarray(fitted.obsm[embedding_key])
    if z.ndim!=2 or z.shape[0]!=len(ids) or not np.isfinite(z).all() or np.linalg.norm(z-z.mean(axis=0))==0:
        raise ValueError('Native embedding is invalid, nonfinite or collapsed')
    output=adata.copy(); output.obsm[embedding_key]=plan.restore_rows(z)
    receipt={**plan.receipt(),'method':'STAligner','scope':'group_list_nuisance_permutation_only',
        'within_group_order':'preserved, not repaired','graph_rebuilt':True,'input_modified':False,
        'native_runtime_required':'Fixed seed/hash seed, deterministic kernels and single-thread neighbor search; independent process per concurrent fit.'}
    output.uns['stia_group_order_guard']=receipt
    return output,receipt
