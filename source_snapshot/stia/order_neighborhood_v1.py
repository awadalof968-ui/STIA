"""Opt-in identity-aligned local-order diagnostic, independent of centroid OSI.

The fixed numerical envelope comprises representation-preserving transforms,
not biological-effect cutoffs or fitted outcome labels. The historical audit
engine and historical frozen benchmarks are deliberately left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist

VARIANT = '1.3.8+order-neighborhood.v1'
NUMERIC_FLOOR = 100 * float(np.finfo(np.float32).eps)
CONTROL_NAMES = ('identity', 'reflection', 'translation', 'scale', 'float32_roundtrip')


def _valid(values):
    z = np.asarray(values, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 2 or not np.isfinite(z).all():
        raise ValueError('Embedding must be finite, two-dimensional and nonempty')
    if np.linalg.norm(z - z.mean(axis=0)) == 0:
        raise ValueError('Collapsed embedding cannot establish order stability')
    return z


def _aligned(arms):
    if not isinstance(arms, dict) or len(arms) < 2:
        raise ValueError('At least two uniquely named order arms are required')
    result, target, shape = {}, None, None
    for name, pair in arms.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Order name must be explicit')
        z, ids = _valid(pair[0]), np.asarray(pair[1])
        if ids.ndim != 1 or len(ids) != len(z):
            raise ValueError('Order identity row counts mismatch')
        if any(v is None or isinstance(v, (float, np.floating)) and np.isnan(v) for v in ids):
            raise ValueError('Missing spot identity')
        ids = ids.astype(str)
        if any(not v.strip() for v in ids) or len(set(ids)) != len(ids):
            raise ValueError('Spot identities must be nonempty and unique')
        if target is None:
            target = np.asarray(sorted(ids), dtype=str)
            shape = z.shape
        if set(ids) != set(target) or z.shape != shape:
            raise ValueError('Order spot identity sets or embedding dimensions mismatch')
        positions = {v: i for i, v in enumerate(ids)}
        result[name] = z[[positions[v] for v in target]]
    return result, target


def _neighbors(z, ks, chunk_rows):
    # Translation/positive-scale normalization improves numerical conditioning.
    # Row order is already canonical, so stable sorting resolves distance ties
    # by spot identity rather than by original input row order.
    x = z - z.mean(axis=0)
    x /= np.linalg.norm(x)
    n = len(x)
    nearest = np.empty((n, max(ks)), dtype=np.int64)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        distances = cdist(x[start:stop], x, metric='sqeuclidean')
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.argsort(distances, axis=1, kind='stable')[:, :max(ks)]
    return {k: nearest[:, :k] for k in ks}


def _mismatch(a, b):
    k = a.shape[1]
    intersections = np.fromiter((len(set(x) & set(y)) for x, y in zip(a, b)), dtype=float, count=len(a))
    return float(np.mean(1 - intersections / (2 * k - intersections)))


def analyze_orders(arms, ks=(10, 30), chunk_rows=256):
    """Measure all-pair local-neighborhood drift after explicit ID alignment.

    FAIL means numerical order invariance was not established, not that an
    embedding is biologically inferior. Missing/invalid evidence raises;
    ``audit_extended`` converts it to a non-passing decision with its reason.
    """
    arrays, ids = _aligned(arms)
    ks = tuple(ks)
    if not ks or any(isinstance(k, bool) or not isinstance(k, Integral) or not 0 < k < len(ids) for k in ks) or len(set(ks)) != len(ks):
        raise ValueError('Neighbor sizes must be distinct integers in [1,n-1]')
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, Integral) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    ks = tuple(sorted(int(k) for k in ks))
    neighbors = {name: _neighbors(z, ks, chunk_rows) for name, z in arrays.items()}
    controls = []
    envelope = {k: 0.0 for k in ks}
    for name, z in arrays.items():
        transforms = (z, -z, z + 5, z * 2.5, z.astype(np.float32).astype(np.float64))
        for control_name, transformed in zip(CONTROL_NAMES, transforms):
            checked = _neighbors(_valid(transformed), ks, chunk_rows)
            values = {k: _mismatch(neighbors[name][k], checked[k]) for k in ks}
            controls.append({'arm': name, 'control': control_name,
                             'mismatch_by_k': {str(k): v for k, v in values.items()}})
            for k, value in values.items(): envelope[k] = max(envelope[k], value)
    limits = {k: envelope[k] + NUMERIC_FLOOR for k in ks}
    pairs = []
    for a, b in itertools.combinations(arrays, 2):
        values = {k: _mismatch(neighbors[a][k], neighbors[b][k]) for k in ks}
        pairs.append({'left': a, 'right': b, 'mismatch_by_k': {str(k): v for k, v in values.items()},
                      'detected': any(v > limits[k] for k, v in values.items())})
    detected = any(p['detected'] for p in pairs)
    return {'schema_version': 'stia.order_neighborhood.v1', 'variant': VARIANT,
            'status': 'FAIL' if detected else 'PASS', 'detected': detected,
            'scope': 'numerical_order_invariance_only', 'neighbor_sizes': list(ks),
            'n_spots': len(ids), 'n_orders': len(arrays),
            'spot_id_sha256': hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest(),
            'max_neighbor_mismatch': max(v for p in pairs for v in p['mismatch_by_k'].values()),
            'limits_by_k': {str(k): v for k, v in limits.items()},
            'numeric_floor': NUMERIC_FLOOR, 'control_names': list(CONTROL_NAMES),
            'controls': controls, 'pairs': pairs,
            'biological_harm_inferred': False}


def _checked_path(root, item, expected_hashes):
    value = item.get('path') if isinstance(item, dict) else item
    if not isinstance(value, str): raise ValueError('Order evidence requires a file path')
    path = (Path(root) / value).resolve()
    expected = expected_hashes.get(str(path))
    if not expected: raise ValueError('Frozen hash is required for every order file')
    if not path.is_file(): raise FileNotFoundError(str(path))
    from .io import sha256_file
    if sha256_file(path) != expected: raise ValueError('Order file hash mismatch')
    return path


def load_frozen_orders(root, cfg, expected_hashes):
    orders = cfg.get('orders')
    if isinstance(orders, dict): orders = [dict(item, name=name) for name, item in orders.items()]
    if not isinstance(orders, list) or len(orders) < 2:
        raise FileNotFoundError('At least two frozen order files are required')
    arrays = {}
    for item in orders:
        name = item.get('name')
        if not isinstance(name, str) or not name or name in arrays:
            raise ValueError('Unique explicit order names are required')
        embedding = _checked_path(root, item.get('embedding'), expected_hashes)
        identity = _checked_path(root, item.get('spot_ids'), expected_hashes)
        with np.load(embedding, allow_pickle=False) as archive:
            z = archive[item.get('embedding_key', 'latent')].copy()
        with np.load(identity, allow_pickle=False) as archive:
            ids = archive[item.get('spot_ids_key', 'spot_ids')].copy()
        arrays[name] = (z, ids)
    return arrays


def audit_extended(case_root, config, expected_hashes, output):
    """Opt-in full audit plus local-order guard; never overwrite an old bundle."""
    from .engine import audit
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = audit(case_root, config, output=out / 'base_audit')
    try:
        guard = analyze_orders(load_frozen_orders(case_root, config, expected_hashes))
    except FileNotFoundError as exc:
        guard = {'status': 'UNRESOLVED', 'reason': str(exc)}
    except (ValueError, TypeError, KeyError) as exc:
        guard = {'status': 'FAIL', 'reason': str(exc), 'evidence_invalid': True}
    decision = base.decision
    if decision == 'REJECT' or guard['status'] == 'FAIL': decision = 'REJECT'
    elif decision in ('ABSTAIN', 'REPAIR') or guard['status'] == 'UNRESOLVED': decision = 'ABSTAIN'
    value = {'schema_version': 'stia.extended_audit.v1', 'variant': VARIANT,
             'base_decision': base.decision, 'decision': decision, 'order_neighborhood': guard,
             'base_bundle': str(out / 'base_audit'), 'historical_default_audit_unchanged': True}
    with (out / 'EXTENDED_DECISION.json').open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.case_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    hashes = manifest['file_sha256']
    config_path = _checked_path(root, args.config, hashes)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    value = audit_extended(root, cfg, hashes, args.output)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
