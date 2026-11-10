"""Opt-in section-specific physical-neighborhood reference check.

Measures coordinate-to-embedding kNN recall, not annotation purity. A failed
check is a numerical loss relative to the supplied frozen reference, not a
claim of biological damage. Historical/default audit code is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist

SCHEMA = 'stia.spatial_reference.v1'
NUMERIC_FLOOR = 100 * float(np.finfo(np.float32).eps)
CONTROL_NAMES = ('identity', 'reflection', 'rotation', 'translation', 'scale', 'float32_roundtrip')


def _strings(values, name, unique=False):
    a = np.asarray(values)
    if a.ndim != 1 or any(v is None or isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in a):
        raise ValueError(f'Invalid {name}')
    a = a.astype(str)
    if any(not v.strip() for v in a) or unique and len(set(a)) != len(a):
        raise ValueError(f'Empty or duplicate {name}')
    return a


def _matrix(values, name):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite with at least two rows and dimensions')
    return a


def _aligned(values, ids, coordinates, coordinate_ids, sections):
    z, xy = _matrix(values, 'Embedding'), _matrix(coordinates, 'Coordinates')
    zi, ci = _strings(ids, 'spot identities', True), _strings(coordinate_ids, 'coordinate identities', True)
    groups = _strings(sections, 'section identities')
    if len(zi) != len(z) or len(ci) != len(xy) or len(groups) != len(xy) or set(zi) != set(ci):
        raise ValueError('Embedding/coordinate identities are not a bijection')
    target = np.asarray(sorted(ci), dtype=str)
    ixz, ixc = {v: i for i, v in enumerate(zi)}, {v: i for i, v in enumerate(ci)}
    pz, pc = [ixz[v] for v in target], [ixc[v] for v in target]
    return z[pz], xy[pc], target, groups[pc]


def _sizes(ks, sections, chunk_rows):
    ks = tuple(ks)
    if not ks or len(set(ks)) != len(ks) or any(isinstance(k, bool) or not isinstance(k, Integral) or k < 1 for k in ks):
        raise ValueError('Neighbor sizes must be distinct positive integers')
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, Integral) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if any(np.count_nonzero(sections == s) <= max(ks) for s in np.unique(sections)):
        raise ValueError('Each section requires more observations than the largest neighbor size')
    return tuple(sorted(int(k) for k in ks))


def _neighbors(z, k, chunk_rows):
    if np.linalg.norm(z - z.mean(axis=0)) == 0:
        raise ValueError('Collapsed embedding or physical coordinate section')
    # Match the existing physical-recall protocol: raw Euclidean float64.
    # The arrays are identity-sorted before slicing, making exact distance ties
    # deterministic by spot ID. Section boundaries are never crossed.
    nearest = np.empty((len(z), k), dtype=np.int64)
    for start in range(0, len(z), chunk_rows):
        stop = min(start + chunk_rows, len(z))
        distances = cdist(z[start:stop], z, metric='euclidean')
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite pairwise distance')
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.argsort(distances, axis=1, kind='stable')[:, :k]
    return nearest


def _score(z, xy, groups, ks, chunk_rows):
    rows = []
    for section in np.unique(groups):
        mask = groups == section
        physical = _neighbors(xy[mask], max(ks), chunk_rows)
        latent = _neighbors(z[mask], max(ks), chunk_rows)
        for k in ks:
            recall = float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(physical, latent)]))
            rows.append({'section': str(section), 'n_spots': int(mask.sum()), 'k': k, 'recall': recall})
    means = {str(k): float(np.mean([r['recall'] for r in rows if r['k'] == k])) for k in ks}
    return rows, means


def spatial_recall(values, ids, coordinates, coordinate_ids, sections, ks=(10, 30), chunk_rows=256):
    """Identity-aligned physical-neighbor recall, aggregated equally by section.

    Donor-level aggregation/inference must be supplied by a study-specific
    caller; algorithm seeds and sections are not treated as new donors here.
    """
    z, xy, target, groups = _aligned(values, ids, coordinates, coordinate_ids, sections)
    ks = _sizes(ks, groups, chunk_rows)
    rows, means = _score(z, xy, groups, ks, chunk_rows)
    return {'schema_version': SCHEMA, 'n_spots': len(target), 'neighbor_sizes': list(ks),
            'aggregation': 'equal_section_mean', 'distance': 'raw_euclidean_float64',
            'tie_policy': 'stable_canonical_spot_id', 'sections': rows, 'mean_by_k': means}


def _digest(z, xy, ids, groups):
    h = hashlib.sha256()
    h.update(json.dumps({'ids': ids.tolist(), 'sections': groups.tolist()}, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    for a in (z, xy):
        h.update(json.dumps(list(a.shape)).encode('ascii'))
        h.update(np.ascontiguousarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def calibrate_reference(reference, reference_ids, coordinates, coordinate_ids, sections, ks=(10, 30), chunk_rows=256):
    """Freeze limits using reference-only, fixed numerical neutral controls."""
    z, xy, ids, groups = _aligned(reference, reference_ids, coordinates, coordinate_ids, sections)
    ks = _sizes(ks, groups, chunk_rows)
    baseline, _ = _score(z, xy, groups, ks, chunk_rows)
    rotation, _ = np.linalg.qr(np.random.default_rng(812).normal(size=(z.shape[1], z.shape[1])))
    transformed = (z, -z, z @ rotation, z + 5, z * 2.5, z.astype(np.float32).astype(np.float64))
    controls, envelope = [], np.zeros(len(baseline))
    for name, values in zip(CONTROL_NAMES, transformed):
        rows, _ = _score(values, xy, groups, ks, chunk_rows)
        envelope = np.maximum(envelope, [abs(b['recall'] - r['recall']) for b, r in zip(baseline, rows)])
        controls.append({'name': name, 'sections': rows})
    rules = [{'section': b['section'], 'k': b['k'], 'limit': float(e + NUMERIC_FLOOR)} for b, e in zip(baseline, envelope)]
    return {'schema_version': 'stia.spatial_reference.calibration.v1', 'reference_sha256': _digest(z, xy, ids, groups),
            'neighbor_sizes': list(ks), 'distance': 'raw_euclidean_float64',
            'tie_policy': 'stable_canonical_spot_id', 'reference_scores': baseline,
            'rules': rules, 'controls': controls, 'numeric_floor': NUMERIC_FLOOR,
            'scope': 'reference_relative_numerical_spatial_retention_only'}


def compare_reference(candidate, candidate_ids, reference, reference_ids, coordinates, coordinate_ids, sections,
                      ks=(10, 30), chunk_rows=256, calibration=None):
    """Compare physical-neighbor recall to an independently fixed reference.

    Supplied calibration is recomputed and checked, rather than trusting an
    editable action threshold. No candidate is used to set the action limit.
    """
    expected = calibrate_reference(reference, reference_ids, coordinates, coordinate_ids, sections, ks, chunk_rows)
    if calibration is not None and calibration != expected:
        raise ValueError('Supplied calibration differs from reference-only recomputation')
    result = spatial_recall(candidate, candidate_ids, coordinates, coordinate_ids, sections, ks, chunk_rows)
    contrasts, failed = [], []
    for row, base, rule in zip(result['sections'], expected['reference_scores'], expected['rules']):
        loss = base['recall'] - row['recall']
        contrasts.append({'section': row['section'], 'k': row['k'], 'reference_recall': base['recall'],
                          'candidate_recall': row['recall'], 'loss': loss, 'limit': rule['limit']})
        if loss > rule['limit']: failed.append({'section': row['section'], 'k': row['k']})
    encoded = json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    return {'schema_version': SCHEMA, 'status': 'FAIL' if failed else 'PASS',
            'scope': 'reference_relative_numerical_spatial_retention_only',
            'scores': result, 'contrasts': contrasts, 'failed_section_k': failed,
            'calibration_sha256': hashlib.sha256(encoded).hexdigest(),
            'reference_sha256': expected['reference_sha256'], 'biological_harm_inferred': False}
