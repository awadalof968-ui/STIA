"""Opt-in composition of the frozen core and two independently tested guards."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .engine import audit as _core_audit
from .io import validate_canonical_bundle
from .order_neighborhood_v1 import analyze_orders, load_frozen_orders, _checked_path
from .spatial_reference_v1 import compare_reference

VARIANT = '1.3.8+order-neighborhood.v1+spatial-reference.v1'


def compose_native_decision(base, statuses):
    if base not in ('ACCEPT', 'ACCEPT_WITH_SCOPE', 'ABSTAIN', 'REPAIR', 'REJECT'):
        raise ValueError('Unknown native core decision')
    if any(s not in ('PASS', 'FAIL', 'UNRESOLVED') for s in statuses):
        raise ValueError('Unknown optional guard status')
    if base == 'REJECT' or 'FAIL' in statuses: return 'REJECT'
    if base in ('ABSTAIN', 'REPAIR') or 'UNRESOLVED' in statuses: return 'ABSTAIN'
    return 'ACCEPT'


def _order_check(root, cfg, hashes):
    return analyze_orders(load_frozen_orders(root, cfg, hashes))


def _spatial_check(root, cfg, hashes):
    spec = cfg.get('physical_spatial_reference')
    if not isinstance(spec, dict): raise FileNotFoundError('Physical spatial reference is unavailable')
    metadata = _checked_path(root, spec.get('metadata'), hashes)
    reference = _checked_path(root, spec.get('embedding'), hashes)
    candidate = _checked_path(root, cfg.get('embedding'), hashes)
    with np.load(metadata, allow_pickle=False) as a:
        xy, coordinate_ids, sections = a['coordinates'].copy(), a['spot_ids'].astype(str), a['section_ids'].astype(str)
    with np.load(reference, allow_pickle=False) as a:
        z, ids = a[spec.get('latent_key', 'latent')].copy(), a[spec.get('spot_ids_key', 'spot_ids')].astype(str)
    with np.load(candidate, allow_pickle=False) as a:
        values, candidate_ids = a[cfg['embedding'].get('latent_key', 'latent')].copy(), a[cfg['embedding'].get('spot_ids_key', 'spot_ids')].astype(str)
    return compare_reference(values, candidate_ids, z, ids, xy, coordinate_ids, sections,
                             ks=tuple(spec.get('neighbor_sizes', [10, 30])), chunk_rows=256)


def audit_native_extended(case_root, config, expected_hashes, output):
    """Run actual native core, order check and physical check in a new bundle.

    Optional numerical calibration is reference-only. Neither gold labels nor
    generator parameters are arguments or read here. Defaults remain intact.
    """
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = _core_audit(case_root, config, output=out / 'base_audit')
    if not validate_canonical_bundle(out / 'base_audit')['valid']:
        raise ValueError('Incomplete native core bundle')
    findings = json.loads((out / 'base_audit' / 'findings.json').read_text(encoding='utf-8'))
    families = set(findings['diagnostics']['fault_families'])
    checks = {}
    for name, function, family in [('order_neighborhood', _order_check, 'stability'), ('spatial_reference', _spatial_check, 'quality')]:
        try:
            value = function(case_root, config, expected_hashes)
        except FileNotFoundError as exc:
            value = {'status': 'UNRESOLVED', 'reason': str(exc)}
        except (ValueError, TypeError, KeyError) as exc:
            value = {'status': 'FAIL', 'reason': str(exc), 'evidence_invalid': True}
        checks[name] = value
        if value['status'] == 'FAIL': families.add(family)
    value = {'schema_version': 'stia.native_workflow.v1', 'variant': VARIANT,
             'base_decision': base.decision,
             'decision': compose_native_decision(base.decision, [v['status'] for v in checks.values()]),
             'predicted_families': sorted(families), 'base_bundle': str(out / 'base_audit'),
             **checks, 'default_engine_modified': False, 'gold_read_by_workflow': False}
    with (out / 'NATIVE_DECISION.json').open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    return value
