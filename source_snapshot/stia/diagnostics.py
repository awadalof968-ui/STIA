"""Check-level diagnostic semantics, independent of gold labels and decisions.

An evidence prerequisite failure is not a measured biological/stability defect.
Every failed check remains visible; concurrent computed failures are never hidden.
This is a diagnostic taxonomy, not a claim of experimentally proven unique cause.
"""
FAMILIES = ('custody', 'identity', 'design', 'features', 'statistics', 'stability', 'quality')
DIAGNOSTIC_SCHEMA = 'stia.check_diagnostics.v1'


class EvidenceFault(ValueError):
    def __init__(self, message, family, code):
        if family not in FAMILIES:
            raise ValueError('Unknown diagnostic family')
        super().__init__(message)
        self.family, self.code = family, code

    def fields(self):
        return {'fault_family': self.family, 'fault_code': self.code,
                'diagnostic_kind': 'invalid_evidence'}


def error_fields(error):
    return error.fields() if isinstance(error, EvidenceFault) else {
        'fault_code': 'unclassified_invalid_evidence', 'diagnostic_kind': 'invalid_evidence'}


def _classification(mid, check):
    name = check.get('name', '')
    if check.get('fault_family') in FAMILIES:
        return check['fault_family'], check.get('diagnostic_kind', 'invalid_evidence')
    if check.get('fault_code') == 'unclassified_invalid_evidence':
        return None, 'invalid_evidence'
    if mid in ('M1', 'M2', 'M3', 'M4', 'M5'):
        return dict(zip(('M1', 'M2', 'M3', 'M4', 'M5'), FAMILIES[:5]))[mid], 'gate_failure'
    if mid == 'M6':
        if name in ('osi', 'computed_seed_robustness'):
            return 'stability', 'measured_failure'
        if name.startswith(('donor_alignment', 'order:')) or name in ('order_names_unique', 'order_identity_evidence'):
            return 'identity', 'invalid_evidence'
        if name == 'perturbation_contract':
            return 'custody', 'invalid_evidence'
    if mid == 'M7':
        if name == 'embedding_identity':
            return 'identity', 'invalid_evidence'
        if name.startswith('reference_relative:') or name == 'global_conditional_reversal':
            return 'quality', 'measured_failure'
    return None, 'unclassified'


def diagnose(modules):
    """Return evidence-linked findings; never change any module status/action."""
    findings, blocked, unresolved, unknown = [], [], [], []
    for mid, value in modules.items():
        if mid in ('M8', 'M9'):  # governance/decision remain in original modules
            continue
        m = value.as_dict() if hasattr(value, 'as_dict') else value
        failed = [c for c in m.get('checks', []) if c.get('status') == 'FAIL']
        if m.get('status') == 'FAIL' and not failed:
            unknown.append({'module_id': mid, 'check': None, 'reason': 'FAIL without typed failed check'})
        if m.get('status') == 'UNRESOLVED':
            unresolved.append({'module_id': mid, 'reason': m.get('summary'), 'details': m.get('unresolved', [])})
        for check in m.get('checks', []):
            if check.get('status') == 'UNRESOLVED':
                unresolved.append({'module_id': mid, 'check': check.get('name'), 'reason': check.get('detail')})
        for check in failed:
            family, kind = _classification(mid, check)
            row = {'module_id': mid, 'check': check.get('name'), 'family': family, 'kind': kind,
                   'fault_code': check.get('fault_code'), 'detail': check.get('detail'),
                   'evidence_ref': f"modules/{mid}/checks/{check.get('name')}"}
            findings.append(row)
            if family is None:
                unknown.append(row)
            if kind == 'invalid_evidence' and mid in ('M6', 'M7'):
                blocked.append({'target': 'stability' if mid == 'M6' else 'quality',
                                'blocked_by': row['evidence_ref'], 'family': family,
                                'reason': 'Prerequisite evidence invalid; do not interpret this check as measured deterioration'})
    return {'schema': DIAGNOSTIC_SCHEMA,
            'fault_families': sorted({r['family'] for r in findings if r['family']}),
            'findings': findings, 'measured_failures': [r for r in findings if r['kind'] == 'measured_failure'],
            'blocked_checks': blocked, 'unresolved': unresolved, 'unclassified_failures': unknown,
            'complete': not unresolved and not unknown,
            'decision_policy': 'Reporting only; all original checks and non-compensatory safety gates retained',
            'boundary': 'Diagnostic classification, not proof of unique physical cause or clinical validity'}
