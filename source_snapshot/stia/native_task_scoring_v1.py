"""Shared scoring of native decisions, never shared decision postprocessing.

This module accepts frozen gold and actual workflow receipts. It computes
neither model metrics nor workflow decisions and cannot fill missing gold.
"""
from __future__ import annotations
from numbers import Integral
import math
import statistics

TASKS = ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
ACTIONS = ('ACCEPT', 'REJECT', 'ABSTAIN')


def _systems(systems):
    systems = tuple(systems)
    if not systems or len(set(systems)) != len(systems) or any(not isinstance(v, str) or not v.strip() for v in systems):
        raise ValueError('Explicit unique systems are required')
    return systems


def _families(values):
    if not isinstance(values, list) or len(set(values)) != len(values) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError('Explicit unique diagnostic family list is required')
    return set(values)


def score_native_tasks(cases, records, systems):
    """N1-N5 balanced action accuracy; N6 complete diagnosis/action agreement.

    Gold must be ACCEPT or REJECT for these decidable primary cases. ABSTAIN
    in a workflow receipt is retained and scores as an incorrect action.
    Truly indeterminate gold is a separate task, not silently substituted.
    """
    systems = _systems(systems)
    gold, rows = {}, {}
    for case in cases:
        key = case.get('case_id')
        if not isinstance(key, str) or not key.strip() or key in gold:
            raise ValueError('Missing or duplicate case identity')
        if case.get('task') not in TASKS or case.get('expected_action') not in ('ACCEPT', 'REJECT'):
            raise ValueError('Frozen primary task and explicit action gold are required')
        if not isinstance(case.get('source_id'), str) or not case['source_id'].strip():
            raise ValueError('Source grouping identity is required')
        if case['task'] == 'N6':
            families = _families(case.get('expected_families'))
            if (case['expected_action'] == 'ACCEPT' and families) or (case['expected_action'] == 'REJECT' and len(families) < 2):
                raise ValueError('Compound gold requires at least two families; neutral gold requires none')
        gold[key] = case
    if set(c['task'] for c in gold.values()) != set(TASKS):
        raise ValueError('Every N1-N6 task requires an explicitly frozen corpus')
    for record in records:
        key = record.get('case_id'), record.get('system')
        if key in rows or key[0] not in gold or key[1] not in systems:
            raise ValueError('Duplicate or unknown prediction identity')
        if record.get('decision') not in ACTIONS:
            raise ValueError('Explicit native ACCEPT/REJECT/ABSTAIN is required')
        if gold[key[0]]['task'] == 'N6': _families(record.get('predicted_families'))
        rows[key] = record
    expected_keys = {(case, system) for case in gold for system in systems}
    if set(rows) != expected_keys:
        raise ValueError('Incomplete case-by-system native prediction matrix')
    result = {}
    for task in TASKS:
        group = [c for c in gold.values() if c['task'] == task]
        positives = [c for c in group if c['expected_action'] == 'REJECT']
        negatives = [c for c in group if c['expected_action'] == 'ACCEPT']
        if not positives or not negatives:
            raise ValueError('Each task requires both fault and neutral/beneficial controls')
        by_system = {}
        for system in systems:
            def correct(c):
                row = rows[c['case_id'], system]
                action_ok = row['decision'] == c['expected_action']
                return action_ok and (task != 'N6' or _families(row['predicted_families']) == _families(c['expected_families']))
            sensitivity = sum(correct(c) for c in positives) / len(positives)
            specificity = sum(correct(c) for c in negatives) / len(negatives)
            abstentions = sum(rows[c['case_id'], system]['decision'] == 'ABSTAIN' for c in group)
            if task == 'N6':
                score = sum(correct(c) for c in group) / len(group)
            else:
                score = (sensitivity + specificity) / 2
            by_system[system] = {'score': score, 'n_faults': len(positives), 'n_neutral_or_beneficial': len(negatives),
                                 'abstentions': abstentions, 'decision_coverage': 1 - abstentions / len(group)}
            if task == 'N6':
                by_system[system].update(positive_exact_agreement=sensitivity, neutral_exact_agreement=specificity)
            else:
                by_system[system].update(sensitivity=sensitivity, specificity=specificity)
        result[task] = {'metric': 'exact_diagnosis_and_action_agreement' if task == 'N6' else 'balanced_action_accuracy',
                        'case_count': len(group), 'source_count': len(set(c['source_id'] for c in group)), 'by_system': by_system}
    return {'schema_version': 'stia.native_task_scores.v1', 'systems': list(systems), 'by_task': result,
            'decisions_changed_by_scorer': False, 'gold_inferred_from_predictions': False,
            'confidence_intervals': None, 'uncertainty_note': 'Point scoring only; source/template dependence requires separately frozen grouped uncertainty analysis.'}


def score_native_resources(receipts, systems, workload_sha256):
    """N7 median complete workload time, three fresh verified run receipts."""
    systems = _systems(systems)
    if not isinstance(workload_sha256, str) or len(workload_sha256) != 64 or any(v not in '0123456789abcdef' for v in workload_sha256):
        raise ValueError('Frozen workload sha256 is required')
    table, run_ids = {}, set()
    for row in receipts:
        system, round_id = row.get('system'), row.get('round')
        if system not in systems or isinstance(round_id, bool) or not isinstance(round_id, Integral) or round_id not in (1, 2, 3):
            raise ValueError('Unknown system or round')
        if (system, round_id) in table or row.get('workload_sha256') != workload_sha256:
            raise ValueError('Duplicate resource receipt or unmatched workload')
        run_id = row.get('run_id')
        if not isinstance(run_id, str) or not run_id.strip() or run_id in run_ids:
            raise ValueError('Independent unique resource run identities are required')
        seconds = row.get('wall_seconds')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Actual positive finite workload time is required')
        table[system, round_id] = float(seconds)
        run_ids.add(run_id)
    if set(table) != {(system, round_id) for system in systems for round_id in (1, 2, 3)}:
        raise ValueError('Three complete matched rounds per system are required')
    return {'task': 'N7', 'metric': 'median_complete_workload_wall_seconds', 'workload_sha256': workload_sha256,
            'by_system': {system: {'median_wall_seconds': statistics.median([table[system, i] for i in (1, 2, 3)]),
                                  'raw_wall_seconds': [table[system, i] for i in (1, 2, 3)]} for system in systems}}
