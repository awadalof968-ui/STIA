"""Recompute the 768-case numerical stress test from saved case predictions."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SYSTEMS = ('B0', 'B1', 'B2', 'B3', 'B5', 'B6', 'B6_HASH_CONTROL', 'STIA')


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def score():
    truth = {r['case_id']: r for r in load('GOLD.json')}
    predictions = load('PREDICTIONS.json')
    assert len(truth) == 768 and len(predictions) == 768 * len(SYSTEMS)
    assert Counter(r['task'] for r in truth.values()) == {f'N{i}': 128 for i in range(1, 7)}
    assert {(r['case_id'], r['system']) for r in predictions} == {
        (case_id, system) for case_id in truth for system in SYSTEMS
    }
    scores = defaultdict(Counter)
    for row in predictions:
        gold = truth[row['case_id']]
        action = row['decision']
        if action == 'ACCEPT_WITH_SCOPE':
            action = 'ACCEPT'
        fault = gold['expected_action'] == 'REJECT'
        correct = action == gold['expected_action']
        task = gold['task']
        if task == 'N6':
            correct = correct and set(row['predicted_families']) == set(gold['expected_families'])
        scores[task, row['system']].update(
            cases=1, faults=int(fault), controls=int(not fault), correct_faults=int(fault and correct),
            correct_controls=int(not fault and correct), rejected_faults=int(fault and action == 'REJECT'),
        )

    with (ROOT / 'TABLE_1.csv').open(encoding='utf-8', newline='') as f:
        table = {r['system']: r for r in csv.DictReader(f)}
    for (task, system), counts in scores.items():
        if task == 'N6':
            assert (counts['faults'], counts['controls']) == (96, 32)
            complete = (counts['correct_faults'] + counts['correct_controls']) / counts['cases']
            rejection = counts['rejected_faults'] / counts['faults']
            assert table[system]['N6_complete_pct'] == f'{100 * complete:.2f}'
            assert table[system]['N6_rejection_pct'] == f'{100 * rejection:.2f}'
        else:
            assert (counts['faults'], counts['controls']) == (64, 64)
            ba = (counts['correct_faults'] / counts['faults'] +
                  counts['correct_controls'] / counts['controls']) / 2
            assert table[system][f'{task}_BA_pct'] == f'{100 * ba:.2f}'
    return {'status': 'PASS', 'cases': len(truth), 'predictions': len(predictions),
            'table_values_verified': 7 * len(SYSTEMS)}


if __name__ == '__main__':
    print(json.dumps(score(), indent=2))
