"""Describe native decisions without inventing accuracy for unlabeled changes."""
from collections import defaultdict


def summarize_native_profiles(cases, records, systems):
    systems = tuple(systems)
    if not systems or len(set(systems)) != len(systems):
        raise ValueError('Unique native systems required')
    gold = {c['case_id']: c for c in cases}
    if len(gold) != len(cases) or not gold:
        raise ValueError('Unique nonempty cases required')
    actions = {}
    for row in records:
        key = row['case_id'], row['system']
        if key in actions or key[0] not in gold or key[1] not in systems:
            raise ValueError('Duplicate or unknown prediction identity')
        if row['decision'] not in ('ACCEPT', 'REJECT', 'ABSTAIN'):
            raise ValueError('Explicit native action required')
        actions[key] = row['decision']
    if len(actions) != len(gold)*len(systems):
        raise ValueError('Complete native-system by case matrix required')
    groups = defaultdict(list)
    for case in cases:
        groups[case['kind'], case['family']].append(case)
    profiles, totals = [], {s: dict(cases=len(gold), rejected=0, accepted=0, abstained=0,
                                   known_action_cases=0, known_action_agreement=0,
                                   neutral_cases=0, neutral_rejected=0) for s in systems}
    for (kind, family), members in sorted(groups.items()):
        known = all(c.get('expected_action') in ('ACCEPT', 'REJECT', 'ABSTAIN') for c in members)
        by_system = {}
        for system in systems:
            values = [actions[c['case_id'], system] for c in members]
            item = {'cases': len(members), 'rejected': values.count('REJECT'),
                    'accepted': values.count('ACCEPT'), 'abstained': values.count('ABSTAIN'),
                    'gold_action_agreement': sum(a == c['expected_action'] for a, c in zip(values, members)) if known else None}
            by_system[system] = item
            for key in ('rejected', 'accepted', 'abstained'):
                totals[system][key] += item[key]
            if known:
                totals[system]['known_action_cases'] += len(members)
                totals[system]['known_action_agreement'] += item['gold_action_agreement']
            if kind == 'neutral':
                totals[system]['neutral_cases'] += len(members)
                totals[system]['neutral_rejected'] += item['rejected']
        profiles.append({'kind': kind, 'family': family, 'cases': len(members),
                         'gold_action_available': known, 'higher_response_is_better': None,
                         'by_system': by_system})
    signature_groups = defaultdict(list)
    for system in systems:
        signature = tuple(actions[cid, system] for cid in sorted(gold))
        signature_groups[signature].append(system)
    return {'family_profiles': profiles, 'system_totals': totals,
            'decision_profile_groups': list(signature_groups.values()),
            'numeric_response_is_accuracy': False, 'overall_superiority_inferred': False}
