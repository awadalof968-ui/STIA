"""Small synthetic, provider-free example shipped inside the installed wheel.

This is an installation and identity-alignment example, not an accuracy test.
It intentionally lacks independent disease preservation calibration, so a
complete audit abstains rather than certifying a self-asserted positive result.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .contracts import TOOL_VERSION


def build_demo_case(output: str | Path) -> Path:
    root = Path(output).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f'Refusing to replace existing example evidence: {root}')
    root.mkdir(parents=True, exist_ok=True)
    rows, values = [], []
    for ci, condition in enumerate(('control', 'disease')):
        for bi, batch in enumerate(('B1', 'B2')):
            for di in range(3):
                donor = f'{condition}_{batch}_D{di}'
                for si in range(4):
                    rows.append({'spot_id': f's{len(rows):03d}', 'sample_id': donor,
                                 'donor_id': donor, 'disease_or_condition': condition,
                                 'technical_batch': batch, 'layer': f'L{si % 2}',
                                 'group': 'all', 'x': si % 2, 'y': si // 2})
                    values.append([2.0 * ci + di * .05 + si * .01, bi * .2 + si * .02])
    z = np.asarray(values, dtype=float)
    ids = np.asarray([r['spot_id'] for r in rows], dtype='U')
    with (root / 'metadata.csv').open('x', encoding='utf-8', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)

    def save_array(name, array):
        with (root / name).open('xb') as h:
            np.save(h, array, allow_pickle=False)

    save_array('embedding.npy', z)
    save_array('spot_ids.npy', ids)
    arms = []
    permutations = [np.arange(48), np.arange(47, -1, -1),
                    np.r_[np.arange(0, 48, 2), np.arange(1, 48, 2)],
                    np.roll(np.arange(48), 12)]
    donors = np.asarray([r['donor_id'] for r in rows], dtype='U')
    for name, order in zip(('A_original', 'B_reverse', 'C_interleaved', 'D_rotated'), permutations):
        for suffix, arr in [('embedding', z[order]), ('spot_ids', ids[order]), ('donor_keys', donors[order])]:
            save_array(f'{name}_{suffix}.npy', arr)
        arms.append({'name': name, 'embedding': f'{name}_embedding.npy',
                     'spot_ids': f'{name}_spot_ids.npy', 'donor_keys': f'{name}_donor_keys.npy'})
    config = {
        'metadata': 'metadata.csv', 'spot_id': 'spot_id', 'sample_id': 'sample_id',
        'donor_id': 'donor_id', 'disease_or_condition': 'disease_or_condition',
        'technical_batch': 'technical_batch', 'estimand': 'synthetic donor-level demonstration only',
        'provenance': {'case_id': 'stia_installed_demo', 'run_id': 'demo_' + TOOL_VERSION,
                       'software': {'stia': TOOL_VERSION}, 'seeds': {'data': 0, 'model': 0},
                       'orders': [a['name'] for a in arms]},
        'feature_panel': ['G1', 'G2'], 'method_feature_panels': {'synthetic': ['G1', 'G2']},
        'analysis_mode': 'unsupervised_order_audit',
        'preprocessing': {'fit_scope': 'full_unsupervised_frozen', 'leakage_checked': True,
                          'outcome_informed_selection': False},
        'outcome_informed_selection': False, 'require_perturbation_contract': True,
        'perturbation_contract': {'axis': 'SAMPLE_SLICE_BATCH_ORDER', 'unit': 'slice',
            'semantics': 'NUISANCE_PERMUTATION',
            'fixed_fields': ['observations', 'entity_identities', 'hyperparameters', 'random_seed', 'intended_analysis']},
        'embedding': {'path': 'embedding.npy', 'spot_ids': 'spot_ids.npy',
                      'layer_column': 'layer', 'group_column': 'group'},
        'orders': arms, 'osi_orders_required': 4, 'osi_margin': .05, 'seed': 0,
        'disease_axis_metrics': {'mode': 'donor_batch_matched_separation_auc_v1'},
        'evaluator': {'functional_tests': [{'name': 'keyed_synthetic_identity', 'passed': True}],
                      'statistics': {'unit': 'donor', 'ci_method': 'cluster_bootstrap', 'multiple_testing': 'Holm'}},
    }
    with (root / 'config.json').open('x', encoding='utf-8') as h:
        json.dump(config, h, sort_keys=True, indent=2); h.write('\n')
    return root
