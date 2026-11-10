"""Known-component numerical cases; no audit import and no prediction access.

Synthetic donor/condition/batch labels describe the generating design. They
are not actual patients, independent external studies or clinical truth.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_panel(seed, side=8, donors_per_cell=4):
    if side < 6 or donors_per_cell < 2: raise ValueError('Design requires sufficient grid and donor groups')
    rng = np.random.default_rng(seed)
    rows, coordinates, biological, technical, spatial, other = [], [], [], [], [], []
    for batch in range(2):
        for condition in range(2):
            for donor in range(donors_per_cell):
                name = f'p{seed}_b{batch}c{condition}d{donor}'
                donor_noise = rng.normal(0, .6, 12)
                donor_noise[6] = rng.normal(0, 1.4)
                donor_noise[:2] = 0
                for i in range(side):
                    for j in range(side):
                        label = (i >= side / 2) * 2 + (j >= side / 2)
                        rows.append({'spot_id': f'{name}_s{i:02}{j:02}', 'sample_id': name, 'donor_id': name,
                                     'disease_or_condition': f'c{condition}', 'technical_batch': f'b{batch}',
                                     'cell_type': f't{label}', 'section_id': name})
                        coordinates.append([float(i), float(j)])
                        bio, batch_effect, location = np.zeros(12), np.zeros(12), np.zeros(12)
                        bio[6] = (condition * 2 - 1) * 1.4
                        batch_effect[7] = (batch * 2 - 1) * .35
                        location[:2] = (np.array([i, j]) - (side - 1) / 2) * .65
                        location[2 + label] = 3.0
                        biological.append(bio); technical.append(batch_effect); spatial.append(location)
                        other.append(donor_noise + rng.normal(0, .12, 12))
    components = {'biological': np.array(biological), 'technical': np.array(technical),
                  'spatial': np.array(spatial), 'unprotected': np.array(other)}
    reference = sum(components.values())
    return {'seed': int(seed), 'metadata': pd.DataFrame(rows), 'coordinates': np.array(coordinates),
            'components': components, 'reference': reference,
            'expression': np.expm1(np.clip(reference, -2, 3) + 2),
            'scope': 'known_component_simulation_no_clinical_truth', 'biological_studies': 0}


def perturb(panel, kind, severity, seed):
    z, metadata = panel['reference'], panel['metadata']
    rng = np.random.default_rng(seed)
    main, changed_seed, order = z.copy(), z.copy(), z.copy()
    positive = kind in ('biological_loss', 'batch_residual', 'spatial_loss', 'seed_loss', 'local_order', 'donor_order')
    if positive and not 0 < severity <= 3: raise ValueError('Positive severity is required')
    if kind == 'biological_loss': main -= severity * panel['components']['biological']
    elif kind == 'batch_residual': main += severity * panel['components']['technical']
    elif kind == 'beneficial':
        if not 0 < severity <= 1: raise ValueError('Correction fraction must be in (0,1]')
        main -= severity * panel['components']['technical']
    elif kind in ('spatial_loss', 'seed_loss'):
        target = main if kind == 'spatial_loss' else changed_seed
        for _, frame in metadata.groupby(['sample_id', 'cell_type'], sort=True):
            ix = frame.index.to_numpy(dtype=int)
            selected = rng.permutation(ix)[:max(2, min(len(ix), round(severity * len(ix))))]
            if kind == 'spatial_loss': target[selected, :2] = z[np.roll(selected, 1), :2]
            else: target[selected] = z[np.roll(selected, 1)]
    elif kind == 'local_order':
        for _, frame in metadata.groupby('sample_id', sort=True):
            ix = frame.index.to_numpy(dtype=int)
            noise = rng.normal(size=(len(ix), z.shape[1]))
            order[ix] += severity * (noise - noise.mean(0)) * z.std(0)
    elif kind == 'donor_order':
        first = sorted(metadata.sample_id.unique())[0]
        ix = metadata.sample_id.eq(first).to_numpy()
        direction = rng.normal(size=z.shape[1]); direction /= np.linalg.norm(direction)
        order[ix] += severity * direction * z.std(0) * np.sqrt(z.shape[1])
    elif kind == 'identity': pass
    elif kind == 'reflection': main = -z
    elif kind == 'translation': main = z + 5
    elif kind == 'scale': main = z * 2.5
    elif kind == 'rotation':
        q, _ = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
        main = z @ q
    elif kind == 'float32': main = z.astype(np.float32).astype(float)
    else: raise ValueError('Unknown independent intervention: ' + kind)
    if kind not in ('seed_loss', 'local_order', 'donor_order'):
        changed_seed, order = main.copy(), main.copy()
    return {'main': main, 'seed': changed_seed, 'order': order,
            'expected_action': 'REJECT' if positive else 'ACCEPT',
            'known_component_or_relation_changed': kind, 'severity': float(severity),
            'gold_scope': 'controlled_numerical_intervention_not_clinical_truth'}
