"""Opt-in, isolated method-control orchestration; no automatic causal claims.

The five research backends are NOT bundled. Their explicit adapters must honor
the control contract; adapter declarations are recorded, not independently
verified. PCA's built-in full-SVD backend is only a deterministic reference.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import itertools
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from numbers import Integral

import numpy as np

VERSION = 'stia.method_diagnostics.v1'
ORDERS = ('reference', 'reverse_groups', 'rotate_groups', 'hash_groups')
BOUNDARY = (
    'Descriptive controlled comparison, conditional on adapter correctness. '
    'Technical seeds are not biological replicates. Reduction does not establish '
    'a unique/primary cause, biological preservation, or a deployable repair. '
    'No universal OSI pass threshold or inferential significance is supplied.'
)


def _recipe(hypothesis, variants, boundary):
    return dict(hypothesis=hypothesis, variants=variants, boundary=boundary,
                builtin_adapter=False)


_RECIPES = {
    'PCA': dict(hypothesis='Deterministic negative control, not a stochastic mechanism.',
                variants={'native': {}}, builtin_adapter=True,
                boundary='Full-SVD PCA only; randomized/incremental PCA is outside this reference.'),
    'Harmony': _recipe('Does initial center identity contribute to order response?',
        {'native': {}, 'identity_centers': {'initial_centers': 'stable_spot_identity'}},
        'Hold PCA, initial-center count, refinement and other settings fixed; centers are selected by stable identity, not array position.'),
    'INSPIRE': _recipe('Does the adjacent-section adversarial objective contribute?',
        {'native': {}, 'no_adversarial': {'adjacent_adversarial_objective': False}},
        'Disabling the objective changes the model. Improvement in consistency is not evidence that biological signal is preserved.'),
    'GraphST': _recipe('Does corruption-to-spot assignment contribute?',
        {'native': {}, 'identity_corruption': {'corruption_assignment': 'stable_spot_identity'}},
        'Hold random draws, graph, objective and training schedule fixed; change only which spot receives each draw.'),
    'STAligner': _recipe('Does adaptive matching feedback propagate early differences?',
        {'native': {}, 'reference_trajectory': {'matching_trajectory': 'reference_replay'}},
        'This is a reference-informed, variant-specific diagnostic requiring a recorded reference trajectory; NOT an ordinary deployable repair or proof about every native implementation.'),
    'PRECAST': _recipe('Do Harmony preprocessing and K-means initialization interact?',
        {'native': {}, 'harmony_fixed': {'harmony_identity_control': True},
         'kmeans_fixed': {'kmeans_identity_control': True},
         'joint_fixed': {'harmony_identity_control': True, 'kmeans_identity_control': True}},
        'Four-arm control: hold downstream fit and all other settings fixed. Joint benefit does not identify either stage as the sole cause.'),
}


def method_recipes():
    """Return independent copies of all six control contracts."""
    return copy.deepcopy(_RECIPES)


def _method_recipe(method, config):
    if not isinstance(method, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,79}', method):
        raise ValueError('method must be a short identifier, not a path or free text')
    if not isinstance(config, dict):
        raise ValueError('Configuration must be an object')
    known = {key.lower(): key for key in _RECIPES}
    if method.lower() in known:
        method = known[method.lower()]
        if 'recipe' in config:
            raise ValueError('Do not override a built-in recipe; use a distinct custom method name')
        return method, method_recipes()[method]
    recipe = config.get('recipe')
    if recipe is None:
        return method, _recipe('Order-sensitivity detection only; no internal mechanism intervention configured.',
                                {'native': {}}, 'Unlisted method: no method-specific cause is inferred.')
    if not isinstance(recipe, dict) or set(recipe) != {'hypothesis', 'variants', 'boundary'}:
        raise ValueError('Custom recipe needs exactly hypothesis, variants and boundary')
    if any(not isinstance(recipe[k], str) or not recipe[k].strip() for k in ('hypothesis', 'boundary')):
        raise ValueError('Custom hypothesis and boundary must be nonempty text')
    variants = recipe['variants']
    if not isinstance(variants, dict) or not variants or variants.get('native') != {}:
        raise ValueError('Custom variants must include native with empty controls')
    if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', v)
           or not isinstance(c, dict) or (v != 'native' and not c) for v, c in variants.items()):
        raise ValueError('Variant names must be safe identifiers; controls must be objects and interventions nonempty')
    # Force native first even if the custom JSON lists an intervention first.
    ordered = {'native': {}, **{v: c for v, c in variants.items() if v != 'native'}}
    return method, copy.deepcopy(_recipe(recipe['hypothesis'], ordered, recipe['boundary']))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _ids(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in 'US':
        raise ValueError(name + ' must be a one-dimensional string array (no pickle)')
    result = array.astype(str)
    if not len(result) or any(not v.strip() for v in result):
        raise ValueError(name + ' contains empty identities')
    return result


def _input(path):
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data['X'])
        if x.dtype.kind not in 'fiu':
            raise ValueError('X must contain real numeric values')
        x = x.astype(np.float64)
        ids, groups = _ids(data['spot_ids'], 'spot_ids'), _ids(data['group_ids'], 'group_ids')
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('X must be a finite dense n-by-d matrix, n,d >= 2')
    if len(x) != len(ids) or len(x) != len(groups) or len(set(ids)) != len(ids):
        raise ValueError('Input identities must be unique and aligned to X and groups')
    blocks = [str(groups[0])]
    blocks.extend(str(groups[i]) for i in range(1, len(groups)) if groups[i] != groups[i-1])
    if len(set(blocks)) != len(blocks):
        raise ValueError('v1 requires contiguous group blocks; materialize a documented reference first')
    if len(blocks) < 3:
        raise ValueError('At least three groups are needed for informative normalized geometry')
    return x, ids, groups, blocks


def _permutations(groups, blocks, orders):
    values = {}
    for order in orders:
        sequence = list(blocks)
        if order == 'reverse_groups':
            sequence.reverse()
        elif order == 'rotate_groups':
            sequence = sequence[1:] + sequence[:1]
        elif order == 'hash_groups':
            sequence.sort(key=lambda v: (hashlib.sha256(v.encode('utf-8')).hexdigest(), v))
        rows = np.concatenate([np.flatnonzero(groups == g) for g in sequence])
        if any(np.array_equal(rows, old) for old in values.values()):
            raise ValueError('Requested orders produce duplicate permutations; choose distinct group orders')
        values[order] = rows
    return values


def _config(method, config, recipe):
    if not isinstance(config, dict):
        raise ValueError('Configuration must be an object')
    unknown = set(config) - {'seeds', 'orders', 'variants', 'adapter', 'fixed_settings', 'timeout_seconds', 'recipe', 'neighbor_sizes'}
    if unknown:
        raise ValueError('Unknown configuration fields: ' + ', '.join(sorted(unknown)))
    # Reject non-JSON data and NaN before writing any output.
    cfg = json.loads(json.dumps(config, allow_nan=False))
    seeds = cfg.setdefault('seeds', [1, 2, 3])
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct unsigned 32-bit integers')
    orders = cfg.setdefault('orders', list(ORDERS))
    if not isinstance(orders, list) or len(orders) < 2 or any(o not in ORDERS for o in orders) or len(set(orders)) != len(orders) or orders[0] != 'reference':
        raise ValueError('orders must start with reference and include distinct supported perturbations')
    variants = cfg.setdefault('variants', list(recipe['variants']))
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) or v not in recipe['variants'] for v in variants) or len(set(variants)) != len(variants) or variants[0] != 'native':
        raise ValueError('variants must start with native and use the selected method recipe')
    adapter = cfg.setdefault('adapter', None)
    if adapter is not None and (not isinstance(adapter, str) or adapter.count(':') != 1 or
                                not all(part and all(p.isidentifier() for p in part.split('.')) for part in adapter.split(':'))):
        raise ValueError('adapter must be module:callable')
    if not isinstance(cfg.setdefault('fixed_settings', {}), dict):
        raise ValueError('fixed_settings must be an object')
    timeout = cfg.setdefault('timeout_seconds', 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
        raise ValueError('timeout_seconds must be in (0,86400]')
    ks = cfg.setdefault('neighbor_sizes', [])
    if not isinstance(ks, list) or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError('neighbor_sizes must be distinct positive integers; empty disables quadratic neighbor analysis')
    return cfg


def summarize_embeddings(arms, spot_ids, group_ids, orders, *, neighbor_sizes=()):
    """All-pair maximum OSI and separate same-input repeat, after ID restoration.

    A degenerate group geometry is unresolved, never zero. Pairwise distances
    eliminate rigid translations/rotations and Frobenius normalization removes
    a uniform positive scale. This is group geometry, not spatial-domain ARI.
    """
    try:
        if not isinstance(orders, (list, tuple)) or len(orders) < 2 or orders[0] != 'reference' or 'repeat' in orders or len(set(orders)) != len(orders):
            raise ValueError('Distinct orders must start with reference; repeat is separate')
        target, groups = _ids(spot_ids, 'spot_ids'), _ids(group_ids, 'group_ids')
        if len(target) != len(groups) or len(set(target)) != len(target):
            raise ValueError('Invalid canonical identity mapping')
        names = sorted(set(groups))
        if len(names) < 3:
            raise ValueError('Need at least three groups')
        matrices, restored, original, shape = {}, {}, {}, None
        for name in list(orders) + ['repeat']:
            z, ids = arms[name]
            z, ids = np.asarray(z), _ids(ids, 'output spot_ids')
            if z.dtype.kind not in 'fiu':
                raise ValueError('Embedding must contain real numeric values')
            z = z.astype(np.float64)
            if z.ndim != 2 or not z.shape[1] or len(z) != len(ids) or not np.isfinite(z).all():
                raise ValueError('Invalid or nonfinite output embedding')
            if len(set(ids)) != len(ids) or set(ids) != set(target):
                raise ValueError('Output identity is not a bijection of input identities')
            if shape is not None and z.shape != shape:
                raise ValueError('Embedding shape changes across orders')
            shape = z.shape
            ix = {v: i for i, v in enumerate(ids)}
            z = z[[ix[v] for v in target]]
            original[name] = z
            # Remove a common translation before centroid summation.
            z = z - z.mean(axis=0)
            restored[name] = z
            means = np.vstack([z[groups == g].mean(axis=0) for g in names])
            distances = np.linalg.norm(means[:, None] - means[None, :], axis=2)
            norm = float(np.linalg.norm(distances))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError('Collapsed or numerically degenerate group geometry')
            matrices[name] = distances / norm
        pairs = [dict(left=a, right=b, drift=float(np.linalg.norm(matrices[a]-matrices[b])))
                 for a, b in itertools.combinations(orders, 2)]
        repeat = float(np.linalg.norm(matrices['reference'] - matrices['repeat']))
        osi_value = max(p['drift'] for p in pairs)
        result = dict(status='VALID', osi=osi_value, pairs=pairs,
                    repeat_drift=repeat,
                    repeat_array_equal=bool(np.array_equal(original['reference'], original['repeat'])),
                    array_comparison_scope='ID-aligned float64 arrays, not original byte streams',
                    repeat_centered_array_equal=bool(np.array_equal(restored['reference'], restored['repeat'])),
                    repeat_bitwise_determinism_proven=False,
                    order_detection='EXCEEDS_SINGLE_REPEAT' if osi_value > repeat + 1e-12 else 'NO_EXCESS_OVER_SINGLE_REPEAT',
                    detection_scope='Descriptive comparison to ONE observed repeat, not a calibrated pass/fail test.',
                    scope='normalized_group_centroid_geometry', n_groups=len(names))
        if neighbor_sizes:
            from .order_neighborhood_v1 import analyze_orders
            result['neighborhood'] = analyze_orders({name: (original[name], target) for name in orders}, ks=neighbor_sizes)
            result['repeat_neighborhood'] = analyze_orders({name: (original[name], target) for name in ('reference', 'repeat')}, ks=neighbor_sizes)
        return result
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return dict(status='UNRESOLVED', osi=None, repeat_drift=None, reason=str(exc))


def _pca(request):
    x = request['X'] - request['X'].mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    n = request['fixed_settings'].get('n_components', min(20, x.shape[1], x.shape[0]-1))
    if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= min(x.shape):
        raise ValueError('n_components must be a positive integer no larger than min(X.shape)')
    if set(request['fixed_settings']) - {'n_components'}:
        raise ValueError('Built-in PCA only supports n_components')
    return dict(embedding=u[:, :n] * s[:n], spot_ids=request['spot_ids'], applied_controls={})


def _worker(request_path):
    request_path = Path(request_path)
    meta = json.loads(request_path.read_text(encoding='utf-8'))
    cell = request_path.parent
    try:
        source = Path(meta['input'])
        if _hash(source) != meta['input_sha256']:
            raise ValueError('Frozen input hash changed')
        if _hash(__file__) != meta['executor_sha256']:
            raise ValueError('Executor source changed after planning')
        x, ids, groups, blocks = _input(source)
        rows = _permutations(groups, blocks, [meta['order']])[meta['order']]
        request = dict(copy.deepcopy(meta), X=x[rows], spot_ids=ids[rows], group_ids=groups[rows], row_indices=rows,
                       ordered_group_ids=list(dict.fromkeys(groups[rows].tolist())),
                       work_dir=str(cell), seed_work_dir=str(cell.parent.parent))
        np.random.seed(meta['seed'])
        if meta['adapter']:
            module_name, attribute = meta['adapter'].split(':')
            fit = importlib.import_module(module_name)
            for part in attribute.split('.'):
                fit = getattr(fit, part)
            adapter_path = inspect.getsourcefile(fit)
            if not adapter_path:
                raise ValueError('Adapter callable must have a hashable Python source file')
            adapter_hash = _hash(adapter_path)
            result = fit(request)
            if _hash(adapter_path) != adapter_hash:
                raise ValueError('Adapter source changed during fit')
        else:
            adapter_path, adapter_hash, result = __file__, _hash(__file__), _pca(request)
        if not isinstance(result, dict) or _json_hash(result.get('applied_controls')) != _json_hash(meta['controls']):
            raise ValueError('Adapter must declare exactly the requested applied_controls')
        z, output_ids = np.asarray(result['embedding']), _ids(result['spot_ids'], 'output spot_ids')
        if z.ndim != 2 or z.dtype.kind not in 'fiu' or z.shape[0] != len(output_ids):
            raise ValueError('Adapter must return a numeric embedding aligned with output spot_ids')
        # Retain finite/collapse failures as raw evidence for fail-closed analysis.
        raw = cell / 'RAW.npz'
        with raw.open('xb') as handle:
            np.savez_compressed(handle, embedding=z, spot_ids=output_ids)
        if _hash(source) != meta['input_sha256']:
            raise ValueError('Input changed during fit')
        torch = sys.modules.get('torch')
        deterministic = bool(torch.are_deterministic_algorithms_enabled()) if torch is not None else None
        _write(cell / 'RECEIPT.json', dict(pid=os.getpid(), raw_sha256=_hash(raw),
            adapter_source=str(Path(adapter_path).resolve()), adapter_sha256=adapter_hash,
            applied_controls=result['applied_controls'], controls_independently_verified=False,
            fixed_settings_sha256=_json_hash(meta['fixed_settings']),
            input_sha256=meta['input_sha256'], request_sha256=_hash(request_path),
            environment=dict(python=sys.version, numpy=np.__version__,
                             torch_deterministic_algorithms_enabled=deterministic)))
        return 0
    except Exception as exc:
        _write(cell / 'FAILURE.json', dict(error=type(exc).__name__, message=str(exc)))
        return 1


def _reduce(seed, variant, native, candidate):
    row = dict(seed=seed, variant=variant, native_osi=native.get('osi'),
               controlled_osi=candidate.get('osi'), absolute_reduction=None,
               relative_reduction_percent=None, native_repeat_drift=native.get('repeat_drift'),
               controlled_repeat_drift=candidate.get('repeat_drift'), status='UNRESOLVED')
    if native['status'] == candidate['status'] == 'VALID':
        a, b = native['osi'], candidate['osi']
        row.update(status='DESCRIPTIVE', absolute_reduction=a-b,
                   relative_reduction_percent=100*(a-b)/a if a > 1e-12 else None,
                   order_exceeds_observed_repeat=bool(a > native['repeat_drift'] + 1e-12),
                   explanation='OSI and repeat are separate; one repeat estimates neither a noise distribution nor statistical significance.')
    return row


def diagnose_method(method, input_path, config, output, *, execute=False):
    """Plan by default; explicit execute launches configured adapters separately.

    Each callback receives a dictionary (see docs/method_diagnostics_v1.md).
    Missing native backends are UNCONFIGURED, not six-method PASS. Failed cells
    are retained. Existing outputs are never resumed or replaced in v1.
    """
    method, recipe = _method_recipe(method, config)
    cfg = _config(method, config, recipe)
    source, out = Path(input_path).resolve(), Path(output).resolve()
    source_hash = _hash(source)
    x, ids, groups, blocks = _input(source)
    permutations = _permutations(groups, blocks, cfg['orders'])
    if any(k >= len(ids) for k in cfg['neighbor_sizes']):
        raise ValueError('neighbor_sizes must be smaller than the spot count')
    if out == source or out in source.parents:
        raise ValueError('Output directory must be separate from input')
    count = len(cfg['seeds']) * len(cfg['variants']) * (len(cfg['orders']) + 1)
    report = dict(schema_version=VERSION, method=method, status='PLANNED', config=cfg,
        input=str(source), input_sha256=source_hash, executor_sha256=_hash(__file__),
        config_sha256=_json_hash(cfg), recipe=recipe, planned_cells=count, completed_cells=0,
        n_spots=len(x), n_groups=len(blocks), selected_group_order_unit='input group_ids',
        order_permutation_sha256={name: hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()
                                  for name, rows in permutations.items()},
        comparisons_per_seed=len(cfg['orders'])-1,
        osi_definition='Maximum Frobenius drift over all order pairs of Frobenius-normalized group-centroid distance matrices; repeat excluded.',
        same_input_repeat_per_seed_variant=1, independent_process_per_cell=True,
        native_backend_bundled=method == 'PCA' and cfg['adapter'] is None,
        unique_cause_established=False, biological_preservation_established=False,
        controls_independently_verified=False, input_equivalence_independently_verified=False,
        capability='CONTROL_PROBE' if len(cfg['variants']) > 1 else 'ORDER_DETECTION_ONLY',
        execution_mode='ISOLATED_ADAPTERS',
        claim_boundary=BOUNDARY, seed_results=[], reductions=[], failures=[])
    out.mkdir(parents=True, exist_ok=False)
    _write(out / 'PLAN.json', report)
    if not execute:
        return report
    if method != 'PCA' and not cfg['adapter']:
        report.update(status='UNCONFIGURED', reason='Provide an explicit native backend adapter (module:callable); not installed by this interface.')
    else:
        adapter_hashes = set()
        for seed in cfg['seeds']:
            result = dict(seed=seed, variants={})
            for variant in cfg['variants']:
                arms = {}
                for label in list(cfg['orders']) + ['repeat']:
                    cell = out / 'cells' / str(seed) / variant / label
                    cell.mkdir(parents=True, exist_ok=False)
                    request = dict(method=method, seed=seed, variant=variant, order='reference' if label == 'repeat' else label,
                        is_same_input_repeat=label == 'repeat', adapter=cfg['adapter'],
                        controls=recipe['variants'][variant], fixed_settings=cfg['fixed_settings'],
                        input=str(source), input_sha256=source_hash, executor_sha256=report['executor_sha256'])
                    request_path = cell / 'REQUEST.json'
                    _write(request_path, request)
                    env = dict(os.environ, PYTHONHASHSEED=str(seed), CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    try:
                        with (cell / 'stdout.log').open('x', encoding='utf-8') as stdout, (cell / 'stderr.log').open('x', encoding='utf-8') as stderr:
                            process = subprocess.run([sys.executable, '-m', 'stia.method_diagnostics_v1', '--worker', str(request_path)],
                                                     env=env, stdout=stdout, stderr=stderr,
                                                     timeout=cfg['timeout_seconds'], check=False)
                        if process.returncode:
                            raise RuntimeError('Adapter worker exit code ' + str(process.returncode))
                        receipt = json.loads((cell / 'RECEIPT.json').read_text(encoding='utf-8'))
                        if _hash(cell / 'RAW.npz') != receipt['raw_sha256']:
                            raise ValueError('Raw output hash mismatch')
                        adapter_hashes.add(receipt['adapter_sha256'])
                        with np.load(cell / 'RAW.npz', allow_pickle=False) as raw:
                            arms[label] = (raw['embedding'].copy(), raw['spot_ids'].copy())
                        report['completed_cells'] += 1
                    except Exception as exc:
                        if not (cell / 'FAILURE.json').exists():
                            _write(cell / 'FAILURE.json', dict(error=type(exc).__name__, message=str(exc)))
                        report['failures'].append(dict(seed=seed, variant=variant, cell=label, reason=str(exc)))
                result['variants'][variant] = summarize_embeddings(arms, ids, groups, cfg['orders'], neighbor_sizes=cfg['neighbor_sizes'])
            for variant in cfg['variants'][1:]:
                report['reductions'].append(_reduce(seed, variant, result['variants']['native'], result['variants'][variant]))
            report['seed_results'].append(result)
        valid = all(v['status'] == 'VALID' for r in report['seed_results'] for v in r['variants'].values())
        frozen = _hash(source) == source_hash and _hash(__file__) == report['executor_sha256'] and len(adapter_hashes) <= 1
        report['provenance_consistent'] = frozen
        report['status'] = 'COMPLETE_DESCRIPTIVE' if valid and frozen and not report['failures'] else 'INCOMPLETE'
        if not frozen:
            # No controlled inference may survive a changed input or adapter.
            report['reductions'] = []
            report['failures'].append(dict(reason='Frozen input, executor or adapter source changed across cells'))
    _write_report(out, report)
    return report


def _write_report(out, report):
    method, recipe = report['method'], report['recipe']
    _write(out / 'REPORT.json', report)
    with (out / 'REPORT.md').open('x', encoding='utf-8') as handle:
        handle.write('# %s method diagnosis\n\nStatus: %s\n\n%s\n\n%s\n\n' %
                     (method, report['status'], recipe['boundary'], BOUNDARY))
        handle.write('OSI: normalized group-centroid geometry; smaller is more consistent. '
                     'Repeat: supplied same-input rerun, not part of order comparisons. '
                     'Technical seeds do not provide independent biological replication.\n\n')
        if report['execution_mode'] == 'SAVED_OUTPUTS_ONLY':
            handle.write('Independent process provenance is not verified for saved outputs. '
                         'Input equivalence and intervention correctness are not independently verified.\n\n')
        else:
            handle.write('Each executed cell uses a fresh Python process; this alone does not prove bitwise determinism.\n\n')
        for result in report['seed_results']:
            for variant, value in result['variants'].items():
                handle.write('- Seed %s, %s: %s; OSI=%s; same-input repeat=%s.\n' %
                             (result['seed'], variant, value['status'], value.get('osi'), value.get('repeat_drift')))
        handle.write('\n')
        for row in report['reductions']:
            handle.write('- Seed %s, %s: OSI %s → %s; absolute reduction %s; relative reduction %s%%; '
                         'same-input repeat %s → %s.\n' % (row['seed'], row['variant'], row['native_osi'],
                         row['controlled_osi'], row['absolute_reduction'], row['relative_reduction_percent'],
                         row['native_repeat_drift'], row['controlled_repeat_drift']))
        handle.write('\nMissing/invalid evidence remains unresolved. See REPORT.json and cell receipts/logs.\n')


def diagnose_saved_method(method, manifest_path, config, output):
    """Diagnose frozen, already computed outputs without importing any backend.

    The manifest supplies spot/group identity, seed/variant/order, NPZ paths and
    SHA-256 hashes. Input equivalence and control correctness are NOT established
    by an output file alone. Missing or damaged cells remain unresolved.
    """
    method, recipe = _method_recipe(method, config)
    cfg = _config(method, config, recipe)
    source, out = Path(manifest_path).resolve(), Path(output).resolve()
    manifest_hash = _hash(source)
    manifest = json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 'stia.method_outputs.v1':
        raise ValueError('Saved outputs require schema_version stia.method_outputs.v1')
    ids, groups = _ids(manifest['spot_ids'], 'spot_ids'), _ids(manifest['group_ids'], 'group_ids')
    if len(ids) != len(groups) or len(set(ids)) != len(ids) or len(set(groups)) < 3:
        raise ValueError('Saved output metadata requires unique aligned spots and at least three groups')
    if any(k >= len(ids) for k in cfg['neighbor_sizes']):
        raise ValueError('neighbor_sizes must be smaller than spot count')
    expected = {(s, v, o) for s in cfg['seeds'] for v in cfg['variants'] for o in cfg['orders'] + ['repeat']}
    inventory = {}
    observations = manifest.get('observations')
    if not isinstance(observations, list):
        raise ValueError('observations must be a list')
    for item in observations:
        if not isinstance(item, dict) or set(item) != {'seed', 'variant', 'order', 'path', 'sha256'}:
            raise ValueError('Each observation needs exactly seed, variant, order, path, sha256')
        if type(item['seed']) is not int or not all(isinstance(item[k], str) for k in ('variant', 'order', 'path', 'sha256')):
            raise ValueError('Malformed observation key/path/hash')
        key = (item['seed'], item['variant'], item['order'])
        if key not in expected or key in inventory:
            raise ValueError('Unexpected or duplicate seed/variant/order observation')
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('Each observation needs a lowercase SHA-256 digest')
        path = Path(item['path'])
        path = (source.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if path == out or out in path.parents or path == source:
            raise ValueError('Output must be separate from manifest and saved evidence')
        inventory[key] = (path, item['sha256'])
    if out == source or out in source.parents:
        raise ValueError('Output must be separate from manifest')
    report = dict(schema_version=VERSION, method=method, config=cfg, recipe=recipe,
        status='INCOMPLETE', execution_mode='SAVED_OUTPUTS_ONLY', input=str(source),
        input_sha256=manifest_hash, config_sha256=_json_hash(cfg), executor_sha256=_hash(__file__),
        planned_cells=len(expected), completed_cells=0, n_spots=len(ids), n_groups=len(set(groups)),
        capability='CONTROL_PROBE' if len(cfg['variants']) > 1 else 'ORDER_DETECTION_ONLY',
        native_backend_bundled=False, controls_independently_verified=False,
        input_equivalence_independently_verified=False, independent_process_per_cell=False,
        unique_cause_established=False, biological_preservation_established=False,
        claim_boundary=BOUNDARY, observations=observations, seed_results=[], reductions=[], failures=[])
    out.mkdir(parents=True, exist_ok=False)
    _write(out / 'PLAN.json', report)
    for seed in cfg['seeds']:
        result = dict(seed=seed, variants={})
        for variant in cfg['variants']:
            arms = {}
            for order in cfg['orders'] + ['repeat']:
                try:
                    path, digest = inventory[(seed, variant, order)]
                    if _hash(path) != digest:
                        raise ValueError('Frozen saved-output hash mismatch')
                    with np.load(path, allow_pickle=False) as data:
                        arms[order] = (data['embedding'].copy(), data['spot_ids'].copy())
                    if _hash(path) != digest:
                        raise ValueError('Saved output changed during read')
                    report['completed_cells'] += 1
                except Exception as exc:
                    report['failures'].append(dict(seed=seed, variant=variant, order=order, reason=str(exc)))
            result['variants'][variant] = summarize_embeddings(arms, ids, groups, cfg['orders'], neighbor_sizes=cfg['neighbor_sizes'])
        for variant in cfg['variants'][1:]:
            report['reductions'].append(_reduce(seed, variant, result['variants']['native'], result['variants'][variant]))
        report['seed_results'].append(result)
    valid = all(v['status'] == 'VALID' for r in report['seed_results'] for v in r['variants'].values())
    frozen_outputs = True
    for path, digest in inventory.values():
        try:
            frozen_outputs = frozen_outputs and _hash(path) == digest
        except OSError:
            frozen_outputs = False
    report['provenance_consistent'] = _hash(source) == manifest_hash and frozen_outputs and _hash(__file__) == report['executor_sha256']
    if not report['provenance_consistent']:
        report['reductions'] = []
        report['failures'].append(dict(reason='Manifest, saved evidence or executor does not match the frozen hashes'))
    if valid and report['provenance_consistent'] and not report['failures']:
        report['status'] = 'COMPLETE_DESCRIPTIVE'
    _write_report(out, report)
    return report


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        raise SystemExit(_worker(sys.argv[2]))
    raise SystemExit('Internal worker only; use stia diagnose-method --help')
