# -*- coding: utf-8 -*-
"""Command line entry point for the versioned STIA audit interface."""
from __future__ import annotations

import argparse
import json
import sys

from .engine import audit
from .splits import grouped_holdout_split, grouped_nested_cv_split, write_split_plan
from .cross_ai import run_cross_ai
from .contracts import TOOL_VERSION
from .demo import build_demo_case
from .io import canonical_write
from .tradeoff_v1 import compare_axes
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="stia", description="Spatial Transcriptomics Integration Audit (read-only audit; explicit opt-in method experiments)")
    parser.add_argument('--version', action='version', version='stia ' + TOOL_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    method_parser = sub.add_parser('diagnose-method', help='plan or explicitly execute isolated method-specific controls (backends required except PCA)')
    method_parser.add_argument('--method', required=True, help='built-in method name or a new method identifier')
    method_source = method_parser.add_mutually_exclusive_group(required=True)
    method_source.add_argument('--input', help='NPZ containing dense X, unique string spot_ids and contiguous group_ids')
    method_source.add_argument('--results', help='frozen saved-output manifest; no fitting or adapter import')
    method_parser.add_argument('--config', help='JSON configuration; omitted uses safe planning defaults')
    method_parser.add_argument('--output', required=True, help='new versioned directory; never overwritten')
    method_parser.add_argument('--execute', action='store_true', help='authorize actual adapter execution; default is plan only')
    demo_parser = sub.add_parser('demo', help='generate an offline synthetic example; no model fitting')
    demo_parser.add_argument('--output', required=True, help='new, empty example directory')
    axes_parser = sub.add_parser('compare-axes', help='report paired changes in eligible, higher-is-better axis scores')
    axes_parser.add_argument('--candidate', required=True, help='JSON mapping axis names to scores or null')
    axes_parser.add_argument('--reference', required=True, help='identity-matched reference score mapping')
    axes_parser.add_argument('--intervals', help='optional JSON mapping axes to paired difference intervals')
    axes_parser.add_argument('--output', required=True, help='new JSON output file; never overwritten')
    infer_parser = sub.add_parser('infer-axes', help='optional paired-axis inference with Holm directional and global tradeoff tests')
    infer_parser.add_argument('--input', required=True, help='NPY independent-unit by eligible-axis paired differences')
    infer_parser.add_argument('--axis-names', nargs='+', required=True, help='prespecified axis names in matrix column order')
    infer_parser.add_argument('--alpha', type=float, default=0.05)
    infer_parser.add_argument('--output', required=True, help='new JSON output file; never overwritten')
    audit_parser = sub.add_parser("audit", help="run the versioned M1-M9 audit")
    audit_parser.add_argument("--case-root", required=True, help="case directory containing frozen inputs")
    audit_parser.add_argument("--config", required=True, help="frozen YAML/JSON configuration")
    audit_parser.add_argument("--output", required=True, help="separate output directory")
    split_parser = sub.add_parser("split", help="create a donor-safe validation split plan")
    split_parser.add_argument("--metadata", required=True, help="metadata CSV/TSV/Parquet/H5AD")
    split_parser.add_argument("--output", required=True, help="new versioned output directory")
    split_parser.add_argument("--mode", required=True, choices=("holdout", "nested-cv"))
    split_parser.add_argument("--donor-col", default="donor_id")
    split_parser.add_argument("--condition-col", default="disease_or_condition")
    split_parser.add_argument("--spot-col", default="spot_id")
    split_parser.add_argument("--seed", type=int, default=20260823)
    split_parser.add_argument("--train-fraction", type=float, default=0.6)
    split_parser.add_argument("--validation-fraction", type=float, default=0.2)
    split_parser.add_argument("--test-fraction", type=float, default=0.2)
    split_parser.add_argument("--outer-folds", type=int, default=5)
    split_parser.add_argument("--inner-folds", type=int, default=3)
    cross_parser = sub.add_parser("cross-ai", aliases=("cross_ai", "cross-ai-score", "cross_ai_score"), help="ingest and compare independent answer manifests offline")
    cross_parser.add_argument("--case", required=True, help="frozen case/prompt identity JSON/YAML")
    cross_parser.add_argument("--answers", required=True, help="answer manifest JSON/YAML")
    cross_parser.add_argument("--scorers", help="optional scorer manifest JSON/YAML")
    cross_parser.add_argument("--gold", help="optional frozen gold/adjudication manifest JSON/YAML")
    cross_parser.add_argument("--output", required=True, help="new versioned output directory")
    cross_parser.add_argument("--strict", action="store_true", help="fail instead of emitting a non-independent descriptive result")
    args = parser.parse_args(argv)
    if args.command == 'diagnose-method':
        try:
            from .method_diagnostics_v1 import diagnose_method, diagnose_saved_method
            config = json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {}
            if args.results:
                if args.execute:
                    raise ValueError('--results is read-only; do not combine it with --execute')
                report = diagnose_saved_method(args.method, args.results, config, args.output)
            else:
                report = diagnose_method(args.method, args.input, config, args.output, execute=args.execute)
        except Exception as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 2
        print(json.dumps({key: report[key] for key in ('method', 'status', 'planned_cells', 'completed_cells', 'unique_cause_established')}, ensure_ascii=False))
        return 0 if report['status'] in ('PLANNED', 'COMPLETE_DESCRIPTIVE') else 1
    if args.command == 'infer-axes':
        try:
            import numpy as np
            from .inference_optimization_v1 import infer_axes, _jsonable
            values = np.load(args.input, allow_pickle=False)
            names = args.axis_names
            if values.ndim != 2 or len(names) != values.shape[1] or len(set(names)) != len(names):
                raise ValueError('Unique axis names must match input columns')
            report = _jsonable(infer_axes(values, args.alpha))
            report['schema_version'] = 'stia.axis_inference.v1'
            report['axes'] = names
            report['independent_units'] = len(values)
            report['input_sha256'] = __import__('hashlib').sha256(Path(args.input).read_bytes()).hexdigest()
            report['interpretation'] = 'Holm supplies simultaneous directional declarations; IUT tests only the existence of opposing effects.'
            target = Path(args.output).expanduser().resolve()
            if target == Path(args.input).resolve():
                raise ValueError('Output must be separate from input')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x', encoding='utf-8') as file:
                json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        except Exception as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 2
        print(json.dumps({'output': str(target), 'axes': names,
                          'holm_tradeoff': report['holm_directional']['tradeoff'],
                          'iut_tradeoff': report['tradeoff_iut']['tradeoff']}, ensure_ascii=False))
        return 0
    if args.command == 'demo':
        try:
            root = build_demo_case(args.output)
        except Exception as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 2
        print(json.dumps({'case_root': str(root), 'expected_audit_decision': 'ABSTAIN',
                          'reason': 'Synthetic example does not supply independent disease preservation calibration.'}))
        return 0
    if args.command == 'compare-axes':
        try:
            def load(path):
                return json.loads(Path(path).read_text(encoding='utf-8-sig'))
            candidate, reference = load(args.candidate), load(args.reference)
            if not isinstance(candidate, dict) or not isinstance(reference, dict):
                raise ValueError('Axis inputs must be JSON objects')
            value = compare_axes(candidate, reference, load(args.intervals) if args.intervals else None)
            target = Path(args.output).expanduser().resolve()
            if target in {Path(args.candidate).resolve(), Path(args.reference).resolve()}:
                raise ValueError('Output must be separate from input')
            target.parent.mkdir(parents=True, exist_ok=True)
            canonical_write(target, value)
        except Exception as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 2
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit":
        try:
            run = audit(args.case_root, args.config, output=args.output)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(run.as_dict(), ensure_ascii=False, sort_keys=True))
        return 0 if run.decision in ("ACCEPT", "ACCEPT_WITH_SCOPE") else 1
    if args.command == "split":
        try:
            common = {
                "donor_col": args.donor_col,
                "condition_col": args.condition_col,
                "spot_col": args.spot_col,
                "seed": args.seed,
            }
            if args.mode == "holdout":
                plan = grouped_holdout_split(
                    args.metadata,
                    train_fraction=args.train_fraction,
                    validation_fraction=args.validation_fraction,
                    test_fraction=args.test_fraction,
                    **common,
                )
            else:
                plan = grouped_nested_cv_split(
                    args.metadata,
                    outer_folds=args.outer_folds,
                    inner_folds=args.inner_folds,
                    **common,
                )
            paths = write_split_plan(plan, args.output)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"mode": plan.mode, "outputs": {name: str(path) for name, path in paths.items()}}, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command in ("cross-ai", "cross_ai", "cross-ai-score", "cross_ai_score"):
        try:
            result = run_cross_ai(args.case, args.answers, scorers=args.scorers, gold=args.gold, output=args.output, strict=args.strict)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "status": result["status"],
            "mode": result["mode"],
            "independent": result["independent"],
            "agreement": {key: result["agreement"].get(key) for key in ("decision_agreement", "cohen_kappa", "finding_jaccard", "root_cause_agreement", "coverage")},
            "gold_status": result["gold_scoring"]["status"],
        }, ensure_ascii=False, sort_keys=True))
        return 0 if result["status"] in ("PASS", "SAME_MODEL_ONLY") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
