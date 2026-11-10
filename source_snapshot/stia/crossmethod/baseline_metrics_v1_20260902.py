# -*- coding: utf-8 -*-
"""Build the F1A baseline-comparison and framework-quality metric pack.

This module is deliberately an evidence assembler, not an experiment runner.
It reads only named terminal acceptance documents and (optionally) a small,
explicit development fixture.  Acceptance documents contribute capability and
coverage fields; they are never treated as truth-labelled benchmark records.
Numeric performance metrics are delegated to :mod:`stia.baselines` only for
records carrying both ``frozen_truth=True`` and ``observable_prediction=True``.
Such records remain visibly ``DEVELOPMENT_NON_INDEPENDENT`` and cannot be
promoted to independent validation.

All writers are additive (``open(..., 'x')``) and the payload omits wall-clock
timestamps so repeated builds from unchanged evidence are byte deterministic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from ..baselines import baseline_cells, baseline_comparisons, framework_quality_metrics
from ..contracts import BASELINE_IDS, PRIMARY_METRIC_IDS, SCHEMA_VERSION


PACK_SCHEMA_VERSION = "stia.f1a_baseline_metric_pack.v1.20260902"
DEVELOPMENT_SCOPE = "DEVELOPMENT_NON_INDEPENDENT"
TERMINAL_DECISIONS = {"PASS", "CONDITIONAL_PASS"}
METHODS = ("PCA", "GraphST", "Harmony", "INSPIRE", "STAligner", "PRECAST")

ACCEPTANCE_SOURCES: Dict[str, str] = {
    "internal_six_method": "docs/SIX_METHOD_ORDER_COVERAGE_ACCEPTANCE_20260831_r1.json",
    "PCA": "docs/PCA_EXTERNAL_GSE254652_FINAL_ACCEPTANCE_20260831_r1.json",
    "GraphST": "docs/GRAPHST_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "Harmony": "docs/HARMONY_GSE254652_FIVE_SEED_FINAL_ACCEPTANCE_20260901_r1.json",
    "INSPIRE": "docs/INSPIRE_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "STAligner": "docs/STALIGNER_EXTERNAL_GSE254652_FINAL_ACCEPTANCE_20260831_r1.json",
    "PRECAST": "docs/PRECAST_GSE254652_SEED20260819_TERMINAL_ACCEPTANCE_20260901_r1.json",
}
LITERATURE_REFERENCE = "docs/ORDER_SENSITIVITY_SYSTEMATIC_PRIOR_ART_AUDIT_20260901_r1.json"


class BaselineMetricError(ValueError):
    """Raised for malformed development fixtures or output destinations."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise BaselineMetricError(f"missing JSON evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser error is environment-specific
        raise BaselineMetricError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BaselineMetricError(f"JSON evidence root must be an object: {path}")
    return value


def _source_identity(root: Path, relative: str, *, terminal: bool = False) -> Dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return {"path": relative, "exists": False, "sha256": None, "bytes": None, "terminal": terminal}
    return {
        "path": relative,
        "exists": True,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "terminal": terminal,
    }


def _decision(document: Mapping[str, Any]) -> str:
    value = document.get("decision", document.get("acceptance_decision", "UNRESOLVED"))
    return str(value).strip().upper() if value is not None else "UNRESOLVED"


def _execution_status(document: Mapping[str, Any]) -> str:
    value = document.get("execution_acceptance")
    checks = document.get("completion_checks")
    if value is None and isinstance(checks, Mapping):
        value = checks.get("execution_validity")
    if value is None and isinstance(checks, Mapping) and checks.get("fit_complete") is True:
        value = "PASS"
    if isinstance(value, Mapping):
        # A mapping of per-seed receipts is not itself a terminal receipt.
        return "UNRESOLVED"
    return str(value).strip().upper() if value is not None else "UNRESOLVED"


def _seeds(document: Mapping[str, Any], method: str) -> list[Any]:
    identity = document.get("experiment_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    for key in ("seeds", "model_seeds", "requested_seeds"):
        value = identity.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
    if identity.get("seed") is not None:
        return [identity["seed"]]
    # PCA is a deterministic fit, not an absent stochastic seed.
    return ["DETERMINISTIC_SINGLE_FIT"] if method == "PCA" else []


def _numeric_values(document: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    candidates = []
    for key in ("osi_results", "positive_negative_uncertain_results", "statistical_audit", "numeric_acceptance"):
        value = document.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        value = candidate.get("osi")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        by_seed = candidate.get("by_seed")
        if isinstance(by_seed, Mapping):
            for nested in by_seed.values():
                if isinstance(nested, Mapping):
                    nested_values = nested.values()
                else:
                    nested_values = (nested,)
                for item in nested_values:
                    if isinstance(item, (int, float)) and not isinstance(item, bool):
                        values.append(float(item))
    return values


def _method_record(root: Path, method: str, relative: str) -> Dict[str, Any]:
    identity: Mapping[str, Any] = {}
    document: Mapping[str, Any] = {}
    source = _source_identity(root, relative, terminal=True)
    if source["exists"]:
        try:
            document = _read_json(root / relative)
            identity = document.get("experiment_identity") if isinstance(document.get("experiment_identity"), Mapping) else {}
        except BaselineMetricError:
            # Preserve a deterministic unresolved row; the source error is
            # represented in ``source_error`` rather than guessed away.
            source["source_error"] = "MALFORMED_JSON"
    else:
        source["source_error"] = "MISSING"
    decision = _decision(document)
    execution = _execution_status(document)
    seeds = _seeds(document, method)
    values = _numeric_values(document)
    accepted_arms = identity.get("accepted_arms")
    expected_arms = identity.get("expected_arms")
    return {
        "method": method,
        "dataset": identity.get("dataset"),
        "terminal_decision": decision if decision in TERMINAL_DECISIONS else "UNRESOLVED",
        "source_decision": decision,
        "execution_validity": execution if execution in {"PASS", "FAIL", "UNRESOLVED"} else "UNRESOLVED",
        "terminal_acceptance": decision in TERMINAL_DECISIONS,
        "seed_ids": seeds,
        "seed_count": len(seeds),
        "deterministic_single_fit": method == "PCA" and seeds == ["DETERMINISTIC_SINGLE_FIT"],
        "orders_per_seed": identity.get("orders_per_seed", identity.get("n_orders")),
        "expected_arms": expected_arms,
        "accepted_arms": accepted_arms,
        "arm_coverage_fraction": (float(accepted_arms) / float(expected_arms)
                                   if isinstance(accepted_arms, (int, float)) and isinstance(expected_arms, (int, float)) and expected_arms > 0
                                   else None),
        "observable_osi": bool(values),
        "osi_value_count": len(values),
        "osi_values_observed": values,
        "external_exposure": identity.get("exposure"),
        "project_exposed": str(identity.get("exposure", "")).upper().find("PROJECT_EXPOSED") >= 0,
        "source": source,
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _baseline_row(baseline_id: str, name: str, definition: str, status: str,
                  capabilities: Mapping[str, Any], coverage: Mapping[str, Any],
                  evidence_refs: Sequence[str], limitations: Sequence[str]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "name": name,
        "definition": definition,
        "status": status,
        "decision": "DESCRIPTIVE_ONLY",
        "capability_fields": dict(capabilities),
        "coverage_fields": dict(coverage),
        "evidence_refs": list(evidence_refs),
        "limitations": list(limitations),
        "threshold_status": "PROVISIONAL",
        "accuracy_claim": "UNRESOLVED",
        "independent_validation": "UNRESOLVED",
    }


def _build_baselines(root: Path, internal: Mapping[str, Any], methods: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    terminal = [row for row in methods if row.get("terminal_acceptance")]
    execution = [row for row in terminal if row.get("execution_validity") == "PASS"]
    seed5 = [row for row in terminal if row.get("seed_count") == 5]
    osi = [row for row in terminal if row.get("observable_osi")]
    internal_pass = _decision(internal) == "PASS"
    internal_methods = internal.get("methods") if isinstance(internal.get("methods"), Mapping) else {}
    internal_complete = internal_pass and set(internal_methods) == set(METHODS)
    refs = [row["source"]["path"] for row in methods if row.get("source", {}).get("exists")]
    b0 = _baseline_row(
        "B0", "default_single_run_evaluation", "default single-run evaluation",
        "CONDITIONAL_PASS" if terminal else "UNRESOLVED",
        {"single_run": True, "terminal_acceptance_evidence": bool(terminal), "seed_robustness": False,
         "osi_order_invariance": False, "three_axis_validity": False, "claim_governance": False},
        {"terminal_acceptance_documents": len(terminal), "execution_valid_documents": len(execution),
         "method_count": len(METHODS), "method_coverage_fraction": _fraction(len(terminal), len(METHODS))},
        refs, ["A terminal method receipt is not a framework accuracy benchmark.", "External receipts are project-exposed where stated."],
    )
    b1 = _baseline_row(
        "B1", "standard_scib_istbench_metric_suite", "standard scIB/iSTBench-style metric suite", "UNRESOLVED",
        {"single_run": False, "terminal_acceptance_evidence": False, "standard_metric_suite": False,
         "seed_robustness": False, "osi_order_invariance": False, "three_axis_validity": False},
        {"terminal_acceptance_documents": 0, "method_count": 0, "method_coverage_fraction": None}, [],
        ["No frozen truth-labelled scIB/iSTBench suite is present in this F1A evidence set."],
    )
    b2 = _baseline_row(
        "B2", "standard_robustness_seed_downsample_hyperparameter", "standard robustness (seed/downsample/hyperparameter)",
        "CONDITIONAL_PASS" if seed5 else "UNRESOLVED",
        {"single_run": False, "terminal_acceptance_evidence": bool(seed5), "standard_metric_suite": False,
         "seed_robustness": bool(seed5), "downsample_robustness": False, "hyperparameter_robustness": False},
        {"five_seed_method_count": len(seed5), "method_count": len(METHODS),
         "five_seed_method_coverage_fraction": _fraction(len(seed5), len(METHODS)),
         "terminal_acceptance_documents": len(seed5)},
        [row["source"]["path"] for row in seed5],
        ["Only explicit seed coverage is counted; downsampling and hyperparameter sweeps are unresolved.",
         "Computational seed repeats are not biological replicates."],
    )
    b3 = _baseline_row(
        "B3", "osi_only_audit", "OSI-only audit", "CONDITIONAL_PASS" if osi else "UNRESOLVED",
        {"single_run": False, "terminal_acceptance_evidence": bool(osi), "osi_order_invariance": bool(osi),
         "four_order_audit": bool(osi), "standard_metric_suite": False, "three_axis_validity": False},
        {"methods_with_observable_osi": len(osi), "method_count": len(METHODS),
         "osi_method_coverage_fraction": _fraction(len(osi), len(METHODS)),
         "observable_osi_values": sum(int(row.get("osi_value_count", 0)) for row in osi)},
        [row["source"]["path"] for row in osi],
        ["OSI margin is provisional and not a p-value.", "OSI alone does not establish whole-framework validity."],
    )
    b4 = _baseline_row(
        "B4", "full_integrated_framework", "full integrated framework",
        # The internal six-method matrix is useful capability/coverage
        # evidence, but it is not a frozen truth-labelled framework
        # benchmark.  Keep the B4 cell unresolved until that independent
        # benchmark exists; never promote matrix completeness to framework
        # validity.
        "UNRESOLVED",
        {"single_run": True, "terminal_acceptance_evidence": bool(terminal), "standard_metric_suite": False,
         "seed_robustness": bool(seed5), "osi_order_invariance": bool(osi),
         "three_axis_validity": False, "claim_governance": False, "internal_matrix_coverage": internal_complete,
         "frozen_truth_benchmark": False},
        {"internal_matrix_methods": len(internal_methods), "internal_matrix_complete": internal_complete,
         "external_terminal_methods": len(terminal), "external_method_coverage_fraction": _fraction(len(terminal), len(METHODS)),
         "frozen_truth_benchmark_records": 0},
        ([ACCEPTANCE_SOURCES["internal_six_method"]] + refs),
        ["Full integrated framework quality/accuracy remains unresolved without an independent frozen benchmark.",
         "The current external dataset is project-exposed, not blind held-out confirmation."],
    )
    return [b0, b1, b2, b3, b4]


def _eligible_fixture_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Dict[str, Any]]]:
    eligible: list[Mapping[str, Any]] = []
    excluded: list[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            excluded.append({"index": index, "reason": "NOT_OBJECT"})
            continue
        reasons = []
        if record.get("frozen_truth") is not True:
            reasons.append("frozen_truth_not_true")
        if record.get("observable_prediction") is not True:
            reasons.append("observable_prediction_not_true")
        if reasons:
            excluded.append({"index": index, "record_id": record.get("record_id"), "reason": "+".join(reasons)})
        else:
            eligible.append(record)
    return eligible, excluded


def _load_fixtures(root: Path, fixture_path: Path | str | None) -> tuple[list[Mapping[str, Any]], Dict[str, Any]]:
    if fixture_path is None:
        return [], {"path": None, "exists": False, "scope": DEVELOPMENT_SCOPE, "records": 0, "eligible_records": 0, "excluded_records": 0}
    path = Path(fixture_path)
    if not path.is_absolute():
        path = root / path
    source = {"path": _relative(path, root), "exists": path.is_file(), "scope": DEVELOPMENT_SCOPE}
    if not path.is_file():
        source.update({"records": 0, "eligible_records": 0, "excluded_records": 0, "error": "MISSING"})
        return [], source
    value = _read_json(path)
    records = value.get("records") if isinstance(value, Mapping) else value
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise BaselineMetricError("development fixture JSON must be a list or an object with records")
    eligible, excluded = _eligible_fixture_records(records)
    source.update({"sha256": _sha256(path), "records": len(records), "eligible_records": len(eligible), "excluded_records": len(excluded), "excluded": excluded})
    return eligible, source


def _literature_reference(root: Path) -> Dict[str, Any]:
    source = _source_identity(root, LITERATURE_REFERENCE, terminal=False)
    return {
        "reference_id": "descoeudres_row_order_ari_sd",
        "metric": "row-order ARI/SD",
        "source": source,
        "reference_only": True,
        "executed_in_this_pack": False,
        "numeric_result": None,
        "claim_boundary": "literature antecedent/reference baseline; no executed result is fabricated",
    }


def _comparison_rows(cells: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    fields = ("terminal_acceptance_documents", "method_coverage_fraction", "five_seed_method_count",
              "five_seed_method_coverage_fraction", "methods_with_observable_osi", "osi_method_coverage_fraction",
              "internal_matrix_complete", "frozen_truth_benchmark_records")
    for left_index, left in enumerate(cells):
        for right in cells[left_index + 1:]:
            left_cov = left.get("coverage_fields") if isinstance(left.get("coverage_fields"), Mapping) else {}
            right_cov = right.get("coverage_fields") if isinstance(right.get("coverage_fields"), Mapping) else {}
            delta: Dict[str, Any] = {}
            for field in fields:
                lv, rv = left_cov.get(field), right_cov.get(field)
                if isinstance(lv, (int, float)) and not isinstance(lv, bool) and isinstance(rv, (int, float)) and not isinstance(rv, bool):
                    delta[field] = float(rv) - float(lv)
                elif isinstance(lv, bool) and isinstance(rv, bool):
                    delta[field] = int(rv) - int(lv)
                else:
                    delta[field] = None
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "left": left.get("baseline_id"),
                "right": right.get("baseline_id"),
                "coverage_delta_right_minus_left": delta,
                "capability_comparison": "EXPLICIT_FIELDS_ONLY",
                "decision": "DESCRIPTIVE_ONLY",
                "ranking": None,
                "accuracy_inference": None,
                "threshold_status": "PROVISIONAL",
            })
    return rows


def build_pack(project_root: Path | str, *, development_fixtures: Path | str | None = None) -> Dict[str, Any]:
    """Build an additive, deterministic F1A payload from frozen evidence."""
    root = Path(project_root).resolve()
    internal_path = root / ACCEPTANCE_SOURCES["internal_six_method"]
    internal: Mapping[str, Any] = {}
    internal_source = _source_identity(root, ACCEPTANCE_SOURCES["internal_six_method"], terminal=True)
    if internal_source["exists"]:
        try:
            internal = _read_json(internal_path)
        except BaselineMetricError:
            internal_source["source_error"] = "MALFORMED_JSON"
    else:
        internal_source["source_error"] = "MISSING"
    method_rows = [_method_record(root, method, ACCEPTANCE_SOURCES[method]) for method in METHODS]
    cells = _build_baselines(root, internal, method_rows)
    comparisons = _comparison_rows(cells)
    fixtures, fixture_source = _load_fixtures(root, development_fixtures)
    # The baselines module computes only explicit record-level quantities.  A
    # missing fixture therefore leaves every primary value null/UNRESOLVED.
    quality = framework_quality_metrics({}, cells=baseline_cells({"baselines": cells}), benchmark_records=fixtures)
    for row in quality:
        row["evidence_scope"] = DEVELOPMENT_SCOPE if fixtures else "NO_ELIGIBLE_BENCHMARK_RECORDS"
        row["independent_validation"] = False if fixtures else "UNRESOLVED"
        row["accuracy_claim"] = "UNRESOLVED"
    sources = {"internal_six_method": internal_source}
    sources.update({method: row["source"] for method, row in zip(METHODS, method_rows)})
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "stia_schema_version": SCHEMA_VERSION,
        "project_root": str(root),
        "evidence_policy": {
            "promotion_rule": "terminal acceptance documents and explicit development fixtures only",
            "numeric_rule": "numeric performance requires frozen_truth=True and observable_prediction=True",
            "development_scope": DEVELOPMENT_SCOPE,
            "missing_metric_value": None,
            "missing_metric_status": "UNRESOLVED",
            "threshold_status": "PROVISIONAL_UNLESS_FROZEN_CALIBRATION_IDENTITY_EXISTS",
            "no_inferred_accuracy": True,
        },
        "sources": sources,
        "method_records": method_rows,
        "baseline_cells": cells,
        "baseline_comparisons": comparisons,
        "framework_quality_metrics": quality,
        "development_fixture": fixture_source,
        "literature_reference_baselines": [_literature_reference(root)],
        "summary": {
            "baseline_count": len(cells),
            "comparison_count": len(comparisons),
            "primary_metric_count": sum(row.get("metric_id") in PRIMARY_METRIC_IDS for row in quality),
            "primary_metrics_with_values": sum(row.get("metric_id") in PRIMARY_METRIC_IDS and row.get("value") is not None for row in quality),
            "terminal_external_method_count": sum(row.get("terminal_acceptance") is True for row in method_rows),
            "external_five_seed_method_count": sum(row.get("seed_count") == 5 for row in method_rows),
            "observable_osi_method_count": sum(row.get("observable_osi") is True for row in method_rows),
            "independent_validation": False,
            "whole_framework_accuracy": "UNRESOLVED",
        },
        "claim_boundary": [
            "B0-B4 rows compare explicit capability/coverage fields and do not rank methods or infer accuracy.",
            "Development fixture metrics are non-independent contract evidence, not external validation.",
            "A same-source/project-exposed pilot is not blind held-out confirmation.",
            "Descoeudres-style row-order ARI/SD is retained as a reference antecedent only; no executed value is supplied.",
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field), ensure_ascii=False, sort_keys=True) if isinstance(row.get(field), (dict, list)) else row.get(field) for field in fields})


def write_pack(payload: Mapping[str, Any], output_dir: Path | str) -> Dict[str, Path]:
    """Write the canonical JSON and three flat CSV views, refusing overwrite."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "f1a_baseline_metric_pack_20260902.json",
        "cells_csv": destination / "f1a_baseline_cells_20260902.csv",
        "comparisons_csv": destination / "f1a_baseline_comparisons_20260902.csv",
        "metrics_csv": destination / "f1a_framework_quality_metrics_20260902.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"refusing to overwrite F1A artifacts in {destination}")
    _write_json(paths["json"], payload)
    _write_csv(paths["cells_csv"], payload.get("baseline_cells", []),
               ("schema_version", "baseline_id", "name", "definition", "status", "decision", "capability_fields", "coverage_fields", "evidence_refs", "threshold_status", "accuracy_claim", "independent_validation"))
    _write_csv(paths["comparisons_csv"], payload.get("baseline_comparisons", []),
               ("schema_version", "left", "right", "coverage_delta_right_minus_left", "capability_comparison", "decision", "ranking", "accuracy_inference", "threshold_status"))
    _write_csv(paths["metrics_csv"], payload.get("framework_quality_metrics", []),
               ("schema_version", "metric_id", "value", "unit", "threshold", "threshold_status", "status", "metric_role", "calibration_identity", "evidence_scope", "independent_validation", "accuracy_claim"))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the additive F1A baseline and quality-metric pack")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-fixtures", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = build_pack(args.project_root, development_fixtures=args.development_fixtures)
        paths = write_pack(payload, args.output_dir)
    except (BaselineMetricError, OSError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "schema_version": payload["schema_version"], "summary": payload["summary"], "outputs": {key: str(path) for key, path in paths.items()}}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
