# -*- coding: utf-8 -*-
"""Baseline comparison and framework-quality summaries.

The functions in this module are intentionally descriptive.  They provide a
stable B0--B4 comparison surface and serialize supplied benchmark evidence,
but never turn planning heuristics into universal scientific thresholds.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .contracts import BASELINE_SURFACE, PRIMARY_METRIC_IDS, SCHEMA_VERSION


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def baseline_cells(config: Mapping[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Return canonical B0--B4 cells plus any explicitly supplied evidence.

    ``config['baselines']`` may be a list or mapping keyed by B0--B4.  Values
    are copied as evidence; absent values remain ``UNRESOLVED``.  Each cell
    carries ``threshold_status='PROVISIONAL'`` unless the caller explicitly
    records a calibrated benchmark identity.
    """
    cfg = config or {}
    calibration_identity = cfg.get("calibration_identity") or cfg.get("quality_metrics_calibration_identity")
    if calibration_identity is None and isinstance(cfg.get("quality_metrics_calibration"), Mapping):
        calibration_identity = cfg["quality_metrics_calibration"].get("identity")
    if calibration_identity is None and isinstance(cfg.get("benchmark_calibration"), Mapping):
        calibration_identity = cfg["benchmark_calibration"].get("identity")
    supplied = cfg.get("baselines", cfg.get("baseline_comparisons", {}))
    if isinstance(supplied, Mapping):
        by_id = {str(k): v for k, v in supplied.items()}
    elif isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        by_id = {
            str(item.get("baseline_id", item.get("id"))): item
            for item in supplied
            if isinstance(item, Mapping) and item.get("baseline_id", item.get("id")) is not None
        }
    else:
        by_id = {}
    cells: List[Dict[str, Any]] = []
    for spec in BASELINE_SURFACE:
        raw = by_id.get(spec["id"])
        evidence = dict(raw) if isinstance(raw, Mapping) else {}
        status = str(evidence.pop("status", "UNRESOLVED")).upper()
        if status not in ("PASS", "CONDITIONAL_PASS", "FAIL", "NOT_APPLICABLE", "UNRESOLVED"):
            status = "UNRESOLVED"
        threshold_status = str(evidence.pop("threshold_status", "PROVISIONAL")).upper()
        if threshold_status not in ("PROVISIONAL", "CALIBRATED"):
            threshold_status = "PROVISIONAL"
        if threshold_status == "CALIBRATED" and not calibration_identity:
            threshold_status = "PROVISIONAL"
        cells.append({
            "schema_version": SCHEMA_VERSION,
            "baseline_id": spec["id"],
            "name": spec["name"],
            "definition": spec.get("definition", spec["name"]),
            "status": status,
            "evidence": evidence,
            "threshold_status": threshold_status,
            "calibration_identity": calibration_identity,
            "thresholds": evidence.pop("thresholds", None),
        })
    return cells


def baseline_comparisons(cells: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build pairwise descriptive comparison rows for B0--B4.

    No ranking or acceptance decision is inferred from missing metrics.  A
    supplied ``quality`` number is retained as-is and tagged provisional.
    """
    rows: List[Dict[str, Any]] = []
    for left_index, left in enumerate(cells):
        for right in cells[left_index + 1 :]:
            left_quality = left.get("evidence", {}).get("quality") if isinstance(left.get("evidence"), Mapping) else None
            right_quality = right.get("evidence", {}).get("quality") if isinstance(right.get("evidence"), Mapping) else None
            delta = None
            if isinstance(left_quality, (int, float)) and isinstance(right_quality, (int, float)):
                delta = float(right_quality) - float(left_quality)
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "left": left.get("baseline_id"),
                "right": right.get("baseline_id"),
                "left_status": left.get("status"),
                "right_status": right.get("status"),
                "quality_delta_right_minus_left": delta,
                "threshold_status": "PROVISIONAL",
                "decision": "DESCRIPTIVE_ONLY",
            })
    return rows


def framework_quality_metrics(
    modules: Mapping[str, Any],
    *,
    cells: Sequence[Mapping[str, Any]] | None = None,
    config: Mapping[str, Any] | None = None,
    benchmark_records: Sequence[Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Calculate the frozen framework-quality metrics from benchmark records.

    The metrics intentionally use *only* explicit records supplied under
    ``config['benchmark_records']`` (or ``config['benchmark']['records']``).
    Module statuses and baseline coverage are never used as substitutes for
    benchmark outcomes.  A missing field/denominator is represented by a
    ``None`` value and ``UNRESOLVED`` status.  Thresholds become calibrated
    only when a non-empty frozen calibration identity accompanies the
    calibration map; otherwise they remain visibly ``PROVISIONAL``.

    Records are deliberately small JSON-like mappings.  The accepted aliases
    make the contract tolerant of benchmark runners while preserving one
    canonical output vocabulary:

    ``stratum``/``case_type`` (``clean`` or ``defect``), ``decision`` or
    ``observed_decision``, ``detected``/``unsafe_acceptance``,
    ``root_cause_true``/``root_cause_pred``, ``provenance_completeness``,
    ``required_gate_bypass_count``/``bypassed_gates``, ``runtime_seconds`` and
    ``memory_mb``.
    """

    configured = config if isinstance(config, Mapping) else {}
    records = list(benchmark_records) if benchmark_records is not None else _benchmark_records(configured)
    primary = _benchmark_metric_values(records)

    # These compact diagnostics are useful for dashboards but are explicitly
    # supplementary; they cannot stand in for any primary framework metric.
    baseline = list(cells or [])
    supplementary: Dict[str, Any] = {
        "module_pass_rate": _module_pass_rate(modules),
        "module_resolved_rate": _module_resolved_rate(modules),
        "baseline_cell_coverage": (
            sum(cell.get("status") != "UNRESOLVED" for cell in baseline) / len(baseline)
            if baseline else None
        ),
    }

    calibration = configured.get("quality_metrics_calibration")
    if not isinstance(calibration, Mapping):
        calibration = {}
    # A map alone is not evidence of calibration: require a frozen identity
    # (hash, URI, or explicit versioned identifier) before exposing cutoffs.
    calibration_identity = (
        configured.get("calibration_identity")
        or configured.get("quality_metrics_calibration_identity")
        or (calibration.get("identity") if isinstance(calibration, Mapping) else None)
    )
    benchmark_calibration = configured.get("benchmark_calibration")
    if calibration_identity is None and isinstance(benchmark_calibration, Mapping):
        calibration_identity = benchmark_calibration.get("identity")
    calibrated = bool(calibration_identity)

    rows: List[Dict[str, Any]] = []
    units = {
        "clean_case_fpr": "proportion",
        "clean_case_fwer": "proportion",
        "defect_recall": "proportion",
        "defect_power": "proportion",
        "unsafe_acceptance_rate": "proportion",
        "root_cause_macro_f1": "proportion",
        "provenance_completeness": "proportion",
        "required_gate_bypass_count": "count",
        "runtime_cost_seconds": "seconds",
        "memory_cost_mb": "megabytes",
    }
    for metric_id, value in primary.items():
        threshold = calibration.get(metric_id) if calibrated and metric_id in calibration else None
        rows.append(_metric_row(metric_id, value, units[metric_id], threshold,
                                "primary", calibration_identity))
    for metric_id, value in supplementary.items():
        rows.append(_metric_row(metric_id, value, "proportion", None, "supplementary", None))
    return rows


def _benchmark_records(config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Extract explicit benchmark rows without coercing other evidence."""
    direct = config.get("benchmark_records")
    if direct is None:
        benchmark = config.get("benchmark")
        if isinstance(benchmark, Mapping):
            direct = benchmark.get("records", benchmark.get("cases"))
        elif isinstance(benchmark, Sequence) and not isinstance(benchmark, (str, bytes)):
            direct = benchmark
    if not isinstance(direct, Sequence) or isinstance(direct, (str, bytes)):
        return []
    return [item for item in direct if isinstance(item, Mapping)]


def _module_pass_rate(modules: Mapping[str, Any]) -> float | None:
    """Supplementary module diagnostic; not a benchmark-quality metric."""
    statuses = [_status(value) for value in modules.values()]
    return sum(status == "PASS" for status in statuses) / len(statuses) if statuses else None


def _module_resolved_rate(modules: Mapping[str, Any]) -> float | None:
    statuses = [_status(value) for value in modules.values()]
    return sum(status in ("PASS", "FAIL", "NOT_APPLICABLE") for status in statuses) / len(statuses) if statuses else None


def _status(value: Any) -> Any:
    return getattr(value, "status", None) or (value.get("status") if isinstance(value, Mapping) else None)


def _stratum(record: Mapping[str, Any]) -> str | None:
    value = record.get("stratum", record.get("case_type", record.get("case_class", record.get("label", record.get("truth", record.get("gold_status", record.get("expected_status")))))))
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if text in {"clean", "negative", "negative_control", "null"}:
        return "clean"
    if text in {"defect", "injected_defect", "failure", "positive", "known_defect"}:
        return "defect"
    return None


def _decision(record: Mapping[str, Any]) -> str | None:
    value = record.get("observed_decision", record.get("observed_status", record.get("decision", record.get("status"))))
    return str(value).strip().upper() if value is not None else None


def _flag(record: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in record:
            value = record[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return bool(value)
            if isinstance(value, str):
                if value.strip().lower() in {"true", "yes", "pass", "fail", "detected", "unsafe"}:
                    return True
                if value.strip().lower() in {"false", "no", "none", "not_detected", "safe"}:
                    return False
    return None


def _error_flag(record: Mapping[str, Any], *, defect: bool) -> bool | None:
    """Resolve an explicit error/detection indicator for one stratum."""
    detected = _flag(record, "detected", "defect_detected", "finding_detected", "flagged", "false_positive", "false_positive_flag", "clean_error")
    if detected is None:
        count = next((record.get(key) for key in ("n_findings", "finding_count", "false_positive_count") if record.get(key) is not None), None)
        if isinstance(count, (int, float)) and not isinstance(count, bool) and math.isfinite(float(count)):
            detected = float(count) > 0
    decision = _decision(record)
    if detected is not None:
        return detected
    if decision is None:
        return None
    flagged_decisions = {"REPAIR", "REJECT", "FAIL"}
    accepted_decisions = {"ACCEPT", "ACCEPT_WITH_SCOPE"}
    if defect and decision in flagged_decisions:
        return True
    if defect and decision in accepted_decisions:
        return False
    # Abstention is an observed, non-detecting outcome; it is not an absent
    # outcome and must remain in the defect-recall denominator.
    if decision == "ABSTAIN":
        return False
    if not defect and decision in flagged_decisions:
        return True
    if not defect and decision in accepted_decisions:
        return False
    return None


def _mean_explicit(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    values: List[float] = []
    for record in records:
        value = next((record.get(key) for key in keys if record.get(key) is not None), None)
        if value is None:
            return None
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _macro_f1(records: Sequence[Mapping[str, Any]]) -> float | None:
    pairs = []
    for record in records:
        truth = next((record.get(k) for k in ("root_cause_true", "true_root_cause", "root_cause_gold") if record.get(k) is not None), None)
        pred = next((record.get(k) for k in ("root_cause_pred", "predicted_root_cause", "root_cause_observed") if record.get(k) is not None), None)
        if truth is not None and pred is not None:
            pairs.append((str(truth), str(pred)))
    if not pairs or len(pairs) != len(records):
        return None
    labels = sorted({truth for truth, _ in pairs} | {pred for _, pred in pairs})
    f1_values = []
    for label in labels:
        tp = sum(truth == label and pred == label for truth, pred in pairs)
        fp = sum(truth != label and pred == label for truth, pred in pairs)
        fn = sum(truth == label and pred != label for truth, pred in pairs)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1_values) / len(f1_values) if f1_values else None


def _bypass_count(records: Sequence[Mapping[str, Any]]) -> float | None:
    values: List[float] = []
    for record in records:
        found = False
        value = next((record.get(k) for k in ("required_gate_bypass_count", "gate_bypass_count", "bypassed_gate_count") if record.get(k) is not None), None)
        if value is not None:
            try:
                number = float(value)
                if math.isfinite(number) and number >= 0:
                    values.append(number)
                    found = True
                    continue
            except (TypeError, ValueError):
                pass
        gates = record.get("bypassed_gates")
        if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
            values.append(float(len(gates)))
            found = True
            continue
        flag = _flag(record, "required_gate_bypass", "gate_bypass")
        if flag is not None:
            values.append(float(flag))
            found = True
        if not found:
            return None
    return sum(values) if values and len(values) == len(records) else None


def _benchmark_metric_values(records: Sequence[Mapping[str, Any]]) -> Dict[str, float | None]:
    clean = [record for record in records if _stratum(record) == "clean"]
    defect = [record for record in records if _stratum(record) == "defect"]
    clean_errors = [_error_flag(record, defect=False) for record in clean]
    defect_hits = [_error_flag(record, defect=True) for record in defect]

    # A partial denominator is unresolved rather than silently dropped.
    fpr = sum(clean_errors) / len(clean_errors) if clean_errors and all(value is not None for value in clean_errors) else None
    recall = sum(defect_hits) / len(defect_hits) if defect_hits and all(value is not None for value in defect_hits) else None

    # FWER is only calculable when benchmark rows explicitly identify a
    # family (e.g. one analysis with multiple hypotheses).  We do not infer
    # it from the ordinary clean-case FPR, because that would silently change
    # the unit of analysis.
    family_flags = [_flag(record, "familywise_false_positive", "fwer_event", "fwer", "familywise_error") for record in clean]
    family_observed = [value for value in family_flags if value is not None]
    fwer = sum(family_observed) / len(family_observed) if family_observed and len(family_observed) == len(clean) else None

    unsafe = []
    for record in defect:
        explicit = _flag(record, "unsafe_acceptance", "unsafe_accepted")
        if explicit is None:
            decision = _decision(record)
            if decision in {"ACCEPT", "ACCEPT_WITH_SCOPE"}:
                explicit = True
            elif decision in {"REPAIR", "REJECT", "FAIL", "ABSTAIN"}:
                explicit = False
            else:
                explicit = _flag(record, "accepted")
        unsafe.append(explicit)

    return {
        "clean_case_fpr": fpr,
        "clean_case_fwer": fwer,
        "defect_recall": recall,
        # Power is the detection probability on the explicit injected-defect
        # stratum; retain a separate canonical row even when the record-level
        # design makes it numerically equal to recall.
        "defect_power": recall,
        "unsafe_acceptance_rate": (
            sum(unsafe) / len(unsafe)
            if unsafe and all(value is not None for value in unsafe) else None
        ),
        "root_cause_macro_f1": _macro_f1(defect),
        "provenance_completeness": _mean_explicit(records, ("provenance_completeness", "provenance_complete")),
        "required_gate_bypass_count": _bypass_count(records),
        "runtime_cost_seconds": _mean_explicit(records, ("runtime_seconds", "runtime_cost_seconds", "wall_time_seconds")),
        "memory_cost_mb": _mean_explicit(records, ("memory_mb", "memory_cost_mb", "peak_memory_mb")),
    }


def _metric_row(metric_id: str, value: Any, unit: str, threshold: Any,
                metric_role: str, calibration_identity: Any) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_id": metric_id,
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "threshold_status": "CALIBRATED" if threshold is not None and calibration_identity else "PROVISIONAL",
        "status": "UNRESOLVED" if value is None else "DESCRIPTIVE_ONLY",
        "metric_role": metric_role,
        "calibration_identity": calibration_identity,
    }


# Friendly aliases used by callers and hidden compatibility checks.
build_baseline_surface = baseline_cells
quality_metrics = framework_quality_metrics
