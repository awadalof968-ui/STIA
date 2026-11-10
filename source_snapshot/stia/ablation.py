# -*- coding: utf-8 -*-
"""Prospective, lightweight targeted-ablation contracts for STIA.

This module freezes the *shape* of a future framework ablation without running
models or treating missing benchmark records as outcomes.  Supplied records
are delegated to :func:`stia.baselines.framework_quality_metrics`; absent or
underpowered values remain explicitly ``UNRESOLVED``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

from .baselines import framework_quality_metrics
from .contracts import MODULES, PRIMARY_METRIC_IDS


PROTOCOL_SCHEMA_VERSION = "stia.targeted_ablation_protocol.v1.20260902"
RESULT_SCHEMA_VERSION = "stia.targeted_ablation_result.v1.20260902"
# Naming aliases mirror the existing cross-AI/publication modules.
ABLATION_PROTOCOL_VERSION = PROTOCOL_SCHEMA_VERSION
ABLATION_RESULT_VERSION = RESULT_SCHEMA_VERSION
ABLATION_METRIC_IDS = PRIMARY_METRIC_IDS


_MODULE_IDS = tuple(item["id"] for item in MODULES)
_REQUIRED_PROTOCOL_FIELDS = (
    "protocol_id",
    "schema_version",
    "status",
    "dataset_eligibility",
    "splits",
    "analysis_unit",
    "endpoints",
    "exclusions",
    "hypotheses",
    "calibration",
    "negative_controls",
    "robustness",
    "null_result_retention",
    "ablation_variants",
    "provenance",
)
_REQUIRED_NEGATIVE_CONTROL_IDS = {
    "NC-order-identity",
    "NC-disease-null",
    "NC-provenance-missing",
    "NC-forbidden-claim",
}
_REQUIRED_ROBUSTNESS_IDS = {
    "order_arms",
    "condition_reversal",
    "seed_repeat",
    "evidence_lineage",
}


def ablation_variants() -> list[dict[str, Any]]:
    """Return the frozen G1C variant inventory as independent dictionaries."""
    return deepcopy([
        {
            "id": "full_framework",
            "label": "full M1-M9 framework",
            "target": "reference",
            "disabled_modules": [],
        },
        {
            "id": "drop_order_audit",
            "label": "without order-audit gate",
            "target": "order_audit",
            "disabled_modules": ["M6"],
        },
        {
            "id": "drop_three_axis_condition_reversal",
            "label": "without three-axis/condition/reversal gate",
            "target": "three_axis_condition_reversal",
            "disabled_modules": ["M7"],
        },
        {
            "id": "drop_provenance_evidence",
            "label": "without provenance/evidence gates",
            "target": "provenance_evidence_gates",
            "disabled_modules": ["M1"],
        },
        {
            "id": "drop_claim_governance",
            "label": "without claim-governance/final-decision gates",
            "target": "claim_governance",
            "disabled_modules": ["M8", "M9"],
        },
    ])


def _error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def validate_ablation_protocol(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a prospective protocol without opening data or outcomes.

    The return value is a compact validation receipt.  Validation is strict on
    identity, split/endpoint declarations, controls and retention policy, but
    deliberately does not claim that any benchmark family is calibrated.
    """
    errors: list[str] = []
    if not isinstance(protocol, Mapping):
        return {"valid": False, "errors": ["protocol must be a mapping"]}

    for field in _REQUIRED_PROTOCOL_FIELDS:
        if field not in protocol:
            _error(errors, f"missing required field: {field}")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        _error(errors, f"schema_version must be {PROTOCOL_SCHEMA_VERSION}")
    if protocol.get("status") != "FROZEN":
        _error(errors, "status must be FROZEN before outcomes are opened")
    if protocol.get("outcomes_opened", False) is True:
        _error(errors, "prospective protocol cannot declare outcomes_opened=true")
    if not isinstance(protocol.get("protocol_id"), str) or not str(protocol.get("protocol_id")).strip():
        _error(errors, "protocol_id must be a non-empty string")

    eligibility = protocol.get("dataset_eligibility")
    if not isinstance(eligibility, Mapping):
        _error(errors, "dataset_eligibility must be a mapping")
    else:
        for field in ("source_roles", "outcome_independence_required", "minimum_donors_per_condition", "batch_crosswalk_required", "license_and_privacy_required"):
            if field not in eligibility:
                _error(errors, f"dataset_eligibility.{field} is required")
        if eligibility.get("outcome_independence_required") is not True:
            _error(errors, "dataset_eligibility.outcome_independence_required must be true")
        if eligibility.get("batch_crosswalk_required") is not True:
            _error(errors, "dataset_eligibility.batch_crosswalk_required must be true")
        if eligibility.get("license_and_privacy_required") is not True:
            _error(errors, "dataset_eligibility.license_and_privacy_required must be true")

    splits = protocol.get("splits")
    if not isinstance(splits, Mapping):
        _error(errors, "splits must be a mapping")
    else:
        for name in ("development", "calibration", "locked_test"):
            if not splits.get(name):
                _error(errors, f"splits.{name} must be declared")
        if splits.get("group_disjointness") is not True:
            _error(errors, "splits.group_disjointness must be true")
        if splits.get("locked_labels_sealed") is not True:
            _error(errors, "splits.locked_labels_sealed must be true")

    if protocol.get("analysis_unit") != "donor":
        _error(errors, "analysis_unit must be donor")

    endpoints = protocol.get("endpoints")
    endpoint_ids = []
    if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)):
        _error(errors, "endpoints must be a list")
    else:
        for endpoint in endpoints:
            endpoint_id = endpoint.get("id") if isinstance(endpoint, Mapping) else endpoint
            if endpoint_id is not None:
                endpoint_ids.append(str(endpoint_id))
        missing = sorted(set(PRIMARY_METRIC_IDS) - set(endpoint_ids))
        if missing:
            _error(errors, "endpoints omit required metrics: " + ", ".join(missing))

    for field in ("exclusions", "hypotheses", "negative_controls", "robustness"):
        value = protocol.get(field)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            _error(errors, f"{field} must be a non-empty list")

    hypotheses = protocol.get("hypotheses")
    if isinstance(hypotheses, Sequence) and not isinstance(hypotheses, (str, bytes)):
        for index, item in enumerate(hypotheses):
            if not isinstance(item, Mapping) or not item.get("id") or not item.get("statement") or not item.get("direction"):
                _error(errors, f"hypotheses[{index}] requires id, statement and direction")

    for field, required_ids in (("negative_controls", _REQUIRED_NEGATIVE_CONTROL_IDS), ("robustness", _REQUIRED_ROBUSTNESS_IDS)):
        value = protocol.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            present_ids = {str(item.get("id")) for item in value if isinstance(item, Mapping) and item.get("id") is not None}
            missing_ids = sorted(required_ids - present_ids)
            if missing_ids:
                _error(errors, f"{field} omit required controls: " + ", ".join(missing_ids))
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    _error(errors, f"{field}[{index}] must be a mapping")
                    continue
                if not item.get("id") or not item.get("target", item.get("axis")):
                    _error(errors, f"{field}[{index}] requires id and target/axis")
                if field == "negative_controls" and not item.get("expected_behavior"):
                    _error(errors, f"negative_controls[{index}].expected_behavior is required")
                if field == "robustness" and (not isinstance(item.get("fixed_fields"), Sequence) or isinstance(item.get("fixed_fields"), (str, bytes)) or not item.get("fixed_fields")):
                    _error(errors, f"robustness[{index}].fixed_fields must be a non-empty list")

    calibration = protocol.get("calibration")
    if not isinstance(calibration, Mapping):
        _error(errors, "calibration must be a mapping")
    else:
        if calibration.get("status") != "PROVISIONAL":
            _error(errors, "calibration.status must remain PROVISIONAL until frozen benchmark records exist")
        if not calibration.get("identity"):
            _error(errors, "calibration.identity is required")
        if calibration.get("thresholds_locked") is not True:
            _error(errors, "calibration.thresholds_locked must be true")
        if not calibration.get("selection_rule"):
            _error(errors, "calibration.selection_rule is required")

    retention = protocol.get("null_result_retention")
    if not isinstance(retention, Mapping):
        _error(errors, "null_result_retention must be a mapping")
    else:
        for field in ("required", "report_all_strata", "no_outcome_driven_exclusion"):
            if retention.get(field) is not True:
                _error(errors, f"null_result_retention.{field} must be true")
        if retention.get("unavailable_value") != "UNRESOLVED":
            _error(errors, "null_result_retention.unavailable_value must be UNRESOLVED")

    variants = protocol.get("ablation_variants")
    expected_variants = {item["id"]: item for item in ablation_variants()}
    seen: set[str] = set()
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        _error(errors, "ablation_variants must be a list")
    else:
        for item in variants:
            if not isinstance(item, Mapping) or not item.get("id"):
                _error(errors, "each ablation variant requires an id")
                continue
            variant_id = str(item["id"])
            if variant_id in seen:
                _error(errors, f"duplicate ablation variant: {variant_id}")
            seen.add(variant_id)
            if variant_id not in expected_variants:
                _error(errors, f"unknown ablation variant: {variant_id}")
                continue
            for field in ("target", "disabled_modules"):
                if field not in item:
                    _error(errors, f"ablation variant {variant_id} missing {field}")
            disabled = item.get("disabled_modules")
            if not isinstance(disabled, Sequence) or isinstance(disabled, (str, bytes)) or any(str(module) not in _MODULE_IDS for module in disabled):
                _error(errors, f"ablation variant {variant_id} has invalid disabled_modules")
        missing_variants = sorted(set(expected_variants) - seen)
        if missing_variants:
            _error(errors, "ablation_variants omit required variants: " + ", ".join(missing_variants))

    provenance = protocol.get("provenance")
    if not isinstance(provenance, Mapping):
        _error(errors, "provenance must be a mapping")
    else:
        required_nodes = provenance.get("required_fields")
        if not isinstance(required_nodes, Sequence) or isinstance(required_nodes, (str, bytes)):
            _error(errors, "provenance.required_fields must be a list")
        else:
            for field in ("case_id", "config_hash", "environment_identity", "seed_identity", "output_manifest_hash"):
                if field not in required_nodes:
                    _error(errors, f"provenance.required_fields omits {field}")
        if provenance.get("immutable") is not True:
            _error(errors, "provenance.immutable must be true")

    return {
        "valid": not errors,
        "errors": errors,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": protocol.get("protocol_id"),
        "outcomes_opened": bool(protocol.get("outcomes_opened", False)),
        "required_metrics": list(PRIMARY_METRIC_IDS),
    }


def score_targeted_ablations(
    protocol: Mapping[str, Any],
    records_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Score explicit benchmark records for each frozen ablation variant.

    This is deliberately a records-only adapter: it does not call ``audit``,
    fit a model, infer missing variant outcomes, or promote a synthetic row to
    validation.  A missing variant receives all required metric rows with
    ``status=UNRESOLVED``.
    """
    checked = validate_ablation_protocol(protocol)
    if not checked["valid"]:
        raise ValueError("invalid ablation protocol: " + "; ".join(checked["errors"]))
    if not isinstance(records_by_variant, Mapping):
        raise TypeError("records_by_variant must be a mapping keyed by ablation variant id")

    variants = {item["id"]: item for item in ablation_variants()}
    unknown = sorted(set(str(key) for key in records_by_variant) - set(variants))
    if unknown:
        raise ValueError("records supplied for unknown ablation variant(s): " + ", ".join(unknown))

    results: list[dict[str, Any]] = []
    for variant_id, variant in variants.items():
        supplied = records_by_variant.get(variant_id)
        if supplied is None:
            records: list[Mapping[str, Any]] = []
            records_status = "UNRESOLVED_MISSING_RECORDS"
        else:
            if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
                raise TypeError(f"records for {variant_id} must be a list of mappings")
            records = []
            for index, record in enumerate(supplied):
                if not isinstance(record, Mapping):
                    raise TypeError(f"records for {variant_id}[{index}] must be a mapping")
                declared_variant = record.get("ablation_id")
                if declared_variant is not None and str(declared_variant) != variant_id:
                    raise ValueError(f"record {variant_id}[{index}] declares ablation_id={declared_variant!r}")
                records.append(dict(record))
            records_status = "DESCRIPTIVE_ONLY" if records else "UNRESOLVED_MISSING_RECORDS"

        metrics = framework_quality_metrics({}, benchmark_records=records)
        primary = [row for row in metrics if row.get("metric_id") in PRIMARY_METRIC_IDS]
        results.append({
            "ablation_id": variant_id,
            "target": variant["target"],
            "disabled_modules": list(variant["disabled_modules"]),
            "records_supplied": bool(records),
            "record_count": len(records),
            "status": records_status,
            "metrics": primary,
            "threshold_status": "PROVISIONAL",
        })

    overall = "DESCRIPTIVE_ONLY" if all(item["records_supplied"] for item in results) else "UNRESOLVED"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "status": overall,
        "variants": results,
        "required_metrics": list(PRIMARY_METRIC_IDS),
        "outcome_inference": False,
        "claim_boundary": "Explicit records are descriptive development evidence; missing or underpowered values remain UNRESOLVED and do not establish framework validation.",
    }


# Friendly aliases for callers that prefer an execution-oriented name.
run_targeted_ablations = score_targeted_ablations
validate_protocol = validate_ablation_protocol


__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "ABLATION_PROTOCOL_VERSION",
    "ABLATION_RESULT_VERSION",
    "ABLATION_METRIC_IDS",
    "ablation_variants",
    "validate_ablation_protocol",
    "validate_protocol",
    "score_targeted_ablations",
    "run_targeted_ablations",
]
