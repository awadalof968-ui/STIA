# -*- coding: utf-8 -*-
"""Complete M1-M9 and targeted-interaction ablation evaluator."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import MODULES


SCHEMA_VERSION = "stia.complete_ablation.v2.20260909"
MODULE_IDS = tuple(item["id"] for item in MODULES)
SAFE_DECISIONS = {"ACCEPT", "ACCEPT_WITH_SCOPE"}


def complete_ablation_variants() -> list[dict[str, Any]]:
    variants = [{"id": "full_framework", "disabled_modules": [], "family": "reference"}]
    variants.extend(
        {"id": f"drop_{module}", "disabled_modules": [module], "family": "leave_one_module_out"}
        for module in MODULE_IDS
    )
    variants.extend([
        {"id": "drop_M1_M6", "disabled_modules": ["M1", "M6"], "family": "targeted_interaction",
         "interaction": "provenance_by_order"},
        {"id": "drop_M7_M8", "disabled_modules": ["M7", "M8"], "family": "targeted_interaction",
         "interaction": "three_axis_by_claim_governance"},
    ])
    return variants


def _validate_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ValueError("cases must be a non-empty sequence")
    output = []
    seen = set()
    parent_identity_present = []
    for index, source in enumerate(cases):
        if not isinstance(source, Mapping):
            raise TypeError(f"case {index} is not a mapping")
        case = dict(source)
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("case_id must be unique and non-empty")
        seen.add(case_id)
        parent_id = case.get("parent_id")
        if parent_id is not None and (not isinstance(parent_id, str) or not parent_id.strip()):
            raise ValueError(f"case {case_id} parent_id must be a non-empty string")
        parent_identity_present.append(parent_id is not None)
        truth = case.get("truth")
        if truth not in {"clean", "fault"}:
            raise ValueError(f"case {case_id} truth must be clean or fault")
        statuses = case.get("module_statuses")
        if not isinstance(statuses, Mapping) or set(statuses) != set(MODULE_IDS):
            raise ValueError(f"case {case_id} must contain exactly M1-M9 statuses")
        if any(status not in {"PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"} for status in statuses.values()):
            raise ValueError(f"case {case_id} contains invalid module status")
        true_modules = tuple(str(value) for value in case.get("true_fault_modules", []))
        if truth == "clean" and true_modules:
            raise ValueError(f"clean case {case_id} cannot declare fault modules")
        if truth == "fault" and (not true_modules or any(value not in MODULE_IDS for value in true_modules)):
            raise ValueError(f"fault case {case_id} requires valid true_fault_modules")
        case["true_fault_modules"] = true_modules
        case["module_statuses"] = {str(key): str(value) for key, value in statuses.items()}
        output.append(case)
    if any(parent_identity_present) and not all(parent_identity_present):
        raise ValueError("parent_id must be supplied for every case or no cases")
    return output


def _decision(statuses: Mapping[str, str], disabled: set[str]) -> str:
    enabled = [statuses[module] for module in MODULE_IDS if module not in disabled]
    if any(status == "FAIL" for status in enabled):
        return "REJECT"
    if any(status == "UNRESOLVED" for status in enabled):
        return "ABSTAIN"
    return "ACCEPT"


def _macro_root_f1(rows: Sequence[dict[str, Any]]) -> float | None:
    scores = []
    for module in MODULE_IDS:
        truth = np.asarray([module in row["true_modules"] for row in rows], dtype=bool)
        prediction = np.asarray([module in row["predicted_modules"] for row in rows], dtype=bool)
        if not truth.any():
            continue
        tp = int(np.sum(truth & prediction)); fp = int(np.sum(~truth & prediction)); fn = int(np.sum(truth & ~prediction))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else None


def _rate_interval(values: np.ndarray, rng: np.random.Generator, draws: int,
                   cluster_ids: Sequence[str] | None = None) -> dict[str, Any]:
    if values.size == 0:
        return {"value": None, "ci95": None, "n": 0, "n_clusters": 0,
                "status": "UNRESOLVED"}
    if cluster_ids is not None:
        if len(cluster_ids) != values.size:
            raise ValueError("cluster IDs and rate values must have identical length")
        unique, inverse = np.unique(np.asarray(cluster_ids, dtype=str), return_inverse=True)
        n_clusters = int(unique.size)
        if n_clusters < 2:
            return {"value": float(values.mean()), "ci95": None, "n": int(values.size),
                    "n_clusters": n_clusters, "status": "DESCRIPTIVE_INSUFFICIENT_CLUSTERS"}
        cluster_sum = np.bincount(inverse, weights=values, minlength=n_clusters)
        cluster_n = np.bincount(inverse, minlength=n_clusters)
        sampled = rng.integers(0, n_clusters, size=(draws, n_clusters))
        samples = cluster_sum[sampled].sum(axis=1) / cluster_n[sampled].sum(axis=1)
    else:
        n_clusters = None
        samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {"value": float(values.mean()),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "n": int(values.size), "n_clusters": n_clusters,
            "status": "DESCRIPTIVE_WITH_PARENT_CLUSTER_BOOTSTRAP" if cluster_ids is not None
            else "DESCRIPTIVE_WITH_CASE_BOOTSTRAP"}


def evaluate_complete_ablations(
    cases: Sequence[Mapping[str, Any]], *, variants: Sequence[Mapping[str, Any]] | None = None,
    bootstrap_draws: int = 4000, bootstrap_seed: int = 20260909,
) -> dict[str, Any]:
    """Evaluate frozen full/drop-module decisions without compensatory averaging."""
    records = _validate_cases(cases)
    clustered = "parent_id" in records[0]
    selected = [dict(item) for item in (variants if variants is not None else complete_ablation_variants())]
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    variant_ids = [str(item.get("id", "")) for item in selected]
    if len(variant_ids) != len(set(variant_ids)) or any(not value for value in variant_ids):
        raise ValueError("variant ids must be unique and non-empty")
    for item in selected:
        disabled = item.get("disabled_modules")
        if not isinstance(disabled, Sequence) or isinstance(disabled, (str, bytes)) or any(value not in MODULE_IDS for value in disabled):
            raise ValueError(f"invalid disabled_modules for {item.get('id')}")
    rng = np.random.default_rng(int(bootstrap_seed))
    full_decisions = {case["case_id"]: _decision(case["module_statuses"], set()) for case in records}
    results = []
    for variant in selected:
        disabled = set(variant["disabled_modules"])
        rows = []
        for case in records:
            decision = _decision(case["module_statuses"], disabled)
            predicted = tuple(module for module in MODULE_IDS if module not in disabled and case["module_statuses"][module] == "FAIL")
            rows.append({"case_id": case["case_id"], "truth": case["truth"],
                         "true_modules": case["true_fault_modules"], "predicted_modules": predicted,
                         "decision": decision, "detected": decision == "REJECT",
                         "unsafe_acceptance": case["truth"] == "fault" and decision in SAFE_DECISIONS,
                         "decision_changed_from_full": decision != full_decisions[case["case_id"]],
                         **({"parent_id": case["parent_id"]} if clustered else {})})
        clean = [row for row in rows if row["truth"] == "clean"]
        faults = [row for row in rows if row["truth"] == "fault"]
        fpr = np.asarray([row["detected"] for row in clean], dtype=float)
        recall = np.asarray([row["detected"] for row in faults], dtype=float)
        unsafe = np.asarray([row["unsafe_acceptance"] for row in faults], dtype=float)
        abstain = np.asarray([row["decision"] == "ABSTAIN" for row in rows], dtype=float)
        clean_parents = [row["parent_id"] for row in clean] if clustered else None
        fault_parents = [row["parent_id"] for row in faults] if clustered else None
        all_parents = [row["parent_id"] for row in rows] if clustered else None
        runtime_maps = [case.get("module_runtime_seconds") for case in records]
        costs = []
        if all(isinstance(value, Mapping) and set(value) >= set(MODULE_IDS) for value in runtime_maps):
            costs = [sum(float(value[module]) for module in MODULE_IDS if module not in disabled) for value in runtime_maps]
        results.append({
            "variant_id": variant["id"], "family": variant.get("family"),
            "disabled_modules": sorted(disabled), "case_rows": rows,
            "metrics": {
                "clean_case_fpr": _rate_interval(fpr, rng, bootstrap_draws, clean_parents),
                "defect_recall": _rate_interval(recall, rng, bootstrap_draws, fault_parents),
                "unsafe_acceptance_rate": _rate_interval(unsafe, rng, bootstrap_draws, fault_parents),
                "abstention_rate": _rate_interval(abstain, rng, bootstrap_draws, all_parents),
                "root_cause_macro_f1": {"value": _macro_root_f1(rows), "status": "DESCRIPTIVE"},
                "decision_changes_from_full": {"value": int(sum(row["decision_changed_from_full"] for row in rows)), "status": "DESCRIPTIVE"},
                "mean_runtime_seconds": {"value": float(np.mean(costs)) if costs else None,
                                         "status": "DESCRIPTIVE" if costs else "UNRESOLVED"},
            },
        })
    return {"schema_version": SCHEMA_VERSION, "status": "COMPLETE_FROZEN_CASE_SCOPE",
            "case_count": len(records), "clean_cases": sum(case["truth"] == "clean" for case in records),
            "fault_cases": sum(case["truth"] == "fault" for case in records), "variants": results,
            "bootstrap": {"draws": bootstrap_draws, "seed": bootstrap_seed,
                          "unit": "parent_id" if clustered else "benchmark case"},
            "claim_boundary": "Truth-labelled injected/development cases estimate controlled fault detection; they do not establish real-world prevalence or clinical accuracy."}


__all__ = ["SCHEMA_VERSION", "complete_ablation_variants", "evaluate_complete_ablations"]
