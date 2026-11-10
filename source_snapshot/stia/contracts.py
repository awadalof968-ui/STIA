# -*- coding: utf-8 -*-
"""Versioned STIA contracts.

The contract is deliberately data-oriented: every module has one of the four
allowed statuses and missing evidence is represented as ``UNRESOLVED``.  The
engine never converts an unresolved check into a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


TOOL_NAME = "stia"
# T1R1 expands the canonical bundle and its benchmark contract.  Keep the
# predecessor identities immutable so consumers can distinguish old bundles.
TOOL_VERSION = "1.3.8"
SCHEMA_VERSION = "1.2.1"
STATUSES = ("PASS", "FAIL", "NOT_APPLICABLE", "UNRESOLVED")
DECISIONS = ("ACCEPT", "ACCEPT_WITH_SCOPE", "REPAIR", "REJECT", "ABSTAIN")

# The baseline surface is part of the executable contract.  It deliberately
# names comparison modes without assigning scientific pass/fail cut-offs;
# calibration of those quality metrics belongs to the locked benchmark phase.
BASELINE_SURFACE: List[Dict[str, str]] = [
    {"id": "B0", "name": "default_single_run_evaluation",
     "definition": "default single-run evaluation"},
    {"id": "B1", "name": "standard_scib_istbench_metric_suite",
     "definition": "standard scIB/iSTBench-style metric suite"},
    {"id": "B2", "name": "standard_robustness_seed_downsample_hyperparameter",
     "definition": "standard robustness (seed/downsample/hyperparameter)"},
    {"id": "B3", "name": "osi_only_audit",
     "definition": "OSI-only audit"},
    {"id": "B4", "name": "full_integrated_framework",
     "definition": "full integrated framework"},
]
# Backwards/forwards-friendly aliases for callers that refer to the rows as
# simply ``BASELINES`` or need the ordered identifiers.
BASELINES = BASELINE_SURFACE
BASELINE_IDS = tuple(item["id"] for item in BASELINE_SURFACE)

# Canonical files are always emitted for a complete bundle.  ``AuditRun.paths``
# retains the original five-file API for compatibility; ``all_paths`` exposes
# this complete set to new callers.
CANONICAL_BUNDLE_FILES = (
    "findings.json",
    "decision.json",
    "claim_review.json",
    "provenance.json",
    "run_manifest.json",
    "cells.json",
    "comparisons.json",
    "evidence_inventory.json",
    "repair_actions.json",
    "metrics.csv",
)

# Frozen descriptive framework-quality metrics.  Their values require
# explicit benchmark records; thresholds remain provisional until calibration
# identity is supplied.
PRIMARY_METRIC_IDS = (
    "clean_case_fpr",
    "clean_case_fwer",
    "defect_recall",
    "defect_power",
    "unsafe_acceptance_rate",
    "root_cause_macro_f1",
    "provenance_completeness",
    "required_gate_bypass_count",
    "runtime_cost_seconds",
    "memory_cost_mb",
)


MODULES: List[Dict[str, str]] = [
    {"id": "M1", "name": "manifest_provenance_and_input_integrity"},
    {"id": "M2", "name": "identity_unit_and_nesting"},
    {"id": "M3", "name": "estimand_identifiability_and_confounding"},
    {"id": "M4", "name": "feature_preprocessing_representation_fairness"},
    {"id": "M5", "name": "evaluator_and_statistical_validity"},
    {"id": "M6", "name": "robustness_order_and_component_invariance"},
    {"id": "M7", "name": "three_axis_conditional_and_cross_method_validity"},
    {"id": "M8", "name": "claim_governance"},
    {"id": "M9", "name": "repair_and_final_decision_safety"},
]


@dataclass
class ModuleResult:
    """Canonical serialisable result for one M1-M9 module."""

    module_id: str
    name: str
    status: str
    summary: str
    checks: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)
    repair_actions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"invalid module status: {self.status}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "checks": self.checks,
            "evidence": self.evidence,
            "unresolved": self.unresolved,
            "repair_actions": self.repair_actions,
        }


def module_info(module_id: str) -> Dict[str, str]:
    for item in MODULES:
        if item["id"] == module_id:
            return item
    raise KeyError(module_id)


def result(module_id: str, status: str, summary: str, **kwargs: Any) -> ModuleResult:
    item = module_info(module_id)
    return ModuleResult(module_id, item["name"], status, summary, **kwargs)
