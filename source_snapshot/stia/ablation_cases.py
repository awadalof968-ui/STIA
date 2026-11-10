# -*- coding: utf-8 -*-
"""Executable, truth-labelled cases for the complete STIA ablation benchmark.

Every fault's target status is obtained by running the actual STIA
module/decision implementation. Non-target modules are held PASS so that a
leave-one-module-out comparison has an identifiable interpretation. The clean
control is a contract control rather than a fitted biological dataset; callers
must retain that limitation in manuscript claims.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation_v2 import evaluate_complete_ablations
from .engine import _module_m3, _module_m7
from .regression import execute_regression_case


SCHEMA_VERSION = "stia.executable_ablation_cases.v1.20260909"
MODULE_IDS = tuple(f"M{i}" for i in range(1, 10))


def _pass_statuses() -> dict[str, str]:
    return {module: "PASS" for module in MODULE_IDS}


def _timed_regression(case_id: str) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    row = execute_regression_case(case_id)
    return row, time.perf_counter() - start


def _execute_m3_confounded_design() -> tuple[dict[str, Any], float]:
    metadata = pd.DataFrame({
        "spot_id": [f"s{i}" for i in range(8)],
        "sample_id": ["sa", "sa", "sb", "sb", "sc", "sc", "sd", "sd"],
        "donor_id": ["da", "da", "db", "db", "dc", "dc", "dd", "dd"],
        "disease_or_condition": ["case"] * 4 + ["control"] * 4,
        "technical_batch": ["batch_case"] * 4 + ["batch_control"] * 4,
    })
    start = time.perf_counter()
    module = _module_m3(metadata, {}, None)
    elapsed = time.perf_counter() - start
    return {
        "case_id": "confounded_design_no_common_support",
        "target_module": "M3",
        "expected_status": "FAIL",
        "observed_status": module.status,
        "passed": module.status == "FAIL",
        "assertion": "exact batch-condition confounding without a bridge is not identifiable",
        "module_result": module.as_dict(),
        "execution": "actual_module_call",
    }, elapsed


def _execute_m7_global_conditional_reversal() -> tuple[dict[str, Any], float]:
    rng = np.random.default_rng(20260909)
    n = 32
    metadata = pd.DataFrame({
        "spot_id": [f"s{i:03d}" for i in range(n)],
        "sample_id": [f"sample{i // 4:02d}" for i in range(n)],
        "donor_id": [f"donor{i // 8:02d}" for i in range(n)],
        "disease_or_condition": ["case" if i < n // 2 else "control" for i in range(n)],
        "technical_batch": [f"batch{i % 2}" for i in range(n)],
        "layer": [f"L{(i % 4) + 1}" for i in range(n)],
        "group": [f"G{i % 2}" for i in range(n)],
    })
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        np.savez_compressed(
            root / "embedding.npz",
            latent=rng.normal(size=(n, 6)).astype(np.float32),
            spot_ids=metadata["spot_id"].to_numpy(dtype=str),
        )
        cfg = {
            "embedding": {"path": "embedding.npz", "latent_key": "latent", "layer_column": "layer", "group_column": "group"},
            "seed": 20260909, "k_ilisi": 8, "k_graph": 4, "max_per_stratum": 32,
            "disease_axis_metrics": {"status": "PASS", "unit": "donor"},
            "global_conditional_reversal": {
                "reversed": True, "global_best": "method_A", "conditional_best": "method_B", "locked_before_audit": True,
            },
        }
        start = time.perf_counter()
        module = _module_m7(root, cfg, metadata, None)
        elapsed = time.perf_counter() - start
    return {
        "case_id": "locked_global_conditional_reversal",
        "target_module": "M7",
        "expected_status": "FAIL",
        "observed_status": module.status,
        "passed": module.status == "FAIL",
        "assertion": "a frozen global/conditional winner reversal is a hard M7 failure",
        "module_result": module.as_dict(),
        "execution": "actual_module_call",
    }, elapsed


def build_executable_ablation_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return evaluator cases plus their actual target-execution evidence."""
    mappings = {
        "M1": "mixed_run_provenance", "M2": "identity_mapping_mismatch",
        "M4": "representation_definition_mismatch", "M5": "pseudoreplication",
        "M6": "invalid_order_control", "M8": "exploratory_as_confirmatory",
        "M9": "not_applicable_misuse",
    }
    evidence: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = [{
        "case_id": "contract_clean_control", "truth": "clean", "true_fault_modules": [],
        "module_statuses": _pass_statuses(), "case_scope": "contract_control_not_biological_fit",
    }]
    for module in MODULE_IDS:
        if module == "M3":
            row, elapsed = _execute_m3_confounded_design()
        elif module == "M7":
            row, elapsed = _execute_m7_global_conditional_reversal()
        else:
            row, elapsed = _timed_regression(mappings[module])
        if row.get("passed") is not True:
            raise AssertionError(f"ablation fixture {row.get('case_id')} missed its frozen expectation: {row}")
        observed = str(row["observed_status"])
        target_status = "FAIL" if observed in {"FAIL", "REJECT"} else "UNRESOLVED" if observed == "ABSTAIN" else observed
        if target_status not in {"FAIL", "UNRESOLVED"}:
            raise AssertionError(f"unsupported target status {target_status} for {module}")
        statuses = _pass_statuses()
        statuses[module] = target_status
        cases.append({
            "case_id": str(row["case_id"]), "truth": "fault", "true_fault_modules": [module],
            "module_statuses": statuses,
            "module_runtime_seconds": {key: elapsed if key == module else 0.0 for key in MODULE_IDS},
            "case_scope": "controlled_fault_with_actual_target_module_execution",
        })
        evidence.append({**row, "runtime_seconds": elapsed, "evaluator_target_status": target_status})
    return cases, evidence


def run_executable_ablation_benchmark(*, bootstrap_draws: int = 4000, bootstrap_seed: int = 20260909) -> dict[str, Any]:
    cases, evidence = build_executable_ablation_cases()
    evaluation = evaluate_complete_ablations(cases, bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_CONTROLLED_EXECUTABLE_SCOPE",
        "execution_evidence": evidence,
        "cases": cases,
        "evaluation": evaluation,
        "limitations": [
            "The clean row is a contract control, not a biological dataset run.",
            "Non-target modules are held PASS to identify the marginal necessity of the target gate.",
            "Controlled faults support fault-detection claims only; real-data ablation remains separately required.",
        ],
    }


__all__ = ["SCHEMA_VERSION", "build_executable_ablation_cases", "run_executable_ablation_benchmark"]
