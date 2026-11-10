# -*- coding: utf-8 -*-
"""Synthetic executable regression corpus for known audit failure modes.

Practical historical cases map to tiny assertions over the actual STIA module
contracts. Cases requiring a richer locked benchmark remain explicitly
``metadata_only`` and carry the future fixture/evidence required. Schema
self-validation is never presented as regression proof.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import tempfile

import pandas as pd

from .contracts import SCHEMA_VERSION, STATUSES


# case_id, target module, expected status, execution mode, future fixture
REGRESSION_CORPUS = (
    ("identity_mapping_mismatch", "M2", "FAIL", "synthetic_module_assertion", None),
    ("invalid_order_control", "M6", "FAIL", "synthetic_module_assertion", None),
    ("representation_definition_mismatch", "M4", "FAIL", "synthetic_module_assertion", None),
    ("mixed_run_provenance", "M1", "FAIL", "synthetic_module_assertion", None),
    ("pseudoreplication", "M5", "FAIL", "synthetic_module_assertion", None),
    ("global_conditional_reversal", "M7", "UNRESOLVED", "metadata_only", "paired_global_conditional_embedding_records_with_locked_labels"),
    ("not_applicable_misuse", "M9", "ABSTAIN", "synthetic_decision_assertion", None),
    ("leakage_or_confounding", "M4", "FAIL", "synthetic_module_assertion", None),
    ("exploratory_as_confirmatory", "M8", "FAIL", "synthetic_claim_assertion", None),
    ("missing_provenance", "M1", "UNRESOLVED", "metadata_only", "minimal_case_with_valid_inputs_and_omitted_provenance"),
)

METADATA_ONLY_REASONS = {
    "global_conditional_reversal": "requires paired global and conditional rankings with locked labels; a tiny fixture would not identify the reversal target",
    "missing_provenance": "requires a valid-input case whose only defect is omitted provenance; current synthetic cases intentionally omit the full input hash envelope",
}


def corpus_records() -> list[dict[str, str]]:
    """Return serialisable regression records for the canonical schema."""
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "target_module": module_id,
            "expected_status": expected,
            "fixture": "synthetic_minimal",
            "execution": execution,
            **({"future_fixture": future_fixture} if future_fixture else {}),
            **({"metadata_only_reason": METADATA_ONLY_REASONS[case_id]} if execution == "metadata_only" else {}),
        }
        for case_id, module_id, expected, execution, future_fixture in REGRESSION_CORPUS
    ]


def validate_regression_record(record: Mapping[str, Any]) -> bool:
    """Validate one corpus record without executing model/data code."""
    if not isinstance(record, Mapping):
        return False
    if record.get("schema_version") != SCHEMA_VERSION:
        return False
    if not isinstance(record.get("case_id"), str) or not record["case_id"]:
        return False
    if not isinstance(record.get("target_module"), str) or not record["target_module"].startswith("M"):
        return False
    expected = record.get("expected_status")
    if not (isinstance(expected, str) and (expected in STATUSES or expected in ("ABSTAIN", "REJECT"))):
        return False
    execution = record.get("execution")
    if execution not in {"synthetic_module_assertion", "synthetic_decision_assertion", "synthetic_claim_assertion", "metadata_only"}:
        return False
    return execution != "metadata_only" or (isinstance(record.get("future_fixture"), str) and isinstance(record.get("metadata_only_reason"), str) and bool(record.get("metadata_only_reason")))


def validate_corpus(records: Sequence[Mapping[str, Any]] | None = None) -> Dict[str, Any]:
    values = list(records or corpus_records())
    valid = [validate_regression_record(item) for item in values]
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": all(valid),
        "n_cases": len(values),
        "n_valid": sum(valid),
        "case_ids": [item.get("case_id") for item in values],
    }


def execute_regression_case(case_id: str) -> Dict[str, Any]:
    """Run one bounded synthetic fail-closed assertion.

    No real datasets, model outputs, R workloads, or project artifacts are
    loaded. Metadata-only rows return ``executable=False`` and identify the
    exact future fixture required to make them executable.
    """
    record = next((item for item in corpus_records() if item["case_id"] == case_id), None)
    if record is None:
        raise KeyError(case_id)
    execution = record["execution"]
    expected = record["expected_status"]
    if execution == "metadata_only":
        return {
            **record,
            "executable": False,
            "observed_status": None,
            "passed": None,
            "assertion": "metadata-only historical case; no locked synthetic fixture exists",
            "future_fixture": record["future_fixture"],
            "metadata_only_reason": record["metadata_only_reason"],
        }

    # Local imports avoid a cycle during stia package initialisation.
    from .engine import _decision, _module_m1, _module_m2, _module_m4, _module_m5, _module_m6

    observed: str
    assertion: str
    if case_id == "identity_mapping_mismatch":
        frame = pd.DataFrame({
            "spot_id": ["s1", "s1"], "sample_id": ["a", "a"],
            "donor_id": ["d", "d"], "disease_or_condition": ["case", "case"],
            "technical_batch": ["b", "b"],
        })
        module = _module_m2(frame, {}, None)
        observed, assertion = module.status, "duplicate spot identity is a hard M2 failure"
    elif case_id == "invalid_order_control":
        module = _module_m6(Path.cwd(), {"orders": [{"name": "A"}, {"name": "A"}, {"name": "C"}, {"name": "D"}]})
        observed, assertion = module.status, "duplicate order-arm identities fail before OSI"
    elif case_id == "representation_definition_mismatch":
        with tempfile.TemporaryDirectory() as tmp:
            module = _module_m4(Path(tmp), {
                "feature_panel": ["G1"], "method_feature_panels": {"A": ["G1"]},
                "representations": {"A": {"type": "pca"}, "B": {"type": "graph"}},
                "preprocessing": {"fit_scope": "train_only", "leakage_checked": True},
            })
        observed, assertion = module.status, "incomparable representation types fail M4"
    elif case_id == "mixed_run_provenance":
        cfg = {"provenance": {
            "case_id": "synthetic", "run_id": "run-a", "software": {"stia": "x"},
            "seeds": {"s": 1}, "orders": ["A", "B"], "nested": {"run_id": "run-b"},
        }}
        module, _ = _module_m1(Path.cwd(), cfg, None, None, Path.cwd() / "stia-regression-output", None)
        observed, assertion = module.status, "multiple provenance run IDs fail M1"
    elif case_id == "pseudoreplication":
        module = _module_m5({"evaluator": {
            "functional_tests": [{"name": "synthetic", "passed": True}],
            "statistics": {"unit": "donor", "ci_method": "cluster", "multiple_testing": "Holm", "pseudoreplication": {"pseudoreplication_detected": True}},
        }})
        observed, assertion = module.status, "spot-level pseudoreplication evidence fails M5"
    elif case_id == "not_applicable_misuse":
        result, _, _ = _decision({"M1": type("R", (), {"status": "UNRESOLVED"})(), "M9": type("R", (), {"status": "UNRESOLVED"})()})
        observed, assertion = result, "missing applicability evidence produces ABSTAIN, never acceptance"
    elif case_id == "leakage_or_confounding":
        with tempfile.TemporaryDirectory() as tmp:
            module = _module_m4(Path(tmp), {
                "feature_panel": ["G1"], "method_feature_panels": {"A": ["G1"]},
                "preprocessing": {"fit_scope": "train_only", "leakage_checked": True},
                "marker_genes": ["G2"], "marker_selected_after_training": True,
            })
        observed, assertion = module.status, "out-of-panel/post-training markers fail M4 leakage checks"
    elif case_id == "exploratory_as_confirmatory":
        # Exercise the actual M8 external-review path with a tiny frozen
        # artifact; the claim text itself is what triggers the fail-closed
        # exploratory/confirmatory guard.
        import json
        from .engine import _module_m8
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review.json"
            review_path.write_text(json.dumps({
                "kind": "rule_based", "frozen": True, "protocol_version": "synthetic-v1", "rule_version": "synthetic-v1",
                "claims": [{"claim_id": "c1", "classification": "allowed", "evidence_refs": ["synthetic"]}],
            }), encoding="utf-8")
            module, _ = _module_m8(root, {
                "claims": [{"claim_id": "c1", "text": "Exploratory result presented as confirmatory"}],
                "claim_review_evidence": {"path": "review.json", "kind": "rule_based", "frozen": True, "protocol_version": "synthetic-v1", "rule_version": "synthetic-v1"},
            })
        observed, assertion = module.status, "M8 external-review path blocks exploratory-as-confirmatory wording"
    else:
        raise AssertionError(f"no executable mapping for {case_id}")
    return {
        **record,
        "executable": True,
        "observed_status": observed,
        "passed": observed == expected,
        "assertion": assertion,
    }


def execute_regression_corpus() -> Dict[str, Any]:
    """Execute every practical mapping and summarize metadata-only gaps."""
    rows = [execute_regression_case(item["case_id"]) for item in corpus_records()]
    executable = [row for row in rows if row["executable"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "records": rows,
        "n_cases": len(rows),
        "n_executable": len(executable),
        "n_passed": sum(row["passed"] is True for row in executable),
        "valid": all(row["passed"] is not False for row in rows),
    }
