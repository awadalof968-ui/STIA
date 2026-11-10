# -*- coding: utf-8 -*-
"""Build the publication-facing six-method evidence registry.

Only explicitly named, terminal acceptance documents are eligible inputs.
Live result directories, timestamps and producer ``COMPLETED`` markers are not
promotion evidence.  The writer is additive and refuses to reuse an output
directory containing either canonical artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


SCHEMA_VERSION = "stia.publication_evidence_registry.v1.20260902"
METHODS = ("PCA", "GraphST", "Harmony", "INSPIRE", "STAligner", "PRECAST")
INTERNAL_ACCEPTANCE = "docs/SIX_METHOD_ORDER_COVERAGE_ACCEPTANCE_20260831_r1.json"
EXTERNAL_ACCEPTANCES = {
    "PCA": "docs/PCA_EXTERNAL_GSE254652_FINAL_ACCEPTANCE_20260831_r1.json",
    "GraphST": "docs/GRAPHST_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "Harmony": "docs/HARMONY_GSE254652_FIVE_SEED_FINAL_ACCEPTANCE_20260901_r1.json",
    "INSPIRE": "docs/INSPIRE_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "STAligner": "docs/STALIGNER_EXTERNAL_GSE254652_FINAL_ACCEPTANCE_20260831_r1.json",
    "PRECAST": "docs/PRECAST_GSE254652_SEED20260819_TERMINAL_ACCEPTANCE_20260901_r1.json",
}


class RegistryError(ValueError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RegistryError(f"required acceptance document is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RegistryError(f"invalid JSON acceptance document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"acceptance document must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_values(method: str, record: Mapping[str, Any]) -> list[Any]:
    identity = record.get("experiment_identity", {})
    if method == "PCA":
        return ["DETERMINISTIC_SINGLE_FIT"]
    seeds = identity.get("seeds", identity.get("model_seeds", identity.get("requested_seeds")))
    if seeds is None and identity.get("seed") is not None:
        seeds = [identity["seed"]]
    return list(seeds or [])


def _osi_values(method: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    source = record.get("osi_results", record.get("positive_negative_uncertain_results", {}))
    if method == "PCA":
        value = source.get("osi")
        return {"overall": value} if value is not None else {}
    if method in {"GraphST", "Harmony"}:
        return {str(key): value for key, value in (source.get("by_seed") or {}).items()}
    if method == "INSPIRE":
        return {str(key): value for key, value in (source.get("by_seed") or {}).items()}
    value = source.get("osi")
    return {str(_seed_values(method, record)[0]): value} if value is not None and _seed_values(method, record) else {}


def _numeric_classification(method: str, record: Mapping[str, Any]) -> str:
    source = record.get("osi_results", {})
    if method == "PCA":
        return "ORDER_INVARIANT_WITHIN_MARGIN"
    if method in {"GraphST", "Harmony"}:
        threshold = float(source.get("operational_threshold", 0.05))
        values = list((source.get("by_seed") or {}).values())
        return "ORDER_INVARIANT_WITHIN_MARGIN" if values and all(float(value) <= threshold for value in values) else "ORDER_SENSITIVE_OR_UNRESOLVED"
    if method == "INSPIRE":
        return "COMPONENT_MIXED_LATENT_SHARED_FAIL_GAMMA_PASS_FULL_4_OF_5_PASS"
    return str(source.get("numeric_classification", record.get("numeric_acceptance", "UNRESOLVED")))


def _planned_seed_target(method: str) -> Any:
    return "DETERMINISTIC_NOT_APPLICABLE" if method == "PCA" else 5


def _coverage_status(method: str, seeds: list[Any]) -> str:
    if method == "PCA":
        return "COMPLETE_DETERMINISTIC"
    return "COMPLETE_5_OF_5" if len(seeds) == 5 else f"ACCEPTED_{len(seeds)}_OF_5_PENDING_TERMINAL_ACCEPTANCE"


def build_registry(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    internal_path = root / INTERNAL_ACCEPTANCE
    internal = _read_json(internal_path)
    if internal.get("decision") != "PASS" or set((internal.get("methods") or {}).keys()) != set(METHODS):
        raise RegistryError("internal six-method acceptance is not a complete PASS matrix")

    sources = {
        "internal": {"path": INTERNAL_ACCEPTANCE, "sha256": _sha256(internal_path)},
        "external": {},
    }
    rows = []
    for method in METHODS:
        relative = EXTERNAL_ACCEPTANCES[method]
        path = root / relative
        external = _read_json(path)
        external_method = str((external.get("experiment_identity") or {}).get("method", method))
        if external_method.lower() != method.lower():
            raise RegistryError(f"{relative} method={external_method!r}; expected {method!r}")
        decision = str(external.get("decision", external.get("acceptance_decision", "UNRESOLVED")))
        if decision not in {"PASS", "CONDITIONAL_PASS"}:
            raise RegistryError(f"{relative} is not a terminal accepted evidence document: {decision}")
        seeds = _seed_values(method, external)
        sources["external"][method] = {"path": relative, "sha256": _sha256(path)}
        internal_method = internal["methods"][method]
        rows.append({
            "method": method,
            "dataset": str((external.get("experiment_identity") or {}).get("dataset", "GSE254652_P1")),
            "internal_matrix_decision": internal["decision"],
            "internal_applicable_cells": internal_method.get("applicable_cells"),
            "internal_execution_valid_cells": internal_method.get("execution_valid_cells"),
            "internal_numeric_classification_counts": internal_method.get("numeric_classification_counts"),
            "external_acceptance_decision": decision,
            "external_execution_status": str(external.get("execution_acceptance", (external.get("completion_checks") or {}).get("execution_validity", "PASS"))),
            "external_numeric_classification": _numeric_classification(method, external),
            "accepted_seed_ids": seeds,
            "accepted_seed_count": len(seeds),
            "planned_seed_target": _planned_seed_target(method),
            "seed_coverage_status": _coverage_status(method, seeds),
            "osi_values": _osi_values(method, external),
            "operational_threshold": (external.get("osi_results") or {}).get("operational_threshold", (external.get("statistical_audit") or {}).get("osi_margin", 0.05)),
            "threshold_semantics": "PROVISIONAL_OPERATIONAL_NOT_P_VALUE",
            "external_blinding": "PROJECT_EXPOSED_NOT_BLIND_HELD_OUT",
            "allowed_claims": external.get("allowed_manuscript_claims", []),
            "forbidden_claims": external.get("forbidden_manuscript_claims", []),
            "residual_limitations": external.get("residual_limitations", external.get("blocking_issues", [])),
            "source_acceptance": relative,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "promotion_rule": "only explicit terminal acceptance documents; live outputs and timestamps are excluded",
        "perturbation_contract": {
            "axis": "SAMPLE_SLICE_BATCH_ORDER",
            "unit": "slice",
            "semantics": "NUISANCE_PERMUTATION",
            "fixed_fields": ["observations", "entity_identities", "hyperparameters", "random_seed", "intended_analysis"],
        },
        "sources": sources,
        "methods": rows,
        "summary": {
            "methods": len(rows),
            "internal_complete_methods": sum(row["internal_execution_valid_cells"] is not None for row in rows),
            "external_five_seed_complete": [row["method"] for row in rows if row["seed_coverage_status"] == "COMPLETE_5_OF_5"],
            "external_deterministic_complete": [row["method"] for row in rows if row["seed_coverage_status"] == "COMPLETE_DETERMINISTIC"],
            "external_pending_terminal_seed_acceptance": [row["method"] for row in rows if "PENDING" in row["seed_coverage_status"]],
            "whole_framework_externally_confirmed": False,
        },
        "claim_boundary": [
            "Internal matrix coverage does not establish external validity or framework accuracy.",
            "GSE254652 is project-exposed external reproduction, not blind held-out confirmation.",
            "Seed repetitions are computational replicates, not biological replicates.",
            "The 0.05 OSI boundary is operational and must not be reported as a p-value.",
            "No live STAligner or PRECAST continuation is promoted until a separate terminal acceptance exists.",
        ],
    }


def write_registry(payload: Mapping[str, Any], output_dir: Path) -> Dict[str, Path]:
    out = Path(output_dir)
    json_path = out / "publication_evidence_registry.json"
    csv_path = out / "publication_evidence_registry.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"refusing to overwrite canonical registry artifacts in {out}")
    out.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    fields = [
        "method", "dataset", "internal_matrix_decision", "internal_applicable_cells",
        "internal_execution_valid_cells", "external_acceptance_decision", "external_execution_status",
        "external_numeric_classification", "accepted_seed_count", "planned_seed_target",
        "seed_coverage_status", "operational_threshold", "threshold_semantics", "external_blinding",
        "source_acceptance",
    ]
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["methods"])
    return {"json": json_path, "csv": csv_path}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build terminal-acceptance-only six-method publication registry")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_registry(Path(args.project_root))
        paths = write_registry(payload, Path(args.output_dir))
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "PASS", "methods": len(payload["methods"]), "summary": payload["summary"], "outputs": {key: str(value) for key, value in paths.items()}}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
