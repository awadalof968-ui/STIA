"""G3A publication integration for the 2026-09-02 gate.

This module is intentionally additive.  It reuses the frozen figure plan and
the existing F2A failure inventory, but reads the G2B STAligner receipt and
the G2C records-only result explicitly.  Only terminally accepted numeric
rows are promoted to the publication tables; unresolved STAligner seed 20260820
is retained as an unresolved audit row and is never plotted.  G2C remains
development-only evidence and is carried in separate tables/figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


METHODS = ("PCA", "GraphST", "Harmony", "INSPIRE", "STAligner", "PRECAST")
TERMINAL_DECISIONS = {"PASS", "CONDITIONAL_PASS"}
DEFAULT_TARGET_SEEDS = [20260819, 20260820, 20260821, 20260822, 20260823]
SCHEMA_VERSION = "stia.g3a_publication_integration.v1.20260902"
REGISTRY_SCHEMA_VERSION = "stia.publication_evidence_registry.v1.20260902"
FIGURE_DATA_SCHEMA_VERSION = "stia.f2a_figure_data.v1.20260902"

INTERNAL_ACCEPTANCE = "docs/SIX_METHOD_ORDER_COVERAGE_ACCEPTANCE_20260831_r1.json"
FIGURE_PLAN = "docs/STIA_PAPER_FIGURE_PLAN_20260821.json"
F1_PACK = "results/framework_evaluation/f1a_baseline_metrics_20260902_r3/f1a_baseline_metric_pack_20260902.json"
EXTERNAL_ACCEPTANCES = {
    "PCA": "docs/PCA_EXTERNAL_GSE254652_FINAL_ACCEPTANCE_20260831_r1.json",
    "GraphST": "docs/GRAPHST_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "Harmony": "docs/HARMONY_GSE254652_FIVE_SEED_FINAL_ACCEPTANCE_20260901_r1.json",
    "INSPIRE": "docs/INSPIRE_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "STAligner": "docs/STALIGNER_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260902_r1.json",
    "PRECAST": "docs/PRECAST_GSE254652_SEED20260819_TERMINAL_ACCEPTANCE_20260901_r1.json",
}
G2B_AUDIT = "Lunacy/runs/publication-gates-20260902/evidence/G2/STALIGNER_G2B_RAW_RECOMPUTE_20260902_r2.json"
G2B_REGISTRY = "Lunacy/runs/publication-gates-20260902/evidence/G2/EXTERNAL_METHOD_REGISTRY_20260902_r1.json"
G2C_PROTOCOL = "docs/STIA_TARGETED_ABLATION_PROTOCOL_20260902.json"
G2C_FIXTURE = "Lunacy/runs/publication-gates-20260902/evidence/G2C/g2c_truth_labelled_development_fixture_20260902.json"
G2C_RESULT = "Lunacy/runs/publication-gates-20260902/evidence/G2C/g2c_records_only_result_20260902.json"


class G3AError(ValueError):
    """Raised when a publication source is missing, malformed or ineligible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(root: Path, relative: str, *, required: bool = True) -> Dict[str, Any]:
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        if required:
            raise G3AError(f"required source is missing: {relative}")
        return {"path": str(relative).replace("\\", "/"), "exists": False,
                "bytes": None, "sha256": None}
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()
    return {"path": rel, "exists": True, "bytes": path.stat().st_size,
            "sha256": _sha256(path)}


def _read(root: Path, relative: str, *, required: bool = True) -> Mapping[str, Any]:
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        if required:
            raise G3AError(f"required JSON source is missing: {relative}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - parser wording varies
        raise G3AError(f"invalid JSON source {relative}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise G3AError(f"JSON source must be an object: {relative}")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _decision(doc: Mapping[str, Any]) -> str:
    return str(doc.get("decision", doc.get("acceptance_decision", "UNRESOLVED"))).strip().upper()


def _identity(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    value = doc.get("experiment_identity", {})
    return value if isinstance(value, Mapping) else {}


def _seed_ids(method: str, doc: Mapping[str, Any]) -> list[Any]:
    identity = _identity(doc)
    if method == "PCA":
        return ["DETERMINISTIC_SINGLE_FIT"]
    for key in ("seeds", "accepted_seed_ids", "model_seeds"):
        value = identity.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
    if identity.get("seed") is not None:
        return [identity["seed"]]
    return []


def _target_seed_ids(method: str, doc: Mapping[str, Any], accepted: Sequence[Any]) -> list[Any]:
    identity = _identity(doc)
    for key in ("target_seeds", "requested_seeds"):
        value = identity.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
    if method == "PCA":
        return ["DETERMINISTIC_SINGLE_FIT"]
    # The frozen external program targets five computational seeds even when
    # a one-seed formal receipt (currently PRECAST) does not repeat that field.
    return list(DEFAULT_TARGET_SEEDS)


def _osi_values(method: str, doc: Mapping[str, Any], accepted: Sequence[Any]) -> Dict[str, Any]:
    osi = doc.get("osi_results", {})
    if not isinstance(osi, Mapping):
        osi = {}
    accepted_keys = {str(seed) for seed in accepted}
    by_seed = osi.get("by_seed")
    if isinstance(by_seed, Mapping):
        return {str(seed): value for seed, value in by_seed.items()
                if str(seed) in accepted_keys and (
                    _finite(value) or isinstance(value, Mapping))}
    value = osi.get("osi")
    if _finite(value) and accepted:
        return {str(accepted[0]): value}
    positive = doc.get("positive_negative_uncertain_results", {})
    if isinstance(positive, Mapping) and _finite(positive.get("osi")) and accepted:
        return {str(accepted[0]): positive["osi"]}
    statistical = doc.get("statistical_audit", {})
    if method in {"PCA", "STAligner"} and isinstance(statistical, Mapping) and _finite(statistical.get("osi")):
        return {str(accepted[0] if accepted else "UNSPECIFIED"): statistical["osi"]}
    return {}


def _coverage_status(method: str, accepted: Sequence[Any], target: Sequence[Any], unresolved: Sequence[Any]) -> str:
    if method == "PCA":
        return "COMPLETE_DETERMINISTIC"
    if unresolved:
        return f"CONDITIONAL_{len(accepted)}_OF_{len(target)}_TERMINALITY_UNRESOLVED"
    if len(accepted) == len(target) == 5:
        return "COMPLETE_5_OF_5"
    return f"ACCEPTED_{len(accepted)}_OF_{len(target)}_PENDING_TERMINAL_ACCEPTANCE"


def _numeric_classification(method: str, doc: Mapping[str, Any]) -> str:
    osi = doc.get("osi_results", {})
    if not isinstance(osi, Mapping):
        osi = {}
    value = osi.get("numeric_classification")
    if value is not None:
        return str(value)
    if method == "PCA":
        return "ORDER_INVARIANT_WITHIN_MARGIN"
    return "UNRESOLVED"


def _execution_status(method: str, doc: Mapping[str, Any]) -> str:
    checks = doc.get("completion_checks", {})
    if isinstance(checks, Mapping):
        for key in ("terminal_execution_validity", "execution_validity", "producer_exit_receipts"):
            if checks.get(key) is not None:
                return str(checks[key])
    value = doc.get("execution_acceptance")
    return str(value) if value is not None else "PASS"


def _development_rows(root: Path, result: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for variant in result.get("scored_variants", []):
        if not isinstance(variant, Mapping):
            continue
        variant_id = str(variant.get("ablation_id", "UNSPECIFIED"))
        target = variant.get("target")
        for metric in variant.get("metrics", []):
            if not isinstance(metric, Mapping):
                continue
            rows.append({
                "variant_id": variant_id,
                "target": target,
                "metric_id": metric.get("metric_id"),
                "value": metric.get("value") if _finite(metric.get("value")) else None,
                "unit": metric.get("unit"),
                "status": metric.get("status", "UNRESOLVED"),
                "threshold_status": metric.get("threshold_status", "PROVISIONAL"),
                "record_count": variant.get("record_count"),
                "scope": result.get("scope", "DEVELOPMENT_NON_INDEPENDENT"),
                "independent_validation": False,
                "source_result": _source(root, G2C_RESULT),
            })
    rows.sort(key=lambda row: (str(row["variant_id"]), str(row["metric_id"])))
    return rows


def build_registry(project_root: Path | str, *, revision: str = "20260902_r1") -> Dict[str, Any]:
    root = Path(project_root).resolve()
    internal = _read(root, INTERNAL_ACCEPTANCE)
    if internal.get("decision") != "PASS" or set((internal.get("methods") or {}).keys()) != set(METHODS):
        raise G3AError("internal six-method acceptance is not a complete PASS matrix")
    rows: list[Dict[str, Any]] = []
    external_sources: Dict[str, Any] = {}
    for method in METHODS:
        relative = EXTERNAL_ACCEPTANCES[method]
        doc = _read(root, relative)
        identity = _identity(doc)
        if str(identity.get("method", method)).lower() != method.lower():
            raise G3AError(f"{relative} does not identify method {method}")
        decision = _decision(doc)
        if decision not in TERMINAL_DECISIONS:
            raise G3AError(f"{relative} is not terminally accepted: {decision}")
        accepted = _seed_ids(method, doc)
        target = _target_seed_ids(method, doc, accepted)
        unresolved = list(identity.get("unresolved_seed_ids", [])) if isinstance(identity.get("unresolved_seed_ids", []), Sequence) else []
        external_sources[method] = _source(root, relative)
        internal_method = internal["methods"][method]
        rows.append({
            "method": method,
            "dataset": str(identity.get("dataset", "GSE254652_P1")),
            "internal_matrix_decision": internal["decision"],
            "internal_applicable_cells": internal_method.get("applicable_cells"),
            "internal_execution_valid_cells": internal_method.get("execution_valid_cells"),
            "internal_numeric_classification_counts": internal_method.get("numeric_classification_counts"),
            "external_acceptance_decision": decision,
            "external_execution_status": _execution_status(method, doc),
            "external_numeric_classification": _numeric_classification(method, doc),
            "accepted_seed_ids": accepted,
            "accepted_seed_count": len(accepted),
            "target_seed_ids": target,
            "planned_seed_target": "DETERMINISTIC_NOT_APPLICABLE" if method == "PCA" else len(target),
            "unresolved_seed_ids": unresolved,
            "seed_coverage_status": _coverage_status(method, accepted, target, unresolved),
            "osi_values": _osi_values(method, doc, accepted),
            "unresolved_numeric_osi_values": {
                str(seed): value for seed, value in ((doc.get("osi_results", {}) or {}).get("by_seed", {}) or {}).items()
                if str(seed) in {str(item) for item in unresolved} and _finite(value)
            },
            "operational_threshold": (doc.get("osi_results", {}) or {}).get("operational_threshold", 0.05),
            "threshold_semantics": "PROVISIONAL_OPERATIONAL_NOT_P_VALUE",
            "external_blinding": "PROJECT_EXPOSED_NOT_BLIND_HELD_OUT",
            "allowed_claims": doc.get("allowed_manuscript_claims", []),
            "forbidden_claims": doc.get("forbidden_manuscript_claims", []),
            "residual_limitations": doc.get("residual_limitations", doc.get("blocking_issues", [])),
            "source_acceptance": relative,
            "source_sha256": external_sources[method]["sha256"],
        })
    g2c = _read(root, G2C_RESULT)
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "revision": revision,
        "project_root": str(root),
        "promotion_rule": "only explicit terminal acceptance documents; unresolved terminality and development evidence are never promoted as accepted external numerics",
        "perturbation_contract": {
            "axis": "SAMPLE_SLICE_BATCH_ORDER",
            "unit": "slice",
            "semantics": "NUISANCE_PERMUTATION",
            "fixed_fields": ["observations", "entity_identities", "hyperparameters", "random_seed", "intended_analysis"],
        },
        "sources": {
            "internal": _source(root, INTERNAL_ACCEPTANCE),
            "external": external_sources,
            "g2b": {
                "formal_acceptance": _source(root, EXTERNAL_ACCEPTANCES["STAligner"]),
                "raw_audit": _source(root, G2B_AUDIT),
                "external_method_registry": _source(root, G2B_REGISTRY),
            },
            "g2c": {
                "protocol": _source(root, G2C_PROTOCOL),
                "fixture": _source(root, G2C_FIXTURE),
                "records_only_result": _source(root, G2C_RESULT),
            },
        },
        "methods": rows,
        "development_evidence": {
            "scope": str(g2c.get("scope", "DEVELOPMENT_NON_INDEPENDENT")),
            "status": str(g2c.get("status", "UNRESOLVED")),
            "fixture_id": g2c.get("fixture_id"),
            "protocol_id": g2c.get("protocol_id"),
            "record_counts": g2c.get("record_counts", {}),
            "metrics": _development_rows(root, g2c),
            "negative_controls": g2c.get("negative_control_results", []),
            "claim_boundary": [str(g2c.get("claim_boundary", ""))],
        },
        "summary": {
            "methods": len(rows),
            "internal_complete_methods": sum(row["internal_execution_valid_cells"] is not None for row in rows),
            "external_five_seed_complete": [row["method"] for row in rows if row["seed_coverage_status"] == "COMPLETE_5_OF_5"],
            "external_deterministic_complete": [row["method"] for row in rows if row["seed_coverage_status"] == "COMPLETE_DETERMINISTIC"],
            "external_terminal_seed_counts": {row["method"]: row["accepted_seed_count"] for row in rows},
            "external_pending_terminal_seed_acceptance": [row["method"] for row in rows if "UNRESOLVED" in row["seed_coverage_status"] or "PENDING" in row["seed_coverage_status"]],
            "g2c_scope": str(g2c.get("scope", "DEVELOPMENT_NON_INDEPENDENT")),
            "whole_framework_externally_confirmed": False,
        },
        "claim_boundary": [
            "Internal matrix coverage does not establish external validity or framework accuracy.",
            "GSE254652 is project-exposed external reproduction, not blind held-out confirmation.",
            "Seed repetitions are computational replicates, not biological replicates.",
            "The 0.05 OSI boundary is operational and must not be reported as a p-value.",
            "STAligner seed 20260820 numeric raw-array recomputation is retained as unresolved terminality and is not promoted as terminal execution.",
            "G2C is eight-donor development-only records evidence; FWER, runtime and memory remain unresolved and no independent-validation claim is licensed.",
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = {}
            for field in fields:
                value = row.get(field)
                values[field] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            writer.writerow(values)


def write_registry(payload: Mapping[str, Any], output_dir: Path | str) -> Dict[str, Path]:
    destination = Path(output_dir)
    paths = {"json": destination / "publication_evidence_registry.json",
             "csv": destination / "publication_evidence_registry.csv"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"refusing to overwrite registry artifacts in {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(paths["json"], payload)
    _write_csv(paths["csv"], payload["methods"], (
        "method", "dataset", "internal_matrix_decision", "internal_applicable_cells",
        "internal_execution_valid_cells", "external_acceptance_decision", "external_execution_status",
        "external_numeric_classification", "accepted_seed_ids", "accepted_seed_count", "target_seed_ids",
        "unresolved_seed_ids", "seed_coverage_status", "osi_values", "unresolved_numeric_osi_values",
        "operational_threshold", "threshold_semantics", "external_blinding", "source_acceptance", "source_sha256"))
    return paths


def _plan_contracts(root: Path) -> list[Dict[str, Any]]:
    plan = _read(root, FIGURE_PLAN)
    contracts: list[Dict[str, Any]] = []
    for section in ("main_figures", "supplementary_figures"):
        entries = plan.get(section, [])
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            panels = entry.get("panels", {})
            contracts.append({
                "figure_id": str(entry.get("figure_id", "")),
                "section": "main" if section == "main_figures" else "supplementary",
                "title_en": entry.get("title_en", entry.get("title")),
                "title_cn": entry.get("title_cn"),
                "file_stem": entry.get("file_stem"),
                "purpose": entry.get("purpose", entry.get("content")),
                "panels": dict(panels) if isinstance(panels, Mapping) else panels,
                "source_data": entry.get("source_data"),
                "plan_status": entry.get("status"),
                "acceptance_gate": entry.get("acceptance_gate", entry.get("acceptance")),
                "allowed_claim": entry.get("allowed_claim"),
                "forbidden_claim": entry.get("forbidden_claim"),
                "render_rule": "TERMINAL_ACCEPTANCE_ONLY; unsupported panels remain UNRESOLVED",
            })
    return contracts


def _terminal_rows(root: Path, registry: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    rows: list[Dict[str, Any]] = []
    unresolved: list[Dict[str, Any]] = []
    for record in registry.get("methods", []):
        method = str(record["method"])
        source_rel = str(record["source_acceptance"])
        source = _source(root, source_rel)
        accepted = {str(seed) for seed in record.get("accepted_seed_ids", [])}
        unresolved_ids = {str(seed) for seed in record.get("unresolved_seed_ids", [])}
        values = record.get("osi_values", {})
        doc = _read(root, source_rel)
        identity = _identity(doc)
        evidence = doc.get("evidence_inventory", {})
        evidence_refs = [str(value) for value in evidence.values()] if isinstance(evidence, Mapping) else []
        if method == "INSPIRE":
            for seed, components in values.items():
                if str(seed) not in accepted or not isinstance(components, Mapping):
                    continue
                for component in ("latent", "shared", "gamma", "full"):
                    value = components.get(component)
                    if not _finite(value):
                        continue
                    rows.append({
                        "figure_ids": ["Figure_4", "Figure_5"], "panel": "component_OSI",
                        "method": method, "dataset": identity.get("dataset"), "condition": identity.get("condition"),
                        "seed": seed, "component": component, "osi": float(value),
                        "operational_threshold": float(record.get("operational_threshold", 0.05)),
                        "threshold_semantics": "PROVISIONAL_OPERATIONAL_NOT_P_VALUE",
                        "numeric_classification": _numeric_classification(method, doc),
                        "terminal_decision": record["external_acceptance_decision"],
                        "seed_coverage_status": record["seed_coverage_status"],
                        "evidence_scope": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION",
                        "independent_validation": False, "source_acceptance": source["path"],
                        "source_sha256": source["sha256"], "evidence_refs": evidence_refs,
                        "claim_status": "ALLOWED_ONLY_WITH_ACCEPTANCE_SCOPE",
                    })
        else:
            for seed, value in values.items():
                if str(seed) not in accepted or not _finite(value):
                    continue
                rows.append({
                    "figure_ids": ["Figure_4"], "panel": "method_OSI", "method": method,
                    "dataset": identity.get("dataset"), "condition": identity.get("condition"), "seed": seed,
                    "component": "overall", "osi": float(value),
                    "operational_threshold": float(record.get("operational_threshold", 0.05)),
                    "threshold_semantics": "PROVISIONAL_OPERATIONAL_NOT_P_VALUE",
                    "numeric_classification": _numeric_classification(method, doc),
                    "terminal_decision": record["external_acceptance_decision"],
                    "seed_coverage_status": record["seed_coverage_status"],
                    "evidence_scope": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION",
                    "independent_validation": False, "source_acceptance": source["path"],
                    "source_sha256": source["sha256"], "evidence_refs": evidence_refs,
                    "claim_status": "ALLOWED_ONLY_WITH_ACCEPTANCE_SCOPE",
                })
        for seed, value in (record.get("unresolved_numeric_osi_values") or {}).items():
            unresolved.append({
                "method": method, "seed": seed, "component": "overall", "osi": value,
                "status": "UNRESOLVED_TERMINALITY", "reason": "raw numeric recomputation exists but terminal receipt is missing",
                "source_acceptance": source["path"], "source_sha256": source["sha256"],
            })
    rows.sort(key=lambda row: (METHODS.index(str(row["method"])), str(row["seed"]), str(row["component"])))
    unresolved.sort(key=lambda row: (str(row["method"]), str(row["seed"])))
    return rows, unresolved


def build_figure_data(project_root: Path | str, registry: Mapping[str, Any],
                      registry_relative: str = "results/framework_evaluation/publication_registry_20260902_r1/publication_evidence_registry.json") -> Dict[str, Any]:
    root = Path(project_root).resolve()
    rows, unresolved = _terminal_rows(root, registry)
    g2c = registry.get("development_evidence", {})
    # Reuse the established, source-separated failure inventory implementation.
    try:
        from .figure_data_v1_20260902 import _failure_inventory  # local import avoids a competing implementation
    except ImportError:  # direct ``python stia/crossmethod/...py`` invocation
        from stia.crossmethod.figure_data_v1_20260902 import _failure_inventory
    internal = _read(root, INTERNAL_ACCEPTANCE)
    failure_inventory = _failure_inventory(root, rows, internal)
    return {
        "schema_version": FIGURE_DATA_SCHEMA_VERSION,
        "revision": "20260902_r3",
        "project_root": str(root),
        "upstream_sources": {
            "figure_plan": _source(root, FIGURE_PLAN),
            "publication_registry": {"path": str(registry_relative).replace("\\", "/")},
            "f1a_pack": _source(root, F1_PACK),
            "internal_acceptance": _source(root, INTERNAL_ACCEPTANCE),
            "external_acceptances": registry.get("sources", {}).get("external", {}),
            "g2b_raw_audit": _source(root, G2B_AUDIT),
            "g2c_protocol": _source(root, G2C_PROTOCOL),
            "g2c_fixture": _source(root, G2C_FIXTURE),
            "g2c_result": _source(root, G2C_RESULT),
        },
        "publication_rule": "Only explicit terminal acceptance documents populate numeric figure rows; unresolved terminality and development metrics remain separate and are never promoted as external validation.",
        "figure_contracts": _plan_contracts(root),
        "figure_data_rows": rows,
        "unresolved_numeric_rows": unresolved,
        "development_ablation_rows": g2c.get("metrics", []),
        "development_negative_control_rows": g2c.get("negative_controls", []),
        "failure_inventory": failure_inventory,
        "summary": {
            "figure_contract_count": len(_plan_contracts(root)),
            "numeric_row_count": len(rows),
            "numeric_methods": sorted({str(row["method"]) for row in rows}),
            "numeric_components": sorted({str(row["component"]) for row in rows}),
            "unresolved_numeric_row_count": len(unresolved),
            "development_ablation_row_count": len(g2c.get("metrics", [])),
            "development_negative_control_count": len(g2c.get("negative_controls", [])),
            "failure_case_count": failure_inventory["summary"]["case_count"],
            "rendered_figures": ["Figure_4", "Figure_5", "Figure_6"],
            "whole_framework_accuracy": "UNRESOLVED",
            "independent_validation": False,
        },
        "claim_boundary": [
            "Figure contracts describe planned panels; only supported terminal rows are rendered.",
            "STAligner seed 20260820 is retained as unresolved terminality and excluded from numeric figure rows.",
            "G2C metrics are development-only, eight-donor, descriptive records evidence; FWER/runtime/memory remain unresolved.",
            "OSI is an operational margin, not a p-value; GSE254652 is project-exposed rather than blind held out.",
            "Whole-framework accuracy and independent validation remain UNRESOLVED.",
        ],
    }


def write_figure_data(payload: Mapping[str, Any], output_dir: Path | str) -> Dict[str, Path]:
    destination = Path(output_dir)
    paths = {
        "contract_json": destination / "f2a_figure_data_contract_20260902.json",
        "figure_rows_csv": destination / "f2a_figure_data_rows_20260902.csv",
        "failure_json": destination / "f2a_failure_case_inventory_20260902.json",
        "failure_csv": destination / "f2a_failure_case_inventory_20260902.csv",
        "development_ablation_csv": destination / "f2a_g2c_development_ablation_rows_20260902.csv",
        "development_negative_controls_csv": destination / "f2a_g2c_negative_control_rows_20260902.csv",
        "unresolved_numeric_csv": destination / "f2a_unresolved_numeric_rows_20260902.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"refusing to overwrite figure-data artifacts in {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(paths["contract_json"], payload)
    _write_csv(paths["figure_rows_csv"], payload.get("figure_data_rows", []), (
        "figure_ids", "panel", "method", "dataset", "condition", "seed", "component", "osi",
        "operational_threshold", "threshold_semantics", "numeric_classification", "terminal_decision",
        "seed_coverage_status", "evidence_scope", "independent_validation", "source_acceptance",
        "source_sha256", "evidence_refs", "claim_status"))
    inventory = payload.get("failure_inventory", {})
    _write_json(paths["failure_json"], inventory)
    _write_csv(paths["failure_csv"], inventory.get("cases", []) if isinstance(inventory, Mapping) else [], (
        "case_id", "category", "subtype", "method", "status", "numeric_value", "numeric_classification",
        "root_cause", "observed", "disposition", "evidence_refs", "source_document", "source_sha256",
        "preserved", "claim_boundary"))
    _write_csv(paths["development_ablation_csv"], payload.get("development_ablation_rows", []), (
        "variant_id", "target", "metric_id", "value", "unit", "status", "threshold_status", "record_count",
        "scope", "independent_validation"))
    _write_csv(paths["development_negative_controls_csv"], payload.get("development_negative_control_rows", []),
               ("id", "surface", "expected", "by_variant"))
    _write_csv(paths["unresolved_numeric_csv"], payload.get("unresolved_numeric_rows", []),
               ("method", "seed", "component", "osi", "status", "reason", "source_acceptance", "source_sha256"))
    return paths


def _exclusive_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_one(fig, stem: str, destination: Path) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        path = destination / f"{stem}.{suffix}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite figure {path}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        out[suffix] = path.name
    plt.close(fig)
    return out


def render_figures(payload: Mapping[str, Any], output_dirs: Sequence[Path | str]) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(payload.get("figure_data_rows", []))
    dev = list(payload.get("development_ablation_rows", []))
    rendered: Dict[str, Any] = {}
    palettes = {"PCA": "#0072B2", "GraphST": "#009E73", "Harmony": "#E69F00",
                "INSPIRE": "#CC79A7", "STAligner": "#D55E00", "PRECAST": "#56B4E9"}
    methods = list(METHODS)

    def fig4():
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        vals: Dict[str, list[float]] = {m: [] for m in methods}
        for row in rows:
            if row["component"] == "overall" or (row["method"] == "INSPIRE" and row["component"] == "full"):
                vals[str(row["method"])].append(float(row["osi"]))
        for idx, method in enumerate(methods):
            data = vals[method]
            if not data:
                continue
            jitter = np.linspace(-0.17, 0.17, len(data)) if len(data) > 1 else np.array([0.0])
            ax.scatter(np.full(len(data), idx) + jitter, data, s=34, color=palettes[method], edgecolor="white", linewidth=0.45, zorder=3)
            ax.plot([idx - 0.21, idx + 0.21], [float(np.mean(data))] * 2, color="black", linewidth=1.2, zorder=4)
        ax.axhline(0.05, color="#444444", linestyle="--", linewidth=1, label="0.05 provisional operational margin")
        ax.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("OSI (six pairwise drifts; log scale)")
        ax.set_title("Project-exposed GSE254652 order sensitivity: terminally accepted seeds")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig

    def fig5():
        fig, ax = plt.subplots(figsize=(7.7, 4.8))
        components = ["latent", "shared", "gamma", "full"]
        colors = ["#CC79A7", "#D55E00", "#009E73", "#0072B2"]
        for idx, component in enumerate(components):
            data = [float(row["osi"]) for row in rows if row["method"] == "INSPIRE" and row["component"] == component]
            jitter = np.linspace(-0.16, 0.16, len(data)) if len(data) > 1 else np.array([0.0])
            ax.scatter(np.full(len(data), idx) + jitter, data, s=38, color=colors[idx], edgecolor="white", linewidth=0.45, zorder=3)
            if data:
                ax.plot([idx - 0.2, idx + 0.2], [float(np.mean(data))] * 2, color="black", linewidth=1.2, zorder=4)
        ax.axhline(0.05, color="#444444", linestyle="--", linewidth=1, label="0.05 provisional operational margin")
        ax.set_xticks(range(len(components)), components)
        ax.set_yscale("log")
        ax.set_ylabel("OSI (six pairwise drifts; log scale)")
        ax.set_title("INSPIRE component-level order sensitivity (five accepted seeds)")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig

    def fig6():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.7), sharex=True, gridspec_kw={"height_ratios": [3, 1.35]})
        variants = ["full_framework", "drop_order_audit", "drop_three_axis_condition_reversal", "drop_provenance_evidence", "drop_claim_governance"]
        labels = ["full", "−M6", "−M7", "−M1", "−M8/M9"]
        metrics = ["clean_case_fpr", "defect_recall", "unsafe_acceptance_rate", "root_cause_macro_f1", "provenance_completeness"]
        x = np.arange(len(variants))
        width = 0.15
        for j, metric in enumerate(metrics):
            vals = []
            for variant in variants:
                hit = next((r for r in dev if r.get("variant_id") == variant and r.get("metric_id") == metric), {})
                vals.append(float(hit["value"]) if _finite(hit.get("value")) else np.nan)
            ax1.bar(x + (j - 2) * width, vals, width, label=metric, color=plt.cm.viridis(j / max(1, len(metrics) - 1)), edgecolor="white", linewidth=0.3)
        ax1.set_ylim(0, 1.08)
        ax1.set_ylabel("Descriptive proportion")
        ax1.set_title("G2C records-only targeted ablation (development, n=8 donors)")
        ax1.grid(axis="y", alpha=0.22)
        ax1.legend(ncol=2, fontsize=7, frameon=False, loc="upper left")
        bypass = []
        for variant in variants:
            hit = next((r for r in dev if r.get("variant_id") == variant and r.get("metric_id") == "required_gate_bypass_count"), {})
            bypass.append(float(hit["value"]) if _finite(hit.get("value")) else np.nan)
        ax2.bar(x, bypass, color="#999999", width=0.55)
        ax2.set_ylabel("Gate bypass count")
        ax2.set_xticks(x, labels)
        ax2.grid(axis="y", alpha=0.22)
        fig.text(0.01, 0.01, "Development-only; provisional thresholds. FWER, runtime and memory are unresolved; no independent-validation claim.", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return fig

    builders = {"Figure_4": ("Fig4_G3A_external_order_OSI", fig4),
                "Figure_5": ("Fig5_G3A_INSPIRE_components", fig5),
                "Figure_6": ("Fig6_G3A_G2C_development_ablation", fig6)}
    for figure_id, (stem, builder) in builders.items():
        destinations = []
        file_sets = []
        for directory in output_dirs:
            destination = Path(directory)
            expected = {suffix: destination / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")}
            if all(path.exists() for path in expected.values()):
                file_sets.append({suffix: path.name for suffix, path in expected.items()})
            else:
                file_sets.append(_render_one(builder(), stem, destination))
            destinations.append(destination)
        rendered[figure_id] = {"stem": stem, "files": file_sets[0], "supported": True}

    source_manifest = {
        "schema_version": "stia.g3a_figure_source_manifest.v1.20260902",
        "revision": "20260902_r1",
        "rendered_figures": rendered,
        "source_tables": payload.get("upstream_sources", {}),
        "excluded_unresolved_rows": payload.get("unresolved_numeric_rows", []),
        "unsupported_plan_figures": [contract["figure_id"] for contract in payload.get("figure_contracts", []) if contract["figure_id"] not in builders],
        "rule": "No unsupported panel is rendered; planned but unsupported panels remain UNRESOLVED.",
    }
    stats = {
        "schema_version": "stia.g3a_figure_statistics.v1.20260902",
        "revision": "20260902_r1",
        "numeric_row_count": payload["summary"]["numeric_row_count"],
        "unresolved_numeric_row_count": payload["summary"]["unresolved_numeric_row_count"],
        "development_ablation_row_count": payload["summary"]["development_ablation_row_count"],
        "threshold": 0.05,
        "threshold_semantics": "PROVISIONAL_OPERATIONAL_NOT_P_VALUE",
        "figure_value_counts": {figure_id: sum(1 for row in rows if figure_id in row.get("figure_ids", [])) for figure_id in ("Figure_4", "Figure_5")},
        "g2c_numeric_metric_count": sum(1 for row in dev if _finite(row.get("value"))),
    }
    claims = {
        "schema_version": "stia.g3a_figure_claims.v1.20260902",
        "allowed": payload.get("claim_boundary", []),
        "figure_claims": {
            "Figure_4": "Descriptive OSI under the project-exposed GSE254652 frozen four-order protocol; accepted seeds only.",
            "Figure_5": "INSPIRE latent/shared/gamma/full component pattern under the accepted five-seed external receipt.",
            "Figure_6": "Development-only targeted-ablation sensitivity on eight truth-labelled donor records; descriptive and provisional.",
        },
        "forbidden": ["blind held-out confirmation", "biological replicate inference", "p-value/significance interpretation of OSI", "universal method or framework accuracy", "independent validation from G2C"],
    }
    captions = """# G3A figure captions (2026-09-02 revision)\n\n""" \
        "**Figure 4.** Project-exposed GSE254652 input-order OSI for terminally accepted seeds under the frozen CPU/four-order protocol. Points are computational seed replicates; horizontal bars are means. The dashed line is the provisional 0.05 operational margin, not a p-value. STAligner seed 20260820 is excluded because terminal receipts are unresolved.\n\n" \
        "**Figure 5.** INSPIRE component-level OSI for latent, shared, gamma and full readouts across five accepted seeds. Values are descriptive and project-exposed; component behavior does not establish causality.\n\n" \
        "**Figure 6.** G2C targeted-ablation records-only development fixture (eight donors, five variants). Bars show descriptive proportions and gate-bypass counts. FWER, runtime and memory are unresolved; no independent-validation or superiority claim is licensed.\n"
    for directory in output_dirs:
        destination = Path(directory)
        _exclusive_text(destination / "figure_source_manifest.json", json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _exclusive_text(destination / "figure_statistics.json", json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _exclusive_text(destination / "figure_claims.json", json.dumps(claims, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _exclusive_text(destination / "figure_captions.md", captions)
        _exclusive_text(destination / "figure_render_log.txt", "G3A renderer: stia/crossmethod/g3a_publication_v1_20260902.py\nMatplotlib backend: Agg\nFormats: PNG 600 dpi, PDF, SVG\nNo unsupported plan panels rendered; unresolved terminal rows retained only in data sidecar.\n")
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build additive G3A publication registry, figure data and figures")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, default=Path("results/framework_evaluation/publication_registry_20260902_r7"))
    parser.add_argument("--figure-data-dir", type=Path, default=Path("results/framework_evaluation/f2a_figure_data_20260902_r7"))
    parser.add_argument("--results-figures-dir", type=Path, default=Path("results/framework_evaluation/g3a_publication_figures_20260902_r5"))
    parser.add_argument("--paper-figures-dir", type=Path, default=Path("paper_package/figures/g3a_publication_20260902_r5"))
    args = parser.parse_args(argv)
    try:
        registry_dir = args.project_root / args.registry_dir if not args.registry_dir.is_absolute() else args.registry_dir
        registry = build_registry(args.project_root, revision=registry_dir.name)
        registry_paths = write_registry(registry, registry_dir)
        registry_relative = (args.registry_dir / "publication_evidence_registry.json").as_posix() if not args.registry_dir.is_absolute() else str(registry_paths["json"])
        figure_data = build_figure_data(args.project_root, registry, registry_relative)
        figure_data_paths = write_figure_data(figure_data, args.project_root / args.figure_data_dir if not args.figure_data_dir.is_absolute() else args.figure_data_dir)
        result_dir = args.project_root / args.results_figures_dir if not args.results_figures_dir.is_absolute() else args.results_figures_dir
        paper_dir = args.project_root / args.paper_figures_dir if not args.paper_figures_dir.is_absolute() else args.paper_figures_dir
        rendered = render_figures(figure_data, [result_dir, paper_dir])
    except (G3AError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "schema_version": SCHEMA_VERSION,
                      "registry": {key: str(value) for key, value in registry_paths.items()},
                      "figure_data": {key: str(value) for key, value in figure_data_paths.items()},
                      "summary": figure_data["summary"], "rendered": rendered}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
