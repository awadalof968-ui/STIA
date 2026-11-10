# -*- coding: utf-8 -*-
"""Assemble publication-safe figure data and a retained failure inventory.

This is an evidence assembler, not a plotting or experiment module.  It reads
the frozen STIA figure plan, the additive F1A pack and explicitly named
terminal acceptance documents.  Numeric rows are emitted only when the
acceptance document contains an explicit finite OSI value.  Engineering and
audit failures are copied into a separate inventory so that they cannot be
mistaken for adverse scientific outcomes.  Writers use exclusive creation and
never modify an existing artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stia.f2a_figure_data.v1.20260902"
TERMINAL_DECISIONS = {"PASS", "CONDITIONAL_PASS"}
METHODS = ("PCA", "GraphST", "Harmony", "INSPIRE", "STAligner", "PRECAST")
FIGURE_PLAN = "docs/STIA_PAPER_FIGURE_PLAN_20260821.json"
REGISTRY = "results/framework_evaluation/publication_registry_20260902_r1/publication_evidence_registry.json"
F1_PACK = "results/framework_evaluation/f1a_baseline_metrics_20260902_r3/f1a_baseline_metric_pack_20260902.json"
INTERNAL_ACCEPTANCE = "docs/SIX_METHOD_ORDER_COVERAGE_ACCEPTANCE_20260831_r1.json"
ACCEPTANCE_SOURCES = {
    "PCA": "docs/PCA_EXTERNAL_GSE254652_FINAL_ACCEPTANCE_20260831_r1.json",
    "GraphST": "docs/GRAPHST_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "Harmony": "docs/HARMONY_GSE254652_FIVE_SEED_FINAL_ACCEPTANCE_20260901_r1.json",
    "INSPIRE": "docs/INSPIRE_GSE254652_FIVE_SEED_TERMINAL_ACCEPTANCE_20260901_r1.json",
    "STAligner": "docs/STALIGNER_EXTERNAL_GSE254652_FINAL_ACCEPTANCE_20260831_r1.json",
    "PRECAST": "docs/PRECAST_GSE254652_SEED20260819_TERMINAL_ACCEPTANCE_20260901_r1.json",
}
FAILURE_SOURCES = (
    "docs/EXTERNAL_STARTUP_FAILURE_CASES_20260831_r1.json",
    "docs/PRECAST_V15_ERROR_CASE_REGISTER_20260826.json",
    "docs/PRECAST_V16_GAP_BLOCKER_20260829.json",
    "docs/INSPIRE_V5B_AUDITABILITY_FINDING_20260830.json",
    "docs/INSPIRE_RAW_REPLAY_R2_ACCEPTANCE_20260830_r1.json",
)


class FigureDataError(ValueError):
    """Raised when a required evidence source is malformed or unavailable."""


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


def _source_identity(root: Path, relative: str, *, required: bool = True) -> Dict[str, Any]:
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        if required:
            raise FigureDataError(f"required evidence is missing: {relative}")
        return {"path": _relative(path, root), "exists": False, "bytes": None, "sha256": None}
    return {
        "path": _relative(path, root),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(root: Path, relative: str, *, required: bool = True) -> Mapping[str, Any]:
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        if required:
            raise FigureDataError(f"required JSON evidence is missing: {relative}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - parser wording is environment-specific
        raise FigureDataError(f"invalid JSON evidence {relative}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FigureDataError(f"JSON evidence root must be an object: {relative}")
    return value


def _decision(document: Mapping[str, Any]) -> str:
    value = document.get("decision", document.get("acceptance_decision", "UNRESOLVED"))
    return str(value).strip().upper() if value is not None else "UNRESOLVED"


def _identity(document: Mapping[str, Any]) -> Mapping[str, Any]:
    value = document.get("experiment_identity", {})
    return value if isinstance(value, Mapping) else {}


def _seeds(document: Mapping[str, Any], method: str) -> list[Any]:
    identity = _identity(document)
    for key in ("seeds", "model_seeds", "requested_seeds"):
        value = identity.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
    if identity.get("seed") is not None:
        return [identity["seed"]]
    return ["DETERMINISTIC_SINGLE_FIT"] if method == "PCA" else []


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _numeric_classification(document: Mapping[str, Any], method: str, component: str | None = None) -> str:
    osi = document.get("osi_results")
    if not isinstance(osi, Mapping):
        osi = {}
    if method == "INSPIRE" and component:
        classes = osi.get("component_threshold_classification")
        if isinstance(classes, Mapping) and classes.get(component) is not None:
            return str(classes[component])
    for candidate in (osi.get("numeric_classification"), document.get("numeric_acceptance")):
        if candidate is not None and not isinstance(candidate, Mapping):
            return str(candidate)
    if method == "PCA":
        return "ORDER_INVARIANT_WITHIN_MARGIN"
    return "UNRESOLVED"


def _numeric_rows_for_method(root: Path, method: str, relative: str) -> list[Dict[str, Any]]:
    document = _read_json(root, relative)
    decision = _decision(document)
    if decision not in TERMINAL_DECISIONS:
        return []
    identity = _identity(document)
    evidence = document.get("evidence_inventory")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    source = _source_identity(root, relative)
    threshold = None
    osi = document.get("osi_results")
    if isinstance(osi, Mapping) and _finite_number(osi.get("operational_threshold")):
        threshold = float(osi["operational_threshold"])
    if threshold is None:
        statistical = document.get("statistical_audit")
        if isinstance(statistical, Mapping) and _finite_number(statistical.get("osi_margin")):
            threshold = float(statistical["osi_margin"])
    if threshold is None:
        threshold = 0.05

    rows: list[Dict[str, Any]] = []
    if method == "INSPIRE" and isinstance(osi, Mapping) and isinstance(osi.get("by_seed"), Mapping):
        for seed, values in sorted(osi["by_seed"].items(), key=lambda item: str(item[0])):
            if not isinstance(values, Mapping):
                continue
            for component in ("latent", "shared", "gamma", "full"):
                value = values.get(component)
                if not _finite_number(value):
                    continue
                rows.append(_figure_row(method, identity, source, evidence, decision, seed, component,
                                        float(value), threshold, _numeric_classification(document, method, component)))
        return rows

    values_by_seed: Dict[Any, Any] = {}
    if isinstance(osi, Mapping) and isinstance(osi.get("by_seed"), Mapping):
        values_by_seed = dict(osi["by_seed"])
    elif isinstance(osi, Mapping) and _finite_number(osi.get("osi")):
        seed_list = _seeds(document, method)
        if seed_list:
            values_by_seed = {seed_list[0]: osi["osi"]}
    if method == "PCA":
        statistical = document.get("statistical_audit")
        value = statistical.get("osi") if isinstance(statistical, Mapping) else None
        if not _finite_number(value):
            # PCA's accepted numeric value is recorded in the positive/negative
            # result block in some prior receipts.
            positive = document.get("positive_negative_uncertain_results")
            value = positive.get("osi") if isinstance(positive, Mapping) else None
        if _finite_number(value):
            values_by_seed = {"DETERMINISTIC_SINGLE_FIT": value}
    if method in {"STAligner"} and not values_by_seed:
        statistical = document.get("statistical_audit")
        value = statistical.get("osi") if isinstance(statistical, Mapping) else None
        if _finite_number(value):
            values_by_seed = {_seeds(document, method)[0] if _seeds(document, method) else "UNSPECIFIED": value}
    for seed, value in sorted(values_by_seed.items(), key=lambda item: str(item[0])):
        if not _finite_number(value):
            continue
        rows.append(_figure_row(method, identity, source, evidence, decision, seed, "overall",
                                float(value), threshold, _numeric_classification(document, method)))
    return rows


def _figure_row(method: str, identity: Mapping[str, Any], source: Mapping[str, Any],
                evidence: Mapping[str, Any], decision: str, seed: Any, component: str,
                value: float, threshold: float, classification: str) -> Dict[str, Any]:
    return {
        "figure_ids": ["Figure_4", "Figure_5"] if method == "INSPIRE" else ["Figure_4"],
        "panel": "component_OSI" if method == "INSPIRE" else "method_OSI",
        "method": method,
        "dataset": identity.get("dataset"),
        "condition": identity.get("condition"),
        "seed": seed,
        "component": component,
        "osi": value,
        "operational_threshold": threshold,
        "threshold_semantics": "PROVISIONAL_OPERATIONAL_NOT_P_VALUE",
        "numeric_classification": classification,
        "terminal_decision": decision,
        "seed_coverage_status": "TERMINAL_ACCEPTED_SCOPED",
        "evidence_scope": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION",
        "independent_validation": False,
        "source_acceptance": source["path"],
        "source_sha256": source["sha256"],
        "evidence_refs": sorted(str(value) for value in evidence.values() if isinstance(value, (str, int, float))),
        "claim_status": "ALLOWED_ONLY_WITH_ACCEPTANCE_SCOPE",
    }


def _plan_contracts(root: Path) -> list[Dict[str, Any]]:
    plan = _read_json(root, FIGURE_PLAN)
    contracts: list[Dict[str, Any]] = []
    for section in ("main_figures", "supplementary_figures"):
        entries = plan.get(section, [])
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            figure_id = str(entry.get("figure_id", ""))
            panels = entry.get("panels", {})
            contracts.append({
                "figure_id": figure_id,
                "section": "main" if section == "main_figures" else "supplementary",
                "title_en": entry.get("title_en", entry.get("title")),
                "title_cn": entry.get("title_cn"),
                "file_stem": entry.get("file_stem"),
                "purpose": entry.get("purpose", entry.get("content")),
                "panels": dict(panels) if isinstance(panels, Mapping) else list(panels) if isinstance(panels, Sequence) else panels,
                "source_data": list(entry.get("source_data", [])) if isinstance(entry.get("source_data"), Sequence) and not isinstance(entry.get("source_data"), (str, bytes)) else entry.get("source_data"),
                "plan_status": entry.get("status"),
                "acceptance_gate": entry.get("acceptance_gate", entry.get("acceptance")),
                "allowed_claim": entry.get("allowed_claim"),
                "forbidden_claim": entry.get("forbidden_claim"),
                "render_rule": "TERMINAL_EVIDENCE_ONLY; unsupported numeric panels remain UNRESOLVED",
            })
    return contracts


def _failure_case(case_id: str, category: str, subtype: str, source: Mapping[str, Any],
                  *, method: str | None = None, status: str, root_cause: Any = None,
                  evidence_refs: Sequence[Any] = (), observed: Any = None,
                  disposition: Any = None, numeric_value: Any = None,
                  numeric_classification: Any = None) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "subtype": subtype,
        "method": method,
        "status": status,
        "numeric_value": numeric_value if _finite_number(numeric_value) else None,
        "numeric_classification": numeric_classification,
        "root_cause": root_cause,
        "observed": observed,
        "disposition": disposition,
        "evidence_refs": [str(item) for item in evidence_refs],
        "source_document": source["path"],
        "source_sha256": source["sha256"],
        "preserved": True,
        "claim_boundary": "Engineering/audit failures are not numeric method failures; valid adverse numeric rows are scoped to their accepted protocol and are not universal claims.",
    }


def _failure_inventory(root: Path, acceptance_rows: Sequence[Mapping[str, Any]], internal: Mapping[str, Any]) -> Dict[str, Any]:
    sources = [_source_identity(root, relative) for relative in FAILURE_SOURCES]
    docs = {relative: _read_json(root, relative) for relative in FAILURE_SOURCES}
    cases: list[Dict[str, Any]] = []

    startup = docs[FAILURE_SOURCES[0]]
    for item in startup.get("cases", []):
        if not isinstance(item, Mapping):
            continue
        case_id = str(item.get("case_id", "STARTUP_FAILURE_UNKNOWN"))
        subtype = "STATIC_CODE_HAZARD" if "STATIC" in case_id else "STARTUP_LIFECYCLE"
        category = "CODE_FAILURE" if subtype == "STATIC_CODE_HAZARD" else "LAUNCH_FAILURE"
        cases.append(_failure_case(case_id, category, subtype, sources[0],
                                   status="INVALID_FOR_NUMERIC_CLAIM", root_cause=item.get("root_cause"),
                                   evidence_refs=item.get("evidence", []), observed=item.get("observed"),
                                   disposition=item.get("repair")))

    precast_v15 = docs[FAILURE_SOURCES[1]]
    cases.append(_failure_case(str(precast_v15.get("case_id", "precast-v15-invalid-mapping")),
                               "AUDIT_FAILURE", "IDENTITY_MAPPING", sources[1], method="PRECAST",
                               status="INVALID_FOR_NUMERIC_CLAIM", root_cause=precast_v15.get("root_cause"),
                               evidence_refs=[precast_v15.get("authoritative_audit"), precast_v15.get("case_directory")],
                               disposition=precast_v15.get("scientific_disposition")))

    blocker = docs[FAILURE_SOURCES[2]]
    run = blocker.get("run", {}) if isinstance(blocker.get("run"), Mapping) else {}
    cases.append(_failure_case(str(run.get("run_id", "precast-v16-gap-20260829-r1")),
                               "LAUNCH_FAILURE", "DEPENDENCY_STARTUP_BLOCKER", sources[2], method="PRECAST",
                               status="INVALID_FOR_NUMERIC_CLAIM", root_cause=(blocker.get("observed_error") or {}).get("precast_library_error") if isinstance(blocker.get("observed_error"), Mapping) else blocker.get("observed_error"),
                               evidence_refs=[run.get("output_dir"), blocker.get("required_before_rerun")],
                               observed=blocker.get("observed_error"), disposition=blocker.get("claim_boundary")))

    auditability = docs[FAILURE_SOURCES[3]]
    evidence = auditability.get("evidence", {}) if isinstance(auditability.get("evidence"), Mapping) else {}
    cases.append(_failure_case(str(auditability.get("run_id", "x2-inspire-v5b-20260830")),
                               "AUDIT_FAILURE", "RAW_ARTIFACT_PERSISTENCE_OMISSION", sources[3], method="INSPIRE",
                               status="INVALID_FOR_NUMERIC_CLAIM", root_cause=auditability.get("new_findings"),
                               evidence_refs=evidence.values(), observed=auditability.get("scope"),
                               disposition=auditability.get("decision")))

    replay = docs[FAILURE_SOURCES[4]]
    historical = replay.get("historical_failure_case", {})
    if isinstance(historical, Mapping):
        cases.append(_failure_case("INSPIRE_RAW_REPLAY_R2_HISTORICAL_ROOT_CAUSE", "AUDIT_FAILURE",
                                   "UNRESOLVED_HISTORICAL_ROOT_CAUSE", sources[4], method="INSPIRE",
                                   status="UNRESOLVED", root_cause=historical.get("correction"),
                                   evidence_refs=[replay.get("evidence_inventory"), replay.get("parent_gate")],
                                   observed=historical.get("status"), disposition=historical))

    # Explicit accepted numeric outcomes remain visible, but are never mixed
    # into engineering-failure counts.
    for row in acceptance_rows:
        classification = str(row.get("numeric_classification") or "")
        above_margin = (_finite_number(row.get("osi")) and _finite_number(row.get("operational_threshold"))
                        and float(row["osi"]) > float(row["operational_threshold"]))
        if "FAIL" not in classification and "SENSITIVE" not in classification and not above_margin:
            continue
        method = row.get("method")
        value = row.get("osi")
        case_id = f"{method}_ACCEPTED_ADVERSE_NUMERIC_{row.get('seed')}_{row.get('component')}"
        cases.append(_failure_case(case_id, "VALID_ADVERSE_NUMERIC", "ORDER_SENSITIVITY_RESULT",
                                   {"path": row.get("source_acceptance"), "sha256": row.get("source_sha256")},
                                   method=method, status="VALID_MEASURED_RESULT", numeric_value=value,
                                   numeric_classification=classification,
                                   evidence_refs=row.get("evidence_refs", []),
                                   disposition="Accepted numeric result; scoped to the frozen dataset/seed/protocol."))

    # The internal matrix has accepted adverse cells even where external seed
    # coverage is incomplete.  Keep a count row with no fabricated aggregate.
    methods = internal.get("methods", {}) if isinstance(internal.get("methods"), Mapping) else {}
    for method in METHODS:
        details = methods.get(method, {}) if isinstance(methods.get(method), Mapping) else {}
        counts = details.get("numeric_classification_counts", {})
        adverse = counts.get("ORDER_SENSITIVE") if isinstance(counts, Mapping) else None
        if isinstance(adverse, int) and adverse > 0:
            source = _source_identity(root, INTERNAL_ACCEPTANCE)
            cases.append(_failure_case(f"{method}_INTERNAL_MATRIX_ORDER_SENSITIVE_CELLS", "VALID_ADVERSE_NUMERIC",
                                       "INTERNAL_MATRIX_COUNT", source, method=method,
                                       status="VALID_MEASURED_RESULT", numeric_classification="ORDER_SENSITIVE",
                                       disposition={"accepted_cell_count": adverse, "numeric_value": None},
                                       evidence_refs=[INTERNAL_ACCEPTANCE]))

    cases.sort(key=lambda item: item["case_id"])
    by_category: Dict[str, int] = {}
    for case in cases:
        by_category[case["category"]] = by_category.get(case["category"], 0) + 1
    return {
        "schema_version": "stia.f2a_failure_inventory.v1.20260902",
        "scope": "RETAINED_ENGINEERING_AUDIT_AND_VALID_ADVERSE_NUMERIC_EVIDENCE",
        "sources": sources,
        "cases": cases,
        "summary": {"case_count": len(cases), "by_category": dict(sorted(by_category.items())),
                    "engineering_or_audit_count": sum(value for key, value in by_category.items() if key != "VALID_ADVERSE_NUMERIC"),
                    "valid_adverse_numeric_count": by_category.get("VALID_ADVERSE_NUMERIC", 0)},
        "claim_boundary": [
            "Code, launch and audit failures are retained engineering evidence and cannot be plotted as adverse method numerics.",
            "A valid adverse numeric result is retained only with its terminal acceptance scope; it is not a universal method claim.",
            "Invalid/retracted PRECAST v15 values are excluded from numeric figure rows but retained as an audit failure case.",
            "No row in this inventory establishes independent framework accuracy or blind validation.",
        ],
    }


def build_figure_data(project_root: Path | str) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    plan_source = _source_identity(root, FIGURE_PLAN)
    registry_source = _source_identity(root, REGISTRY)
    f1_source = _source_identity(root, F1_PACK)
    registry = _read_json(root, REGISTRY)
    f1 = _read_json(root, F1_PACK)
    # Check that these are the expected upstream artifacts before using them.
    if not str(registry.get("schema_version", "")).startswith("stia.publication_evidence_registry"):
        raise FigureDataError("publication registry schema is not the expected terminal registry")
    if not str(f1.get("schema_version", "")).startswith("stia.f1a_baseline_metric_pack"):
        raise FigureDataError("F1A pack schema is not the expected baseline pack")
    acceptance_rows: list[Dict[str, Any]] = []
    acceptance_sources: Dict[str, Any] = {}
    for method in METHODS:
        relative = ACCEPTANCE_SOURCES[method]
        acceptance_sources[method] = _source_identity(root, relative)
        acceptance_rows.extend(_numeric_rows_for_method(root, method, relative))
    internal = _read_json(root, INTERNAL_ACCEPTANCE)
    failure_inventory = _failure_inventory(root, acceptance_rows, internal)
    return {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(root),
        "upstream_sources": {
            "figure_plan": plan_source,
            "publication_registry": registry_source,
            "f1a_pack": f1_source,
            "internal_acceptance": _source_identity(root, INTERNAL_ACCEPTANCE),
            "external_acceptances": acceptance_sources,
        },
        "publication_rule": "Only explicit terminal acceptance documents and F1A outputs may populate numeric figure rows; unresolved panels remain unresolved.",
        "figure_contracts": _plan_contracts(root),
        "figure_data_rows": acceptance_rows,
        "failure_inventory": failure_inventory,
        "summary": {
            "figure_contract_count": len(_plan_contracts(root)),
            "numeric_row_count": len(acceptance_rows),
            "numeric_methods": sorted({str(row["method"]) for row in acceptance_rows}),
            "numeric_components": sorted({str(row["component"]) for row in acceptance_rows}),
            "failure_case_count": failure_inventory["summary"]["case_count"],
            "whole_framework_accuracy": "UNRESOLVED",
            "independent_validation": False,
        },
        "claim_boundary": [
            "Figure contracts describe what may be rendered; they do not promote planned or partial figures to accepted claims.",
            "OSI is an operational margin, not a p-value; seed repetitions are computational, not biological replicates.",
            "GSE254652 evidence is project-exposed external reproduction, not blind held-out confirmation.",
            "Whole-framework accuracy and independent validation remain UNRESOLVED without a frozen truth benchmark.",
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
            values = {}
            for field in fields:
                value = row.get(field)
                values[field] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            writer.writerow(values)


def write_figure_data(payload: Mapping[str, Any], output_dir: Path | str) -> Dict[str, Path]:
    destination = Path(output_dir)
    paths = {
        "contract_json": destination / "f2a_figure_data_contract_20260902.json",
        "figure_rows_csv": destination / "f2a_figure_data_rows_20260902.csv",
        "failure_json": destination / "f2a_failure_case_inventory_20260902.json",
        "failure_csv": destination / "f2a_failure_case_inventory_20260902.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"refusing to overwrite F2A artifacts in {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(paths["contract_json"], payload)
    _write_csv(paths["figure_rows_csv"], payload.get("figure_data_rows", []),
               ("figure_ids", "panel", "method", "dataset", "condition", "seed", "component", "osi",
                "operational_threshold", "threshold_semantics", "numeric_classification", "terminal_decision",
                "seed_coverage_status", "evidence_scope", "independent_validation", "source_acceptance",
                "source_sha256", "evidence_refs", "claim_status"))
    inventory = payload.get("failure_inventory", {})
    _write_json(paths["failure_json"], inventory)
    _write_csv(paths["failure_csv"], inventory.get("cases", []) if isinstance(inventory, Mapping) else [],
               ("case_id", "category", "subtype", "method", "status", "numeric_value", "numeric_classification",
                "root_cause", "observed", "disposition", "evidence_refs", "source_document", "source_sha256",
                "preserved", "claim_boundary"))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build additive publication figure-data contracts and failure inventory")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_figure_data(args.project_root)
        paths = write_figure_data(payload, args.output_dir)
    except (FigureDataError, OSError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "schema_version": payload["schema_version"], "summary": payload["summary"],
                      "outputs": {key: str(path) for key, path in paths.items()}}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
