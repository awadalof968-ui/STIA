"""Six-method evidence consolidation (U8).

This successor composes the immutable v5 registry with the independently
audited H5 STAligner receipt.  It changes no scientific producer or prior
receipt and keeps execution validity separate from the frozen numeric OSI
classification.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import evidence_registry_v5_20260830 as _v5

PROJECT_DEFAULT = _v5.PROJECT_DEFAULT
METHODS = _v5.METHODS
CONDITIONS = _v5.CONDITIONS
SEEDS = _v5.SEEDS
STATUSES = _v5.STATUSES
H5_ACCEPTANCE_DEFAULT = Path("docs/STALIGNER_STRICT_ORDER_AUDIT_ACCEPTANCE_20260831_r1.json")
SCHEMA_VERSION = "stia.evidence_registry.v6.20260831"
BOUND_MANIFEST_SCHEMA = "stia.evidence_registry.v6.bound_manifest.v1"


class EvidenceValidationError(ValueError):
    """Raised when a method acceptance cannot be bound to its audit receipt."""


def _resolve(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceValidationError(f"missing evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvidenceValidationError(f"malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"evidence root must be an object: {path}")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(float(value)) != float("inf")


def _empty_staligner(source: str, reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [{"method": "STAligner", "condition": condition, "seed": seed, "status": "UNRESOLVED", "evidence_tier": "h5_acceptance_missing_or_invalid", "source": source, "reason": reason, "execution_validity": "UNRESOLVED"} for condition in CONDITIONS for seed in SEEDS]
    return rows, {"status": "UNRESOLVED", "source": source, "accepted_cells": 0, "execution_valid_cells": 0, "numeric_fail_cells": 0, "target_cells": 20, "reason": reason}


def ingest_staligner(project_root: Path | str, acceptance: Path | str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind the H5 formal acceptance to its independent terminal audit."""
    root = Path(project_root); value = acceptance if acceptance is not None else H5_ACCEPTANCE_DEFAULT; path = _resolve(root, value); source = str(value)
    try:
        doc = _json(path)
        if doc.get("document_type") != "METHOD_ORDER_AUDIT_ACCEPTANCE" or doc.get("schema_version") != "inspire.method_order_audit.acceptance.v1" or doc.get("method") != "STAligner" or doc.get("decision") not in {"PASS", "CONDITIONAL_PASS"} or doc.get("execution_validity") != "PASS":
            raise EvidenceValidationError("H5 STAligner acceptance envelope is invalid")
        matrix = doc.get("matrix"); numeric = doc.get("numeric_classification"); independent = doc.get("independent_audit")
        if not isinstance(matrix, Mapping) or matrix.get("cells") != 20 or matrix.get("raw_order_arms") != 80 or matrix.get("execution_valid_cells") != 20 or matrix.get("unresolved_cells") != 0 or not isinstance(numeric, Mapping) or numeric.get("operational_osi_margin") != 0.05 or numeric.get("pass_cells") != 11 or numeric.get("fail_cells") != 9 or not isinstance(independent, Mapping) or independent.get("verdict") != "COMPLETE":
            raise EvidenceValidationError("H5 matrix/numeric/independent summary is inconsistent")
        audit_value = independent.get("source_evidence")
        audit_path = _resolve(root, audit_value) if isinstance(audit_value, str) else None
        if audit_path is None:
            raise EvidenceValidationError("H5 independent source evidence path is missing")
        audit = _json(audit_path)
        if audit.get("schema") != "inspire.h5.staligner_terminal_audit.v1" or audit.get("verdict") != "COMPLETE" or audit.get("execution_validity") != "PASS" or audit.get("errors") not in ([], None):
            raise EvidenceValidationError("H5 terminal audit envelope is invalid")
        audit_cells = audit.get("cells"); doc_cells = doc.get("condition_summary")
        if not isinstance(audit_cells, list) or len(audit_cells) != 20:
            raise EvidenceValidationError("H5 terminal audit does not contain 20 cells")
        rows: list[dict[str, Any]] = []; keys: set[tuple[str, int]] = set(); pass_count = 0; fail_count = 0
        for cell in audit_cells:
            if not isinstance(cell, Mapping) or cell.get("audit") != "PASS" or cell.get("execution_status") != "PASS" or cell.get("condition") not in CONDITIONS or not isinstance(cell.get("seed"), int) or cell.get("seed") not in SEEDS or cell.get("numeric_status") not in {"PASS", "FAIL"} or cell.get("n_pairs") != 6 or not _finite(cell.get("osi")):
                raise EvidenceValidationError("H5 terminal cell is malformed or unresolved")
            key = (str(cell["condition"]), int(cell["seed"]))
            if key in keys:
                raise EvidenceValidationError(f"H5 duplicate cell: {key}")
            keys.add(key)
            status = str(cell["numeric_status"]); pass_count += status == "PASS"; fail_count += status == "FAIL"
            rows.append({"method": "STAligner", "condition": key[0], "seed": key[1], "status": status, "evidence_tier": "accepted_h5_raw_receipt", "source": str(path), "audit_source": str(audit_path), "reason": "H5 sidecar-aware independent terminal audit; execution and raw provenance PASS", "execution_validity": "PASS", "numeric_status": status, "numeric": {"osi": float(cell["osi"]), "margin": 0.05, "classification": "ORDER_INVARIANT_WITHIN_MARGIN" if status == "PASS" else "ORDER_SENSITIVE"}, "raw_recomputable": True})
        expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
        if keys != expected or pass_count != 11 or fail_count != 9 or audit.get("numeric_acceptance", {}).get("n_pass") != 11 or audit.get("numeric_acceptance", {}).get("n_fail") != 9 or audit.get("numeric_acceptance", {}).get("n_unresolved") != 0:
            raise EvidenceValidationError("H5 terminal/numeric counts do not cover the frozen 20-cell matrix")
        return rows, {"status": "PARTIAL", "source": str(path), "audit_source": str(audit_path), "accepted_cells": pass_count, "execution_valid_cells": 20, "numeric_fail_cells": fail_count, "target_cells": 20, "decision": doc.get("decision"), "execution_validity": "PASS", "audit_verdict": audit.get("verdict")}
    except (EvidenceValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return _empty_staligner(source, str(exc))


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _precast_missing(registry: Mapping[str, Any]) -> dict[str, Any]:
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    present = {(row.get("condition"), int(row.get("seed"))) for row in registry.get("ledger", []) if row.get("method") == "PRECAST" and row.get("status") == "PASS"}
    missing = sorted(expected - present)
    return {"target_cells": 20, "independent_pass_cells": len(present), "missing_independent_keys": [f"{condition}:seed{seed}" for condition, seed in missing], "valid_elsewhere": [{"path": "docs/PRECAST_R2_DISEASE_CELL_ACCEPTANCE_20260830.json", "same_key": "disease_only_no_batch:seed20260819", "role": "valid historical receipt for the missing key; same-seed replay"}, {"path": "docs/PRECAST_GAP4_FINAL_ACCEPTANCE_20260830_r1.json", "same_key": "disease_only_no_batch:seed20260819", "role": "valid current terminal receipt; duplicate replay"}], "replay_excluded": True, "reason": "The only missing independent key is disease seed20260819; both available receipts are the same P2R key and cannot be counted twice."}


def build_registry(project_root: Path | str = PROJECT_DEFAULT, *, staligner_acceptance: Path | str | None = None, precast_acceptance: Path | str | None = None, pca_acceptance: Path | str | None = None, graphst_anchor: Path | str | None = _v5.GRAPHST_ANCHOR_DEFAULT, split_manifest: Path | str | None = None, source_snapshot: Path | str | None = None, checked_at: str | None = None) -> dict[str, Any]:
    root = Path(project_root); when = checked_at or _datetime.datetime.now().astimezone().isoformat()
    base = _v5.build_registry(root, precast_acceptance=precast_acceptance, pca_acceptance=pca_acceptance, graphst_anchor=graphst_anchor, split_manifest=split_manifest, source_snapshot=source_snapshot, checked_at=when)
    st_rows, st_source = ingest_staligner(root, staligner_acceptance)
    ledger = [row for row in base["ledger"] if row.get("method") != "STAligner"] + st_rows
    ledger.sort(key=lambda row: (METHODS.index(row["method"]), CONDITIONS.index(row["condition"]), int(row["seed"])))
    coverage = {method: {status: sum(row.get("method") == method and row.get("status") == status for row in ledger) for status in STATUSES} for method in METHODS}
    method_status = dict(base.get("method_status", {})); method_status["STAligner"] = {"status": "PARTIAL" if coverage["STAligner"]["PASS"] else "UNRESOLVED", "accepted_cells": coverage["STAligner"]["PASS"], "numeric_fail_cells": coverage["STAligner"]["FAIL"], "execution_valid_cells": st_source.get("execution_valid_cells", 0), "target_cells": 20, "execution_status": st_source.get("execution_validity", "UNRESOLVED")}
    result = {"schema_version": SCHEMA_VERSION, "checked_at": when, "project_root": str(root), "contract": base["contract"], "sources": {**base.get("sources", {}), "STAligner": st_source}, "ledger": ledger, "coverage": coverage, "method_status": method_status, "axes": base["axes"], "split": base["split"], "graphst_anchor": base.get("graphst_anchor"), "pca_source": base.get("pca_source"), "precast_source": base.get("precast_source"), "staligner_source": st_source, "precast_unique": _precast_missing(base), "source_snapshots": base.get("source_snapshots"), "claim_boundary": ["Each method's execution/provenance and numeric behavior remain separate; STAligner H5 contributes 20 execution-valid cells with 11 numeric PASS and 9 numeric FAIL.", "PRECAST disease seed20260819 is a valid same-seed replay available in two receipts and is excluded from the independent 20-cell aggregate.", "Harmony remains unresolved pending the independent H7 audit; no producer COMPLETE state is promoted to PASS.", "No external validation, framework-wide success, causal disease/batch claim, or universal 0.05 threshold claim is licensed."]}
    return result


def write_registry(registry: Mapping[str, Any], output_dir: Path | str, *, filename: str = "evidence_registry_v6_20260831.json") -> Path:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def write_bound_manifest(registry: Mapping[str, Any], registry_path: Path, output_dir: Path | str, *, filename: str = "U8_BOUND_MANIFEST_20260831.json") -> Path:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); path = destination / filename
    candidates = [registry.get("staligner_source", {}).get("source"), registry.get("staligner_source", {}).get("audit_source"), registry.get("pca_source", {}).get("source"), registry.get("precast_source", {}).get("source"), registry.get("graphst_anchor", {}).get("anchor"), registry.get("sources", {}).get("INSPIRE", {}).get("source")]
    inputs = []
    for value in candidates:
        if isinstance(value, str):
            source = Path(value)
            if source.is_file():
                inputs.append(_fingerprint(source))
    project_root = Path(__file__).resolve().parents[2]
    implementation = [_fingerprint(Path(__file__).resolve())]
    test_path = project_root / "tests" / "test_evidence_registry_v6_20260831.py"
    if test_path.is_file():
        implementation.append(_fingerprint(test_path))
    payload = {"schema_version": BOUND_MANIFEST_SCHEMA, "checked_at": registry.get("checked_at"), "registry": _fingerprint(registry_path), "implementation": implementation, "inputs": inputs, "coverage": registry.get("coverage"), "precast_unique": registry.get("precast_unique"), "claim_boundary": registry.get("claim_boundary")}
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Six-method STIA evidence consolidation v6")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--staligner-acceptance", type=Path, default=H5_ACCEPTANCE_DEFAULT)
    parser.add_argument("--precast-acceptance", type=Path)
    parser.add_argument("--pca-acceptance", type=Path)
    parser.add_argument("--graphst-anchor", type=Path, default=_v5.GRAPHST_ANCHOR_DEFAULT)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="evidence_registry_v6_20260831.json")
    parser.add_argument("--bound-manifest-filename", default="U8_BOUND_MANIFEST_20260831.json")
    args = parser.parse_args(argv)
    registry = build_registry(args.project_root, staligner_acceptance=args.staligner_acceptance, precast_acceptance=args.precast_acceptance, pca_acceptance=args.pca_acceptance, graphst_anchor=args.graphst_anchor, split_manifest=args.split_manifest, source_snapshot=args.source_snapshot)
    output = None; bound = None
    if args.output_dir:
        output = write_registry(registry, args.output_dir, filename=args.filename)
        bound = write_bound_manifest(registry, output, args.output_dir, filename=args.bound_manifest_filename)
    print(json.dumps({"status": "READY", "schema_version": registry["schema_version"], "ledger_cells": len(registry["ledger"]), "coverage": registry["coverage"], "method_status": registry["method_status"], "staligner_source": registry["staligner_source"], "precast_unique": registry["precast_unique"], "output": str(output) if output else None, "bound_manifest": str(bound) if bound else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
