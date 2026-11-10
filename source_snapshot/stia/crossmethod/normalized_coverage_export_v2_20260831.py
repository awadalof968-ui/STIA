"""Fail-closed successor to the U10 normalized coverage exporter.

The v2 helper keeps the U9 registry unchanged while refusing to infer
execution validity from an arbitrary status string.  Numeric fields are
published only for applicable cells with explicit execution evidence and
finite OSI/margin values.
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import normalized_coverage_export_v1_20260831 as _v1
from . import evidence_registry_v7_20260831 as _v7

PROJECT_DEFAULT = _v1.PROJECT_DEFAULT
REGISTRY_DEFAULT = _v1.REGISTRY_DEFAULT
BOUND_MANIFEST_DEFAULT = _v1.BOUND_MANIFEST_DEFAULT
SCHEMA_VERSION = "stia.normalized_coverage.v2.20260831"


class ExportValidationError(ValueError):
    """Raised when normalized evidence cannot be proven safe to publish."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExportValidationError(f"JSON root must be an object: {path}")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


_EXPLICIT_RECEIPT_TIERS = {"current_independent_anchor", "accepted_raw_receipt", "raw_recomputable", "accepted_final_raw_receipt", "canonical_current_final_raw_receipt", "accepted_harmony_raw_receipt", "accepted_harmony_explicit_na", "accepted_h5_raw_receipt"}


def _execution_validity(row: Mapping[str, Any]) -> str:
    explicit = row.get("execution_validity")
    if explicit in {"PASS", "FAIL", "UNRESOLVED"}:
        return str(explicit)
    if row.get("status") == "PASS" and row.get("evidence_tier") in _EXPLICIT_RECEIPT_TIERS and isinstance(row.get("source"), str) and row["source"]:
        return "PASS"
    return "UNRESOLVED"


def _components(row: Mapping[str, Any]) -> list[tuple[str | None, Mapping[str, Any] | None]]:
    numeric = row.get("numeric")
    if numeric is None:
        return [(None, None)]
    if not isinstance(numeric, Mapping):
        raise ExportValidationError("numeric field is not an object")
    names = [name for name in ("latent", "shared", "gamma", "full") if isinstance(numeric.get(name), Mapping)]
    return [(name, numeric[name]) for name in names] if names else [("primary", numeric)]


def normalize_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize v7 rows with conservative execution/metric semantics."""
    if registry.get("schema_version") != "stia.evidence_registry.v7.20260831" or not isinstance(registry.get("ledger"), list):
        raise ExportValidationError("registry is not a v7 ledger")
    rows: list[dict[str, Any]] = []; cell_keys: set[str] = set()
    for source_row in registry["ledger"]:
        if not isinstance(source_row, Mapping) or source_row.get("method") not in _v7.METHODS or source_row.get("condition") not in _v7.CONDITIONS or not isinstance(source_row.get("seed"), int) or source_row.get("seed") not in _v7.SEEDS:
            raise ExportValidationError("registry ledger row identity is malformed")
        method = str(source_row["method"]); condition = str(source_row["condition"]); seed = int(source_row["seed"]); cell_key = f"{method}:{condition}:seed{seed}"; cell_keys.add(cell_key)
        execution = _execution_validity(source_row)
        if method == "Harmony" and source_row.get("status") == "NOT_APPLICABLE":
            applicability = "NOT_APPLICABLE"
        elif source_row.get("status") in {"PASS", "FAIL"} and execution == "PASS":
            applicability = "APPLICABLE"
        else:
            applicability = "UNKNOWN"
        numeric_obj = source_row.get("numeric")
        if applicability == "NOT_APPLICABLE" and numeric_obj is not None:
            raise ExportValidationError(f"explicit NOT_APPLICABLE cell carries a metric: {cell_key}")
        for component, numeric in _components(source_row):
            has_osi = isinstance(numeric, Mapping) and "osi" in numeric
            if has_osi and not _finite(numeric.get("osi")):
                raise ExportValidationError(f"non-finite OSI in {cell_key}:{component or 'none'}")
            has_margin = isinstance(numeric, Mapping) and "margin" in numeric
            if has_margin and not _finite(numeric.get("margin")):
                raise ExportValidationError(f"non-finite numeric margin in {cell_key}:{component or 'none'}")
            metric_available = applicability == "APPLICABLE" and execution == "PASS" and has_osi and _finite(numeric.get("osi"))
            metric_name = "OSI" if metric_available else None
            metric_scale = "max_pair_drift" if metric_available else None
            osi = float(numeric["osi"]) if metric_available else None
            margin_value = numeric.get("margin") if metric_available else None
            if metric_available and margin_value is None:
                margin_value = registry.get("contract", {}).get("margin")
            if metric_available and not _finite(margin_value):
                raise ExportValidationError(f"numeric margin is missing/non-finite in {cell_key}:{component or 'none'}")
            classification = numeric.get("classification") if metric_available else ("NOT_APPLICABLE" if applicability == "NOT_APPLICABLE" else "UNRESOLVED")
            metric_row_key = f"{cell_key}:{component or 'none'}"
            rows.append({"cell_key": cell_key, "metric_row_key": metric_row_key, "method": method, "condition": condition, "seed": seed, "applicability": applicability, "execution_validity": execution, "metric_available": bool(metric_available), "metric_name": metric_name, "metric_component": component, "metric_scale": metric_scale, "osi": osi, "numeric_margin": float(margin_value) if metric_available else None, "numeric_classification": classification, "source_receipt": str(source_row.get("source")) if source_row.get("source") is not None else None, "known_limitations": _v1._limitations(source_row, component)})
    if len(cell_keys) != 120 or len(rows) != 180:
        raise ExportValidationError(f"normalized output identity/count mismatch: cells={len(cell_keys)} rows={len(rows)}")
    if len({row["metric_row_key"] for row in rows}) != len(rows):
        raise ExportValidationError("normalized metric row keys are duplicated")
    return rows


def build_export(project_root: Path | str = PROJECT_DEFAULT, registry_path: Path | str = REGISTRY_DEFAULT, manifest_path: Path | str = BOUND_MANIFEST_DEFAULT) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root); binding = _v1.verify_u9_binding(root, registry_path, manifest_path); reg_file = _v1._resolve(root, registry_path)
    if reg_file is None:
        raise ExportValidationError("registry path is not resolvable")
    registry = _json(reg_file); rows = normalize_registry(registry); draft = _v1.coverage_draft(rows, binding); draft["schema_version"] = "stia.normalized_coverage_acceptance_draft.v2.20260831"; draft["execution_semantics"] = "explicit_execution_validity_or_accepted_receipt_evidence; unknown_status_fails_closed"; return {"schema_version": SCHEMA_VERSION, "binding": binding, "coverage": draft, "rows": rows}, rows, draft


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed U10R normalized coverage export")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--registry", type=Path, default=REGISTRY_DEFAULT)
    parser.add_argument("--bound-manifest", type=Path, default=BOUND_MANIFEST_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-filename", default="normalized_coverage_v2_20260831.json")
    parser.add_argument("--csv-filename", default="normalized_coverage_v2_20260831.csv")
    parser.add_argument("--coverage-filename", default="U10R_COVERAGE_ACCEPTANCE_DRAFT_20260831.json")
    args = parser.parse_args(argv)
    payload, rows, draft = build_export(args.project_root, args.registry, args.bound_manifest); _v1.write_json(payload, args.output_dir / args.json_filename); _v1.write_csv(rows, args.output_dir / args.csv_filename); _v1.write_json(draft, args.output_dir / args.coverage_filename)
    print(json.dumps({"status": "READY", "schema_version": SCHEMA_VERSION, "normalized_rows": len(rows), "distinct_cell_keys": draft["distinct_cell_keys"], "applicable_cells": draft["applicable_cells"], "not_applicable_cells": draft["not_applicable_cells"], "binding": payload["binding"], "output_dir": str(args.output_dir)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
