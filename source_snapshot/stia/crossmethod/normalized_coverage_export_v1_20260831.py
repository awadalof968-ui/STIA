"""Uniform coverage export for the accepted U9 registry.

This is a presentation/export layer, not a second registry.  It keeps
execution validity, applicability and metric outcomes in separate columns and
expands INSPIRE's four measured representations into long-form component rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import evidence_registry_v7_20260831 as _v7

PROJECT_DEFAULT = _v7.PROJECT_DEFAULT
REGISTRY_DEFAULT = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/evidence/U9/evidence_registry_v7_20260831_r2.json")
BOUND_MANIFEST_DEFAULT = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/evidence/U9/U9_BOUND_MANIFEST_20260831_r2.json")
SCHEMA_VERSION = "stia.normalized_coverage.v1.20260831"
FIELDS = ("cell_key", "metric_row_key", "method", "condition", "seed", "applicability", "execution_validity", "metric_available", "metric_name", "metric_component", "metric_scale", "osi", "numeric_margin", "numeric_classification", "source_receipt", "known_limitations")


class ExportValidationError(ValueError):
    """Raised when the source registry or bound manifest is stale/incomplete."""


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExportValidationError(f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExportValidationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_u9_binding(project_root: Path | str, registry_path: Path | str, manifest_path: Path | str) -> dict[str, Any]:
    """Recheck the U9 registry, all manifest inputs, source snapshot and replay fixture."""
    root = Path(project_root); registry_file = _resolve(root, registry_path); manifest_file = _resolve(root, manifest_path)
    if registry_file is None or manifest_file is None:
        raise ExportValidationError("registry/manifest path is not resolvable")
    registry = _json(registry_file); manifest = _json(manifest_file)
    if registry.get("schema_version") != "stia.evidence_registry.v7.20260831" or manifest.get("schema_version") != "stia.evidence_registry.v7.bound_manifest.v1":
        raise ExportValidationError("U9 registry/manifest schema is not v7")
    declared_registry = _resolve(root, (manifest.get("registry") or {}).get("path")) if isinstance(manifest.get("registry"), Mapping) else None
    declared_hash = (manifest.get("registry") or {}).get("sha256") if isinstance(manifest.get("registry"), Mapping) else None
    if declared_registry is None or declared_registry.resolve() != registry_file.resolve() or not isinstance(declared_hash, str) or _sha256(registry_file) != declared_hash:
        raise ExportValidationError("U9 registry binding is stale")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 11:
        raise ExportValidationError("U9 manifest does not bind the expected 11 current inputs")
    for item in inputs:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ExportValidationError("U9 manifest input row is malformed")
        path = _resolve(root, item["path"])
        if path is None or not path.is_file() or _sha256(path) != item["sha256"] or (item.get("bytes") is not None and path.stat().st_size != item["bytes"]):
            raise ExportValidationError(f"U9 manifest input drift: {item.get('path')}")
    if registry.get("source_snapshots", {}).get("status") != "PASS":
        raise ExportValidationError("U9 source snapshot is not PASS")
    replay = registry.get("precast_replay")
    if not isinstance(replay, Mapping) or replay.get("status") != "PASS" or replay.get("covered_key_count") != 1 or replay.get("duplicate_alias_count") != 1:
        raise ExportValidationError("U9 canonical replay summary is invalid")
    replay_fixture = {"record": {"condition": "disease_only_no_batch", "seed": 20260819, "input_sha256": "a" * 64, "panel_sha256": "b" * 64, "source_sha256": "c" * 64, "r_core_sha256": "d" * 64, "runner_sha256": "e" * 64, "continuation_sha256": "f" * 64, "implementation_sha256": "1" * 64, "margin": 0.05, "config": {"K": 20, "q": 15, "beta_grid": 1, "coreNum": 1, "coreNum_int": 1, "seed": 20260819}, "arms": {order: {"retained_count": 14999, "retained_ids_unordered_sha256": "2" * 64, "donor_keys_ordered_sha256": "3" * 64, "block_sizes_in": [1500, 1499], "block_sizes_out": [1500, 1499]} for order in _v7.ORDERS}}, "cell": {"independent_source_retention": {"retained_count": 14999, "retained_ids_unordered_sha256": "2" * 64, "donor_levels_first_occurrence": ["AD0", "CT0"]}}, "record_path": "fixture-p2.json"}
    current_fixture = json.loads(json.dumps(replay_fixture)); current_fixture["record_path"] = "fixture-gap4.json"
    fixture_result = _v7.select_canonical_replay(replay_fixture, current_fixture)
    if fixture_result.get("covered_key_count") != 1 or fixture_result.get("duplicate_alias_count") != 1 or fixture_result.get("canonical") != "current_gap4":
        raise ExportValidationError("canonical replay fixture did not produce one representative")
    return {"status": "PASS", "registry": str(registry_file), "registry_sha256": _sha256(registry_file), "manifest": str(manifest_file), "manifest_sha256": _sha256(manifest_file), "inputs_checked": len(inputs), "source_snapshot_status": registry.get("source_snapshots", {}).get("status"), "canonical_replay_fixture": fixture_result}


def _limitations(row: Mapping[str, Any], component: str | None) -> str:
    method = str(row.get("method")); limits: list[str] = []
    if method == "PRECAST":
        limits.append("receipt PASS is execution/provenance validity; OSI threshold outcome is separate")
        if row.get("canonical_representative"):
            limits.append("current gap4 disease seed19 is canonical; P2R same-key replay is an excluded alias")
    elif method == "STAligner":
        limits.append("numeric FAIL is distinct from execution-valid raw evidence")
    elif method == "Harmony":
        limits.append("single-batch cells are explicit NOT_APPLICABLE, not stability successes")
        limits.append("H4 Elkan compatibility fallback is not numerically identical to Lloyd")
    elif method == "INSPIRE":
        limits.append("representations remain separate long-form components; no pooling")
    elif method == "GraphST":
        limits.append("historical continuity remains UNKNOWN")
    elif method == "PCA":
        limits.append("full-SVD seed is unused; deterministic replicates are not stochastic")
    if component:
        limits.append(f"component={component}")
    return "; ".join(limits)


def _component_values(row: Mapping[str, Any]) -> list[tuple[str | None, Mapping[str, Any] | None]]:
    numeric = row.get("numeric")
    if not isinstance(numeric, Mapping):
        return [(None, None)]
    components = [name for name in ("latent", "shared", "gamma", "full") if isinstance(numeric.get(name), Mapping)]
    if components:
        return [(name, numeric[name]) for name in components]
    return [("primary", numeric)]


def normalize_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project v7 rows into uniform cell/component records without pooling metrics."""
    if registry.get("schema_version") != "stia.evidence_registry.v7.20260831" or not isinstance(registry.get("ledger"), list):
        raise ExportValidationError("registry is not a v7 ledger")
    rows: list[dict[str, Any]] = []; cell_keys: set[str] = set()
    for row in registry["ledger"]:
        if not isinstance(row, Mapping) or row.get("method") not in _v7.METHODS or row.get("condition") not in _v7.CONDITIONS or not isinstance(row.get("seed"), int) or row.get("seed") not in _v7.SEEDS:
            raise ExportValidationError("registry ledger row identity is malformed")
        method = str(row["method"]); condition = str(row["condition"]); seed = int(row["seed"]); cell_key = f"{method}:{condition}:seed{seed}"; cell_keys.add(cell_key)
        applicability = "NOT_APPLICABLE" if method == "Harmony" and row.get("status") == "NOT_APPLICABLE" else ("UNKNOWN" if row.get("status") == "UNRESOLVED" else "APPLICABLE")
        execution = row.get("execution_validity") if row.get("execution_validity") in {"PASS", "FAIL", "UNRESOLVED"} else ("UNRESOLVED" if row.get("status") == "UNRESOLVED" else "PASS")
        source = row.get("source")
        for component, numeric in _component_values(row):
            metric_available = isinstance(numeric, Mapping) and isinstance(numeric.get("osi"), (int, float)) and not isinstance(numeric.get("osi"), bool)
            metric_name = "OSI" if metric_available else None
            metric_scale = "max_pair_drift" if metric_available else None
            osi = float(numeric["osi"]) if metric_available else None
            margin = numeric.get("margin") if metric_available and isinstance(numeric, Mapping) else None
            if metric_available and margin is None:
                margin = registry.get("contract", {}).get("margin")
            classification = numeric.get("classification") if metric_available and isinstance(numeric, Mapping) else ("NOT_APPLICABLE" if applicability == "NOT_APPLICABLE" else "UNRESOLVED")
            metric_row_key = f"{cell_key}:{component or 'none'}"
            rows.append({"cell_key": cell_key, "metric_row_key": metric_row_key, "method": method, "condition": condition, "seed": seed, "applicability": applicability, "execution_validity": execution, "metric_available": bool(metric_available), "metric_name": metric_name, "metric_component": component, "metric_scale": metric_scale, "osi": osi, "numeric_margin": float(margin) if isinstance(margin, (int, float)) and not isinstance(margin, bool) else None, "numeric_classification": classification, "source_receipt": str(source) if source is not None else None, "known_limitations": _limitations(row, component)})
    if len(cell_keys) != 120 or len(rows) != 180:
        raise ExportValidationError(f"normalized output identity/count mismatch: cells={len(cell_keys)} rows={len(rows)}")
    if len({row["metric_row_key"] for row in rows}) != len(rows):
        raise ExportValidationError("normalized metric row keys are duplicated")
    return rows


def coverage_draft(rows: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method in _v7.METHODS:
        method_rows = [row for row in rows if row.get("method") == method]; cells = {row["cell_key"] for row in method_rows}; applicable = {row["cell_key"] for row in method_rows if row.get("applicability") == "APPLICABLE"}; na = {row["cell_key"] for row in method_rows if row.get("applicability") == "NOT_APPLICABLE"}; execution = {row["cell_key"] for row in method_rows if row.get("execution_validity") == "PASS"}; metric_cells = {row["cell_key"] for row in method_rows if row.get("metric_available") is True}; classes: dict[str, int] = {}
        for row in method_rows:
            cls = str(row.get("numeric_classification")); classes[cls] = classes.get(cls, 0) + 1
        methods[method] = {"distinct_cell_keys": len(cells), "applicable_cells": len(applicable), "not_applicable_cells": len(na), "execution_valid_cells": len(execution), "metric_available_cells": len(metric_cells), "metric_rows": len(method_rows), "numeric_classification_counts": classes}
    all_cells = {row["cell_key"] for row in rows}; applicable_total = sum(item["applicable_cells"] for item in methods.values()); na_total = sum(item["not_applicable_cells"] for item in methods.values())
    return {"schema_version": "stia.normalized_coverage_acceptance_draft.v1.20260831", "binding": dict(binding), "distinct_cell_keys": len(all_cells), "normalized_rows": len(rows), "applicable_cells": applicable_total, "not_applicable_cells": na_total, "coverage_is_separate_from_numeric_outcomes": True, "methods": methods, "claim_boundary": ["110 applicable and 10 explicit NOT_APPLICABLE cells are binding-derived across the six method ledgers; this is coverage, not a pooled ranking.", "INSPIRE latent/shared/gamma/full OSIs remain separate component rows; no favourable component is selected.", "Numeric classifications and thresholds are method-specific observations; no universal cutoff or framework/FPR claim is licensed."]}


def write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    return path


def build_export(project_root: Path | str = PROJECT_DEFAULT, registry_path: Path | str = REGISTRY_DEFAULT, manifest_path: Path | str = BOUND_MANIFEST_DEFAULT) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root); binding = verify_u9_binding(root, registry_path, manifest_path); reg_file = _resolve(root, registry_path)
    if reg_file is None:
        raise ExportValidationError("registry path is not resolvable")
    registry = _json(reg_file); rows = normalize_registry(registry); draft = coverage_draft(rows, binding); return {"schema_version": SCHEMA_VERSION, "binding": binding, "coverage": draft, "rows": rows}, rows, draft


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uniform U9 coverage/validity/numeric export")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--registry", type=Path, default=REGISTRY_DEFAULT)
    parser.add_argument("--bound-manifest", type=Path, default=BOUND_MANIFEST_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-filename", default="normalized_coverage_v1_20260831.json")
    parser.add_argument("--csv-filename", default="normalized_coverage_v1_20260831.csv")
    parser.add_argument("--coverage-filename", default="U10_COVERAGE_ACCEPTANCE_DRAFT_20260831.json")
    args = parser.parse_args(argv)
    payload, rows, draft = build_export(args.project_root, args.registry, args.bound_manifest)
    write_json(payload, args.output_dir / args.json_filename); write_csv(rows, args.output_dir / args.csv_filename); write_json(draft, args.output_dir / args.coverage_filename)
    print(json.dumps({"status": "READY", "schema_version": SCHEMA_VERSION, "normalized_rows": len(rows), "distinct_cell_keys": draft["distinct_cell_keys"], "applicable_cells": draft["applicable_cells"], "not_applicable_cells": draft["not_applicable_cells"], "binding": payload["binding"], "output_dir": str(args.output_dir)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
