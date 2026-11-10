"""Explicit-receipt six-method registry (U6).

This successor keeps the v3 GraphST current-anchor contract and adds strict,
read-only ingestion for the accepted PCA r3 receipt and the completed cells in
the PRECAST gap receipt.  It never fits, loads expression matrices, chooses a
receipt by timestamp, or overwrites an output.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import evidence_registry_v2_20260830 as _v2
from . import evidence_registry_v3_20260830 as _v3

METHODS = _v3.METHODS
CONDITIONS = _v3.CONDITIONS
SEEDS = _v3.SEEDS
ORDERS = _v3.ORDERS
STATUSES = _v3.STATUSES
MARGIN = _v3.MARGIN
PROJECT_DEFAULT = _v3.PROJECT_DEFAULT
PCA_PROTOCOL = "PCA_STRICT_E2_20260830.r3"
PCA_ACCEPTANCE_DEFAULT = Path("docs/PCA_STRICT_R3_FINAL_ACCEPTANCE_20260830_r1.json")
PRECAST_ACCEPTANCE_DEFAULT = Path("docs/PRECAST_GAP4_PARTIAL_ACCEPTANCE_20260830_r3.json")
GRAPHST_ANCHOR_DEFAULT = Path(
    "Lunacy/runs/precast-null-inspire-seeds-20260830/evidence/U5/GRAPHST_CURRENT_ANCHOR_20260830.json"
)
SOURCE_SNAPSHOT_DEFAULT = Path(
    "Lunacy/runs/precast-null-inspire-seeds-20260830/evidence/U6/source_snapshots/V3_FREEZE_20260830.json"
)


class EvidenceValidationError(ValueError):
    """Raised when a supplied receipt cannot bind current bytes."""


def _now() -> str:
    return _datetime.datetime.now().astimezone().isoformat()


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


def _resolve(root: Path, value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_hash(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _cell(method: str, condition: str, seed: int, *, status: str, tier: str,
          source: str | None, reason: str, **extra: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(status)
    row = {"method": method, "condition": condition, "seed": int(seed), "status": status,
           "evidence_tier": tier, "source": source, "reason": reason}
    row.update(extra)
    return row


def _all(method: str, reason: str, source: str | None = None) -> list[dict[str, Any]]:
    return [_cell(method, condition, seed, status="UNRESOLVED", tier="receipt_missing_or_invalid",
                  source=source, reason=reason) for condition in CONDITIONS for seed in SEEDS]


def _acceptance_path(root: Path, value: Path | str | None, default: Path) -> Path:
    path = _resolve(root, value if value is not None else default)
    if path is None:
        raise EvidenceValidationError(f"receipt path is not resolvable: {value or default}")
    return path


def _check_receipt_file(path: Path, expected_hash: Any, label: str) -> None:
    if not path.is_file() or not _valid_hash(expected_hash) or _sha256(path) != expected_hash:
        raise EvidenceValidationError(f"{label} path/hash mismatch")


def _validate_source_closure(root: Path, closure: Any, label: str) -> None:
    """Verify an optional producer source-closure ledger against current bytes."""
    if closure is None:
        return
    if not isinstance(closure, Mapping) or not closure:
        raise EvidenceValidationError(f"{label} source closure is malformed")
    for name, entry in closure.items():
        if not isinstance(entry, Mapping) or not _valid_hash(entry.get("sha256")):
            raise EvidenceValidationError(f"{label} source closure row is malformed: {name}")
        path = _resolve(root, entry.get("path"))
        if path is None or not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise EvidenceValidationError(f"{label} source closure drift: {name}")


def _pca_identity(doc: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[int]]:
    if doc.get("schema_version") != "inspire.pca.strict_acceptance.v1" or doc.get("acceptance_decision") != "PASS":
        raise EvidenceValidationError("PCA strict acceptance envelope is not PASS")
    identity = doc.get("experiment_identity")
    if not isinstance(identity, Mapping) or identity.get("method") != "PCA" or identity.get("protocol") != PCA_PROTOCOL:
        raise EvidenceValidationError("PCA identity/protocol binding is invalid")
    conditions = list(identity.get("conditions", [])); seeds = [int(s) for s in identity.get("seed_labels", [])]
    if conditions != list(CONDITIONS) or seeds != list(SEEDS) or identity.get("pca_dimensions") != 30:
        raise EvidenceValidationError("PCA condition/seed/dimension contract mismatch")
    policy = identity.get("model_seed_policy")
    if not isinstance(policy, Mapping) or policy.get("behavior") != "deterministic_full_svd_seed_unused" or policy.get("model_seed") is not None or policy.get("replicates_are_stochastic") is not False or policy.get("requested_seed_field") != "requested_seed":
        raise EvidenceValidationError("PCA deterministic model-seed policy is missing or altered")
    checks = doc.get("completion_checks")
    if not isinstance(checks, Mapping) or checks.get("completed_cells") != 20 or checks.get("raw_npz") != 80 or checks.get("independent_audit_exit_code") != 0 or checks.get("independent_audit_unresolved") != 0 or checks.get("terminal_summary_consistent") is not True or checks.get("current_actual_source_path_and_hash_binding") is not True:
        raise EvidenceValidationError("PCA acceptance counts/source binding are incomplete")
    return dict(identity), conditions, seeds


def _validate_pca_arm(root: Path, output_root: Path, ref: Mapping[str, Any], *, condition: str, seed: int, expected_hash: Mapping[str, str]) -> dict[str, Any]:
    json_path = _resolve(root, ref.get("json"))
    npz_path = _resolve(root, ref.get("npz"))
    if json_path is None or npz_path is None or not _within(json_path, output_root) or not _within(npz_path, output_root):
        raise EvidenceValidationError("PCA arm path escapes output root")
    rel_json = json_path.relative_to(output_root).as_posix(); rel_npz = npz_path.relative_to(output_root).as_posix()
    if expected_hash.get(rel_json) != _sha256(json_path) or expected_hash.get(rel_npz) != _sha256(npz_path):
        raise EvidenceValidationError("PCA arm receipt hash mismatch")
    arm = _json(json_path)
    if arm.get("arm") not in ORDERS or arm.get("arm") != Path(str(ref.get("json", ""))).name.removesuffix(".json"):
        raise EvidenceValidationError("PCA arm/order binding is invalid")
    if arm.get("condition") != condition or int(arm.get("seed", -1)) != seed or int(arm.get("requested_seed", -1)) != seed:
        raise EvidenceValidationError("PCA arm condition/seed binding is invalid")
    if arm.get("strict_protocol_version") != PCA_PROTOCOL or arm.get("pca_dim") != 30 or arm.get("model_seed") is not None:
        raise EvidenceValidationError("PCA arm protocol/dimension/model seed is invalid")
    if arm.get("npz_sha256") != _sha256(npz_path) or arm.get("npz") not in {str(npz_path), str(npz_path.resolve())}:
        raise EvidenceValidationError("PCA arm NPZ metadata is not bound")
    ok, detail = _v2._npz_check(npz_path, arm.get("npz_sha256"))
    if not ok:
        raise EvidenceValidationError(detail)
    shape = arm.get("latent_shape")
    if not isinstance(shape, list) or len(shape) != 2 or shape[1] != 30 or not isinstance(shape[0], int) or shape[0] <= 0 or arm.get("retained_n") != shape[0]:
        raise EvidenceValidationError("PCA arm shape/retained count is not native-30 bound")
    expected_identity = arm.get("source_expected"); actual_identity = arm.get("source_identity")
    if not isinstance(expected_identity, Mapping) or not isinstance(actual_identity, Mapping):
        raise EvidenceValidationError("PCA arm source identity is missing")
    _validate_source_closure(root, arm.get("source_closure"), "PCA arm")
    if not all(_valid_hash(expected_identity.get(key)) for key in ("donor_map_sha256", "ordered_ids_sha256", "retained_ids_set_sha256")) or not _valid_hash(actual_identity.get("retained_donor_map_sha256")) or expected_identity.get("retained_ids_set_sha256") != actual_identity.get("retained_ids_set_sha256") or expected_identity.get("retained_n") != arm.get("retained_n"):
        raise EvidenceValidationError("PCA arm source identity mismatch")
    return {"arm": arm, "json": str(json_path), "npz": str(npz_path), "npz_sha256": arm["npz_sha256"],
            "retained_ids_set_sha256": actual_identity["retained_ids_set_sha256"],
            "input_sha256": arm.get("input_h5ad_sha256"), "panel_sha256": arm.get("panel_sha256"),
            "source_sha256": arm.get("source_sha256")}


def ingest_pca(project_root: Path | str, acceptance: Path | str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root); source = str(acceptance or PCA_ACCEPTANCE_DEFAULT)
    try:
        acceptance_path = _acceptance_path(root, acceptance, PCA_ACCEPTANCE_DEFAULT)
        doc = _json(acceptance_path); identity, _, _ = _pca_identity(doc)
        inventory = doc.get("evidence_inventory")
        if not isinstance(inventory, Mapping):
            raise EvidenceValidationError("PCA evidence inventory is missing")
        for key in ("plan", "raw_audit"):
            path = _resolve(root, inventory.get(key)); _check_receipt_file(path, inventory.get(f"{key}_sha256"), f"PCA {key}") if path else (_ for _ in ()).throw(EvidenceValidationError(f"PCA {key} path missing"))
        output_root = _resolve(root, identity.get("output"))
        if output_root is None or not output_root.is_dir():
            raise EvidenceValidationError("PCA output root is missing")
        manifest = _json(output_root / "manifest.json")
        if manifest.get("method") != "PCA" or manifest.get("strict_protocol_version") != PCA_PROTOCOL or manifest.get("pca_dim") != 30 or manifest.get("output_dir") != str(output_root):
            raise EvidenceValidationError("PCA manifest identity/protocol/output binding is invalid")
        if manifest.get("conditions") != list(CONDITIONS) or [int(s) for s in manifest.get("seeds", [])] != list(SEEDS) or manifest.get("order_arms") != list(ORDERS):
            raise EvidenceValidationError("PCA manifest matrix contract mismatch")
        policy = manifest.get("model_seed_policy")
        if not isinstance(policy, Mapping) or policy.get("model_seed") is not None or policy.get("behavior") != "deterministic_full_svd_seed_unused":
            raise EvidenceValidationError("PCA manifest deterministic policy mismatch")
        manifest_panel = manifest.get("panel")
        if isinstance(manifest_panel, Mapping):
            panel_path = _resolve(root, manifest_panel.get("path"))
            _check_receipt_file(panel_path, manifest_panel.get("sha256"), "PCA manifest panel") if panel_path else (_ for _ in ()).throw(EvidenceValidationError("PCA manifest panel path missing"))
        manifest_source = manifest.get("source")
        if isinstance(manifest_source, Mapping):
            source_path = _resolve(root, manifest_source.get("path"))
            _check_receipt_file(source_path, manifest_source.get("sha256"), "PCA manifest source") if source_path else (_ for _ in ()).throw(EvidenceValidationError("PCA manifest source path missing"))
        manifest_inputs = manifest.get("condition_inputs")
        if isinstance(manifest_inputs, Mapping):
            for condition in CONDITIONS:
                item = manifest_inputs.get(condition)
                if not isinstance(item, Mapping):
                    raise EvidenceValidationError(f"PCA manifest condition input missing: {condition}")
                input_path = _resolve(root, item.get("h5ad")); sidecar_path = _resolve(root, item.get("sidecar"))
                _check_receipt_file(input_path, item.get("h5ad_sha256"), f"PCA {condition} input") if input_path else (_ for _ in ()).throw(EvidenceValidationError(f"PCA {condition} input path missing"))
                _check_receipt_file(sidecar_path, item.get("sidecar_sha256"), f"PCA {condition} sidecar") if sidecar_path else (_ for _ in ()).throw(EvidenceValidationError(f"PCA {condition} sidecar path missing"))
        output_files = inventory.get("output_files")
        if not isinstance(output_files, list) or len(output_files) < 182:
            raise EvidenceValidationError("PCA output inventory is incomplete")
        expected_hash: dict[str, str] = {}; expected_size: dict[str, int] = {}
        for item in output_files:
            if not isinstance(item, Mapping) or not isinstance(item.get("relative_path"), str) or not _valid_hash(item.get("sha256")):
                raise EvidenceValidationError("PCA output inventory row is malformed")
            rel = Path(item["relative_path"]); path = output_root / rel
            if not _within(path, output_root) or not path.is_file():
                raise EvidenceValidationError(f"PCA output inventory file missing: {item.get('relative_path')}")
            actual = _sha256(path)
            if actual != item["sha256"] or path.stat().st_size != item.get("bytes"):
                raise EvidenceValidationError(f"PCA output inventory drift: {item.get('relative_path')}")
            if rel.as_posix() in expected_hash:
                raise EvidenceValidationError("PCA output inventory contains duplicates")
            expected_hash[rel.as_posix()] = actual; expected_size[rel.as_posix()] = int(item["bytes"])
        summary_path = output_root / "run_summary.json"
        summary = _json(summary_path)
        if summary.get("n_cells") != 20 or summary.get("n_pass") != 20 or summary.get("n_fail") != 0 or summary.get("n_unresolved") != 0 or summary.get("strict_protocol_version") != PCA_PROTOCOL:
            raise EvidenceValidationError("PCA run summary is not terminal 20/20 PASS")
        rows: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            for seed in SEEDS:
                cell_path = output_root / "cells" / f"{condition}_seed{seed}.json"; rel = cell_path.relative_to(output_root).as_posix()
                if expected_hash.get(rel) != _sha256(cell_path):
                    raise EvidenceValidationError(f"PCA cell inventory missing/drifted: {rel}")
                cell = _json(cell_path)
                if cell.get("condition") != condition or int(cell.get("seed", -1)) != seed or cell.get("status") != "PASS" or cell.get("strict_protocol_version") != PCA_PROTOCOL:
                    raise EvidenceValidationError("PCA cell binding/status is invalid")
                refs = cell.get("arms")
                if not isinstance(refs, Mapping) or tuple(refs) != ORDERS:
                    raise EvidenceValidationError("PCA cell arm keys are incomplete")
                arm_rows: list[dict[str, Any]] = []
                for order in ORDERS:
                    checked = _validate_pca_arm(root, output_root, refs[order], condition=condition, seed=seed, expected_hash=expected_hash)
                    if checked["arm"].get("arm") != order:
                        raise EvidenceValidationError("PCA arm order mismatch")
                    arm_rows.append(checked)
                signatures = {(a["input_sha256"], a["panel_sha256"], a["source_sha256"], a["retained_ids_set_sha256"]) for a in arm_rows}
                if len(signatures) != 1:
                    raise EvidenceValidationError("PCA four arms do not share source/panel/identity")
                metric = cell.get("stored_metric")
                raw_drifts = metric.get("pairwise_drifts") if isinstance(metric, Mapping) else None
                if isinstance(raw_drifts, list) and raw_drifts and all(isinstance(v, Mapping) for v in raw_drifts):
                    raw_drifts = [v.get("drift") for v in raw_drifts]
                if not isinstance(metric, Mapping) or not _finite(metric.get("osi")) or not isinstance(raw_drifts, list) or len(raw_drifts) != 6 or not all(_finite(v) for v in raw_drifts):
                    raise EvidenceValidationError("PCA cell numeric summary is missing or non-finite")
                max_drift = max(float(v) for v in raw_drifts)
                classification = "ORDER_INVARIANT" if max_drift <= MARGIN else "ORDER_SENSITIVE"
                rows.append(_cell("PCA", condition, seed, status="PASS", tier="accepted_raw_receipt", source=str(cell_path), reason="strict r3 acceptance, independent 20-cell/80-NPZ receipt and current output hashes verified", raw_recomputable=True, retained_count=int(arm_rows[0]["arm"]["retained_n"]), retained_ids_unordered_sha256=arm_rows[0]["retained_ids_set_sha256"], numeric={"osi": float(metric["osi"]), "pairwise_drifts": [float(v) for v in raw_drifts], "classification": classification}, model_seed_policy=dict(policy), pca_dim=30, source_signature={"input_sha256": arm_rows[0]["input_sha256"], "panel_sha256": arm_rows[0]["panel_sha256"], "source_sha256": arm_rows[0]["source_sha256"]}))
        return rows, {"status": "PASS", "accepted_cells": 20, "target_cells": 20, "source": str(acceptance_path), "output_root": str(output_root), "protocol": PCA_PROTOCOL, "pca_dim": 30, "model_seed_policy": dict(policy)}
    except (EvidenceValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return _all("PCA", str(exc), source), {"status": "UNRESOLVED", "accepted_cells": 0, "target_cells": 20, "source": source, "reason": str(exc)}


def _precast_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("input_sha256", "panel_sha256", "source_sha256", "r_core_sha256", "runner_sha256", "continuation_sha256", "artifact_sha256", "implementation_sha256", "run_id", "plan_hash")}


def _validate_precast_current_cell(root: Path, cell_info: Mapping[str, Any], *, condition: str, seed: int, run_id: str, plan_hash: str) -> dict[str, Any]:
    if cell_info.get("status") != "PASS":
        raise EvidenceValidationError("PRECAST current cell is not PASS")
    cell_path = _resolve(root, cell_info.get("cell_json"))
    if cell_path is None:
        raise EvidenceValidationError("PRECAST current cell JSON is missing")
    record = _json(cell_path)
    _v2.validate_binding(record, method="PRECAST", condition=condition, seed=seed, run_id=run_id, plan_hash=plan_hash)
    if record.get("status") != "OK":
        raise EvidenceValidationError("PRECAST current cell JSON is not terminal OK")
    for name, provenance in (cell_info.get("bound_provenance") or {}).items():
        if not isinstance(provenance, Mapping) or provenance.get("status") != "PASS" or provenance.get("plan_match") is not True or not _valid_hash(provenance.get("expected_sha256")):
            raise EvidenceValidationError(f"PRECAST provenance is not PASS: {name}")
        bound = _resolve(root, provenance.get("bound_path"))
        if bound is None or _sha256(bound) != provenance.get("expected_sha256"):
            raise EvidenceValidationError(f"PRECAST provenance bytes drifted: {name}")
    checks = cell_info.get("cell_checks")
    orders = cell_info.get("order_checks")
    cross = cell_info.get("cross_arm")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()) or not isinstance(orders, Mapping) or not all(value is True for value in orders.values()) or not isinstance(cross, Mapping) or not all(value is True for value in (cross.get("checks") or {}).values()):
        raise EvidenceValidationError("PRECAST current cell checks are incomplete")
    arms = record.get("arms")
    if not isinstance(arms, Mapping) or tuple(arms) != ORDERS:
        raise EvidenceValidationError("PRECAST current arm keys are incomplete")
    arm_rows = []
    for order in ORDERS:
        info = arms[order]
        if not isinstance(info, Mapping) or info.get("status") != "OK":
            raise EvidenceValidationError(f"PRECAST {order} arm is not terminal OK")
        npz_path = _resolve(root, info.get("npz_path")); ok, detail = _v2._npz_check(npz_path, info.get("npz_sha256")) if npz_path else (False, "PRECAST raw NPZ is missing")
        if not ok:
            raise EvidenceValidationError(detail)
        arm_rows.append(info)
    count, identity = _v2._ids_and_counts(arm_rows, "PRECAST")
    numeric = (cross.get("osi_recomputed") or {})
    if not _finite(numeric.get("osi")):
        raise EvidenceValidationError("PRECAST current OSI is missing/non-finite")
    return {"row": _cell("PRECAST", condition, seed, status="PASS", tier="accepted_current_partial_receipt", source=str(cell_path), reason="explicit current gap receipt, bound provenance, four-arm NPZ hashes and independent checks verified", raw_recomputable=True, retained_count=count, retained_ids_unordered_sha256=identity, numeric={"osi": float(numeric["osi"]), "margin": numeric.get("margin"), "pass_reported": numeric.get("pass"), "classification": numeric.get("classification")}, source_signature=_precast_signature(record)), "signature": _precast_signature(record), "cell_path": str(cell_path)}


def ingest_precast(project_root: Path | str, acceptance: Path | str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root); source = str(acceptance or PRECAST_ACCEPTANCE_DEFAULT)
    old_rows, old_summary = _v2.ingest_precast(root)
    try:
        acceptance_path = _acceptance_path(root, acceptance, PRECAST_ACCEPTANCE_DEFAULT)
        doc = _json(acceptance_path)
        if doc.get("schema_version") != "inspire.experiment.partial_acceptance.v1" or doc.get("acceptance_decision") not in {"CONDITIONAL_PASS", "PASS"}:
            raise EvidenceValidationError("PRECAST current acceptance envelope is not conditional/pass")
        identity = doc.get("experiment_identity")
        if not isinstance(identity, Mapping) or not isinstance(identity.get("run_id"), str) or not _valid_hash(identity.get("plan_hash")):
            raise EvidenceValidationError("PRECAST current run/plan identity is missing")
        inventory = doc.get("evidence_inventory")
        if not isinstance(inventory, Mapping):
            raise EvidenceValidationError("PRECAST current evidence inventory is missing")
        audit_path = _resolve(root, inventory.get("audit_json")); audit_csv = _resolve(root, inventory.get("audit_csv"))
        _check_receipt_file(audit_path, inventory.get("audit_json_sha256"), "PRECAST audit JSON") if audit_path else (_ for _ in ()).throw(EvidenceValidationError("PRECAST audit JSON path missing"))
        _check_receipt_file(audit_csv, inventory.get("audit_csv_sha256"), "PRECAST audit CSV") if audit_csv else (_ for _ in ()).throw(EvidenceValidationError("PRECAST audit CSV path missing"))
        audit = _json(audit_path)
        cell_results = audit.get("cell_results")
        if not isinstance(cell_results, Mapping):
            raise EvidenceValidationError("PRECAST audit cell_results map is missing")
        current: dict[tuple[str, int], dict[str, Any]] = {}
        for key, info in cell_results.items():
            if not isinstance(info, Mapping) or not isinstance(key, str) or ":seed" not in key:
                continue
            condition, raw_seed = key.rsplit(":seed", 1)
            try: seed = int(raw_seed)
            except ValueError: continue
            if condition not in CONDITIONS or seed not in SEEDS or info.get("status") != "PASS":
                continue
            current[(condition, seed)] = _validate_precast_current_cell(root, info, condition=condition, seed=seed, run_id=str(identity["run_id"]), plan_hash=str(identity["plan_hash"]))
        rows = [dict(row) for row in old_rows]
        old_pass = {(row.get("condition"), int(row.get("seed"))): row for row in rows if row.get("status") == "PASS"}
        duplicates: list[str] = []; conflicts: list[str] = []
        for key, checked in current.items():
            if key in old_pass:
                duplicates.append(f"{key[0]}:seed{key[1]}")
                old_signature = _precast_signature(_json(Path(old_pass[key]["source"]))) if old_pass[key].get("source") else {}
                if old_signature and checked["signature"] and old_signature != checked["signature"]:
                    conflicts.append(f"{key[0]}:seed{key[1]}")
                continue
            rows = [row for row in rows if (row.get("condition"), int(row.get("seed"))) != key]
            rows.append(checked["row"])
        rows.sort(key=lambda row: (METHODS.index(row["method"]), CONDITIONS.index(row["condition"]), int(row["seed"])))
        accepted = sum(row.get("status") == "PASS" for row in rows)
        return rows, {"status": "PASS" if accepted == 20 else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": 20, "old_accepted_cells": sum(row.get("status") == "PASS" for row in old_rows), "current_accepted_cells": len(current), "source": str(acceptance_path), "audit_json": str(audit_path), "duplicates_skipped": duplicates, "duplicate_conflicts": conflicts, "run_id": identity["run_id"], "plan_hash": identity["plan_hash"]}
    except (EvidenceValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return old_rows, {"status": old_summary.get("status", "UNRESOLVED"), "accepted_cells": old_summary.get("accepted_cells", 0), "target_cells": 20, "old_accepted_cells": old_summary.get("accepted_cells", 0), "current_accepted_cells": 0, "source": source, "reason": str(exc)}


def _source_snapshot(root: Path, value: Path | str | None) -> dict[str, Any]:
    path = _acceptance_path(root, value, SOURCE_SNAPSHOT_DEFAULT)
    try:
        doc = _json(path)
        if doc.get("schema_version") != "stia.v3_source_snapshot.v1":
            raise EvidenceValidationError("source snapshot schema mismatch")
        files = doc.get("files")
        if not isinstance(files, Mapping) or not files:
            raise EvidenceValidationError("source snapshot file ledger is missing")
        checked = []
        for name, info in files.items():
            if not isinstance(info, Mapping) or not _valid_hash(info.get("sha256")):
                raise EvidenceValidationError(f"source snapshot row malformed: {name}")
            target = _resolve(root, info.get("path", name))
            if target is None or not target.is_file() or _sha256(target) != info.get("sha256"):
                raise EvidenceValidationError(f"source snapshot drift: {name}")
            checked.append(name)
        return {"status": "PASS", "source": str(path), "files_checked": checked, "snapshot_sha256": _sha256(path)}
    except (EvidenceValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return {"status": "UNRESOLVED", "source": str(path), "reason": str(exc)}


def build_registry(project_root: Path | str = PROJECT_DEFAULT, *, pca_acceptance: Path | str | None = None, precast_acceptance: Path | str | None = None, graphst_anchor: Path | str | None = GRAPHST_ANCHOR_DEFAULT, split_manifest: Path | str | None = None, source_snapshot: Path | str | None = SOURCE_SNAPSHOT_DEFAULT, checked_at: str | None = None) -> dict[str, Any]:
    root = Path(project_root); when = checked_at or _now()
    base = _v3.build_registry(root, graphst_anchor=graphst_anchor, split_manifest=split_manifest, checked_at=when)
    pca_rows, pca_source = ingest_pca(root, pca_acceptance)
    precast_rows, precast_source = ingest_precast(root, precast_acceptance)
    ledger = [row for row in base["ledger"] if row.get("method") not in {"PCA", "PRECAST"}] + pca_rows + precast_rows
    ledger.sort(key=lambda row: (METHODS.index(row["method"]), CONDITIONS.index(row["condition"]), int(row["seed"])))
    coverage = {method: {status: sum(row.get("method") == method and row.get("status") == status for row in ledger) for status in STATUSES} for method in METHODS}
    method_status = dict(base.get("method_status", {}))
    for method in ("PCA", "PRECAST"):
        accepted = coverage[method]["PASS"]
        method_status[method] = {"status": "PASS" if accepted == 20 else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": 20, "historical_continuity": "UNKNOWN" if method == "PCA" else "PARTIAL_RECEIPT_MERGE"}
    return {"schema_version": "stia.evidence_registry.v4.20260830", "checked_at": when, "project_root": str(root), "contract": base["contract"], "sources": {**base.get("sources", {}), "PCA": pca_source, "PRECAST": precast_source}, "ledger": ledger, "coverage": coverage, "method_status": method_status, "axes": base["axes"], "split": base["split"], "graphst_anchor": base.get("graphst_anchor"), "pca_source": pca_source, "precast_source": precast_source, "source_snapshots": _source_snapshot(root, source_snapshot), "claim_boundary": ["PCA PASS is scoped to the explicit strict r3 acceptance/output inventory; five labels are deterministic repetitions, not stochastic replicates.", "PRECAST rows combine only non-duplicate old14 acceptance cells and explicitly supplied current completed gap cells; missing cells remain UNRESOLVED.", "Current GraphST PASS remains scoped to the U5 fresh independent anchor; historical continuity remains UNKNOWN.", "Harmony and STAligner unresolved/NA axes remain visible; no overall six-method acceptance is implied.", "Operational margin 0.05 is not a validated universal cutoff; high OSI is not relabeled invalid experiment."]}


def write_registry(registry: Mapping[str, Any], output_dir: Path | str, *, filename: str = "evidence_registry_v4_20260830.json") -> Path:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicit-receipt six-method evidence registry v4")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--pca-acceptance", type=Path, default=PCA_ACCEPTANCE_DEFAULT)
    parser.add_argument("--precast-acceptance", type=Path, default=PRECAST_ACCEPTANCE_DEFAULT)
    parser.add_argument("--graphst-anchor", type=Path, default=GRAPHST_ANCHOR_DEFAULT)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--source-snapshot", type=Path, default=SOURCE_SNAPSHOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="evidence_registry_v4_20260830.json")
    args = parser.parse_args(argv)
    registry = build_registry(args.project_root, pca_acceptance=args.pca_acceptance, precast_acceptance=args.precast_acceptance, graphst_anchor=args.graphst_anchor, split_manifest=args.split_manifest, source_snapshot=args.source_snapshot)
    path = write_registry(registry, args.output_dir, filename=args.filename) if args.output_dir else None
    print(json.dumps({"status": "READY", "ledger_cells": len(registry["ledger"]), "coverage": registry["coverage"], "method_status": registry["method_status"], "sources": {"PCA": registry["pca_source"], "PRECAST": registry["precast_source"]}, "output": str(path) if path else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
