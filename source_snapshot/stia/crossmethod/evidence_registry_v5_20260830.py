"""Anchored PRECAST final-receipt successor (U7).

The final receipt has a structured evidence inventory and an independent
accepted-current-file snapshot.  This module validates both before admitting a
cell, so changing a cell's self-reported hash cannot make a stale NPZ pass.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import evidence_registry_v2_20260830 as _v2
from . import evidence_registry_v3_20260830 as _v3
from . import evidence_registry_v4_20260830 as _v4

METHODS = _v3.METHODS
CONDITIONS = _v3.CONDITIONS
SEEDS = _v3.SEEDS
ORDERS = _v3.ORDERS
STATUSES = _v3.STATUSES
MARGIN = _v3.MARGIN
PROJECT_DEFAULT = _v3.PROJECT_DEFAULT
FINAL_ACCEPTANCE_DEFAULT = Path("docs/PRECAST_GAP4_FINAL_ACCEPTANCE_20260830_r1.json")
PCA_ACCEPTANCE_DEFAULT = _v4.PCA_ACCEPTANCE_DEFAULT
GRAPHST_ANCHOR_DEFAULT = _v4.GRAPHST_ANCHOR_DEFAULT
SOURCE_SNAPSHOT_DEFAULT = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/evidence/U7/source_snapshots/V5_FREEZE_20260830_r3.json")


class EvidenceValidationError(ValueError):
    """Raised when a final receipt is malformed, stale or weakly anchored."""


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
    return path if path.is_absolute() else root / path


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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(float(value)) != float("inf")


def _cell(condition: str, seed: int, *, status: str, tier: str, source: str | None, reason: str, **extra: Any) -> dict[str, Any]:
    row = {"method": "PRECAST", "condition": condition, "seed": int(seed), "status": status, "evidence_tier": tier, "source": source, "reason": reason}
    row.update(extra)
    return row


def _empty(reason: str, source: str | None) -> list[dict[str, Any]]:
    return [_cell(condition, seed, status="UNRESOLVED", tier="receipt_missing_or_invalid", source=source, reason=reason) for condition in CONDITIONS for seed in SEEDS]


def _path_hash_map(root: Path, snapshot: Any) -> dict[str, str]:
    if not isinstance(snapshot, list) or not snapshot:
        raise EvidenceValidationError("accepted_current_file_snapshot is missing")
    result: dict[str, str] = {}
    for item in snapshot:
        if not isinstance(item, Mapping) or not _valid_hash(item.get("sha256")):
            raise EvidenceValidationError("accepted snapshot row is malformed")
        path = _resolve(root, item.get("path"))
        if path is None or not path.is_file() or _sha256(path) != item["sha256"] or (item.get("bytes") is not None and path.stat().st_size != int(item["bytes"])):
            raise EvidenceValidationError(f"accepted snapshot path/hash mismatch: {item.get('path')}")
        key = str(path.resolve())
        if key in result:
            raise EvidenceValidationError("accepted snapshot contains duplicate paths")
        result[key] = item["sha256"]
    return result


def _check_inventory(root: Path, inventory: Any) -> dict[str, Path]:
    if not isinstance(inventory, Mapping):
        raise EvidenceValidationError("final evidence inventory is missing")
    paths: dict[str, Path] = {}
    for name in ("audit_json", "audit_csv", "manifest", "plan", "current_environment", "auditor", "checkpoint"):
        item = inventory.get(name)
        if not isinstance(item, Mapping):
            raise EvidenceValidationError(f"final inventory entry missing: {name}")
        path = _resolve(root, item.get("path"))
        if path is None or not path.is_file() or not _valid_hash(item.get("sha256")) or _sha256(path) != item["sha256"]:
            raise EvidenceValidationError(f"final inventory path/hash mismatch: {name}")
        if item.get("bytes") is not None and path.stat().st_size != item["bytes"]:
            raise EvidenceValidationError(f"final inventory size mismatch: {name}")
        paths[name] = path
    return paths


def _signature(record: Mapping[str, Any]) -> dict[str, Any]:
    config = record.get("config") if isinstance(record.get("config"), Mapping) else {}
    return {"input_sha256": record.get("input_sha256"), "panel_sha256": record.get("panel_sha256"), "r_core_sha256": record.get("r_core_sha256"), "margin": record.get("margin"), "model": {key: config.get(key) for key in ("K", "q", "beta_grid", "coreNum", "coreNum_int")}}


def _compatibility(old_rows: list[Mapping[str, Any]], current_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare non-seed scientific contract fields without treating run hashes as equivalence."""
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in old_rows + current_rows:
        if row.get("method") != "PRECAST":
            continue
        if isinstance(row.get("source_signature"), Mapping):
            sig = dict(row["source_signature"])
        elif row.get("status") == "PASS" and row.get("source"):
            try:
                sig = _signature(_json(Path(str(row["source"]))))
            except (EvidenceValidationError, OSError, ValueError):
                continue
        else:
            continue
        by_condition.setdefault(str(row.get("condition")), []).append(sig)
    incompatible: list[str] = []
    for condition, signatures in by_condition.items():
        unique = {json.dumps(sig, sort_keys=True, separators=(",", ":")) for sig in signatures}
        if len(unique) > 1:
            incompatible.append(condition)
    return {"status": "PASS" if not incompatible else "UNRESOLVED", "basis": "condition input/panel, native K/q/beta/coreNum, R-core and four prescribed order semantics are compared; run/seed-specific hashes remain scoped", "cohorts": sorted(by_condition), "incompatible_conditions": incompatible}


def _load_record(root: Path, path: Path, *, condition: str, seed: int, run_id: str, plan_hash: str, snapshot: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    key = str(path.resolve())
    if key not in snapshot or _sha256(path) != snapshot[key]:
        raise EvidenceValidationError("PRECAST cell JSON is not bound to accepted current snapshot")
    record = _json(path)
    _v2.validate_binding(record, method="PRECAST", condition=condition, seed=seed, run_id=run_id, plan_hash=plan_hash)
    if record.get("status") != "OK":
        raise EvidenceValidationError("PRECAST cell JSON is not terminal OK")
    _v2._hash_chain(record, ("input_sha256", "panel_sha256", "source_sha256", "r_core_sha256", "runner_sha256", "continuation_sha256", "artifact_sha256", "implementation_sha256"), "PRECAST cell")
    arms = record.get("arms")
    if not isinstance(arms, Mapping) or tuple(arms) != ORDERS:
        raise EvidenceValidationError("PRECAST cell arm keys are incomplete")
    return record, _signature(record)


def _validate_final_cell(root: Path, cell_info: Mapping[str, Any], *, condition: str, seed: int, run_id: str, plan_hash: str, snapshot: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if cell_info.get("status") != "PASS" or cell_info.get("errors") not in ([], None):
        raise EvidenceValidationError("independent final audit cell is not clean PASS")
    checks = cell_info.get("cell_checks"); orders = cell_info.get("order_checks"); cross = cell_info.get("cross_arm")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()) or not isinstance(orders, Mapping) or not orders or not all(value is True for value in orders.values()) or not isinstance(cross, Mapping) or not isinstance(cross.get("checks"), Mapping) or not cross["checks"] or not all(value is True for value in cross["checks"].values()):
        raise EvidenceValidationError("final audit cell checks are incomplete")
    cell_path = _resolve(root, cell_info.get("cell_json"))
    if cell_path is None:
        raise EvidenceValidationError("final audit cell JSON path is missing")
    record, signature = _load_record(root, cell_path, condition=condition, seed=seed, run_id=run_id, plan_hash=plan_hash, snapshot=snapshot)
    retention = cell_info.get("independent_source_retention")
    if not isinstance(retention, Mapping) or not _valid_hash(retention.get("retained_ids_unordered_sha256")):
        raise EvidenceValidationError("independent source-retention anchor is missing")
    arm_rows = []
    for order in ORDERS:
        audit_arm = (cell_info.get("arms") or {}).get(order)
        record_arm = record["arms"].get(order)
        if not isinstance(audit_arm, Mapping) or not isinstance(record_arm, Mapping):
            raise EvidenceValidationError(f"PRECAST {order} arm is missing")
        if audit_arm.get("path") is None or audit_arm.get("npz_sha256") != record_arm.get("npz_sha256") or not _valid_hash(audit_arm.get("npz_sha256")):
            raise EvidenceValidationError(f"PRECAST {order} independent NPZ hash is not cell-bound")
        if not isinstance(audit_arm.get("checks"), Mapping) or not audit_arm["checks"] or not all(value is True for value in audit_arm["checks"].values()) or not isinstance(audit_arm.get("hash_checks"), Mapping) or not audit_arm["hash_checks"] or not all(value is True for value in audit_arm["hash_checks"].values()):
            raise EvidenceValidationError(f"PRECAST {order} independent checks are incomplete")
        npz_path = _resolve(root, audit_arm.get("path"))
        if npz_path is None or str(_resolve(root, record_arm.get("npz_path"))) != str(npz_path):
            raise EvidenceValidationError(f"PRECAST {order} NPZ path is not cell/audit bound")
        if str(npz_path.resolve()) not in snapshot:
            raise EvidenceValidationError(f"PRECAST {order} NPZ is absent from accepted snapshot")
        ok, detail = _v2._npz_check(npz_path, audit_arm.get("npz_sha256"))
        if not ok:
            raise EvidenceValidationError(detail)
        if audit_arm.get("n_rows") != record_arm.get("retained_count") or audit_arm.get("retained_ids_unordered_sha256") != retention.get("retained_ids_unordered_sha256"):
            raise EvidenceValidationError(f"PRECAST {order} retained identity mismatch")
        arm_rows.append(record_arm)
    count, identity = _v2._ids_and_counts(arm_rows, "PRECAST")
    if count != retention.get("retained_count") or identity != retention.get("retained_ids_unordered_sha256"):
        raise EvidenceValidationError("PRECAST source-derived retention does not bind four arms")
    numeric = cross.get("osi_recomputed")
    if not isinstance(numeric, Mapping) or not _finite(numeric.get("osi")):
        raise EvidenceValidationError("PRECAST independent OSI is missing/non-finite")
    row = _cell(condition, seed, status="PASS", tier="accepted_final_raw_receipt", source=str(cell_path), reason="final receipt, independent audit arm hashes, accepted current snapshot and source-derived retained identity verified", raw_recomputable=True, retained_count=count, retained_ids_unordered_sha256=identity, numeric={"osi": float(numeric["osi"]), "margin": numeric.get("margin"), "pass_reported": numeric.get("pass"), "classification": numeric.get("classification")}, source_signature=signature, run_id=run_id, plan_hash=plan_hash)
    return row, signature


def _ingest_final(root: Path, acceptance_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = _json(acceptance_path)
    if doc.get("schema_version") != "inspire.experiment.final_acceptance.v1" or doc.get("acceptance_decision") not in {"PASS", "CONDITIONAL_PASS"} or doc.get("execution_validity") != "PASS":
        raise EvidenceValidationError("final PRECAST receipt envelope is not valid")
    identity = doc.get("experiment_identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("run_id"), str) or not _valid_hash(identity.get("plan_hash")) or identity.get("orders") != list(ORDERS):
        raise EvidenceValidationError("final PRECAST run/plan/order identity is invalid")
    checks = doc.get("completion_checks")
    if not isinstance(checks, Mapping) or checks.get("status") != "PASS" or checks.get("expected_cells") != 6 or checks.get("observed_cells") != 6 or checks.get("passed_cells") != 6 or checks.get("invalid_cells") != 0 or checks.get("verified_npz") != 24:
        raise EvidenceValidationError("final PRECAST terminal counts are incomplete")
    parent_manifest = doc.get("parent_manifest_snapshot")
    if not isinstance(parent_manifest, Mapping) or parent_manifest.get("status") != "OK" or parent_manifest.get("whole_run_complete") is not True:
        raise EvidenceValidationError("final parent manifest is not terminal")
    completed = parent_manifest.get("completed_cells")
    if not isinstance(completed, list) or len(completed) != 6 or len({tuple(item) for item in completed if isinstance(item, list) and len(item) == 2}) != 6:
        raise EvidenceValidationError("final parent manifest cell list is malformed")
    inventory_paths = _check_inventory(root, doc.get("evidence_inventory"))
    snapshot = _path_hash_map(root, doc.get("accepted_current_file_snapshot"))
    audit = _json(inventory_paths["audit_json"])
    audit_identity = audit.get("experiment_identity")
    if not isinstance(audit_identity, Mapping) or audit_identity.get("run_id") != identity.get("run_id") or audit_identity.get("plan_hash") != identity.get("plan_hash"):
        raise EvidenceValidationError("final audit run/plan identity disagrees with receipt")
    audit_cells = audit.get("cell_results")
    if not isinstance(audit_cells, Mapping):
        raise EvidenceValidationError("final audit cell_results map is missing")
    current_rows: list[dict[str, Any]] = []; signatures: dict[tuple[str, int], dict[str, Any]] = {}
    current_keys = {(condition, 20260819) if condition == "disease_only_no_batch" else (condition, seed) for condition, seed in [("disease_only_no_batch", 20260819)] + [("null_no_batch", seed) for seed in SEEDS]}
    for condition, seed in sorted(current_keys):
        key = f"{condition}:seed{seed}"
        info = audit_cells.get(key)
        if not isinstance(info, Mapping):
            raise EvidenceValidationError(f"final audit cell missing: {key}")
        row, signature = _validate_final_cell(root, info, condition=condition, seed=seed, run_id=str(identity["run_id"]), plan_hash=str(identity["plan_hash"]), snapshot=snapshot)
        current_rows.append(row); signatures[(condition, seed)] = signature
    results = doc.get("results")
    if not isinstance(results, list) or len(results) != 6:
        raise EvidenceValidationError("final receipt results do not cover six cells")
    result_keys = {(item.get("condition"), int(item.get("seed"))) for item in results if isinstance(item, Mapping) and "condition" in item and "seed" in item}
    if result_keys != current_keys:
        raise EvidenceValidationError("final receipt result identities are incomplete/duplicated")
    for item in results:
        key = (item.get("condition"), int(item.get("seed")))
        if item.get("cell_status") != "PASS" or not _finite(item.get("osi")) or item.get("osi") != next(row["numeric"]["osi"] for row in current_rows if (row["condition"], row["seed"]) == key):
            raise EvidenceValidationError(f"final receipt result is not audit-bound: {key}")
    old_rows, old_summary = _v2.ingest_precast(root)
    old_pass = {(row.get("condition"), int(row.get("seed"))): row for row in old_rows if row.get("status") == "PASS"}
    duplicates = []
    # The final disease seed19 is the explicitly documented P2R same-seed replay.
    duplicate_key = ("disease_only_no_batch", 20260819)
    for row in current_rows:
        if (row["condition"], row["seed"]) == duplicate_key:
            duplicates.append("disease_only_no_batch:seed20260819")
            row["status"] = "UNRESOLVED"; row["evidence_tier"] = "valid_duplicate_replay_excluded"; row["reason"] = "valid final cell is the documented P2R disease seed19 replay; excluded from independent aggregate"
    rows = [dict(row) for row in old_rows if (row.get("condition"), int(row.get("seed"))) not in {(r["condition"], r["seed"]) for r in current_rows}] + current_rows
    # Preserve old14 and the five independent null cells; expose, rather than hide, the replay.
    accepted = sum(row.get("status") == "PASS" for row in rows)
    compatibility = _compatibility(old_rows, current_rows)
    return rows, {"status": "PARTIAL" if accepted < 20 else "PASS", "accepted_cells": accepted, "target_cells": 20, "old_accepted_cells": old_summary.get("accepted_cells", 0), "current_terminal_cells": 6, "current_unique_cells": 5, "duplicates_skipped": duplicates, "source": str(acceptance_path), "audit_json": str(inventory_paths["audit_json"]), "run_id": identity["run_id"], "plan_hash": identity["plan_hash"], "compatibility": compatibility, "snapshot_files": len(snapshot)}


def ingest_precast(project_root: Path | str, acceptance: Path | str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root); value = acceptance if acceptance is not None else FINAL_ACCEPTANCE_DEFAULT; path = _resolve(root, value)
    source = str(value)
    try:
        if path is None:
            raise EvidenceValidationError("PRECAST final receipt path is not resolvable")
        doc = _json(path)
        if doc.get("schema_version") == "inspire.experiment.final_acceptance.v1":
            return _ingest_final(root, path)
        # Explicit partial receipts remain available for backward-compatible callers;
        # final receipts never fall through to this weaker parser.
        return _v4.ingest_precast(root, path)
    except (EvidenceValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return _empty(str(exc), source), {"status": "UNRESOLVED", "accepted_cells": 0, "target_cells": 20, "source": source, "reason": str(exc)}


def _source_snapshot(root: Path, value: Path | str | None) -> dict[str, Any]:
    path = _resolve(root, value if value is not None else SOURCE_SNAPSHOT_DEFAULT)
    if path is None or not path.is_file():
        return {"status": "UNRESOLVED", "source": str(value or SOURCE_SNAPSHOT_DEFAULT), "reason": "v5 source snapshot is missing"}
    try:
        doc = _json(path); files = doc.get("files")
        schema = doc.get("schema_version")
        if schema not in {"stia.v4_source_snapshot.v1", "stia.v5_source_snapshot.v1"} or not isinstance(files, Mapping):
            raise EvidenceValidationError("v4/v5 source snapshot schema/ledger invalid")
        for name, item in files.items():
            if not isinstance(item, Mapping) or not _valid_hash(item.get("sha256")):
                raise EvidenceValidationError(f"source snapshot row malformed: {name}")
            target = _resolve(root, item.get("path"));
            if target is None or not target.is_file() or _sha256(target) != item["sha256"] or (item.get("bytes") is not None and target.stat().st_size != int(item["bytes"])):
                raise EvidenceValidationError(f"source snapshot drift: {name}")
        closure = doc.get("source_closure")
        if schema == "stia.v5_source_snapshot.v1":
            if not isinstance(closure, Mapping) or not _valid_hash(closure.get("v4_source_snapshot_sha256")):
                raise EvidenceValidationError("v5 source closure is missing or malformed")
            v4_path = _resolve(root, closure.get("v4_source_snapshot"))
            if v4_path is None or not v4_path.is_file() or _sha256(v4_path) != closure["v4_source_snapshot_sha256"]:
                raise EvidenceValidationError("v4 source snapshot closure drift")
            v4_doc = _json(v4_path)
            v4_files = v4_doc.get("files")
            if v4_doc.get("schema_version") != "stia.v4_source_snapshot.v1" or not isinstance(v4_files, Mapping):
                raise EvidenceValidationError("v4 source snapshot closure schema invalid")
            for name, item in v4_files.items():
                if not isinstance(item, Mapping) or not _valid_hash(item.get("sha256")):
                    raise EvidenceValidationError(f"v4 closure row malformed: {name}")
                target = _resolve(root, item.get("path"));
                if target is None or not target.is_file() or _sha256(target) != item["sha256"] or (item.get("bytes") is not None and target.stat().st_size != int(item["bytes"])):
                    raise EvidenceValidationError(f"v4 closure drift: {name}")
        return {"status": "PASS", "source": str(path), "schema_version": schema, "files_checked": list(files), "source_closure": closure if schema == "stia.v5_source_snapshot.v1" else None, "snapshot_sha256": _sha256(path)}
    except (EvidenceValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return {"status": "UNRESOLVED", "source": str(path), "reason": str(exc)}


def build_registry(project_root: Path | str = PROJECT_DEFAULT, *, precast_acceptance: Path | str | None = None, pca_acceptance: Path | str | None = None, graphst_anchor: Path | str | None = GRAPHST_ANCHOR_DEFAULT, split_manifest: Path | str | None = None, source_snapshot: Path | str | None = SOURCE_SNAPSHOT_DEFAULT, checked_at: str | None = None) -> dict[str, Any]:
    root = Path(project_root); when = checked_at or _now()
    base = _v3.build_registry(root, graphst_anchor=graphst_anchor, split_manifest=split_manifest, checked_at=when)
    pca_rows, pca_source = _v4.ingest_pca(root, pca_acceptance)
    precast_rows, precast_source = ingest_precast(root, precast_acceptance)
    ledger = [row for row in base["ledger"] if row.get("method") not in {"PCA", "PRECAST"}] + pca_rows + precast_rows
    ledger.sort(key=lambda row: (METHODS.index(row["method"]), CONDITIONS.index(row["condition"]), int(row["seed"])))
    coverage = {method: {status: sum(row.get("method") == method and row.get("status") == status for row in ledger) for status in STATUSES} for method in METHODS}
    method_status = dict(base.get("method_status", {}))
    for method in ("PCA", "PRECAST"):
        accepted = coverage[method]["PASS"]
        method_status[method] = {"status": "PASS" if accepted == 20 else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": 20, "historical_continuity": "UNKNOWN" if method == "PCA" else "FINAL_RECEIPT_WITH_REPLAY_EXCLUDED"}
    return {"schema_version": "stia.evidence_registry.v5.20260830", "checked_at": when, "project_root": str(root), "contract": base["contract"], "sources": {**base.get("sources", {}), "PCA": pca_source, "PRECAST": precast_source}, "ledger": ledger, "coverage": coverage, "method_status": method_status, "axes": base["axes"], "split": base["split"], "graphst_anchor": base.get("graphst_anchor"), "pca_source": pca_source, "precast_source": precast_source, "source_snapshots": _source_snapshot(root, source_snapshot), "claim_boundary": ["PRECAST final receipt validates six terminal cells and 24 NPZ against independent audit and accepted-current snapshot; the disease seed19 replay is excluded from the independent aggregate.", "Old PRECAST14 and five current null cells are retained as separate source cohorts; compatibility is limited to explicit input/panel/model/order bindings.", "PCA/GraphST/INSPIRE/Harmony/STAligner boundaries are inherited without upgrading overall six-method acceptance.", "Operational margin 0.05 is not a universal validated cutoff."]}


def write_registry(registry: Mapping[str, Any], output_dir: Path | str, *, filename: str = "evidence_registry_v5_20260830.json") -> Path:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anchored PRECAST final-receipt six-method registry v5")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--precast-acceptance", type=Path, default=FINAL_ACCEPTANCE_DEFAULT)
    parser.add_argument("--pca-acceptance", type=Path, default=PCA_ACCEPTANCE_DEFAULT)
    parser.add_argument("--graphst-anchor", type=Path, default=GRAPHST_ANCHOR_DEFAULT)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--source-snapshot", type=Path, default=SOURCE_SNAPSHOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="evidence_registry_v5_20260830.json")
    args = parser.parse_args(argv)
    registry = build_registry(args.project_root, precast_acceptance=args.precast_acceptance, pca_acceptance=args.pca_acceptance, graphst_anchor=args.graphst_anchor, split_manifest=args.split_manifest, source_snapshot=args.source_snapshot)
    path = write_registry(registry, args.output_dir, filename=args.filename) if args.output_dir else None
    print(json.dumps({"status": "READY", "ledger_cells": len(registry["ledger"]), "coverage": registry["coverage"], "method_status": registry["method_status"], "precast_source": registry["precast_source"], "output": str(path) if path else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
