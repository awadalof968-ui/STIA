"""Current-state GraphST anchor successor for the six-method registry.

GraphST rows are admitted only when a separately produced current independent
audit anchor supplies hashes for the acceptance document, every cell and all
four arm JSON/NPZ artifacts.  The anchor is a new current-state audit, not a
claim that historical bytes were unchanged.  Other methods reuse U2R's
hash/manifest handling unchanged.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import evidence_registry_v2_20260830 as _v2

METHODS = _v2.METHODS
CONDITIONS = _v2.CONDITIONS
SEEDS = _v2.SEEDS
ORDERS = _v2.ORDERS
STATUSES = _v2.STATUSES
MARGIN = _v2.MARGIN
PROJECT_DEFAULT = _v2.PROJECT_DEFAULT


class EvidenceValidationError(ValueError):
    """Raised when a current audit anchor is missing or does not bind bytes."""


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


def _all_graphst(reason: str, *, source: str | None = None) -> list[dict[str, Any]]:
    return [{"method": "GraphST", "condition": condition, "seed": seed, "status": "UNRESOLVED", "evidence_tier": "anchor_missing_or_invalid", "source": source, "reason": reason, "historical_continuity": "UNKNOWN"}
            for condition in CONDITIONS for seed in SEEDS]


def _anchor_path(project_root: Path, anchor: Path | str | None) -> Path | None:
    if anchor is None:
        return None
    path = Path(anchor)
    if not path.is_absolute():
        path = project_root / path
    return path if path.exists() else None


def _within(path: Path, root: Path) -> bool:
    """Return true only for an artifact contained by its declared root."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _artifact_manifest_digest(cells: Iterable[Mapping[str, Any]]) -> str:
    """Canonical digest of the per-cell aggregate/arm/NPZ digest ledger."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        arms = cell.get("arms") if isinstance(cell.get("arms"), Mapping) else {}
        rows.append({
            "condition": cell.get("condition"),
            "seed": cell.get("seed"),
            "aggregate_json": cell.get("aggregate_json"),
            "aggregate_json_sha256": cell.get("aggregate_json_sha256"),
            "arms": {order: {
                "json": (arms.get(order) or {}).get("json"),
                "json_sha256": (arms.get(order) or {}).get("json_sha256"),
                "npz_sha256": (arms.get(order) or {}).get("npz_sha256"),
            } for order in ORDERS},
        })
    payload = json.dumps(sorted(rows, key=lambda item: (str(item["condition"]), int(item["seed"])),), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_snapshot_current(snapshot: Mapping[str, Any], project_root: Path, output_root: Path) -> None:
    """Recheck any full snapshot fields against bytes at ingestion time."""
    if snapshot.get("output_root") is not None:
        declared_root = _anchor_path(project_root, snapshot.get("output_root"))
        if declared_root is None or declared_root.resolve() != output_root.resolve():
            raise EvidenceValidationError("audit snapshot output root disagrees with anchor")
    for key in ("run_plan", "historical_acceptance"):
        record = snapshot.get(key)
        if record is None:
            continue
        if not isinstance(record, Mapping) or not _v2._valid_hash(record.get("sha256")):
            raise EvidenceValidationError(f"audit snapshot {key} hash is missing")
        path = _anchor_path(project_root, record.get("path"))
        if path is None or not path.is_file() or _v2._sha256(path) != record.get("sha256"):
            raise EvidenceValidationError(f"audit snapshot {key} changed")
    files = snapshot.get("files")
    if files is None:
        return
    if not isinstance(files, Mapping):
        raise EvidenceValidationError("audit snapshot file ledger is malformed")
    for kind, entries in files.items():
        if not isinstance(entries, Mapping):
            raise EvidenceValidationError(f"audit snapshot {kind} ledger is malformed")
        for name, expected_hash in entries.items():
            path = output_root / str(name)
            if not _within(path, output_root) or not path.is_file() or not _v2._valid_hash(expected_hash) or _v2._sha256(path) != expected_hash:
                raise EvidenceValidationError(f"audit snapshot artifact changed: {kind}/{name}")


def _validate_anchor(project_root: Path, anchor_path: Path) -> tuple[dict[str, Any], Path, Path]:
    anchor = _json(anchor_path)
    if anchor.get("schema_version") != "stia.graphst.current_anchor.v1":
        raise EvidenceValidationError("anchor schema is not current GraphST anchor v1")
    if anchor.get("audit_kind") != "fresh_current_independent_reaudit":
        raise EvidenceValidationError("anchor is not declared as a fresh current independent re-audit")
    if not isinstance(anchor.get("audit_command"), str) or not anchor["audit_command"].strip():
        raise EvidenceValidationError("anchor audit command is missing")
    if anchor.get("exit_code") != 0:
        raise EvidenceValidationError("anchor audit exit_code is not zero")
    if not _v2._valid_hash(anchor.get("source_script_sha256")):
        raise EvidenceValidationError("anchor source-script hash is missing")
    if not isinstance(anchor.get("audit_started_at"), str) or not isinstance(anchor.get("audit_finished_at"), str):
        raise EvidenceValidationError("anchor audit timestamps are missing")
    acceptance_path = _anchor_path(project_root, anchor.get("acceptance_path"))
    output_root = _anchor_path(project_root, anchor.get("output_root"))
    if acceptance_path is None or not acceptance_path.is_file() or output_root is None or not output_root.is_dir():
        raise EvidenceValidationError("anchor acceptance/output root is missing")
    if not _v2._valid_hash(anchor.get("acceptance_sha256")) or _v2._sha256(acceptance_path) != anchor.get("acceptance_sha256"):
        raise EvidenceValidationError("anchor acceptance digest is missing or stale")
    source_script = _anchor_path(project_root, anchor.get("source_script"))
    if source_script is None or not source_script.is_file():
        raise EvidenceValidationError("anchor source script path is missing")
    if _v2._sha256(source_script) != anchor.get("source_script_sha256"):
        raise EvidenceValidationError("anchor source-script digest is stale")

    acceptance = _json(acceptance_path)
    meta = acceptance.get("document_meta")
    identity = acceptance.get("experiment_identity")
    inventory = acceptance.get("evidence_inventory")
    checks = acceptance.get("completion_checks")
    if not isinstance(meta, Mapping) or meta.get("status") != "PASS":
        raise EvidenceValidationError("linked GraphST acceptance is not PASS")
    if not isinstance(identity, Mapping) or identity.get("method") != "GraphST":
        raise EvidenceValidationError("linked acceptance is not GraphST-bound")
    if not isinstance(inventory, Mapping):
        raise EvidenceValidationError("linked acceptance inventory is missing")
    linked_root = _anchor_path(project_root, inventory.get("output_root"))
    if linked_root is None or linked_root.resolve() != output_root.resolve():
        raise EvidenceValidationError("linked acceptance output root does not match anchor")
    auditor = _anchor_path(project_root, inventory.get("auditor_script"))
    if auditor is None or auditor.resolve() != source_script.resolve():
        raise EvidenceValidationError("anchor source script is not the accepted auditor")
    published_auditor_hash = inventory.get("auditor_script_sha256")
    if published_auditor_hash is not None and (not _v2._valid_hash(published_auditor_hash) or published_auditor_hash != anchor.get("source_script_sha256")):
        raise EvidenceValidationError("anchor source hash disagrees with acceptance inventory")
    if not isinstance(checks, Mapping):
        raise EvidenceValidationError("linked acceptance completion checks are missing")
    if checks.get("cells_checked") != len(CONDITIONS) * len(SEEDS) or checks.get("cells_passed") != checks.get("cells_checked") or checks.get("independent_osi_recomputed") != checks.get("cells_checked"):
        raise EvidenceValidationError("linked acceptance completion counts do not cover the full GraphST matrix")
    if checks.get("four_distinct_orders_checked") is not True or checks.get("same_unordered_spot_set_checked") is not True:
        raise EvidenceValidationError("linked acceptance lacks four-order/source-identity checks")
    if inventory.get("raw_npz_reloaded_allow_pickle_false") is not True:
        raise EvidenceValidationError("linked acceptance lacks non-pickle raw reload check")

    snapshots = (anchor.get("snapshot_before"), anchor.get("snapshot_after"))
    if any(not isinstance(item, Mapping) for item in snapshots):
        raise EvidenceValidationError("anchor input snapshots are missing")
    if snapshots[0] != snapshots[1]:
        raise EvidenceValidationError("audit input/output snapshots differ; artifacts changed during audit")
    for snapshot in snapshots:
        if "acceptance_sha256" in snapshot and snapshot.get("acceptance_sha256") != anchor.get("acceptance_sha256"):
            raise EvidenceValidationError("audit snapshot is not bound to acceptance digest")
        _validate_snapshot_current(snapshot, project_root, output_root)
    cells = anchor.get("cells")
    expected = len(CONDITIONS) * len(SEEDS)
    if not isinstance(cells, list) or len(cells) != expected:
        raise EvidenceValidationError(f"anchor required-cell coverage is incomplete: expected {expected}")
    keys = [(_v2._v1._condition(item.get("condition")), int(item.get("seed", -1))) for item in cells if isinstance(item, Mapping)]
    expected_keys = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys):
        raise EvidenceValidationError("anchor required-cell identities are incomplete or duplicated")
    manifest_digest = _artifact_manifest_digest(cells)
    for snapshot in snapshots:
        if not _v2._valid_hash(snapshot.get("artifact_manifest_sha256")) or snapshot.get("artifact_manifest_sha256") != manifest_digest:
            raise EvidenceValidationError("audit snapshot artifact digest is not bound to anchor cells")
    return anchor, acceptance_path, output_root


def _validate_anchor_cell(project_root: Path, output_root: Path, anchor_cell: Mapping[str, Any], *, condition: str, seed: int) -> dict[str, Any]:
    if _v2._v1._condition(anchor_cell.get("condition")) != condition or int(anchor_cell.get("seed", -1)) != int(seed):
        raise EvidenceValidationError("anchor cell method/condition/seed binding mismatch")
    if anchor_cell.get("status") != "PASS":
        raise EvidenceValidationError("anchor cell is not PASS")
    aggregate_path = output_root / str(anchor_cell.get("aggregate_json"))
    if not _within(aggregate_path, output_root):
        raise EvidenceValidationError("anchor aggregate escapes declared output root")
    if not aggregate_path.is_file() or not _v2._valid_hash(anchor_cell.get("aggregate_json_sha256")) or _v2._sha256(aggregate_path) != anchor_cell.get("aggregate_json_sha256"):
        raise EvidenceValidationError("anchor aggregate path/hash mismatch")
    aggregate = _json(aggregate_path)
    _v2.validate_binding(aggregate, method="GraphST", condition=condition, seed=seed)
    if aggregate.get("status") != "OK":
        raise EvidenceValidationError("aggregate is not terminal OK")
    arms = anchor_cell.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != ORDERS:
        raise EvidenceValidationError("anchor arm coverage/order keys mismatch")
    aggregate_arm_json = aggregate.get("arm_json")
    if not isinstance(aggregate_arm_json, dict) or tuple(aggregate_arm_json) != ORDERS:
        raise EvidenceValidationError("aggregate arm coverage/order keys mismatch")
    numeric_checks = anchor_cell.get("numeric_checks")
    if not isinstance(numeric_checks, Mapping):
        raise EvidenceValidationError("anchor numeric checks are missing")
    expected_numeric = {"osi": aggregate.get("osi"), "pass": aggregate.get("pass"), "classification": aggregate.get("classification")}
    for key, value in expected_numeric.items():
        if key not in numeric_checks or numeric_checks.get(key) != value:
            raise EvidenceValidationError(f"anchor numeric check {key} is not bound to aggregate")
    arm_rows: list[dict[str, Any]] = []
    for order in ORDERS:
        arm_ref = arms[order]
        if not isinstance(arm_ref, Mapping) or aggregate_arm_json.get(order) != arm_ref.get("json"):
            raise EvidenceValidationError(f"anchor {order} arm JSON is not aggregate-bound")
        arm_path = output_root / str(arm_ref.get("json"))
        if not _within(arm_path, output_root):
            raise EvidenceValidationError(f"anchor {order} arm JSON escapes declared output root")
        if not arm_path.is_file() or not _v2._valid_hash(arm_ref.get("json_sha256")) or _v2._sha256(arm_path) != arm_ref.get("json_sha256"):
            raise EvidenceValidationError(f"anchor {order} arm JSON path/hash mismatch")
        arm = _json(arm_path)
        _v2.validate_binding(arm, method="GraphST", condition=condition, seed=seed, order=order)
        if arm.get("status") != "OK":
            raise EvidenceValidationError(f"GraphST {order} arm is not terminal OK")
        _v2._hash_chain(arm, ("input_sha256", "panel_sha256", "wrapper_sha256", "implementation_sha256", "latent_npz_sha256"), f"GraphST {order}")
        npz_path = arm_path.parent / str(arm.get("latent_npz"))
        if not _within(npz_path, output_root):
            raise EvidenceValidationError(f"anchor {order} NPZ escapes declared output root")
        if not _v2._valid_hash(arm_ref.get("npz_sha256")) or arm_ref.get("npz_sha256") != arm.get("latent_npz_sha256"):
            raise EvidenceValidationError(f"anchor {order} NPZ hash is not bound to arm metadata")
        ok, detail = _v2._npz_check(npz_path, arm_ref.get("npz_sha256"))
        if not ok:
            raise EvidenceValidationError(detail)
        if arm.get("retained_count") != arm.get("embedding_shape", [None])[0]:
            raise EvidenceValidationError(f"GraphST {order} count/embedding shape mismatch")
        arm_rows.append(arm)
    count, identity_hash = _v2._ids_and_counts(arm_rows, "GraphST")
    return {"aggregate": aggregate, "source": str(aggregate_path), "retained_count": count, "retained_ids_unordered_sha256": identity_hash}


def ingest_graphst(project_root: Path | str, anchor: Path | str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root)
    if anchor is None:
        reason = "current independent GraphST anchor is required; historical acceptance summary alone is insufficient"
        return _all_graphst(reason), {"status": "UNRESOLVED", "accepted_cells": 0, "target_cells": len(CONDITIONS) * len(SEEDS), "historical_continuity": "UNKNOWN", "anchor": None}
    anchor_path = _anchor_path(root, anchor)
    if anchor_path is None:
        reason = "supplied GraphST current anchor is missing"
        return _all_graphst(reason, source=str(anchor)), {"status": "UNRESOLVED", "accepted_cells": 0, "target_cells": len(CONDITIONS) * len(SEEDS), "historical_continuity": "UNKNOWN", "anchor": str(anchor)}
    try:
        anchor_doc, acceptance_path, output_root = _validate_anchor(root, anchor_path)
        cells_by_key = {(str(_v2._v1._condition(item.get("condition"))), int(item.get("seed"))): item for item in anchor_doc["cells"]}
    except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
        return _all_graphst(str(exc), source=str(anchor_path)), {"status": "UNRESOLVED", "accepted_cells": 0, "target_cells": len(CONDITIONS) * len(SEEDS), "historical_continuity": "UNKNOWN", "anchor": str(anchor_path)}
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            cell = cells_by_key.get((condition, seed))
            try:
                if cell is None:
                    raise EvidenceValidationError("anchor cell is missing")
                checked = _validate_anchor_cell(root, output_root, cell, condition=condition, seed=seed)
                rows.append({"method": "GraphST", "condition": condition, "seed": seed, "status": "PASS", "evidence_tier": "current_independent_anchor", "source": checked["source"], "reason": "current anchor and current aggregate/arm/NPZ hashes match; historical continuity remains unknown", "historical_continuity": "UNKNOWN", "retained_count": checked["retained_count"], "retained_ids_unordered_sha256": checked["retained_ids_unordered_sha256"], "numeric": {"osi": checked["aggregate"].get("osi"), "pass_reported": checked["aggregate"].get("pass"), "classification": checked["aggregate"].get("classification")}})
            except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
                rows.append(_cell_error(condition, seed, str(exc), str(anchor_path)))
    accepted = sum(row["status"] == "PASS" for row in rows)
    expected = len(CONDITIONS) * len(SEEDS)
    return rows, {"status": "PASS" if accepted == expected else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": expected, "historical_continuity": "UNKNOWN", "anchor": str(anchor_path), "acceptance": str(acceptance_path)}


def _cell_error(condition: str, seed: int, reason: str, source: str | None) -> dict[str, Any]:
    return {"method": "GraphST", "condition": condition, "seed": int(seed), "status": "UNRESOLVED", "evidence_tier": "anchor_present_but_invalid", "source": source, "reason": reason, "historical_continuity": "UNKNOWN"}


def build_registry(project_root: Path | str = PROJECT_DEFAULT, *, graphst_anchor: Path | str | None = None, split_manifest: Path | str | None = None, checked_at: str | None = None) -> dict[str, Any]:
    root = Path(project_root)
    base = _v2.build_registry(root, split_manifest=split_manifest, checked_at=checked_at or _now())
    graph_rows, graph_source = ingest_graphst(root, graphst_anchor)
    ledger = [row for row in base["ledger"] if row.get("method") != "GraphST"] + graph_rows
    ledger.sort(key=lambda row: (METHODS.index(row["method"]), CONDITIONS.index(row["condition"]), int(row["seed"])))
    coverage = {method: {status: sum(row["method"] == method and row["status"] == status for row in ledger) for status in STATUSES} for method in METHODS}
    method_status = dict(base.get("method_status", {}))
    accepted = coverage["GraphST"]["PASS"]
    method_status["GraphST"] = {"status": "PASS" if accepted == len(CONDITIONS) * len(SEEDS) else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": len(CONDITIONS) * len(SEEDS), "historical_continuity": "UNKNOWN"}
    sources = dict(base.get("sources", {})); sources["GraphST"] = graph_source
    return {"schema_version": "stia.evidence_registry.v3.20260830", "checked_at": checked_at or base["checked_at"], "project_root": str(root), "contract": base["contract"], "sources": sources, "ledger": ledger, "coverage": coverage, "method_status": method_status, "axes": base["axes"], "split": base["split"], "graphst_anchor": graph_source, "claim_boundary": ["GraphST current PASS is scoped to a supplied fresh current independent anchor.", "Historical continuity is UNKNOWN unless separately proven.", "No anchor or mismatched/tampered bytes yield UNRESOLVED; no current output is relabeled historical PASS.", "No overall six-method acceptance is implied."]}


def write_registry(registry: Mapping[str, Any], output_dir: Path | str, *, filename: str = "evidence_registry_v3_20260830.json") -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Current-anchor-bound six-method evidence registry")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--graphst-anchor", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="evidence_registry_v3_20260830.json")
    args = parser.parse_args(argv)
    registry = build_registry(args.project_root, graphst_anchor=args.graphst_anchor, split_manifest=args.split_manifest)
    path = write_registry(registry, args.output_dir, filename=args.filename) if args.output_dir else None
    print(json.dumps({"status": "READY", "ledger_cells": len(registry["ledger"]), "coverage": registry["coverage"], "method_status": registry["method_status"], "graphst_anchor": registry["graphst_anchor"], "output": str(path) if path else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
