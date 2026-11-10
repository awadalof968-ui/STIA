"""Six-method evidence consolidation v7 (U9).

This additive successor integrates the accepted Harmony H7R receipt and fixes
canonical PRECAST replay accounting.  It never edits the v6 registry or any
producer receipt: a valid same-key replay contributes one canonical row and
one excluded alias.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import evidence_registry_v2_20260830 as _v2
from . import evidence_registry_v5_20260830 as _v5
from . import evidence_registry_v6_20260831 as _v6

PROJECT_DEFAULT = _v6.PROJECT_DEFAULT
METHODS = _v6.METHODS
CONDITIONS = _v6.CONDITIONS
SEEDS = _v6.SEEDS
ORDERS = _v5.ORDERS
STATUSES = _v6.STATUSES
HARMONY_ACCEPTANCE_DEFAULT = Path("docs/HARMONY_H4_FINAL_ACCEPTANCE_20260831_r1.json")
PRECAST_REPLAY_DEFAULT = Path("docs/PRECAST_R2_DISEASE_CELL_ACCEPTANCE_20260830.json")
SCHEMA_VERSION = "stia.evidence_registry.v7.20260831"
BOUND_MANIFEST_SCHEMA = "stia.evidence_registry.v7.bound_manifest.v1"
_HARMONY_APPLICABLE = {"balanced_orthogonal_positive", "batch_only"}
_HARMONY_NOT_APPLICABLE = {"disease_only_no_batch", "null_no_batch"}
_REPLAY_KEY = ("disease_only_no_batch", 20260819)


class EvidenceValidationError(ValueError):
    """Raised when a receipt cannot be bound to its independent evidence."""


def _resolve(root: Path, value: Path | str | None) -> Path | None:
    if value is None:
        return None
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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(float(value)) != float("inf")


def _empty_harmony(source: str, reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [{"method": "Harmony", "condition": condition, "seed": seed, "status": "UNRESOLVED", "evidence_tier": "harmony_acceptance_missing_or_invalid", "source": source, "reason": reason, "execution_validity": "UNRESOLVED"} for condition in CONDITIONS for seed in SEEDS]
    return rows, {"status": "UNRESOLVED", "source": source, "accepted_cells": 0, "applicable_cells": 0, "not_applicable_cells": 0, "execution_valid_cells": 0, "target_cells": 20, "reason": reason}


def ingest_harmony(project_root: Path | str, acceptance: Path | str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the formal H4 acceptance and its independent H7R audit."""
    root = Path(project_root)
    value = acceptance if acceptance is not None else HARMONY_ACCEPTANCE_DEFAULT
    path = _resolve(root, value)
    source = str(value)
    try:
        if path is None:
            raise EvidenceValidationError("Harmony acceptance path is not resolvable")
        doc = _json(path)
        if doc.get("schema_version") != "inspire.method_order_audit.acceptance.v1" or doc.get("method") != "Harmony" or doc.get("decision") not in {"PASS", "CONDITIONAL_PASS"} or doc.get("execution_validity") != "PASS":
            raise EvidenceValidationError("Harmony acceptance envelope is invalid")
        coverage = doc.get("coverage"); numeric = doc.get("numeric_classification"); independent = doc.get("independent_terminal_audit")
        if not isinstance(coverage, Mapping) or coverage.get("applicable_cells") != 10 or coverage.get("explicit_not_applicable_cells") != 10 or coverage.get("raw_arms") != 40 or coverage.get("unresolved") != 0:
            raise EvidenceValidationError("Harmony coverage is not 10 applicable + 10 explicit NA")
        if not isinstance(numeric, Mapping) or numeric.get("margin") != 0.05 or numeric.get("n_pass") != 10 or numeric.get("n_fail") != 0 or numeric.get("n_not_applicable") != 10 or numeric.get("n_unresolved") != 0:
            raise EvidenceValidationError("Harmony numeric classification is inconsistent")
        if not isinstance(independent, Mapping) or not isinstance(independent.get("path"), str) or not _valid_hash(independent.get("sha256")):
            raise EvidenceValidationError("Harmony independent audit binding is missing")
        audit_path = _resolve(root, independent["path"])
        if audit_path is None or _sha256(audit_path) != independent["sha256"]:
            raise EvidenceValidationError("Harmony independent audit hash is stale")
        audit = _json(audit_path)
        if audit.get("execution_validity") != "PASS" or audit.get("errors") not in ([], None) or not isinstance(audit.get("audit_version"), str):
            raise EvidenceValidationError("Harmony terminal audit envelope is invalid")
        cells = audit.get("cells")
        if not isinstance(cells, list) or len(cells) != 20:
            raise EvidenceValidationError("Harmony terminal audit does not cover 20 cells")
        expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}; seen: set[tuple[str, int]] = set(); rows: list[dict[str, Any]] = []; pass_count = 0; na_count = 0
        for cell in cells:
            if not isinstance(cell, Mapping) or cell.get("audit") != "PASS" or cell.get("execution_status") != "PASS" or cell.get("condition") not in CONDITIONS or not isinstance(cell.get("seed"), int) or cell.get("seed") not in SEEDS:
                raise EvidenceValidationError("Harmony terminal cell envelope is malformed")
            key = (str(cell["condition"]), int(cell["seed"]))
            if key in seen:
                raise EvidenceValidationError(f"Harmony duplicate cell: {key}")
            seen.add(key)
            status = cell.get("numeric_status")
            if key[0] in _HARMONY_APPLICABLE:
                if status != "PASS" or cell.get("n_pairs") != 6 or not _finite(cell.get("osi")):
                    raise EvidenceValidationError(f"Harmony applicable cell is not numeric PASS: {key}")
                pass_count += 1
                rows.append({"method": "Harmony", "condition": key[0], "seed": key[1], "status": "PASS", "evidence_tier": "accepted_harmony_raw_receipt", "source": str(path), "audit_source": str(audit_path), "reason": "H7R independent terminal audit; applicable four-order cell execution and OSI PASS", "execution_validity": "PASS", "numeric_status": "PASS", "raw_recomputable": True, "numeric": {"osi": float(cell["osi"]), "margin": 0.05, "classification": "ORDER_INVARIANT_WITHIN_MARGIN"}})
            elif key[0] in _HARMONY_NOT_APPLICABLE:
                if status != "NOT_APPLICABLE":
                    raise EvidenceValidationError(f"Harmony NA cell is not explicitly NOT_APPLICABLE: {key}")
                na_count += 1
                rows.append({"method": "Harmony", "condition": key[0], "seed": key[1], "status": "NOT_APPLICABLE", "evidence_tier": "accepted_harmony_explicit_na", "source": str(path), "audit_source": str(audit_path), "reason": "H7R marks single-batch condition not applicable; not a numerical stability success", "execution_validity": "PASS", "numeric_status": "NOT_APPLICABLE", "raw_recomputable": True})
        if seen != expected or pass_count != 10 or na_count != 10:
            raise EvidenceValidationError("Harmony terminal cell identities/counts are incomplete")
        return rows, {"status": "PASS", "source": str(path), "audit_source": str(audit_path), "accepted_cells": 10, "applicable_cells": 10, "not_applicable_cells": 10, "numeric_pass_cells": 10, "numeric_fail_cells": 0, "execution_valid_cells": 20, "target_cells": 20, "decision": doc.get("decision"), "execution_validity": "PASS", "audit_version": audit.get("audit_version")}
    except (EvidenceValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return _empty_harmony(source, str(exc))


def _record_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    config = record.get("config") if isinstance(record.get("config"), Mapping) else {}
    return {"condition": record.get("condition"), "seed": record.get("seed"), "input_sha256": record.get("input_sha256"), "panel_sha256": record.get("panel_sha256"), "source_sha256": record.get("source_sha256"), "r_core_sha256": record.get("r_core_sha256"), "runner_sha256": record.get("runner_sha256"), "continuation_sha256": record.get("continuation_sha256"), "implementation_sha256": record.get("implementation_sha256"), "margin": record.get("margin"), "config": {key: config.get(key) for key in ("K", "q", "beta_grid", "coreNum", "coreNum_int", "seed")}}


def _retention_signature(candidate: Mapping[str, Any]) -> dict[str, Any]:
    record = candidate["record"]; cell = candidate["cell"]; retention = cell.get("independent_source_retention") if isinstance(cell.get("independent_source_retention"), Mapping) else {}
    arms = record.get("arms") if isinstance(record.get("arms"), Mapping) else {}
    arm_signature: dict[str, Any] = {}
    for order in ORDERS:
        arm = arms.get(order) if isinstance(arms.get(order), Mapping) else {}
        arm_signature[order] = {key: arm.get(key) for key in ("retained_count", "retained_ids_unordered_sha256", "donor_keys_ordered_sha256", "block_sizes_in", "block_sizes_out")}
    return {"retained_count": retention.get("retained_count"), "retained_ids_unordered_sha256": retention.get("retained_ids_unordered_sha256"), "donor_levels_first_occurrence": retention.get("donor_levels_first_occurrence"), "arm_signature": arm_signature}


def select_canonical_replay(p2_candidate: Mapping[str, Any], current_candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Select one representative for a valid replay key; fail closed on mismatch."""
    p2_record = p2_candidate.get("record"); current_record = current_candidate.get("record")
    if not isinstance(p2_record, Mapping) or not isinstance(current_record, Mapping):
        raise EvidenceValidationError("PRECAST replay candidates lack records")
    mismatches: list[str] = []
    if _record_signature(p2_record) != _record_signature(current_record):
        p2_sig = _record_signature(p2_record); current_sig = _record_signature(current_record)
        for key in p2_sig:
            if p2_sig[key] != current_sig[key]:
                mismatches.append(key)
    p2_ret = _retention_signature(p2_candidate); current_ret = _retention_signature(current_candidate)
    if p2_ret != current_ret:
        for key in p2_ret:
            if p2_ret[key] != current_ret[key]:
                mismatches.append(f"retention.{key}")
    if mismatches:
        raise EvidenceValidationError("PRECAST replay candidates are incompatible: " + ", ".join(sorted(set(mismatches))))
    return {"status": "PASS", "key": "disease_only_no_batch:seed20260819", "canonical": "current_gap4", "covered_key_count": 1, "duplicate_alias_count": 1, "duplicate_aliases": [str(p2_candidate.get("record_path"))], "canonical_source": str(current_candidate.get("record_path")), "compatibility": "input/panel/retained-ID/donor/order semantics and method lineage match; run, plan, artifact and arm-output hashes remain run-scoped"}


def _candidate_from_receipt(root: Path, acceptance_path: Path, *, current: bool) -> dict[str, Any]:
    doc = _json(acceptance_path)
    if current:
        if doc.get("schema_version") != "inspire.experiment.final_acceptance.v1" or doc.get("acceptance_decision") not in {"PASS", "CONDITIONAL_PASS"} or doc.get("execution_validity") != "PASS":
            raise EvidenceValidationError("current PRECAST acceptance is invalid")
        inventory = doc.get("evidence_inventory"); audit_ref = inventory.get("audit_json") if isinstance(inventory, Mapping) else None
        audit_path = _resolve(root, audit_ref.get("path") if isinstance(audit_ref, Mapping) else audit_ref)
    else:
        acceptance_decision = doc.get("acceptance_decision")
        if doc.get("schema_version") != "inspire.single_cell_acceptance.v1" or not isinstance(acceptance_decision, Mapping) or acceptance_decision.get("decision") != "PASS":
            raise EvidenceValidationError("P2R PRECAST replay acceptance is invalid")
        inventory = doc.get("evidence_inventory"); audit_path = _resolve(root, inventory.get("audit_json") if isinstance(inventory, Mapping) else None)
    if audit_path is None:
        raise EvidenceValidationError("PRECAST replay audit path is missing")
    audit = _json(audit_path); cells = audit.get("cell_results")
    cell = cells.get("disease_only_no_batch:seed20260819") if isinstance(cells, Mapping) else None
    if not isinstance(cell, Mapping) or cell.get("status") != "PASS":
        raise EvidenceValidationError("PRECAST replay cell is not PASS")
    record_path = _resolve(root, cell.get("cell_json"))
    if record_path is None:
        raise EvidenceValidationError("PRECAST replay cell JSON is missing")
    record = _json(record_path)
    identity = audit.get("experiment_identity") if isinstance(audit.get("experiment_identity"), Mapping) else {}
    run_id = identity.get("run_id") or record.get("run_id"); plan_hash = identity.get("plan_hash") or record.get("plan_hash")
    if not isinstance(run_id, str) or not isinstance(plan_hash, str):
        raise EvidenceValidationError("PRECAST replay run/plan identity is missing")
    _v2.validate_binding(record, method="PRECAST", condition="disease_only_no_batch", seed=20260819, run_id=run_id, plan_hash=plan_hash)
    if record.get("status") != "OK":
        raise EvidenceValidationError("PRECAST replay record is not terminal OK")
    _v2._hash_chain(record, ("input_sha256", "panel_sha256", "source_sha256", "r_core_sha256", "runner_sha256", "continuation_sha256", "artifact_sha256", "implementation_sha256"), "PRECAST replay")
    if tuple(record.get("arms", {})) != ORDERS:
        raise EvidenceValidationError("PRECAST replay arm order contract is incomplete")
    retention = cell.get("independent_source_retention")
    if not isinstance(retention, Mapping) or retention.get("retained_count") != 14999 or not _valid_hash(retention.get("retained_ids_unordered_sha256")):
        raise EvidenceValidationError("PRECAST replay source retention is missing")
    order_checks = cell.get("order_checks")
    if not isinstance(order_checks, Mapping) or not all(value is True for value in order_checks.values()):
        raise EvidenceValidationError("PRECAST replay prescribed-order checks are incomplete")
    audit_arms = cell.get("arms")
    if not isinstance(audit_arms, Mapping):
        raise EvidenceValidationError("PRECAST replay audit arms are missing")
    for order in ORDERS:
        record_arm = record["arms"].get(order); audit_arm = audit_arms.get(order)
        if not isinstance(record_arm, Mapping) or not isinstance(audit_arm, Mapping) or audit_arm.get("npz_sha256") != record_arm.get("npz_sha256") or not _valid_hash(record_arm.get("npz_sha256")):
            raise EvidenceValidationError(f"PRECAST replay {order} NPZ hash is not cell-bound")
        if not isinstance(audit_arm.get("checks"), Mapping) or not all(value is True for value in audit_arm["checks"].values()) or not isinstance(audit_arm.get("hash_checks"), Mapping) or not all(value is True for value in audit_arm["hash_checks"].values()):
            raise EvidenceValidationError(f"PRECAST replay {order} audit checks are incomplete")
        npz_path = _resolve(root, record_arm.get("npz_path"))
        if npz_path is None or not npz_path.is_file() or str(_resolve(root, audit_arm.get("path"))) != str(npz_path):
            raise EvidenceValidationError(f"PRECAST replay {order} NPZ path is not bound")
        ok, detail = _v2._npz_check(npz_path, record_arm.get("npz_sha256"))
        if not ok:
            raise EvidenceValidationError(detail)
        if record_arm.get("retained_count") != retention.get("retained_count") or record_arm.get("retained_ids_unordered_sha256") != retention.get("retained_ids_unordered_sha256"):
            raise EvidenceValidationError(f"PRECAST replay {order} retained identity mismatch")
    return {"acceptance_path": str(acceptance_path), "audit_path": str(audit_path), "cell": cell, "record": record, "record_path": str(record_path), "current": current}


def _canonicalize_precast(root: Path, base_registry: Mapping[str, Any], current_acceptance: Path | str | None, replay_acceptance: Path | str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_path = _resolve(root, current_acceptance if current_acceptance is not None else _v5.FINAL_ACCEPTANCE_DEFAULT)
    replay_path = _resolve(root, replay_acceptance if replay_acceptance is not None else PRECAST_REPLAY_DEFAULT)
    if current_path is None or replay_path is None:
        raise EvidenceValidationError("PRECAST canonical replay paths are not resolvable")
    current = _candidate_from_receipt(root, current_path, current=True); replay = _candidate_from_receipt(root, replay_path, current=False)
    decision = select_canonical_replay(replay, current)
    rows = [dict(row) for row in base_registry.get("ledger", []) if row.get("method") == "PRECAST"]
    key_rows = [row for row in rows if (row.get("condition"), int(row.get("seed"))) == _REPLAY_KEY]
    if len(key_rows) != 1:
        raise EvidenceValidationError("base PRECAST ledger does not contain exactly one replay key row")
    canonical = key_rows[0]
    if canonical.get("source") != current["record_path"]:
        raise EvidenceValidationError("base PRECAST replay row is not bound to current gap4 record")
    canonical.update({"status": "PASS", "evidence_tier": "canonical_current_final_raw_receipt", "reason": "current full gap4 disease seed19 selected as the sole canonical representative; P2R same-key replay retained as excluded alias", "canonical_representative": True, "duplicate_aliases": [replay["record_path"]]})
    for index, row in enumerate(rows):
        if (row.get("condition"), int(row.get("seed"))) == _REPLAY_KEY:
            rows[index] = canonical
    accepted = sum(row.get("status") == "PASS" for row in rows)
    decision.update({"accepted_cells": accepted, "target_cells": 20, "p2r_acceptance": str(replay_path), "current_acceptance": str(current_path), "p2r_audit": replay["audit_path"], "current_audit": current["audit_path"]})
    return rows, decision


def build_registry(project_root: Path | str = PROJECT_DEFAULT, *, staligner_acceptance: Path | str | None = None, harmony_acceptance: Path | str | None = None, precast_acceptance: Path | str | None = None, precast_replay_acceptance: Path | str | None = None, pca_acceptance: Path | str | None = None, graphst_anchor: Path | str | None = _v6._v5.GRAPHST_ANCHOR_DEFAULT, split_manifest: Path | str | None = None, source_snapshot: Path | str | None = None, checked_at: str | None = None) -> dict[str, Any]:
    root = Path(project_root); when = checked_at or _datetime.datetime.now().astimezone().isoformat()
    base = _v6.build_registry(root, staligner_acceptance=staligner_acceptance, precast_acceptance=precast_acceptance, pca_acceptance=pca_acceptance, graphst_anchor=graphst_anchor, split_manifest=split_manifest, source_snapshot=source_snapshot, checked_at=when)
    harmony_rows, harmony_source = ingest_harmony(root, harmony_acceptance)
    precast_rows, precast_replay = _canonicalize_precast(root, base, precast_acceptance, precast_replay_acceptance)
    ledger = [row for row in base["ledger"] if row.get("method") not in {"Harmony", "PRECAST"}] + precast_rows + harmony_rows
    ledger.sort(key=lambda row: (METHODS.index(row["method"]), CONDITIONS.index(row["condition"]), int(row["seed"])))
    coverage = {method: {status: sum(row.get("method") == method and row.get("status") == status for row in ledger) for status in STATUSES} for method in METHODS}
    method_status = dict(base.get("method_status", {}))
    method_status["Harmony"] = {"status": "PASS" if coverage["Harmony"]["PASS"] == 10 and coverage["Harmony"]["NOT_APPLICABLE"] == 10 else "UNRESOLVED", "accepted_cells": coverage["Harmony"]["PASS"], "applicable_cells": coverage["Harmony"]["PASS"], "not_applicable_cells": coverage["Harmony"]["NOT_APPLICABLE"], "execution_valid_cells": harmony_source.get("execution_valid_cells", 0), "target_cells": 20, "numeric_fail_cells": 0, "numeric_significance": "not_applicable_to_the_10_NA_cells"}
    method_status["PRECAST"] = {"status": "PASS" if coverage["PRECAST"]["PASS"] == 20 else "PARTIAL", "accepted_cells": coverage["PRECAST"]["PASS"], "target_cells": 20, "canonical_replay": "one_current_gap4_representative_plus_one_excluded_alias", "duplicate_alias_count": precast_replay.get("duplicate_alias_count", 0)}
    claim_boundary = ["Harmony has 10 applicable numeric PASS cells plus 10 explicit NOT_APPLICABLE cells; NA is not a numerical stability success.", "PRECAST uses one current gap4 disease seed19 canonical representative; the compatible P2R same-key receipt is an excluded alias, not a second replicate.", "STAligner execution validity remains separate from its 11 numeric PASS and 9 numeric FAIL classifications.", "No universal threshold, external validation, causal claim, or framework-wide success is licensed."]
    return {"schema_version": SCHEMA_VERSION, "checked_at": when, "project_root": str(root), "contract": base["contract"], "sources": {**base.get("sources", {}), "Harmony": harmony_source, "PRECAST": precast_replay}, "ledger": ledger, "coverage": coverage, "method_status": method_status, "axes": base["axes"], "split": base["split"], "graphst_anchor": base.get("graphst_anchor"), "pca_source": base.get("pca_source"), "precast_source": precast_replay, "staligner_source": base.get("staligner_source"), "harmony_source": harmony_source, "precast_replay": precast_replay, "source_snapshots": base.get("source_snapshots"), "claim_boundary": claim_boundary}


def _fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def write_registry(registry: Mapping[str, Any], output_dir: Path | str, *, filename: str = "evidence_registry_v7_20260831.json") -> Path:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def write_bound_manifest(registry: Mapping[str, Any], registry_path: Path, output_dir: Path | str, *, filename: str = "U9_BOUND_MANIFEST_20260831.json") -> Path:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); path = destination / filename; root = Path(str(registry.get("project_root", PROJECT_DEFAULT)))
    values: list[Any] = []
    values.extend([registry.get("staligner_source", {}).get("source"), registry.get("staligner_source", {}).get("audit_source"), registry.get("pca_source", {}).get("source"), registry.get("graphst_anchor", {}).get("anchor"), registry.get("sources", {}).get("INSPIRE", {}).get("source"), registry.get("harmony_source", {}).get("source"), registry.get("harmony_source", {}).get("audit_source"), registry.get("precast_replay", {}).get("current_acceptance"), registry.get("precast_replay", {}).get("current_audit"), registry.get("precast_replay", {}).get("p2r_acceptance"), registry.get("precast_replay", {}).get("p2r_audit")])
    inputs: list[dict[str, Any]] = []; seen: set[str] = set()
    for value in values:
        path_value = _resolve(root, value) if isinstance(value, str) else None
        if path_value is None or not path_value.is_file() or str(path_value.resolve()) in seen:
            continue
        seen.add(str(path_value.resolve())); inputs.append(_fingerprint(path_value))
    project_root = Path(__file__).resolve().parents[2]; implementation = [_fingerprint(Path(__file__).resolve())]; test_path = project_root / "tests" / "test_evidence_registry_v7_20260831.py"
    if test_path.is_file():
        implementation.append(_fingerprint(test_path))
    payload = {"schema_version": BOUND_MANIFEST_SCHEMA, "checked_at": registry.get("checked_at"), "registry": _fingerprint(registry_path), "implementation": implementation, "inputs": inputs, "coverage": registry.get("coverage"), "method_status": registry.get("method_status"), "precast_replay": registry.get("precast_replay"), "harmony_source": registry.get("harmony_source"), "claim_boundary": registry.get("claim_boundary")}
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Six-method STIA evidence consolidation v7")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--staligner-acceptance", type=Path)
    parser.add_argument("--harmony-acceptance", type=Path, default=HARMONY_ACCEPTANCE_DEFAULT)
    parser.add_argument("--precast-acceptance", type=Path)
    parser.add_argument("--precast-replay-acceptance", type=Path, default=PRECAST_REPLAY_DEFAULT)
    parser.add_argument("--pca-acceptance", type=Path)
    parser.add_argument("--graphst-anchor", type=Path, default=_v6._v5.GRAPHST_ANCHOR_DEFAULT)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="evidence_registry_v7_20260831.json")
    parser.add_argument("--bound-manifest-filename", default="U9_BOUND_MANIFEST_20260831.json")
    args = parser.parse_args(argv)
    registry = build_registry(args.project_root, staligner_acceptance=args.staligner_acceptance, harmony_acceptance=args.harmony_acceptance, precast_acceptance=args.precast_acceptance, precast_replay_acceptance=args.precast_replay_acceptance, pca_acceptance=args.pca_acceptance, graphst_anchor=args.graphst_anchor, split_manifest=args.split_manifest, source_snapshot=args.source_snapshot)
    output = None; bound = None
    if args.output_dir:
        output = write_registry(registry, args.output_dir, filename=args.filename); bound = write_bound_manifest(registry, output, args.output_dir, filename=args.bound_manifest_filename)
    print(json.dumps({"status": "READY", "schema_version": registry["schema_version"], "ledger_cells": len(registry["ledger"]), "coverage": registry["coverage"], "method_status": registry["method_status"], "precast_replay": registry["precast_replay"], "harmony_source": registry["harmony_source"], "output": str(output) if output else None, "bound_manifest": str(bound) if bound else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
