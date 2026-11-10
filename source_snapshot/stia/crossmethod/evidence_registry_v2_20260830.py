"""Hash- and binding-closed successor to the U2 evidence registry.

This module is intentionally additive.  It reads producer metadata and raw
artifact headers without fitting or loading expression matrices, and keeps
source/acceptance defects as UNRESOLVED rather than manufacturing PASS rows.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import evidence_registry_v1_20260830 as _v1

METHODS = _v1.METHODS
CONDITIONS = _v1.CONDITIONS
SEEDS = _v1.SEEDS
ORDERS = _v1.ORDERS
STATUSES = _v1.STATUSES
MARGIN = _v1.MARGIN
PROJECT_DEFAULT = _v1.PROJECT_DEFAULT


class EvidenceValidationError(ValueError):
    """Raised for a malformed, unbound or tampered producer artifact."""


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
    return path if path.exists() else None


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


def _npz_check(path: Path, expected_sha256: Any) -> tuple[bool, str]:
    if not path.is_file():
        return False, "raw NPZ is missing"
    if not _valid_hash(expected_sha256):
        return False, "published raw NPZ hash is missing or malformed"
    actual = _sha256(path)
    if actual != expected_sha256:
        return False, f"raw NPZ hash mismatch: {actual} != {expected_sha256}"
    if not zipfile.is_zipfile(path):
        return False, "raw artifact is not a valid NPZ/ZIP container"
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            npy_names = [name for name in names if name.endswith(".npy")]
            if not npy_names:
                return False, "NPZ has no NPY members"
            for name in npy_names:
                with archive.open(name) as member:
                    if member.read(10)[:6] != b"\x93NUMPY":
                        return False, f"NPZ member is not NPY: {name}"
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        return False, f"invalid NPZ container: {exc}"
    return True, "hash and NPZ header verified"


def _cell(method: str, condition: str, seed: int, *, status: str, tier: str,
          source: str | None, reason: str, **extra: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(status)
    row = {"method": method, "condition": condition, "seed": int(seed), "status": status,
           "evidence_tier": tier, "source": source, "reason": reason}
    row.update(extra)
    return row


def _all_cells(method: str, reason: str, tier: str = "missing") -> list[dict[str, Any]]:
    return [_cell(method, condition, seed, status="UNRESOLVED", tier=tier, source=None, reason=reason)
            for condition in CONDITIONS for seed in SEEDS]


def validate_binding(record: Mapping[str, Any], *, method: str, condition: str, seed: int,
                     run_id: str | None = None, plan_hash: str | None = None,
                     order: str | None = None) -> None:
    """Check identity fields shared by all producer arm/cell records."""
    if record.get("method") != method:
        raise EvidenceValidationError(f"method mismatch: {record.get('method')!r}")
    if _v1._condition(record.get("condition")) != condition:
        raise EvidenceValidationError(f"condition mismatch: {record.get('condition')!r}")
    try:
        actual_seed = int(record.get("model_seed", record.get("seed")))
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError("seed missing or malformed") from exc
    if actual_seed != int(seed):
        raise EvidenceValidationError(f"seed mismatch: {actual_seed} != {seed}")
    if run_id is not None and record.get("run_id") != run_id:
        raise EvidenceValidationError("run binding mismatch")
    if plan_hash is not None and record.get("plan_hash") != plan_hash:
        raise EvidenceValidationError("plan binding mismatch")
    if order is not None and record.get("arm") != order:
        raise EvidenceValidationError("order/arm binding mismatch")


def classify_harmony_cell(condition: str, record: Mapping[str, Any]) -> str:
    """Keep no-batch applicability distinct from a numeric zero result."""
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    if condition in {"disease_only_no_batch", "null_no_batch"}:
        if summary.get("status") == "NOT_APPLICABLE" and "NOT_APPLICABLE" in str(summary.get("harmony_applicability", "")) + str(summary.get("status")):
            return "NOT_APPLICABLE"
        return "UNRESOLVED"
    return "UNRESOLVED"


def _hash_chain(record: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if not _valid_hash(record.get(field))]
    if missing:
        raise EvidenceValidationError(f"{label} hash chain missing/malformed: {', '.join(missing)}")


def _ids_and_counts(records: Iterable[Mapping[str, Any]], label: str) -> tuple[int, str]:
    rows = list(records)
    counts = [item.get("retained_count", item.get("n_post_qc")) for item in rows]
    ids = [item.get("retained_ids_unordered_sha256", item.get("canonical_id_hash", (item.get("raw_npz") or {}).get("canonical_id_hash"))) for item in rows]
    if not rows or any(not isinstance(value, int) or value <= 0 for value in counts):
        raise EvidenceValidationError(f"{label} retained counts are missing or invalid")
    if any(not _valid_hash(value) for value in ids):
        raise EvidenceValidationError(f"{label} unordered identity hashes are missing")
    if len(set(counts)) != 1 or len(set(ids)) != 1:
        raise EvidenceValidationError(f"{label} arms do not share one source-derived count/identity")
    return int(counts[0]), str(ids[0])


def _graphst_paths(project_root: Path) -> tuple[Path, Path]:
    return (project_root / "docs" / "GRAPHST_V2_INDEPENDENT_ACCEPTANCE_20260829.json",
            project_root / "results" / "external" / "stageC" / "graphst_osi_v2_20260823")


def _graphst_acceptance(project_root: Path) -> tuple[dict[str, Any], Path, Path]:
    acceptance_path, output_root = _graphst_paths(project_root)
    acceptance = _json(acceptance_path)
    meta = acceptance.get("document_meta", {})
    identity = acceptance.get("experiment_identity", {})
    inventory = acceptance.get("evidence_inventory", {})
    checks = acceptance.get("completion_checks", {})
    if meta.get("status") != "PASS" or identity.get("method") != "GraphST":
        raise EvidenceValidationError("GraphST acceptance envelope is not PASS/method-bound")
    linked_root = _resolve(project_root, inventory.get("output_root"))
    if linked_root is None or linked_root.resolve() != output_root.resolve():
        raise EvidenceValidationError("GraphST acceptance output root is not bound to current project root")
    auditor = _resolve(project_root, inventory.get("auditor_script"))
    if auditor is None:
        raise EvidenceValidationError("GraphST independent auditor script is missing")
    if checks.get("cells_passed") != checks.get("cells_checked") or checks.get("independent_osi_recomputed") != checks.get("cells_checked"):
        raise EvidenceValidationError("GraphST acceptance completion counts are inconsistent")
    if checks.get("four_distinct_orders_checked") is not True or checks.get("same_unordered_spot_set_checked") is not True:
        raise EvidenceValidationError("GraphST acceptance did not prove four orders/source identity")
    if inventory.get("raw_npz_reloaded_allow_pickle_false") is not True:
        raise EvidenceValidationError("GraphST acceptance does not attest non-pickle raw reload")
    return acceptance, acceptance_path, output_root


def ingest_graphst(project_root: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root)
    try:
        acceptance, acceptance_path, output_root = _graphst_acceptance(root)
        acceptance_bound = True
    except EvidenceValidationError as exc:
        acceptance = None
        acceptance_path = root / "docs" / "GRAPHST_V2_INDEPENDENT_ACCEPTANCE_20260829.json"
        output_root = root / "results" / "external" / "stageC" / "graphst_osi_v2_20260823"
        acceptance_bound = False
        acceptance_error = str(exc)
    cells: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            aggregate = output_root / f"graphst_osi_v2_{condition}_seed{seed}.json"
            try:
                record = _json(aggregate)
                validate_binding(record, method="GraphST", condition=condition, seed=seed)
                if record.get("status") != "OK":
                    raise EvidenceValidationError("GraphST aggregate is not terminal OK")
                arm_json = record.get("arm_json")
                if not isinstance(arm_json, dict) or tuple(arm_json) != ORDERS:
                    raise EvidenceValidationError("GraphST aggregate arm keys mismatch")
                arms: list[dict[str, Any]] = []
                for order in ORDERS:
                    arm_path = output_root / str(arm_json[order])
                    arm = _json(arm_path)
                    validate_binding(arm, method="GraphST", condition=condition, seed=seed, order=order)
                    if arm.get("status") != "OK":
                        raise EvidenceValidationError(f"GraphST arm {order} is not terminal OK")
                    _hash_chain(arm, ("input_sha256", "panel_sha256", "wrapper_sha256", "implementation_sha256", "latent_npz_sha256"), f"GraphST {order}")
                    npz_path = arm_path.parent / str(arm.get("latent_npz"))
                    ok, detail = _npz_check(npz_path, arm.get("latent_npz_sha256"))
                    if not ok:
                        raise EvidenceValidationError(detail)
                    if arm.get("retained_count") != arm.get("embedding_shape", [None])[0]:
                        raise EvidenceValidationError(f"GraphST {order} count/embedding shape mismatch")
                    arms.append(arm)
                count, identity_hash = _ids_and_counts(arms, "GraphST")
                numeric = {"osi": record.get("osi"), "pass_reported": record.get("pass"), "classification": record.get("classification")}
                status = "PASS" if acceptance_bound else "UNRESOLVED"
                reason = "independent acceptance envelope, source hashes, four arm JSON/NPZ hashes and NPZ headers verified" if acceptance_bound else acceptance_error
                cells.append(_cell("GraphST", condition, seed, status=status, tier="raw_recomputable" if acceptance_bound else "raw_present_unaccepted", source=str(aggregate), reason=reason, raw_recomputable=acceptance_bound, retained_count=count, retained_ids_unordered_sha256=identity_hash, numeric=numeric))
            except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
                cells.append(_cell("GraphST", condition, seed, status="UNRESOLVED", tier="raw_present_unaccepted", source=str(aggregate), reason=str(exc), raw_recomputable=False))
    accepted = sum(item["status"] == "PASS" for item in cells)
    expected = len(CONDITIONS) * len(SEEDS)
    return cells, {"status": "PASS" if accepted == expected else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": expected, "source": str(acceptance_path), "output_root": str(output_root)}


def _precast_acceptance(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    doc_path = project_root / "docs" / "PRECAST_V16_FINAL_ACCEPTANCE_V2_20260828.json"
    doc = _json(doc_path)
    meta = doc.get("document_meta", {})
    identity = doc.get("experiment_identity", {})
    if meta.get("status") != "PASS":
        raise EvidenceValidationError("PRECAST acceptance envelope is not PASS")
    manifest_path = _resolve(project_root, identity.get("manifest"))
    if manifest_path is None:
        raise EvidenceValidationError("PRECAST manifest is missing")
    manifest = _json(manifest_path)
    if manifest.get("run_id") != identity.get("run_id") or manifest.get("plan_hash") != identity.get("plan_hash"):
        raise EvidenceValidationError("PRECAST manifest run/plan binding mismatch")
    if manifest.get("status") != "OK" or manifest.get("is_canary") is not False:
        raise EvidenceValidationError("PRECAST manifest is not terminal non-canary")
    auditor = _resolve(project_root, meta.get("independent_auditor"))
    if auditor is None:
        raise EvidenceValidationError("PRECAST independent auditor script is missing")
    return doc, manifest, doc_path, manifest_path


def ingest_precast(project_root: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root)
    try:
        doc, manifest, doc_path, manifest_path = _precast_acceptance(root)
        accepted_map: dict[tuple[str, int], Path] = {}
        for item in manifest.get("completed_cells", []):
            condition = _v1._condition(item.get("condition"))
            seed = int(item.get("seed"))
            output = _resolve(root, item.get("output_path"))
            if condition in CONDITIONS and output is not None:
                accepted_map[(condition, seed)] = output
        acceptance_error = None
    except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
        doc_path = root / "docs" / "PRECAST_V16_FINAL_ACCEPTANCE_V2_20260828.json"
        manifest_path = root / "results" / "external" / "stageC" / "precast_osi_v16_continuation_20260826-precast-v16-full-20260826-r1" / "precast_v16_continuation_manifest.json"
        manifest = None
        accepted_map = {}
        doc = None
        acceptance_error = str(exc)
    cells: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            output = accepted_map.get((condition, seed))
            if output is None:
                reason = acceptance_error or "v16 manifest does not cover this cell; partial gap runs are not spliced"
                cells.append(_cell("PRECAST", condition, seed, status="UNRESOLVED", tier="missing", source=str(doc_path), reason=reason))
                continue
            try:
                record = _json(output)
                validate_binding(record, method="PRECAST", condition=condition, seed=seed, run_id=manifest.get("run_id"), plan_hash=manifest.get("plan_hash"))
                if record.get("status") != "OK":
                    raise EvidenceValidationError("PRECAST cell is not terminal OK")
                _hash_chain(record, ("input_sha256", "panel_sha256", "source_sha256", "r_core_sha256", "runner_sha256", "continuation_sha256", "artifact_sha256", "implementation_sha256"), "PRECAST cell")
                arms = record.get("arms")
                if not isinstance(arms, dict) or tuple(arms) != ORDERS:
                    raise EvidenceValidationError("PRECAST arm keys mismatch")
                arm_rows: list[dict[str, Any]] = []
                for order in ORDERS:
                    info = arms[order]
                    if info.get("status") != "OK":
                        raise EvidenceValidationError(f"PRECAST arm {order} is not terminal OK")
                    npz_path = _resolve(root, info.get("npz_path"))
                    ok, detail = _npz_check(npz_path, info.get("npz_sha256")) if npz_path else (False, "PRECAST raw NPZ is missing")
                    if not ok:
                        raise EvidenceValidationError(detail)
                    arm_rows.append(info)
                count, identity_hash = _ids_and_counts(arm_rows, "PRECAST")
                cells.append(_cell("PRECAST", condition, seed, status="PASS", tier="raw_recomputable", source=str(output), reason="v16 acceptance, manifest/plan binding and source-hashed four-arm NPZ closure", raw_recomputable=True, retained_count=count, retained_ids_unordered_sha256=identity_hash, numeric={"osi": record.get("summary", {}).get("osi"), "pass_reported": record.get("summary", {}).get("pass"), "classification": record.get("summary", {}).get("classification")}))
            except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
                cells.append(_cell("PRECAST", condition, seed, status="UNRESOLVED", tier="raw_present_unaccepted", source=str(output), reason=str(exc), raw_recomputable=False))
    accepted = sum(item["status"] == "PASS" for item in cells)
    expected = len(CONDITIONS) * len(SEEDS)
    return cells, {"status": "PASS" if accepted == expected else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": expected, "source": str(doc_path), "manifest": str(manifest_path)}


def _inspire_acceptance(project_root: Path) -> tuple[dict[str, Any], Path]:
    path = project_root / "docs" / "INSPIRE_20CELL_MATRIX_FINAL_ACCEPTANCE_20260829.json"
    doc = _json(path)
    identity = doc.get("experiment_identity", {})
    meta = doc.get("document_meta", {})
    if identity.get("method") != "INSPIRE" or doc.get("acceptance_summary", {}).get("decision") != "PASS":
        raise EvidenceValidationError("INSPIRE acceptance envelope is not PASS/method-bound")
    auditor = _resolve(project_root, meta.get("auditor_path"))
    if auditor is None or not _valid_hash(meta.get("auditor_sha256")):
        raise EvidenceValidationError("INSPIRE acceptance auditor/hash anchor is missing")
    return doc, path


def ingest_inspire(project_root: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(project_root)
    try:
        doc, doc_path = _inspire_acceptance(root)
        registry = {(str(_v1._condition(item.get("condition"))), int(item.get("seed"))): item for item in doc.get("canonical_registry", [])}
        acceptance_error = None
    except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
        doc_path = root / "docs" / "INSPIRE_20CELL_MATRIX_FINAL_ACCEPTANCE_20260829.json"
        registry = {}
        acceptance_error = str(exc)
    cells: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = registry.get((condition, seed))
            if not isinstance(entry, dict):
                cells.append(_cell("INSPIRE", condition, seed, status="UNRESOLVED", tier="missing", source=str(doc_path), reason=acceptance_error or "canonical registry cell missing"))
                continue
            try:
                manifest_path = _resolve(root, entry.get("manifest_path"))
                if manifest_path is None:
                    raise EvidenceValidationError("INSPIRE manifest missing")
                if not _valid_hash(entry.get("manifest_sha256")) or _sha256(manifest_path) != entry.get("manifest_sha256"):
                    raise EvidenceValidationError("INSPIRE manifest hash mismatch or missing")
                manifest = _json(manifest_path)
                validate_binding(manifest, method="INSPIRE", condition=condition, seed=seed, run_id=entry.get("run_id"), plan_hash=entry.get("plan_hash"))
                if manifest.get("status") != "ok" or manifest.get("finished") is not True:
                    raise EvidenceValidationError("INSPIRE manifest is not terminal finished")
                if not _valid_hash(manifest.get("frozen_state_sha256")) or not isinstance(manifest.get("input_sha256"), str):
                    raise EvidenceValidationError("INSPIRE manifest source-state/input hash chain is missing")
                plan_path = _resolve(root, entry.get("plan_path"))
                plan = _json(plan_path) if plan_path is not None else None
                if plan is None or not _valid_hash(entry.get("plan_hash")) or plan.get("plan_hash") != entry.get("plan_hash"):
                    raise EvidenceValidationError("INSPIRE plan binding/hash is missing or mismatched")
                arms = entry.get("arms")
                if not isinstance(arms, dict) or tuple(arms) != ORDERS:
                    raise EvidenceValidationError("INSPIRE canonical arm keys mismatch")
                arm_rows: list[dict[str, Any]] = []
                for order in ORDERS:
                    arm_ref = arms[order]
                    arm_path = _resolve(root, arm_ref.get("json_path"))
                    if arm_path is None or not _valid_hash(arm_ref.get("json_sha256")) or _sha256(arm_path) != arm_ref.get("json_sha256"):
                        raise EvidenceValidationError(f"INSPIRE {order} JSON hash/path is missing or mismatched")
                    arm = _json(arm_path)
                    validate_binding(arm, method="INSPIRE", condition=condition, seed=seed, order=order)
                    if str(arm.get("status", "")).lower() != "ok":
                        raise EvidenceValidationError(f"INSPIRE {order} arm is not terminal OK")
                    npz = arm.get("raw_npz")
                    npz_path = _resolve(root, arm_ref.get("npz_path"))
                    if npz_path is None or not _valid_hash(arm_ref.get("npz_sha256")):
                        raise EvidenceValidationError(f"INSPIRE {order} published NPZ hash/path missing")
                    ok, detail = _npz_check(npz_path, arm_ref.get("npz_sha256"))
                    if not ok:
                        raise EvidenceValidationError(detail)
                    if not isinstance(npz, dict) or npz.get("sha256") != arm_ref.get("npz_sha256") or npz.get("pickle") is not False:
                        raise EvidenceValidationError(f"INSPIRE {order} raw NPZ metadata is not bound")
                    for field in ("n_input", "n_post_qc"):
                        if not isinstance(arm.get(field), int) or arm[field] <= 0:
                            raise EvidenceValidationError(f"INSPIRE {order} {field} missing")
                    if arm.get("n_input") != arm.get("n_post_qc"):
                        raise EvidenceValidationError(f"INSPIRE {order} raw/post-QC count mismatch")
                    arm_rows.append(arm)
                count, identity_hash = _ids_and_counts(arm_rows, "INSPIRE")
                cells.append(_cell("INSPIRE", condition, seed, status="PASS" if entry.get("valid") is True else "UNRESOLVED", tier="raw_recomputable" if entry.get("valid") is True else "raw_present_unaccepted", source=str(manifest_path), reason="acceptance auditor anchor, manifest/plan/JSON/NPZ hash chain and four-arm format closure" if entry.get("valid") is True else "canonical entry is not valid", raw_recomputable=entry.get("valid") is True, retained_count=count, retained_ids_unordered_sha256=identity_hash, source_hash_status="acceptance_auditor_anchored; implementation hash not published", representations=list(manifest.get("representations", [])), numeric=entry.get("summary", {})))
            except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
                cells.append(_cell("INSPIRE", condition, seed, status="UNRESOLVED", tier="raw_present_unaccepted", source=str(doc_path), reason=str(exc), raw_recomputable=False))
    accepted = sum(item["status"] == "PASS" for item in cells)
    expected = len(CONDITIONS) * len(SEEDS)
    return cells, {"status": "PASS" if accepted == expected else ("PARTIAL" if accepted else "UNRESOLVED"), "accepted_cells": accepted, "target_cells": expected, "source": str(doc_path)}


def _split_evidence(project_root: Path, split_manifest: Path | str | None) -> dict[str, Any]:
    if split_manifest is None:
        return {"status": "UNRESOLVED", "source": None, "reason": "no concrete split manifest supplied"}
    path = _resolve(project_root, split_manifest)
    if path is None:
        return {"status": "UNRESOLVED", "source": str(split_manifest), "reason": "supplied split manifest is missing"}
    try:
        manifest = _json(path)
        checks = manifest.get("checks")
        if manifest.get("grouping_unit") != "donor" or not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
            raise EvidenceValidationError("split manifest donor/check contract is invalid")
        artifacts = manifest.get("artifacts", {})
        for name, info in artifacts.items():
            artifact = path.parent / name
            if not artifact.is_file() or not _valid_hash(info.get("sha256")) or _sha256(artifact) != info.get("sha256"):
                raise EvidenceValidationError(f"split artifact hash/path invalid: {name}")
        if not _valid_hash(manifest.get("plan_sha256")):
            raise EvidenceValidationError("split plan hash missing")
    except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
        return {"status": "UNRESOLVED", "source": str(path), "reason": str(exc)}
    return {"status": "PASS", "source": str(path), "reason": "concrete donor-level split manifest and artifact hashes verified", "mode": manifest.get("mode"), "grouping_unit": manifest.get("grouping_unit")}


def _axis_evidence(project_root: Path, ledger: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    root = Path(project_root)
    order_refs = [root / "docs" / "INSPIRE_20CELL_MATRIX_FINAL_ACCEPTANCE_20260829.json", root / "docs" / "GRAPHST_V2_INDEPENDENT_ACCEPTANCE_20260829.json", root / "docs" / "PRECAST_V16_FINAL_ACCEPTANCE_V2_20260828.json"]
    three_refs = [root / "docs" / "D19_CHECKPOINTS_D1D2_20260817_REPAIRED.json", root / "docs" / "INTEGRATED_AUDIT_FRAMEWORK_SPEC_AND_EVALUATION_20260821.json"]
    repr_refs = [root / "docs" / "INSPIRE_20CELL_MATRIX_FINAL_ACCEPTANCE_20260829.json"]
    def refs(paths: Iterable[Path]) -> list[str]:
        return [str(path) for path in paths if path.is_file()]
    order_paths, three_paths, repr_paths = refs(order_refs), refs(three_refs), refs(repr_refs)
    return {
        "order_sensitivity": {"status": "SOURCE_MEASURED" if order_paths else "UNRESOLVED", "references": order_paths, "unit_policy": "per-cell OSI/drifts; no cross-method pooling"},
        "three_axis_global_conditional": {"status": "SUMMARY_ONLY_UNIFIED_PENDING" if three_paths else "UNRESOLVED", "references": three_paths, "unit_policy": "historical named axes only; no invented metrics"},
        "representation_attribution": {"status": "SOURCE_MEASURED_INSPIRE_ONLY" if repr_paths else "UNRESOLVED", "references": repr_paths, "representations": ["latent", "shared", "gamma", "full"], "unit_policy": "method-specific representations remain separate"},
        "provenance_qc": {"status": "PER_CELL_LEDGER", "references": sorted({str(row["source"]) for row in ledger if row.get("source")}), "unit_policy": "identity, hashes, format, completion and numeric result are separate"},
    }


def build_registry(project_root: Path | str = PROJECT_DEFAULT, *, split_manifest: Path | str | None = None, checked_at: str | None = None) -> dict[str, Any]:
    root = Path(project_root)
    base = _v1.build_registry(root, checked_at=checked_at or _now())
    strict = {"GraphST": ingest_graphst, "PRECAST": ingest_precast, "INSPIRE": ingest_inspire}
    source_summaries = dict(base.get("sources", {}))
    ledger = [row for row in base["ledger"] if row.get("method") not in strict]
    method_status = {}
    for method, ingest in strict.items():
        rows, source = ingest(root)
        ledger.extend(rows)
        source_summaries[method] = source
    ledger.sort(key=lambda row: (METHODS.index(row["method"]), CONDITIONS.index(row["condition"]), int(row["seed"])))
    method_status.update({method: {"status": "PARTIAL" if method in {"PCA", "Harmony", "STAligner"} else "UNRESOLVED"} for method in METHODS})
    for method in METHODS:
        rows = [row for row in ledger if row["method"] == method]
        accepted = sum(row["status"] == "PASS" for row in rows)
        applicable = sum(row["status"] != "NOT_APPLICABLE" for row in rows)
        if method in strict:
            status = "PASS" if accepted == len(CONDITIONS) * len(SEEDS) else ("PARTIAL" if accepted else "UNRESOLVED")
        elif method == "Harmony":
            status = "PARTIAL"
        elif method == "PCA" or method == "STAligner":
            status = "UNRESOLVED"
        else:
            status = "UNRESOLVED"
        method_status[method] = {"status": status, "accepted_cells": accepted, "applicable_cells": applicable, "target_cells": len(CONDITIONS) * len(SEEDS)}
    return {
        "schema_version": "stia.evidence_registry.v2.20260830",
        "checked_at": checked_at or base["checked_at"],
        "project_root": str(root),
        "contract": {"methods": list(METHODS), "conditions": list(CONDITIONS), "seeds": list(SEEDS), "orders": list(ORDERS), "margin": MARGIN, "statuses": list(STATUSES)},
        "sources": source_summaries,
        "ledger": ledger,
        "coverage": {method: {status: sum(row["status"] == status for row in ledger if row["method"] == method) for status in STATUSES} for method in METHODS},
        "method_status": method_status,
        "axes": _axis_evidence(root, ledger),
        "split": _split_evidence(root, split_manifest),
        "claim_boundary": ["No overall six-method acceptance is implied.", "Operational margin 0.05 is not a validated universal cutoff.", "Missing/malformed hashes, raw formats, plans or splits remain UNRESOLVED.", "Same-seed replays are not independent ledger cells."],
    }


def write_registry(registry: Mapping[str, Any], output_dir: Path | str, *, filename: str = "evidence_registry_v2_20260830.json") -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hash-closed read-only six-method evidence registry")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="evidence_registry_v2_20260830.json")
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args(argv)
    registry = build_registry(args.project_root, split_manifest=args.split_manifest)
    path = write_registry(registry, args.output_dir, filename=args.filename) if args.output_dir else None
    print(json.dumps({"status": "READY", "ledger_cells": len(registry["ledger"]), "coverage": registry["coverage"], "method_status": registry["method_status"], "split": registry["split"], "output": str(path) if path else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
