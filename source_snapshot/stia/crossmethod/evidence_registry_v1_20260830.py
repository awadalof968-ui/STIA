"""Additive six-method evidence registry for the frozen STIA contract.

The registry reads existing JSON acceptance/manifest/summary files only.  It
never loads model arrays, launches a fit, or treats copied prose as an oracle.
Each cell is kept as a separate ledger record so incomplete, inapplicable and
summary-only evidence cannot be silently pooled into a PASS.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

METHODS = ("PCA", "Harmony", "GraphST", "STAligner", "PRECAST", "INSPIRE")
CONDITIONS = (
    "balanced_orthogonal_positive",
    "batch_only",
    "disease_only_no_batch",
    "null_no_batch",
)
SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
ORDERS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
MARGIN = 0.05
STATUSES = ("PASS", "FAIL", "NOT_APPLICABLE", "UNRESOLVED")
PROJECT_DEFAULT = Path(r"LOCAL_INPUT_ROOT\Downloads\INSPIRE-main\INSPIRE-main\INSPIRE")
PCA_SUMMARY_FALLBACK = Path(
    r"LOCAL_INPUT_ROOT\Documents\Codex\2026-08-17\document-meta-title-inspire-created-at"
    r"\work\stia_crossmethod_20260821\runs\pca_controls\four_condition_summary.json"
)
PCA_TERMINAL_FALLBACK = Path(
    r"LOCAL_INPUT_ROOT\Documents\Codex\2026-08-17\document-meta-title-inspire-created-at"
    r"\work\stia_crossmethod_20260821\runs\pca_controls\terminal_validation.json"
)


class EvidenceValidationError(ValueError):
    """Raised when an evidence record cannot be bound to its requested cell."""


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


def resolve_reference(project_root: Path | str, reference: str | Path) -> Path | None:
    """Resolve a producer path without treating a missing path as PASS."""
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return candidate if candidate.is_file() or candidate.is_dir() else None


def _condition(value: Any) -> str | None:
    aliases = {
        "balanced-positive": "balanced_orthogonal_positive",
        "balanced_orthogonal_diffuse_ds10": "balanced_orthogonal_positive",
    }
    if value is None:
        return None
    return aliases.get(str(value), str(value))


def _cell(method: str, condition: str, seed: int, *, status: str, tier: str,
          source: str | None, reason: str, **extra: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"invalid status {status!r}")
    result: dict[str, Any] = {
        "method": method,
        "condition": condition,
        "seed": int(seed),
        "status": status,
        "evidence_tier": tier,
        "source": source,
        "reason": reason,
    }
    result.update(extra)
    return result


def validate_cell_record(
    record: Mapping[str, Any], method: str, condition: str, seed: int,
    *, expected_plan_hash: str | None = None,
) -> dict[str, Any]:
    """Validate method/cell/plan identity before a record enters the ledger."""
    if record.get("method") != method:
        raise EvidenceValidationError(f"method mismatch: {record.get('method')!r} != {method!r}")
    if _condition(record.get("condition")) != condition:
        raise EvidenceValidationError(f"condition mismatch: {record.get('condition')!r} != {condition!r}")
    try:
        actual_seed = int(record.get("seed"))
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError("seed is missing or malformed") from exc
    if actual_seed != int(seed):
        raise EvidenceValidationError(f"seed mismatch: {actual_seed} != {seed}")
    if expected_plan_hash is not None and record.get("plan_hash") != expected_plan_hash:
        raise EvidenceValidationError("plan hash binding mismatch")
    if record.get("status") not in STATUSES:
        raise EvidenceValidationError(f"invalid ledger status: {record.get('status')!r}")
    return dict(record)


def _safe_path(project_root: Path, value: Any) -> Path | None:
    return resolve_reference(project_root, value) if isinstance(value, (str, Path)) else None


def _validate_manifest(
    project_root: Path, path: Path, *, method: str, condition: str, seed: int,
    expected_plan_hash: str | None = None,
) -> dict[str, Any]:
    manifest = _json(path)
    if manifest.get("method") != method:
        raise EvidenceValidationError(f"manifest method mismatch: {manifest.get('method')!r}")
    if _condition(manifest.get("condition")) != condition:
        raise EvidenceValidationError(f"manifest condition mismatch: {manifest.get('condition')!r}")
    try:
        actual_seed = int(manifest.get("model_seed"))
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError("manifest model_seed missing or malformed") from exc
    if actual_seed != int(seed):
        raise EvidenceValidationError(f"manifest seed mismatch: {actual_seed} != {seed}")
    if expected_plan_hash is not None and manifest.get("plan_hash") != expected_plan_hash:
        raise EvidenceValidationError("manifest plan hash mismatch")
    if tuple(manifest.get("orders", ())) != ORDERS:
        raise EvidenceValidationError("manifest order contract mismatch")
    expected_n = 14999 if condition in {"disease_only_no_batch", "null_no_batch"} else 15000
    if int(manifest.get("n_input", -1)) != expected_n or int(manifest.get("n_post_qc", -1)) != expected_n:
        raise EvidenceValidationError(f"manifest input/QC count mismatch (expected {expected_n})")
    output_dir = _safe_path(project_root, manifest.get("output_dir"))
    if output_dir is None or not output_dir.is_dir():
        raise EvidenceValidationError("manifest output_dir is missing")
    return manifest


def _empty_cells(method: str, reason: str, tier: str = "missing") -> list[dict[str, Any]]:
    return [_cell(method, condition, seed, status="UNRESOLVED", tier=tier, source=None, reason=reason)
            for condition in CONDITIONS for seed in SEEDS]


def _ingest_pca(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [project_root / "work" / "stia_crossmethod_20260821" / "runs" / "pca_controls" / "four_condition_summary.json", PCA_SUMMARY_FALLBACK]
    summary_path = next((path for path in candidates if path.is_file()), None)
    terminal_path = project_root / "work" / "stia_crossmethod_20260821" / "runs" / "pca_controls" / "terminal_validation.json"
    if not terminal_path.is_file() and PCA_TERMINAL_FALLBACK.is_file():
        terminal_path = PCA_TERMINAL_FALLBACK
    if summary_path is None:
        return _empty_cells("PCA", "historical summary evidence is missing; strict raw arm evidence unresolved"), {"status": "UNRESOLVED", "source": None}
    try:
        summary = _json(summary_path)
    except EvidenceValidationError as exc:
        return _empty_cells("PCA", str(exc)), {"status": "UNRESOLVED", "source": str(summary_path)}
    if summary.get("method") != "PCA":
        return _empty_cells("PCA", "historical summary method binding failed"), {"status": "UNRESOLVED", "source": str(summary_path)}
    terminal_ok = terminal_path.is_file()
    cells: list[dict[str, Any]] = []
    condition_records = summary.get("conditions", {})
    for condition in CONDITIONS:
        lookup = "balanced_orthogonal_diffuse_ds10" if condition == "balanced_orthogonal_positive" else condition
        source_record = condition_records.get(lookup) if isinstance(condition_records, dict) else None
        for seed in SEEDS:
            source = str(summary_path)
            if not isinstance(source_record, dict):
                reason = "historical condition summary missing; strict raw arm evidence unresolved"
            elif source_record.get("summary", {}).get("n_pass") != 5:
                reason = "historical summary is not complete for this condition"
            else:
                reason = "summary reports completion, but no raw NPZ arms; hard-coded/derived OSI cannot enter strict E2"
            cells.append(_cell("PCA", condition, seed, status="UNRESOLVED", tier="summary_only", source=source, reason=reason, source_terminal_present=terminal_ok, raw_recomputable=False))
    return cells, {"status": "UNRESOLVED", "source": str(summary_path), "terminal_source": str(terminal_path) if terminal_ok else None, "raw_npz_arms": 0}


def _harmony_file(root: Path, condition: str, seed: int) -> Path:
    return root / "results" / "external" / "stageC" / "staligner_osi" / f"harmony_osi_{condition}_seed{seed}.json"


def _ingest_harmony(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    root = project_root / "results" / "external" / "stageC" / "staligner_osi"
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = _harmony_file(project_root, condition, seed)
            if not path.is_file():
                cells.append(_cell("Harmony", condition, seed, status="UNRESOLVED", tier="missing", source=None, reason="Harmony source summary is missing"))
                continue
            try:
                record = _json(path)
                validate_cell_record({"method": record.get("method"), "condition": record.get("condition"), "seed": record.get("seed", seed), "status": "UNRESOLVED"}, "Harmony", condition, seed)
            except EvidenceValidationError as exc:
                cells.append(_cell("Harmony", condition, seed, status="UNRESOLVED", tier="summary_only", source=str(path), reason=str(exc)))
                continue
            summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
            if condition in {"disease_only_no_batch", "null_no_batch"}:
                if summary.get("status") == "NOT_APPLICABLE" and "NOT_APPLICABLE" in str(summary.get("harmony_applicability", "")) + str(summary.get("status")):
                    cells.append(_cell("Harmony", condition, seed, status="NOT_APPLICABLE", tier="applicability", source=str(path), reason=str(summary.get("reason", "Harmony requires >=2 batch levels")), raw_recomputable=False))
                else:
                    cells.append(_cell("Harmony", condition, seed, status="UNRESOLVED", tier="summary_only", source=str(path), reason="no-batch condition is not explicitly marked NOT_APPLICABLE"))
                continue
            numeric = {"osi": summary.get("osi"), "pass_reported": summary.get("pass"), "drifts": summary.get("drifts")}
            if record.get("n_orders", 4) != 4 or record.get("margin") != MARGIN:
                reason = "order/margin contract mismatch"
            elif record.get("retained_ids", {}).get("all_arms_same_set_unordered") is not True:
                reason = "retained spot identity is not proven"
            else:
                reason = "summary-only Harmony result; no raw NPZ or unified acceptance"
            cells.append(_cell("Harmony", condition, seed, status="UNRESOLVED", tier="summary_only", source=str(path), reason=reason, numeric=numeric, raw_recomputable=False, seed_semantics="deterministic repeat"))
    return cells, {"status": "UNRESOLVED", "source_root": str(root), "applicable_cells": 10, "not_applicable_cells": 10, "raw_npz": False}


def _graphst_file(root: Path, condition: str, seed: int) -> Path:
    return root / "results" / "external" / "stageC" / "graphst_osi_v2_20260823" / f"graphst_osi_v2_{condition}_seed{seed}.json"


def _ingest_graphst(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = project_root / "results" / "external" / "stageC" / "graphst_osi_v2_20260823"
    acceptance_path = project_root / "docs" / "GRAPHST_V2_INDEPENDENT_ACCEPTANCE_20260829.json"
    try:
        acceptance = _json(acceptance_path)
        identity = acceptance.get("experiment_identity", {})
        checks = acceptance.get("completion_checks", {})
        acceptance_ok = acceptance.get("document_meta", {}).get("status") == "PASS" and identity.get("method") == "GraphST" and checks.get("cells_passed") == 20 and checks.get("independent_osi_recomputed") == 20 and checks.get("four_distinct_orders_checked") is True
    except EvidenceValidationError:
        acceptance_ok = False
    cells: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = _graphst_file(project_root, condition, seed)
            if not path.is_file():
                cells.append(_cell("GraphST", condition, seed, status="UNRESOLVED", tier="missing", source=None, reason="GraphST aggregate is missing"))
                continue
            try:
                record = _json(path)
                validate_cell_record({"method": record.get("method"), "condition": record.get("condition"), "seed": record.get("seed"), "status": "PASS" if acceptance_ok else "UNRESOLVED"}, "GraphST", condition, seed)
                arm_json = record.get("arm_json")
                if not isinstance(arm_json, dict) or tuple(arm_json) != ORDERS:
                    raise EvidenceValidationError("GraphST arm order keys mismatch")
                for arm, name in arm_json.items():
                    arm_path = root / str(name)
                    npz_path = arm_path.with_suffix(".npz")
                    if not arm_path.is_file() or not npz_path.is_file():
                        raise EvidenceValidationError(f"GraphST raw arm/NPZ missing for {arm}")
                if record.get("status") != "OK":
                    raise EvidenceValidationError("GraphST aggregate status is not OK")
                numeric = {"osi": record.get("osi"), "pass_reported": record.get("pass"), "classification": record.get("classification")}
                cells.append(_cell("GraphST", condition, seed, status="PASS" if acceptance_ok else "UNRESOLVED", tier="raw_recomputable" if acceptance_ok else "raw_present_unaccepted", source=str(path), reason="independent raw-NPZ acceptance and four-arm closure" if acceptance_ok else "raw present but authoritative acceptance document is not PASS", numeric=numeric, raw_recomputable=True))
            except (EvidenceValidationError, TypeError, ValueError) as exc:
                cells.append(_cell("GraphST", condition, seed, status="UNRESOLVED", tier="raw_present_unaccepted", source=str(path), reason=str(exc), raw_recomputable=False))
    return cells, {"status": "PASS" if acceptance_ok else "UNRESOLVED", "source": str(acceptance_path), "output_root": str(root), "raw_recomputable": acceptance_ok}


def _staligner_file(project_root: Path, condition: str) -> Path:
    names = {"balanced_orthogonal_positive": "balanced_orthogonal_diffuse_ds10", "batch_only": "batch_only_diffuse_ds10", "disease_only_no_batch": "disease_only_no_batch_diffuse_ds10", "null_no_batch": "null_no_batch_diffuse_ds10"}
    return project_root / "results" / "external" / "stageC" / "staligner_osi" / f"{names[condition]}.json"


def _ingest_staligner(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        path = _staligner_file(project_root, condition)
        try:
            record = _json(path)
            if record.get("method") != "STAligner":
                raise EvidenceValidationError("STAligner method binding mismatch")
            raw_orders = record.get("orders", ORDERS)
            if record.get("n_orders") != 4 or (raw_orders not in (4, list(ORDERS), tuple(ORDERS), ORDERS)):  # old aggregate has integer orders
                raise EvidenceValidationError("STAligner order contract mismatch")
            units = {int(item["seed"]): item for item in record.get("units", []) if isinstance(item, dict) and "seed" in item}
        except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
            for seed in SEEDS:
                cells.append(_cell("STAligner", condition, seed, status="UNRESOLVED", tier="summary_only", source=str(path), reason=str(exc)))
            continue
        for seed in SEEDS:
            unit = units.get(seed)
            if unit is None:
                cells.append(_cell("STAligner", condition, seed, status="UNRESOLVED", tier="missing", source=str(path), reason="seed unit missing from aggregate"))
                continue
            order_keys = tuple(unit.get("order_keys", ORDERS))
            if int(unit.get("n_orders", 4)) != 4 or order_keys != ORDERS:
                reason = "unit order contract mismatch"
            else:
                reason = "numeric aggregate present but no raw NPZ/source-config closure for unified acceptance"
            cells.append(_cell("STAligner", condition, seed, status="UNRESOLVED", tier="summary_only", source=str(path), reason=reason, numeric={"osi": unit.get("osi"), "pass_reported": unit.get("pass"), "drifts": unit.get("drifts")}, raw_recomputable=False))
    return cells, {"status": "UNRESOLVED", "source_root": str(path.parent), "aggregate_units": 20, "raw_recomputable": False}


def _precast_manifest(project_root: Path) -> tuple[Path | None, dict[str, Any] | None, Path | None]:
    doc_path = project_root / "docs" / "PRECAST_V16_FINAL_ACCEPTANCE_V2_20260828.json"
    try:
        doc = _json(doc_path)
        identity = doc["experiment_identity"]
        manifest_path = _safe_path(project_root, identity["manifest"])
        if manifest_path is None:
            raise EvidenceValidationError("PRECAST manifest missing")
        manifest = _json(manifest_path)
        return doc_path, doc, manifest_path
    except (EvidenceValidationError, KeyError, TypeError):
        return doc_path, None, None


def _ingest_precast(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc_path, doc, manifest_path = _precast_manifest(project_root)
    cells: list[dict[str, Any]] = []
    accepted: dict[tuple[str, int], Path] = {}
    manifest: dict[str, Any] | None = None
    if doc is not None and manifest_path is not None:
        try:
            manifest = _json(manifest_path)
            if manifest.get("run_id") != doc.get("experiment_identity", {}).get("run_id") or manifest.get("plan_hash") != doc.get("experiment_identity", {}).get("plan_hash"):
                raise EvidenceValidationError("PRECAST manifest run/plan binding mismatch")
            for item in manifest.get("completed_cells", []):
                condition = _condition(item.get("condition"))
                seed = int(item.get("seed"))
                output = _safe_path(project_root, item.get("output_path"))
                if condition in CONDITIONS and output is not None:
                    accepted[(condition, seed)] = output
        except (EvidenceValidationError, TypeError, ValueError, KeyError):
            manifest = None
    for condition in CONDITIONS:
        for seed in SEEDS:
            output = accepted.get((condition, seed))
            if output is None:
                cells.append(_cell("PRECAST", condition, seed, status="UNRESOLVED", tier="missing", source=str(doc_path), reason="v16 accepted manifest does not cover this cell; gap runs are not spliced"))
                continue
            try:
                record = _json(output)
                expected_plan = manifest.get("plan_hash") if manifest else None
                validate_cell_record({"method": record.get("method"), "condition": record.get("condition"), "seed": record.get("seed"), "status": "PASS", "plan_hash": record.get("plan_hash")}, "PRECAST", condition, seed, expected_plan_hash=expected_plan)
                if record.get("status") != "OK" or record.get("run_id") != manifest.get("run_id"):
                    raise EvidenceValidationError("PRECAST output run/status binding mismatch")
                arms = record.get("arms", {})
                if tuple(arms) != ORDERS:
                    raise EvidenceValidationError("PRECAST arm order keys mismatch")
                for arm in ORDERS:
                    arm_info = arms[arm]
                    if arm_info.get("status") != "OK" or _safe_path(project_root, arm_info.get("npz_path")) is None:
                        raise EvidenceValidationError(f"PRECAST raw arm missing for {arm}")
                cells.append(_cell("PRECAST", condition, seed, status="PASS", tier="raw_recomputable", source=str(output), reason="v16 source-aware manifest, four-arm NPZ closure and plan binding", plan_hash=record.get("plan_hash"), raw_recomputable=True, numeric={"osi": record.get("summary", {}).get("osi"), "pass_reported": record.get("summary", {}).get("pass"), "classification": record.get("summary", {}).get("classification")}))
            except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
                cells.append(_cell("PRECAST", condition, seed, status="UNRESOLVED", tier="raw_present_unaccepted", source=str(output), reason=str(exc), raw_recomputable=False))
    return cells, {"status": "PASS" if len(accepted) == 14 and manifest is not None else "UNRESOLVED", "source": str(doc_path), "manifest": str(manifest_path) if manifest_path else None, "accepted_cells": len(accepted), "raw_recomputable_cells": sum(cell["status"] == "PASS" for cell in cells)}


def _inspire_doc(project_root: Path) -> Path:
    return project_root / "docs" / "INSPIRE_20CELL_MATRIX_FINAL_ACCEPTANCE_20260829.json"


def _ingest_inspire(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc_path = _inspire_doc(project_root)
    cells: list[dict[str, Any]] = []
    try:
        doc = _json(doc_path)
        identity = doc.get("experiment_identity", {})
        if identity.get("method") != "INSPIRE" or doc.get("acceptance_summary", {}).get("decision") != "PASS":
            raise EvidenceValidationError("INSPIRE acceptance document is not PASS or method-bound")
        registry = {(str(_condition(item.get("condition"))), int(item.get("seed"))): item for item in doc.get("canonical_registry", [])}
    except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
        return _empty_cells("INSPIRE", str(exc)), {"status": "UNRESOLVED", "source": str(doc_path)}
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = registry.get((condition, seed))
            if not isinstance(entry, dict):
                cells.append(_cell("INSPIRE", condition, seed, status="UNRESOLVED", tier="missing", source=str(doc_path), reason="canonical registry cell missing"))
                continue
            try:
                manifest_path = _safe_path(project_root, entry.get("manifest_path"))
                if manifest_path is None:
                    raise EvidenceValidationError("INSPIRE manifest missing")
                manifest = _validate_manifest(project_root, manifest_path, method="INSPIRE", condition=condition, seed=seed, expected_plan_hash=entry.get("plan_hash"))
                arms = entry.get("arms", {})
                if tuple(arms) != ORDERS:
                    raise EvidenceValidationError("INSPIRE canonical arm keys mismatch")
                for arm in ORDERS:
                    arm_entry = arms[arm]
                    if _safe_path(project_root, arm_entry.get("npz_path")) is None:
                        raise EvidenceValidationError(f"INSPIRE raw arm NPZ missing for {arm}")
                if entry.get("valid") is not True:
                    raise EvidenceValidationError("INSPIRE canonical entry is not valid")
                numeric = {name: entry.get("summary", {}).get(name) for name in ("latent", "shared", "gamma", "full")}
                cells.append(_cell("INSPIRE", condition, seed, status="PASS", tier="raw_recomputable", source=str(manifest_path), reason="canonical acceptance, manifest plan binding and four-arm NPZ closure", raw_recomputable=True, representations=list(manifest.get("representations", [])), numeric=numeric))
            except (EvidenceValidationError, TypeError, ValueError, KeyError) as exc:
                cells.append(_cell("INSPIRE", condition, seed, status="UNRESOLVED", tier="raw_present_unaccepted", source=str(doc_path), reason=str(exc), raw_recomputable=False))
    return cells, {"status": "PASS" if sum(cell["status"] == "PASS" for cell in cells) == 20 else "UNRESOLVED", "source": str(doc_path), "accepted_cells": sum(cell["status"] == "PASS" for cell in cells), "raw_recomputable": True}


def _coverage(ledger: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result = {method: {status: 0 for status in STATUSES} for method in METHODS}
    for cell in ledger:
        result.setdefault(str(cell["method"]), {status: 0 for status in STATUSES})[str(cell["status"])] += 1
    return result


def build_registry(project_root: Path | str = PROJECT_DEFAULT, *, checked_at: str | None = None) -> dict[str, Any]:
    """Ingest the existing six-method producer evidence into a new ledger."""
    root = Path(project_root)
    ingest = {
        "PCA": _ingest_pca,
        "Harmony": _ingest_harmony,
        "GraphST": _ingest_graphst,
        "STAligner": _ingest_staligner,
        "PRECAST": _ingest_precast,
        "INSPIRE": _ingest_inspire,
    }
    ledger: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for method in METHODS:
        cells, source = ingest[method](root)
        ledger.extend(cells)
        sources[method] = source
    return {
        "schema_version": "stia.evidence_registry.v1.20260830",
        "checked_at": checked_at or _now(),
        "project_root": str(root),
        "contract": {"methods": list(METHODS), "conditions": list(CONDITIONS), "seeds": list(SEEDS), "orders": list(ORDERS), "margin": MARGIN, "statuses": list(STATUSES)},
        "sources": sources,
        "ledger": ledger,
        "coverage": _coverage(ledger),
        "axes": {"order_sensitivity": "source numeric fields retained per cell; no cross-method averaging", "three_axis": "historical/global/conditional summaries are references only", "representation": "method-specific units retained; INSPIRE latent/shared/gamma/full kept separate", "provenance_qc": "identity, manifest/plan binding, raw availability and numerical result are separate fields"},
        "split_authority": {"status": "UNVERIFIED", "reason": "No concrete frozen discovery/calibration/locked-test assignment artifact was found; policy/API alone is insufficient."},
        "claim_boundary": ["This registry is not an overall six-method acceptance.", "0.05 remains an operational/provisional margin.", "NOT_APPLICABLE is not a numerical negative; unresolved cells are not failures.", "Same-seed replays are excluded from independent replicate counts."],
    }


def write_registry(registry: Mapping[str, Any], output_dir: Path | str, *, filename: str = "evidence_registry_v1_20260830.json") -> Path:
    """Write a versioned output without overwriting an existing artifact."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only six-method evidence registry")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="evidence_registry_v1_20260830.json")
    args = parser.parse_args(argv)
    registry = build_registry(args.project_root)
    path = write_registry(registry, args.output_dir, filename=args.filename) if args.output_dir else None
    print(json.dumps({"status": "READY", "ledger_cells": len(registry["ledger"]), "coverage": registry["coverage"], "output": str(path) if path else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
