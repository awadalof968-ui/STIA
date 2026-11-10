"""Versioned, fail-closed validation for donor-level disease evidence.

This helper validates traceability and contract shape only.  It deliberately
does not implement or infer a disease estimator: ``invest_st.metrics`` still
exposes a placeholder disease metric, and a validated receipt is not a
scientific or external-validation claim.  The current engine is not imported
or monkey-patched here; a reviewed integration patch is supplied separately
by the F4 terminal audit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stia.disease_evidence.v1"
METRIC_ID = "disease_score"
REQUIRED_INPUT_KEYS = (
    "metadata_sha256",
    "embedding_sha256",
    "spot_ids_sha256",
    "sample_ids_sha256",
    "donor_ids_sha256",
    "n_spots",
    "n_donors",
)


class DiseaseEvidenceError(ValueError):
    """Raised only for programmer-facing invalid invocation arguments."""


def _canonical_hash(values: Sequence[Any]) -> str:
    payload = json.dumps([str(value) for value in values], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _check(checks: list[dict[str, Any]], name: str, passed: bool | None, detail: str, **extra: Any) -> None:
    row: dict[str, Any] = {"name": name, "status": "PASS" if passed is True else "FAIL" if passed is False else "UNRESOLVED", "detail": detail}
    row.update(extra)
    checks.append(row)


def _resolve(path_value: Any, root: Path | None) -> Path | None:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else ((root or Path.cwd()) / path).resolve()


def _identity(metadata: Any, config: Mapping[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    if not isinstance(config, Mapping):
        raise DiseaseEvidenceError("config must be a mapping")
    names = {
        "spot": str(config.get("spot_id", "spot_id")),
        "sample": str(config.get("sample_id", "sample_id")),
        "donor": str(config.get("donor_id", "donor_id")),
    }
    columns: dict[str, list[str]] = {}
    for role, column in names.items():
        try:
            values = metadata[column].astype(str).tolist()
        except Exception as exc:
            raise DiseaseEvidenceError(f"metadata column unavailable: {role}={column}") from exc
        if not values or any(not value for value in values):
            raise DiseaseEvidenceError(f"metadata {role} IDs must be non-empty")
        columns[role] = values
    if len(set(columns["spot"])) != len(columns["spot"]):
        raise DiseaseEvidenceError("metadata spot IDs must be unique")
    if len(set(zip(columns["spot"], columns["sample"], columns["donor"]))) != len(columns["spot"]):
        raise DiseaseEvidenceError("metadata spot/sample/donor identity rows must be unique")
    donors = sorted(set(columns["donor"]))
    return columns, donors


def validate_disease_evidence(
    artifact: Mapping[str, Any] | None,
    *,
    metadata: Any,
    embedding: Any,
    embedding_spot_ids: Sequence[Any],
    config: Mapping[str, Any],
    metadata_source: str | Path | None,
    embedding_source: str | Path | None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one donor-level disease receipt against frozen inputs.

    ``None`` means the endpoint/evidence is unavailable and returns
    ``UNRESOLVED``.  Any supplied but malformed or contradictory field is a
    hard ``FAIL``.  The artifact's own ``status`` field is retained as an
    untrusted input and never determines the returned status.
    """
    if artifact is None:
        return {"schema_version": SCHEMA_VERSION, "status": "UNRESOLVED", "reason": "disease evidence artifact is unavailable", "checks": [], "errors": ["artifact is required"]}
    if not isinstance(artifact, Mapping):
        return {"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": "disease evidence artifact must be a mapping", "checks": [], "errors": ["artifact must be a mapping"]}

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    root = Path(source_root).resolve() if source_root is not None else None
    try:
        columns, donors = _identity(metadata, config)
    except DiseaseEvidenceError as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": "metadata identity is invalid", "checks": [], "errors": [str(exc)]}

    schema = artifact.get("schema_version")
    _check(checks, "schema_version", schema == SCHEMA_VERSION, "versioned disease evidence schema is required", expected=SCHEMA_VERSION, observed=schema)
    metric_id = artifact.get("metric_id")
    _check(checks, "metric_id", metric_id == METRIC_ID, "the canonical disease metric identifier is required", expected=METRIC_ID, observed=metric_id)
    value = artifact.get("metric_value")
    value_ok = _number(value) and 0.0 <= float(value) <= 1.0
    _check(checks, "metric_value", value_ok, "metric value must be finite, non-boolean and within the declared [0,1] score domain", observed=value)
    unit = artifact.get("statistical_unit")
    _check(checks, "statistical_unit", unit == "donor", "disease evidence must be donor-level", expected="donor", observed=unit)
    expected_estimand = config.get("estimand")
    observed_estimand = artifact.get("estimand")
    _check(checks, "estimand", bool(expected_estimand) and observed_estimand == expected_estimand, "artifact estimand must match the frozen config estimand", expected=expected_estimand, observed=observed_estimand)

    input_identity = artifact.get("input_identity")
    if not isinstance(input_identity, Mapping):
        _check(checks, "input_identity", False, "metadata/embedding/ID identity envelope is required")
        input_identity = {}
    else:
        _check(checks, "input_identity", True, "input identity envelope supplied")
    metadata_path = _resolve(metadata_source, root)
    embedding_path = _resolve(embedding_source, root)
    metadata_hash = _sha256_file(metadata_path) if metadata_path is not None and metadata_path.is_file() else None
    embedding_hash = _sha256_file(embedding_path) if embedding_path is not None and embedding_path.is_file() else None
    _check(checks, "metadata_source", metadata_hash is not None, "metadata source hash can be measured")
    _check(checks, "embedding_source", embedding_hash is not None, "embedding source hash can be measured")
    try:
        shape = tuple(int(value) for value in getattr(embedding, "shape"))
    except Exception:
        shape = ()
    ids = [str(value) for value in embedding_spot_ids]
    _check(checks, "embedding_shape", len(shape) == 2 and shape[0] == len(columns["spot"]), "embedding rows match metadata spots", observed=shape, expected_rows=len(columns["spot"]))
    _check(checks, "embedding_spot_ids", len(ids) == len(columns["spot"]) and len(set(ids)) == len(ids) and set(ids) == set(columns["spot"]), "embedding spot IDs match metadata by set")
    expected_ids = {
        "metadata_sha256": metadata_hash,
        "embedding_sha256": embedding_hash,
        "spot_ids_sha256": _canonical_hash(columns["spot"]),
        "sample_ids_sha256": _canonical_hash(columns["sample"]),
        "donor_ids_sha256": _canonical_hash(columns["donor"]),
        "n_spots": len(columns["spot"]),
        "n_donors": len(donors),
    }
    for key in REQUIRED_INPUT_KEYS:
        observed = input_identity.get(key)
        _check(checks, f"input_identity:{key}", observed == expected_ids[key], "input identity matches metadata/embedding", expected=expected_ids[key], observed=observed)

    scores = artifact.get("donor_scores")
    scores_ok = isinstance(scores, Mapping) and set(str(key) for key in scores) == set(donors)
    _check(checks, "donor_scores_keys", scores_ok, "one finite score is required for every observed donor", expected=donors, observed=sorted(str(key) for key in scores) if isinstance(scores, Mapping) else None)
    if isinstance(scores, Mapping):
        for donor in donors:
            score = scores.get(donor)
            _check(checks, f"donor_score:{donor}", _number(score), "donor score is finite and non-boolean", observed=score)

    source = artifact.get("source_artifact")
    if not isinstance(source, Mapping):
        _check(checks, "source_artifact", False, "source artifact path/hash/role is required")
    else:
        source_path = _resolve(source.get("path"), root)
        source_hash = _sha256_file(source_path) if source_path is not None and source_path.is_file() else None
        _check(checks, "source_artifact_path", source_hash is not None, "source artifact exists")
        _check(checks, "source_artifact_hash", isinstance(source.get("sha256"), str) and source.get("sha256") == source_hash, "source artifact hash matches bytes", expected=source.get("sha256"), observed=source_hash)
        _check(checks, "source_artifact_role", bool(source.get("role")), "source artifact role is explicit")

    status = "FAIL" if any(check["status"] == "FAIL" for check in checks) else "UNRESOLVED" if any(check["status"] == "UNRESOLVED" for check in checks) else "PASS"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "reason": "traceability and donor-level contract checks completed",
        "checks": checks,
        "errors": errors,
        "untrusted_declared_status": artifact.get("status"),
        "metric_id": metric_id,
        "estimand": observed_estimand,
        "statistical_unit": unit,
        "input_identity": dict(input_identity),
    }


def validate_disease_evidence_file(
    artifact_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load one JSON receipt and delegate to :func:`validate_disease_evidence`."""
    path = Path(artifact_path)
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "status": "UNRESOLVED", "reason": "disease evidence artifact is unavailable", "checks": [], "errors": [f"missing artifact: {path}"]}
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": "disease evidence artifact is not valid JSON", "checks": [], "errors": [str(exc)]}
    return validate_disease_evidence(artifact, **kwargs)


# Compatibility aliases for callers that use the descriptive names from the
# framework plan.  They do not alter any shared import surface.
validate_metric_artifact = validate_disease_evidence
validate_disease_axis_evidence = validate_disease_evidence


__all__ = [
    "SCHEMA_VERSION",
    "METRIC_ID",
    "DiseaseEvidenceError",
    "validate_disease_evidence",
    "validate_disease_evidence_file",
    "validate_metric_artifact",
    "validate_disease_axis_evidence",
]
