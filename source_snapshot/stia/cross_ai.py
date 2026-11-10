# -*- coding: utf-8 -*-
"""Offline cross-AI answer ingestion and scoring for STIA.

The adapter deliberately stops at answer-level comparison.  It does not know
how an answer was generated and never invokes a provider SDK, a network, or a
model.  Inputs are small JSON-like manifests so that a frozen prompt/case,
identity, blinding attestation, and provenance can be reviewed before any
descriptive agreement is reported.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .baselines import framework_quality_metrics
from .contracts import SCHEMA_VERSION, TOOL_NAME, TOOL_VERSION, PRIMARY_METRIC_IDS
from .io import canonical_bytes, canonical_write, load_json_or_yaml


CROSS_AI_PROTOCOL_VERSION = "1.0.0"
CROSS_AI_BUNDLE_FILES = (
    "cross_ai_manifest.json",
    "cross_ai_agreement.json",
    "cross_ai_disagreements.json",
    "cross_ai_gold_metrics.json",
    "cross_ai_result.json",
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_ACCEPTED = {"ACCEPT", "ACCEPT_WITH_SCOPE"}
_FLAGGED = {"REJECT", "REPAIR", "FAIL"}
_UNKNOWN_MODEL_MARKERS = frozenset({
    "", "unknown", "unknown_model", "unknown_scorer", "missing", "none",
    "null", "n/a", "na", "not_provided", "unspecified",
})


class CrossAIValidationError(ValueError):
    """Raised only when a caller explicitly requests strict validation."""


def _value(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return load_json_or_yaml(path)
    return value


def _items(value: Any, key: str) -> List[Mapping[str, Any]]:
    value = _value(value)
    if isinstance(value, Mapping):
        value = value.get(key, value.get("records", value.get("items", value)))
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_base_model_key(value: Any) -> str | None:
    """Return the identity key used for independence checks.

    Provider and version are provenance attributes and intentionally do not
    participate in this key.  Empty/placeholder values are represented as
    ``None`` so that a missing model identity cannot accidentally satisfy a
    distinct-model or scorer-separation gate.
    """
    text = _text(value)
    if text is None:
        return None
    normalized = unicodedata.normalize("NFKC", text).strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return None if normalized in _UNKNOWN_MODEL_MARKERS else normalized


def _raw_model_id(raw: Mapping[str, Any]) -> Any:
    """Read an explicit base model identifier without provider/version data."""
    # ``model_identity`` is deliberately *not* a fallback: it commonly embeds
    # provider/version and is not an independence key.  Keep model_id as the
    # provenance value when both it and an explicit base_model_id are given.
    for key in ("model_id", "model"):
        if key in raw:
            return raw.get(key)
    return None


def _explicit_base_model_id(raw: Mapping[str, Any], raw_model: Any) -> Any:
    """Return an optional caller-supplied base key, else the model_id."""
    for key in ("base_model_id", "model_base_id"):
        if key in raw:
            return raw.get(key)
    return raw_model


def _sha(value: Any) -> str:
    import hashlib
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _hash_or_compute(raw: Mapping[str, Any], names: Sequence[str], payload_names: Sequence[str]) -> str | None:
    for name in names:
        found = _text(raw.get(name))
        if found:
            return found
    for name in payload_names:
        if name in raw and raw.get(name) is not None:
            return _sha(raw[name])
    return None


def _normalize_case(value: Any) -> Dict[str, Any]:
    raw = _value(value)
    if not isinstance(raw, Mapping):
        raise CrossAIValidationError("case manifest must be a mapping")
    nested = raw.get("case_identity")
    source = dict(nested) if isinstance(nested, Mapping) else dict(raw)
    case_id = _text(source.get("case_id", source.get("id")))
    if case_id is None:
        raise CrossAIValidationError("case manifest requires case_id")
    case_hash = _hash_or_compute(source, ("case_hash", "frozen_case_hash"), ("case_payload", "case"))
    prompt_hash = _hash_or_compute(source, ("prompt_hash", "frozen_prompt_hash"), ("prompt", "prompt_payload"))
    if case_hash is None:
        # A compact identity-only fixture may intentionally have no payload.
        case_hash = _sha({"case_id": case_id})
    if prompt_hash is None:
        prompt_hash = _sha({"case_id": case_id, "prompt": ""})
    case_ids = source.get("case_ids")
    if case_ids is None and isinstance(source.get("cases"), Sequence) and not isinstance(source.get("cases"), (str, bytes)):
        case_ids = [_text(item.get("case_id", item.get("id"))) for item in source["cases"] if isinstance(item, Mapping)]
    case_ids = [item for item in (_text(item) for item in (case_ids or [case_id])) if item]
    return {
        "case_id": case_id,
        "case_hash": case_hash,
        "prompt_hash": prompt_hash,
        "case_ids": list(dict.fromkeys(case_ids)),
        "identity_source": "explicit" if any(source.get(k) for k in ("case_hash", "frozen_case_hash", "prompt_hash", "frozen_prompt_hash")) else "computed",
    }


def _blind(value: Any) -> bool:
    if isinstance(value, Mapping):
        value = value.get("attested", value.get("blind", value.get("value", value.get("is_blind"))))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "blind", "blinded", "attested"}
    return False


def _finding_id(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("finding_id", "id", "code", "name", "finding"):
            text = _text(value.get(key))
            if text:
                return text
        return _sha(value)
    return _text(value) or ""


def _finding_set(raw: Mapping[str, Any]) -> List[str]:
    value = raw.get("findings", raw.get("finding_ids", raw.get("issues", [])))
    if isinstance(value, Mapping):
        value = list(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        value = [value] if value not in (None, "") else []
    return sorted(set(item for item in (_finding_id(x) for x in value) if item))


def _evidence_hashes(raw: Mapping[str, Any]) -> List[str]:
    value = raw.get("evidence_hashes", raw.get("evidence_sha256", raw.get("evidence_hash")))
    if isinstance(value, Mapping):
        value = [v.get("sha256", v.get("hash")) if isinstance(v, Mapping) else v for v in value.values()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sorted(set(item for item in (_text(v) for v in value) if item))
    text = _text(value)
    return [text] if text else []


def _normalize_answer(raw: Mapping[str, Any], index: int, case: Mapping[str, Any]) -> Dict[str, Any]:
    answer_id = _text(raw.get("answer_id", raw.get("id"))) or f"answer-{index + 1}"
    raw_model = _raw_model_id(raw)
    model_id = _text(raw_model) or "UNKNOWN_MODEL"
    normalized_model_id = _normalize_base_model_key(raw_model)
    base_model_id = _normalize_base_model_key(_explicit_base_model_id(raw, raw_model))
    provider = _text(raw.get("provider"))
    model_version = _text(raw.get("model_version", raw.get("version")))
    identity = ":".join(item for item in (provider, model_id, model_version) if item)
    session_id = _text(raw.get("session_id", raw.get("run_id"))) or f"session-{index + 1}"
    decision = _text(raw.get("decision", raw.get("status", raw.get("verdict"))))
    case_id = _text(raw.get("case_id", case.get("case_id")))
    case_hash = _text(raw.get("case_hash", raw.get("frozen_case_hash"))) or str(case["case_hash"])
    prompt_hash = _text(raw.get("prompt_hash", raw.get("frozen_prompt_hash"))) or str(case["prompt_hash"])
    root = raw.get("root_cause", raw.get("root_cause_id", raw.get("root_causes")))
    if isinstance(root, Sequence) and not isinstance(root, (str, bytes)):
        root = tuple(sorted(str(item) for item in root))
    else:
        root = _text(root)
    return {
        "answer_id": answer_id,
        "model_id": model_id,
        "normalized_model_id": normalized_model_id,
        "base_model_id": base_model_id,
        "model_identity": identity,
        "session_id": session_id,
        "case_id": case_id,
        "case_hash": case_hash,
        "prompt_hash": prompt_hash,
        "decision": decision.upper() if decision else None,
        "findings": _finding_set(raw),
        "root_cause": root,
        "evidence_hashes": _evidence_hashes(raw),
        "blindness_attestation": _blind(raw.get("blindness_attestation", raw.get("blind", raw.get("blinded")))),
        "scorer_id": _text(raw.get("scorer_id")),
    }


def _normalize_scorer(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    scorer_id = _text(raw.get("scorer_id", raw.get("id"))) or f"scorer-{index + 1}"
    raw_model = _raw_model_id(raw)
    model_id = _text(raw_model) or "UNKNOWN_SCORER"
    normalized_model_id = _normalize_base_model_key(raw_model)
    base_model_id = _normalize_base_model_key(_explicit_base_model_id(raw, raw_model))
    provider = _text(raw.get("provider"))
    model_version = _text(raw.get("model_version", raw.get("version")))
    return {
        "scorer_id": scorer_id,
        "model_id": model_id,
        "normalized_model_id": normalized_model_id,
        "base_model_id": base_model_id,
        "model_identity": ":".join(item for item in (provider, model_id, model_version) if item),
        "session_id": _text(raw.get("session_id", raw.get("run_id"))),
        "evidence_hashes": _evidence_hashes(raw),
        "blindness_attestation": _blind(raw.get("blindness_attestation", raw.get("blind", raw.get("blinded")))),
    }


def _normalize_gold(value: Any) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    if value is None:
        return None, []
    raw = _value(value)
    if isinstance(raw, Mapping):
        metadata = dict(raw)
        records = raw.get("records", raw.get("cases", raw.get("gold", [])))
        if isinstance(records, Mapping):
            records = [records]
    else:
        metadata, records = {}, raw
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        records = []
    rows: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        row["case_id"] = _text(row.get("case_id", row.get("id")))
        row["decision"] = _text(row.get("decision", row.get("gold_decision", row.get("expected_decision"))))
        row["root_cause"] = _text(row.get("root_cause", row.get("root_cause_gold", row.get("true_root_cause"))))
        rows.append(row)
    return metadata, rows


def _gate(gate_id: str, status: str, details: str, **extra: Any) -> Dict[str, Any]:
    return {"id": gate_id, "status": status, "details": details, **extra}


def validate_cross_ai_input(
    case: Any,
    answers: Any,
    *,
    scorers: Any = None,
    gold: Any = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Normalize manifests and evaluate all independence/provenance gates."""
    normalized_case = _normalize_case(case)
    answer_raw = _items(answers, "answers")
    normalized_answers = [_normalize_answer(item, i, normalized_case) for i, item in enumerate(answer_raw)]
    scorer_raw = _items(scorers, "scorers") if scorers is not None else []
    normalized_scorers = [_normalize_scorer(item, i) for i, item in enumerate(scorer_raw)]
    gold_meta, gold_rows = _normalize_gold(gold)
    gates: List[Dict[str, Any]] = []
    model_ids = {row["model_identity"] for row in normalized_answers}
    normalized_model_ids = {row["normalized_model_id"] for row in normalized_answers if row["normalized_model_id"] is not None}
    base_model_ids = {row["base_model_id"] for row in normalized_answers if row["base_model_id"] is not None}
    session_ids = [row["session_id"] for row in normalized_answers]
    answer_ids = [row["answer_id"] for row in normalized_answers]
    gates.append(_gate("minimum_answers", "PASS" if len(normalized_answers) >= 2 else "FAIL", f"{len(normalized_answers)} answer records"))
    known_answer_models = bool(normalized_answers) and all(
        row["normalized_model_id"] is not None and row["base_model_id"] is not None
        for row in normalized_answers
    )
    gates.append(_gate(
        "known_base_model_ids",
        "PASS" if known_answer_models else "FAIL",
        "every answer supplies an explicit model_id" if known_answer_models else "one or more answers lack a known model_id",
        normalized_model_ids=sorted(normalized_model_ids),
        base_model_ids=sorted(base_model_ids),
        unknown_answer_ids=[row["answer_id"] for row in normalized_answers if row["normalized_model_id"] is None or row["base_model_id"] is None],
    ))
    # Independence is defined by normalized *base* model IDs.  Provider and
    # version are retained in model_identity for provenance only and cannot
    # make two runs independent when they use the same underlying model.
    # A caller-supplied base_model_id may conservatively collapse aliases, but
    # it must never split an identical explicit model_id into a false pair.
    distinct = known_answer_models and len(normalized_model_ids) >= 2 and len(base_model_ids) >= 2
    gates.append(_gate(
        "distinct_model_identities",
        "PASS" if distinct else "FAIL",
        f"{len(normalized_model_ids)} explicit and {len(base_model_ids)} base model IDs" if known_answer_models else "distinctness is unresolved because a model_id is missing",
        model_identities=sorted(model_ids),
        normalized_model_ids=sorted(normalized_model_ids),
        base_model_ids=sorted(base_model_ids),
    ))
    unique_sessions = len(session_ids) == len(set(session_ids))
    gates.append(_gate("unique_sessions", "PASS" if unique_sessions else "FAIL", "session IDs are unique" if unique_sessions else "duplicate session IDs"))
    unique_answers = len(answer_ids) == len(set(answer_ids))
    gates.append(_gate("unique_answer_ids", "PASS" if unique_answers else "FAIL", "answer IDs are unique" if unique_answers else "duplicate answer IDs"))
    case_match = all(row["case_id"] in normalized_case["case_ids"] and row["case_hash"] == normalized_case["case_hash"] for row in normalized_answers)
    prompt_match = all(row["prompt_hash"] == normalized_case["prompt_hash"] for row in normalized_answers)
    gates.append(_gate("frozen_case_hash_match", "PASS" if case_match else "FAIL", "all answer case hashes match the frozen case" if case_match else "case identity/hash mismatch"))
    gates.append(_gate("frozen_prompt_hash_match", "PASS" if prompt_match else "FAIL", "all answer prompt hashes match the frozen prompt" if prompt_match else "prompt identity/hash mismatch"))
    blind = all(row["blindness_attestation"] for row in normalized_answers)
    gates.append(_gate("blindness_attestation", "PASS" if blind else "FAIL", "every answer has an explicit blindness attestation" if blind else "missing or false blindness attestation"))
    evidence = all(row["evidence_hashes"] and all(_text(item) for item in row["evidence_hashes"] for _ in [0]) for row in normalized_answers)
    gates.append(_gate("evidence_hash_completeness", "PASS" if evidence else "FAIL", "every answer supplies evidence hashes" if evidence else "one or more answers lack evidence hashes"))
    if normalized_scorers:
        scorer_model_ids = {row["normalized_model_id"] for row in normalized_scorers if row["normalized_model_id"] is not None}
        scorer_base_models = {row["base_model_id"] for row in normalized_scorers if row["base_model_id"] is not None}
        scorer_sessions = {row["session_id"] for row in normalized_scorers if row["session_id"]}
        known_scorer_models = all(
            row["normalized_model_id"] is not None and row["base_model_id"] is not None
            for row in normalized_scorers
        )
        gates.append(_gate(
            "known_scorer_base_model_ids",
            "PASS" if known_scorer_models else "FAIL",
            "every scorer supplies an explicit model_id" if known_scorer_models else "one or more scorers lack a known model_id",
            normalized_model_ids=sorted(scorer_model_ids),
            base_model_ids=sorted(scorer_base_models),
            unknown_scorer_ids=[row["scorer_id"] for row in normalized_scorers if row["normalized_model_id"] is None or row["base_model_id"] is None],
        ))
        # A scorer using the same base model or session as any answerer is
        # self-scoring, even when provider/version labels differ.
        separated = (
            not (scorer_model_ids & normalized_model_ids)
            and not (scorer_base_models & base_model_ids)
            and not (scorer_sessions & set(session_ids))
        )
        scorer_blind = all(row["blindness_attestation"] for row in normalized_scorers)
        scorer_evidence = all(row["evidence_hashes"] for row in normalized_scorers)
        gates.append(_gate(
            "scorer_answerer_separation",
            "PASS" if separated else "FAIL",
            "scorer base models/sessions are separate" if separated else "scorer overlaps an answerer by base model or session",
            answerer_normalized_model_ids=sorted(normalized_model_ids),
            scorer_normalized_model_ids=sorted(scorer_model_ids),
            answerer_base_model_ids=sorted(base_model_ids),
            scorer_base_model_ids=sorted(scorer_base_models),
        ))
        gates.append(_gate("scorer_blindness_attestation", "PASS" if scorer_blind else "FAIL", "scorers attest blindness" if scorer_blind else "scorer blindness is missing"))
        gates.append(_gate("scorer_evidence_hash_completeness", "PASS" if scorer_evidence else "FAIL", "scorers supply evidence hashes" if scorer_evidence else "scorer evidence hashes are incomplete"))
    else:
        gates.append(_gate("scorer_answerer_separation", "NOT_APPLICABLE", "no scorer manifest supplied; agreement is descriptive only"))
    model_same = (
        known_answer_models
        and len(normalized_answers) >= 2
        and (len(normalized_model_ids) == 1 or len(base_model_ids) == 1)
    )
    all_pass = all(row["status"] == "PASS" or row["status"] == "NOT_APPLICABLE" for row in gates)
    independent = all_pass and distinct
    # SAME_MODEL_ONLY is descriptive only when every gate other than the
    # distinct-model requirement passes.  A non-blind, incomplete, or
    # self-scored same-model input is invalid rather than downgraded.
    non_distinct_pass = all(
        row["status"] in {"PASS", "NOT_APPLICABLE"}
        for row in gates
        if row["id"] != "distinct_model_identities"
    )
    if model_same and non_distinct_pass:
        mode = "SAME_MODEL_ONLY"
        status = "SAME_MODEL_ONLY"
    elif independent:
        mode = "CROSS_AI_INDEPENDENT"
        status = "PASS"
    else:
        mode = "INVALID"
        status = "FAIL"
    result = {
        "schema_version": SCHEMA_VERSION,
        "cross_ai_protocol_version": CROSS_AI_PROTOCOL_VERSION,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "status": status,
        "mode": mode,
        "independent": independent,
        "case": normalized_case,
        "answers": normalized_answers,
        "scorers": normalized_scorers,
        "gold_metadata": gold_meta,
        "gold_records": gold_rows,
        "gates": gates,
    }
    if strict and not independent:
        raise CrossAIValidationError(f"cross-AI independence gates did not pass ({mode})")
    return result


def _pair_cases(left: Mapping[str, Any], right: Mapping[str, Any]) -> List[str]:
    lmap = {row.get("case_id"): row for row in left.get("answers", [])}
    rmap = {row.get("case_id"): row for row in right.get("answers", [])}
    return sorted(set(lmap) & set(rmap) - {None})


def _cohen_kappa(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = sorted(set(left) | set(right), key=str)
    expected = sum((left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels)
    if math.isclose(1.0 - expected, 0.0, abs_tol=1e-15):
        return 1.0 if math.isclose(observed, 1.0, abs_tol=1e-15) else None
    return observed if expected == 1.0 else (observed - expected) / (1.0 - expected)


def _root_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _pair_summary(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    lmap = {row.get("case_id"): row for row in left.get("answers", [])}
    rmap = {row.get("case_id"): row for row in right.get("answers", [])}
    cases = _pair_cases(left, right)
    lrows, rrows = [lmap[c] for c in cases], [rmap[c] for c in cases]
    decisions_l = [row.get("decision") for row in lrows]
    decisions_r = [row.get("decision") for row in rrows]
    decision_agreement = sum(a == b for a, b in zip(decisions_l, decisions_r)) / len(cases) if cases else None
    jaccards = []
    for lrow, rrow in zip(lrows, rrows):
        ls, rs = set(lrow.get("findings", [])), set(rrow.get("findings", []))
        jaccards.append(1.0 if not ls and not rs else len(ls & rs) / len(ls | rs))
    roots_l = [_root_value(row.get("root_cause")) for row in lrows]
    roots_r = [_root_value(row.get("root_cause")) for row in rrows]
    root_agreement = sum(a == b for a, b in zip(roots_l, roots_r)) / len(cases) if cases else None
    return {
        "left_answer_id": left.get("answers", [{}])[0].get("answer_id") if left.get("answers") else None,
        "right_answer_id": right.get("answers", [{}])[0].get("answer_id") if right.get("answers") else None,
        "left_model_identity": left.get("answers", [{}])[0].get("model_identity") if left.get("answers") else None,
        "right_model_identity": right.get("answers", [{}])[0].get("model_identity") if right.get("answers") else None,
        "n_common_cases": len(cases),
        "decision_agreement": decision_agreement,
        "cohen_kappa": _cohen_kappa(decisions_l, decisions_r),
        "finding_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
        "root_cause_agreement": root_agreement,
        "coverage": len(cases) / max(len(set(lmap) | set(rmap)), 1),
    }


def agreement_summary(validated: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute descriptive pairwise answer agreement; never infer validity."""
    answers = validated.get("answers", [])
    records = [{"answers": [row]} for row in answers]
    pairs = [_pair_summary(records[i], records[j]) for i, j in itertools.combinations(range(len(records)), 2)]
    fields = ("decision_agreement", "cohen_kappa", "finding_jaccard", "root_cause_agreement", "coverage")
    aggregate = {}
    for field in fields:
        values = [row[field] for row in pairs if row.get(field) is not None]
        aggregate[field] = sum(values) / len(values) if values else None
    return {
        "schema_version": SCHEMA_VERSION,
        "cross_ai_protocol_version": CROSS_AI_PROTOCOL_VERSION,
        "status": "DESCRIPTIVE_ONLY",
        "pairwise": pairs,
        "n_answer_records": len(answers),
        "n_pairs": len(pairs),
        **aggregate,
        "claim_boundary": "AI-AI agreement is supplementary and is not independent human adjudication.",
    }


def disagreement_inventory(validated: Mapping[str, Any]) -> List[Dict[str, Any]]:
    answers = validated.get("answers", [])
    by_case: Dict[str, List[Mapping[str, Any]]] = {}
    for row in answers:
        by_case.setdefault(str(row.get("case_id")), []).append(row)
    output: List[Dict[str, Any]] = []
    for case_id, rows in sorted(by_case.items()):
        fields = {}
        for field in ("decision", "findings", "root_cause"):
            values = [row.get(field) for row in rows]
            if len({json.dumps(value, sort_keys=True, default=str) for value in values}) > 1:
                fields[field] = values
        if fields:
            output.append({"case_id": case_id, "fields": fields, "answer_ids": [row.get("answer_id") for row in rows]})
    return output


def _unresolved_gold() -> Dict[str, Any]:
    rows = []
    for metric_id in PRIMARY_METRIC_IDS:
        rows.append({"metric_id": metric_id, "value": None, "status": "UNRESOLVED", "threshold_status": "PROVISIONAL", "metric_role": "primary"})
    return {
        "schema_version": SCHEMA_VERSION,
        "cross_ai_protocol_version": CROSS_AI_PROTOCOL_VERSION,
        "status": "UNRESOLVED",
        "metrics": rows,
        "by_metric": {row["metric_id"]: row for row in rows},
        "claim_boundary": "No frozen gold/adjudication artifact was supplied; agreement cannot establish performance or publication-grade validation.",
    }


def gold_scoring(validated: Mapping[str, Any]) -> Dict[str, Any]:
    """Score explicit frozen-gold records through STIA's benchmark substrate."""
    metadata = validated.get("gold_metadata")
    gold_rows = validated.get("gold_records", [])
    if not metadata or not gold_rows:
        return _unresolved_gold()
    frozen = bool(metadata.get("frozen", metadata.get("frozen_gold", False)))
    artifact_hash = _text(metadata.get("artifact_hash", metadata.get("gold_hash", metadata.get("sha256"))))
    if not frozen or not artifact_hash:
        result = _unresolved_gold()
        result["claim_boundary"] = "Gold was supplied without a frozen artifact attestation/hash; gold-dependent claims remain UNRESOLVED."
        return result
    gold = {row.get("case_id"): row for row in gold_rows if row.get("case_id")}
    benchmark_records: List[Dict[str, Any]] = []
    missing = 0
    for answer in validated.get("answers", []):
        target = gold.get(answer.get("case_id"))
        if target is None:
            missing += 1
            continue
        stratum = str(target.get("stratum", target.get("case_type", ""))).lower()
        if stratum not in {"clean", "defect"}:
            stratum = "defect" if target.get("is_defect") else "clean"
        answer_decision = answer.get("decision")
        detected = answer_decision in _FLAGGED if answer_decision else None
        expected = _text(target.get("decision"))
        if expected is not None:
            # Explicit gold decision takes precedence for clean/defect labeling;
            # disagreement itself is represented by the detection fields below.
            expected = expected.upper()
        benchmark_records.append({
            "case_id": answer.get("case_id"),
            "stratum": stratum,
            "decision": answer_decision,
            "detected": detected,
            "familywise_false_positive": bool(detected) if stratum == "clean" else False,
            "unsafe_acceptance": answer_decision in _ACCEPTED if stratum == "defect" and answer_decision else None,
            "root_cause_true": target.get("root_cause"),
            "root_cause_pred": answer.get("root_cause"),
            "provenance_completeness": 1.0 if answer.get("evidence_hashes") else 0.0,
        })
    if missing or not benchmark_records:
        result = _unresolved_gold()
        result["claim_boundary"] = f"Frozen gold is incomplete for {missing} answer record(s); gold-dependent claims remain UNRESOLVED."
        return result
    metrics = framework_quality_metrics({}, benchmark_records=benchmark_records)
    primary = [row for row in metrics if row.get("metric_id") in PRIMARY_METRIC_IDS]
    return {
        "schema_version": SCHEMA_VERSION,
        "cross_ai_protocol_version": CROSS_AI_PROTOCOL_VERSION,
        "status": "RESOLVED",
        "gold_artifact_hash": artifact_hash,
        "n_records": len(benchmark_records),
        "metrics": primary,
        "by_metric": {row["metric_id"]: row for row in primary},
        "claim_boundary": "These metrics are conditional on the supplied frozen gold/adjudication artifact; they are not derived from AI-AI agreement.",
    }


def ingest_cross_ai(case: Any, answers: Any, *, scorers: Any = None, gold: Any = None, strict: bool = False) -> Dict[str, Any]:
    """Public ingestion API; returns normalized records and gate evidence."""
    return validate_cross_ai_input(case, answers, scorers=scorers, gold=gold, strict=strict)


def run_cross_ai(case: Any, answers: Any, *, scorers: Any = None, gold: Any = None, output: Path | str | None = None, strict: bool = False) -> Dict[str, Any]:
    """Run offline ingestion, descriptive agreement, and optional gold scoring."""
    validated = validate_cross_ai_input(case, answers, scorers=scorers, gold=gold, strict=strict)
    agreement = agreement_summary(validated)
    disagreements = disagreement_inventory(validated)
    gold_result = gold_scoring(validated)
    result = {
        "schema_version": SCHEMA_VERSION,
        "cross_ai_protocol_version": CROSS_AI_PROTOCOL_VERSION,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "status": validated["status"],
        "mode": validated["mode"],
        "independent": validated["independent"],
        "case": validated["case"],
        "answers": validated["answers"],
        "scorers": validated["scorers"],
        "gates": validated["gates"],
        "agreement": agreement,
        "disagreements": disagreements,
        "gold_scoring": gold_result,
        "provenance": {
            "read_only": True,
            "provider_calls": False,
            "network_calls": False,
            "answer_evidence_hashes_complete": next((row["status"] == "PASS" for row in validated["gates"] if row["id"] == "evidence_hash_completeness"), False),
        },
        "claim_boundary": "AI-AI agreement is supplementary; without frozen gold, performance and publication-grade validation are UNRESOLVED.",
    }
    if output is not None:
        write_cross_ai_bundle(result, output)
    return result


def score_cross_ai(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Alias for :func:`run_cross_ai` used by CLI and downstream callers."""
    return run_cross_ai(*args, **kwargs)


# Friendly names used by downstream notebooks and older adapter prototypes.
validate_cross_ai = validate_cross_ai_input
summarize_agreement = agreement_summary
compute_agreement = agreement_summary
score_cross_ai_answers = run_cross_ai
ingest_answer_manifest = ingest_cross_ai
compute_agreement_metrics = agreement_summary
score = run_cross_ai
validate = validate_cross_ai_input


def write_cross_ai_bundle(result: Mapping[str, Any], output: Path | str) -> Dict[str, Path]:
    """Write one exclusive/versioned five-file bundle and refuse overwrite."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in CROSS_AI_BUNDLE_FILES}
    existing = [path.name for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite cross-AI outputs: {', '.join(existing)}")
    manifest = {key: result[key] for key in ("schema_version", "cross_ai_protocol_version", "tool", "tool_version", "status", "mode", "independent", "case", "answers", "scorers", "gates", "provenance", "claim_boundary")}
    canonical_write(paths["cross_ai_manifest.json"], manifest)
    canonical_write(paths["cross_ai_agreement.json"], result["agreement"])
    canonical_write(paths["cross_ai_disagreements.json"], {"schema_version": SCHEMA_VERSION, "disagreements": result["disagreements"]})
    canonical_write(paths["cross_ai_gold_metrics.json"], result["gold_scoring"])
    canonical_write(paths["cross_ai_result.json"], dict(result))
    return paths


__all__ = [
    "CROSS_AI_PROTOCOL_VERSION", "CROSS_AI_BUNDLE_FILES", "CrossAIValidationError",
    "validate_cross_ai_input", "ingest_cross_ai", "agreement_summary", "disagreement_inventory",
    "gold_scoring", "run_cross_ai", "score_cross_ai", "write_cross_ai_bundle",
    "validate_cross_ai", "summarize_agreement", "compute_agreement", "score_cross_ai_answers",
    "ingest_answer_manifest", "compute_agreement_metrics", "score", "validate",
]
