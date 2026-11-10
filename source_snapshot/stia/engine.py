# -*- coding: utf-8 -*-
"""STIA's additive, read-only audit engine.

The engine intentionally keeps the existing ``invest_st`` checks as the
implementation substrate for metadata, identifiability, three-axis batch and
OSI checks.  STIA adds the stable registry and output/decision contract around
those checks; it does not modify the INSPIRE checkout.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import (BASELINE_SURFACE, CANONICAL_BUNDLE_FILES, DECISIONS,
                        MODULES, SCHEMA_VERSION, TOOL_NAME, TOOL_VERSION,
                        ModuleResult, result)
from .baselines import baseline_cells, baseline_comparisons, framework_quality_metrics
from .perturbations import perturbation_contract
from .regression import corpus_records, execute_regression_corpus, validate_corpus
from .diagnostics import EvidenceFault, error_fields, diagnose
from .io import (canonical_csv_write, canonical_write, environment_identity, file_identity, load_json_or_yaml,
                 load_table, read_array, relpath_or_name, resolve_input, sha256_file)


REQUIRED_COLUMNS = ("spot_id", "sample_id", "donor_id", "disease_or_condition", "technical_batch")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_text(value: Any) -> str:
    return str(value) if value is not None else ""


def _nested_run_ids(value: Any) -> set[str]:
    """Collect run identities from nested provenance records."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"run_id", "runid"} and item is not None:
                found.add(str(item))
            elif key_text in {"run_ids", "artifact_run_ids", "source_run_ids"} and isinstance(item, (list, tuple, set)):
                found.update(str(entry) for entry in item if entry is not None)
            else:
                found.update(_nested_run_ids(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found.update(_nested_run_ids(item))
    return found


def _check(name: str, passed: Optional[bool], detail: str, **extra: Any) -> Dict[str, Any]:
    state = "PASS" if passed is True else "FAIL" if passed is False else "UNRESOLVED"
    out: Dict[str, Any] = {"name": name, "status": state, "detail": detail}
    out.update(extra)
    return out


def _legacy_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Map the stable STIA names to the existing invest_st configuration."""
    out = {
        "spot_id": cfg.get("spot_id", "spot_id"),
        "sample_id": cfg.get("sample_id", "sample_id"),
        "donor_id": cfg.get("donor_id", "donor_id"),
        "disease_or_condition": cfg.get("disease_or_condition", "disease_or_condition"),
        "technical_batch": cfg.get("technical_batch", "technical_batch"),
        "estimand": cfg.get("estimand", "STIA estimand"),
        "min_donor_per_condition": int(cfg.get("min_donor_per_condition", 1)),
        "min_batch_per_condition": int(cfg.get("min_batch_per_condition", 1)),
        "k_ilisi": int(cfg.get("k_ilisi", 90)),
        "k_graph": int(cfg.get("k_graph", 20)),
        "max_per_stratum": int(cfg.get("max_per_stratum", 1500)),
    }
    return out


def _load_config(case_root: Path, config: str | Path | Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    if isinstance(config, Mapping):
        return dict(config), None
    config_path = Path(config)
    if not config_path.is_absolute():
        # CLI users commonly keep the frozen config next to the invocation
        # directory, while case-local configs are also valid.  Prefer an
        # explicitly existing cwd path; otherwise resolve relative to CASE.
        cwd_candidate = config_path.resolve()
        config_path = cwd_candidate if cwd_candidate.exists() else (case_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    raw = load_json_or_yaml(config_path)
    if not isinstance(raw, dict):
        raise ValueError("config top level must be a mapping")
    return dict(raw), config_path


def _metadata_path(case_root: Path, cfg: Mapping[str, Any]) -> Optional[Path]:
    value = cfg.get("metadata", cfg.get("adata"))
    if value:
        return resolve_input(case_root, value)
    for name in ("metadata.csv", "metadata.tsv", "metadata.parquet", "case.h5ad"):
        candidate = case_root / name
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_embedding(case_root: Path, spec: Mapping[str, Any], metadata: pd.DataFrame, cfg: Mapping[str, Any]):
    """Load an embedding only with an explicit spot-key sidecar.

    Positional truncation is forbidden.  NPZ files may carry both ``latent``
    and ``spot_ids``; otherwise the config must name a sidecar.
    """
    path = resolve_input(case_root, spec["path"])
    latent = read_array(path, spec.get("latent_key"))
    latent = np.asarray(latent)
    if latent.ndim != 2:
        raise EvidenceFault("embedding must be a two-dimensional array", 'quality', 'invalid_embedding')
    keys = None
    if spec.get("spot_ids"):
        keys = read_array(resolve_input(case_root, spec["spot_ids"]), spec.get("spot_ids_key"))
    elif path.suffix.lower() == ".npz":
        loaded = np.load(path, allow_pickle=False)
        for key in ("spot_ids", "spot_id", "ids"):
            if key in loaded:
                keys = loaded[key]
                break
    if keys is None:
        raise EvidenceFault("embedding spot IDs are required; positional row alignment is not permitted", 'identity', 'embedding_identity')
    keys = np.asarray(keys).astype(str)
    if len(keys) != len(latent) or len(set(keys)) != len(keys):
        raise EvidenceFault("embedding spot IDs must be unique and match embedding rows", 'identity', 'embedding_identity')
    sid = str(cfg.get("spot_id", "spot_id"))
    meta_keys = metadata[sid].astype(str).to_numpy()
    common = set(keys).intersection(meta_keys)
    if len(common) != len(keys) or len(common) != len(meta_keys):
        missing_meta = sorted(set(meta_keys) - set(keys))[:5]
        missing_latent = sorted(set(keys) - set(meta_keys))[:5]
        raise EvidenceFault(f"embedding/metadata spot-key mismatch; metadata_missing={missing_meta}, embedding_missing={missing_latent}", 'identity', 'embedding_identity')
    pos = {key: i for i, key in enumerate(keys)}
    order = [pos[key] for key in meta_keys]
    return latent[order], metadata.reset_index(drop=True).copy(), path


def _module_m1(case_root: Path, cfg: Mapping[str, Any], config_path: Optional[Path], metadata_path: Optional[Path], output_dir: Path, load_error: Optional[str]) -> Tuple[ModuleResult, Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    inputs: Dict[str, Any] = {}
    if config_path is None:
        checks.append(_check("config_mapping", isinstance(cfg, dict), "configuration supplied as an in-memory mapping"))
    else:
        checks.append(_check("config_exists", config_path.exists(), f"config path: {relpath_or_name(config_path, case_root)}"))
        if config_path.exists():
            inputs["config"] = file_identity(config_path, case_root)
    checks.append(_check("metadata_exists", metadata_path is not None and metadata_path.exists() if metadata_path else False, "metadata input exists" if metadata_path else "metadata input was not specified or found"))
    if metadata_path is not None and metadata_path.exists():
        inputs["metadata"] = file_identity(metadata_path, case_root)
    declared = cfg.get("inputs", {})
    if isinstance(declared, list):
        declared = {f"input_{i}": value for i, value in enumerate(declared)}
    if isinstance(declared, dict):
        for name, value in declared.items():
            specification = value if isinstance(value, Mapping) else None
            path_value = specification.get("path") if specification is not None else value
            if not path_value:
                checks.append(_check(f"input:{name}", False, "declared input has no path"))
                continue
            path = resolve_input(case_root, path_value)
            checks.append(_check(f"input:{name}", path.exists(), f"declared input {relpath_or_name(path, case_root)}"))
            if path.exists():
                identity = file_identity(path, case_root)
                inputs[str(name)] = identity
                if specification is not None and specification.get("sha256") is not None:
                    expected = str(specification["sha256"]).lower()
                    observed = str(identity["sha256"]).lower()
                    checks.append(_check(
                        f"input:{name}:sha256",
                        expected == observed,
                        "declared input SHA256 matches current bytes",
                        expected=expected,
                        observed=observed,
                    ))
                if specification is not None and specification.get("size_bytes") is not None:
                    try:
                        expected_size = int(specification["size_bytes"])
                    except (TypeError, ValueError):
                        checks.append(_check(f"input:{name}:size_bytes", False, "declared input size is not an integer"))
                    else:
                        checks.append(_check(
                            f"input:{name}:size_bytes",
                            expected_size == int(identity["size_bytes"]),
                            "declared input byte count matches current bytes",
                            expected=expected_size,
                            observed=int(identity["size_bytes"]),
                        ))
    if load_error:
        checks.append(_check("metadata_parse", None, load_error))
    if output_dir.resolve() in {p.resolve() for p in (metadata_path, config_path) if p is not None}:
        checks.append(_check("output_separation", False, "output directory must not replace a source input"))
    else:
        checks.append(_check("output_separation", True, "outputs are written separately from source inputs"))
    # A canonical run must carry an explicit provenance envelope.  Legacy
    # mappings remain readable, but are unresolved rather than silently
    # accepted.  Distinct run identities are a hard mixed-run failure.
    provenance = cfg.get("provenance")
    if not isinstance(provenance, Mapping):
        checks.append(_check("provenance_lineage", None, "provenance.case_id/run_id/software/seeds/orders are required"))
    else:
        required_lineage = ("case_id", "run_id", "software", "seeds", "orders")
        for key in required_lineage:
            checks.append(_check(f"provenance:{key}", bool(provenance.get(key)), f"provenance field {key} is frozen"))
        run_ids = _nested_run_ids(provenance)
        if len(run_ids) > 1:
            checks.append(_check("provenance:single_run", False, "mixed-run provenance identities were supplied", run_ids=sorted(run_ids)))
        else:
            checks.append(_check("provenance:single_run", True, "all provenance identities belong to one run", run_id=next(iter(run_ids), None)))
    hard_fail = any(c["status"] == "FAIL" for c in checks)
    status = "FAIL" if hard_fail else "PASS" if not load_error and all(c["status"] == "PASS" for c in checks) else "UNRESOLVED"
    unresolved = [c["detail"] for c in checks if c["status"] == "UNRESOLVED"]
    evidence = {
        "inputs": inputs,
        "case_root": str(case_root),
        "lineage": dict(provenance) if isinstance(provenance, Mapping) else None,
        "code_identity": cfg.get("source_code", cfg.get("code_identity")),
        "environment_identity": cfg.get("environment"),
    }
    return result("M1", status, "input custody and provenance checks completed" if status == "PASS" else "input custody is incomplete or invalid", checks=checks, evidence=evidence, unresolved=unresolved), inputs


def _module_m2(metadata: Optional[pd.DataFrame], cfg: Mapping[str, Any], load_error: Optional[str]) -> ModuleResult:
    if metadata is None:
        return result("M2", "UNRESOLVED", "identity checks could not run without metadata", unresolved=[load_error or "metadata unavailable"])
    from invest_st import validate
    legacy = _legacy_config(cfg)
    checks: List[Dict[str, Any]] = []
    for key in REQUIRED_COLUMNS:
        col = legacy[key]
        checks.append(_check(f"column:{key}", col in metadata.columns, f"mapped column {col!r}"))
    if any(c["status"] == "FAIL" for c in checks):
        return result("M2", "FAIL", "required identity metadata are missing", checks=checks, repair_actions=["restore frozen spot/sample/donor/condition/batch mappings"])
    try:
        v = validate.run_validate(metadata, legacy)
    except Exception as exc:
        return result("M2", "UNRESOLVED", "identity validation raised an exception", checks=checks + [_check("invest_st_validate", None, str(exc))], unresolved=[str(exc)])
    checks.append(_check("invest_st_validate", v.get("status") == "PASS", "reused invest_st metadata/nesting validator", details=v))
    status = "PASS" if v.get("status") == "PASS" else "FAIL"
    return result("M2", status, "identity, nesting and count checks completed" if status == "PASS" else "identity or nesting checks failed", checks=checks, evidence={"validator": v}, repair_actions=[] if status == "PASS" else ["repair metadata identity or satisfy frozen donor/batch counts"])


def _module_m3(metadata: Optional[pd.DataFrame], cfg: Mapping[str, Any], load_error: Optional[str]) -> ModuleResult:
    if metadata is None:
        return result("M3", "UNRESOLVED", "identifiability could not run without metadata", unresolved=[load_error or "metadata unavailable"])
    from invest_st import identifiability
    try:
        evidence = identifiability.classify(metadata, _legacy_config(cfg))
    except Exception as exc:
        return result("M3", "UNRESOLVED", "identifiability check raised an exception", unresolved=[str(exc)])
    cls = evidence.get("class")
    status = "FAIL" if cls == "NOT_IDENTIFIABLE" else "PASS"
    summary = f"design classified as {cls}" if status == "PASS" else "design is not identifiable"
    repairs = [] if status == "PASS" else ["restrict estimand to common support or repair batch×condition design"]
    return result("M3", status, summary, checks=[_check("design_class", status == "PASS", summary, class_name=cls)], evidence=evidence, repair_actions=repairs)


def _panel_values(case_root: Path, value: Any) -> Tuple[Optional[List[str]], Optional[Path], Optional[str]]:
    if isinstance(value, list):
        return [str(x) for x in value], None, None
    if isinstance(value, str):
        path = resolve_input(case_root, value)
        if not path.exists():
            return None, path, f"feature panel does not exist: {path}"
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return values, path, None
    return None, None, "feature panel evidence was not supplied"


def _module_m4(case_root: Path, cfg: Mapping[str, Any]) -> ModuleResult:
    checks: List[Dict[str, Any]] = []
    feature = cfg.get("feature_panel")
    if feature is None:
        return result("M4", "UNRESOLVED", "common feature-panel evidence is unavailable", unresolved=["feature_panel is required"])
    base, base_path, err = _panel_values(case_root, feature)
    if err:
        return result("M4", "UNRESOLVED", "feature-panel evidence is unavailable", unresolved=[err])
    assert base is not None
    expected_hash = cfg.get("feature_panel_sha256")
    actual_hash = hashlib.sha256(("\n".join(base) + "\n").encode("utf-8")).hexdigest()
    if base_path is not None:
        actual_hash = sha256_file(base_path)
    checks.append(_check("feature_panel_hash", expected_hash is None or expected_hash == actual_hash, "frozen panel hash check", expected=expected_hash, observed=actual_hash))
    methods = cfg.get("method_feature_panels")
    panel_identity = True
    method_hashes: Dict[str, str] = {}
    if isinstance(methods, dict):
        for method, value in methods.items():
            vals, path, method_err = _panel_values(case_root, value)
            if method_err or vals is None:
                checks.append(_check(f"method_panel:{method}", None, method_err or "method panel unavailable"))
                panel_identity = False
                continue
            same = vals == base
            panel_identity &= same
            method_hashes[str(method)] = sha256_file(path) if path is not None else hashlib.sha256(("\n".join(vals) + "\n").encode("utf-8")).hexdigest()
            checks.append(_check(f"method_panel:{method}", same, "method panel matches frozen common panel"))
    else:
        checks.append(_check("method_panel_identity", None, "method_feature_panels not supplied"))
    prep = cfg.get("preprocessing")
    analysis_mode = str(cfg.get("analysis_mode", "supervised_or_tuned"))
    unsupervised_modes = {"unsupervised_order_audit", "unsupervised_integration_audit"}
    if analysis_mode in unsupervised_modes:
        prep_ok = (
            isinstance(prep, dict)
            and prep.get("fit_scope") in ("full_dataset_frozen", "full_unsupervised_frozen")
            and prep.get("leakage_checked") is True
            and prep.get("outcome_informed_selection") is False
        )
        prep_detail = "unsupervised preprocessing is frozen, outcome-blind, and leakage-checked"
        prep_repair = "freeze full-dataset unsupervised preprocessing before order perturbation and document that outcome-informed selection is false"
    else:
        prep_ok = (
            isinstance(prep, dict)
            and prep.get("fit_scope") in ("train_only", "training_fold_only")
            and prep.get("leakage_checked") is True
        )
        prep_detail = "supervised or tuned preprocessing is fit within training scope and leakage-checked"
        prep_repair = "fit all supervised preprocessing, feature selection, and tuning within the training split or training fold"
    checks.append(_check("preprocessing_leakage", prep_ok, prep_detail, analysis_mode=analysis_mode,
                         fit_scope=prep.get("fit_scope") if isinstance(prep, dict) else None))
    # Representation definitions are optional evidence for legacy cases, but
    # when supplied they must be comparable.  A type mismatch is a hard
    # fairness failure; dimensions alone are retained as a warning.
    representations = cfg.get("representations", cfg.get("method_representations"))
    if representations is not None:
        if not isinstance(representations, Mapping) or not representations:
            checks.append(_check("representation_definitions", False, "representations must be a non-empty method mapping"))
        else:
            parsed = []
            invalid = False
            for method, spec in representations.items():
                if not isinstance(spec, Mapping) or not spec.get("type"):
                    invalid = True
                    checks.append(_check(f"representation:{method}", False, "representation requires an explicit type"))
                    continue
                parsed.append((str(method), str(spec.get("type")), spec.get("dimension")))
            if parsed and not invalid:
                ref_type = parsed[0][1]
                mismatches = [item for item in parsed if item[1] != ref_type]
                checks.append(_check("representation_type_identity", not mismatches, "method representations share a comparable semantic type", reference_type=ref_type, mismatches=mismatches))
                dims = [item[2] for item in parsed if isinstance(item[2], (int, float)) and item[2] > 0]
                if len(dims) > 1:
                    ratio = max(dims) / max(1.0, min(dims))
                    checks.append(_check("representation_dimension", True, "dimension differences are disclosed; direction comparisons require a common probe", ratio=ratio, warning=ratio > 2))
    markers = cfg.get("marker_genes")
    if markers is not None:
        if not isinstance(markers, list):
            checks.append(_check("marker_leakage", False, "marker_genes must be a list"))
        else:
            marker_set = set(str(value) for value in markers)
            out_of_panel = sorted(marker_set - set(base))
            post_hoc = cfg.get("marker_selected_after_training") is True
            leak_ok = not out_of_panel and not post_hoc
            checks.append(_check("marker_leakage", leak_ok, "marker genes are frozen inside the common panel and selected before training", out_of_panel=out_of_panel[:20], post_hoc=post_hoc))
    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        status = "FAIL"
    elif "UNRESOLVED" in statuses:
        status = "UNRESOLVED"
    else:
        status = "PASS"
    return result("M4", status, "feature and preprocessing fairness checks completed" if status == "PASS" else "feature or preprocessing evidence is incomplete", checks=checks, evidence={"panel_sha256": actual_hash, "method_panel_hashes": method_hashes, "analysis_mode": analysis_mode}, unresolved=[c["detail"] for c in checks if c["status"] == "UNRESOLVED"], repair_actions=[] if status == "PASS" else ["freeze and record one common feature panel", prep_repair])


def _module_m5(cfg: Mapping[str, Any]) -> ModuleResult:
    evaluator = cfg.get("evaluator")
    if not isinstance(evaluator, dict):
        return result("M5", "UNRESOLVED", "evaluator and statistical-validity evidence is unavailable", unresolved=["evaluator.functional_tests and evaluator.statistics are required"])
    tests = evaluator.get("functional_tests")
    stats = evaluator.get("statistics")
    checks: List[Dict[str, Any]] = []
    if not isinstance(tests, list) or not tests:
        checks.append(_check("functional_tests", None, "functional evaluator tests are missing"))
    else:
        for item in tests:
            if not isinstance(item, dict) or "name" not in item or "passed" not in item:
                checks.append(_check("functional_test_schema", None, "functional test requires name and explicit passed boolean"))
            else:
                checks.append(_check(f"functional:{item['name']}", bool(item["passed"]), _safe_text(item.get("detail", "functional evaluator test")), observed=item.get("observed"), expected=item.get("expected")))
    required = ("unit", "ci_method", "multiple_testing")
    if not isinstance(stats, dict):
        checks.append(_check("statistical_protocol", None, "statistical protocol is missing"))
    else:
        for key in required:
            checks.append(_check(f"statistics:{key}", bool(stats.get(key)), f"statistical protocol field {key}"))
        # A between-donor condition effect has donors as biological units.
        # When that estimand is explicitly declared, a spot-as-independent
        # analysis is invalid even if the caller omits a self-reported
        # pseudoreplication flag. Other spot-level estimands are not assumed
        # invalid merely because the word "spot" appears in the config.
        target = str(stats.get("inference_target", ""))
        if target in ("between_donor_condition_effect", "donor_level_condition_effect"):
            unit = str(stats.get("unit", "")).lower()
            checks.append(_check("between_donor_condition_unit", unit in ("donor", "donor_cluster"),
                                 "between-donor condition inference requires donor-level independent units",
                                 inference_target=target, observed_unit=unit))
        pseudo = stats.get("pseudoreplication", evaluator.get("pseudoreplication"))
        if pseudo is not None:
            if not isinstance(pseudo, Mapping):
                checks.append(_check("pseudoreplication", False, "pseudoreplication evidence must be a mapping"))
            else:
                detected = pseudo.get("pseudoreplication_detected", pseudo.get("detected"))
                checks.append(_check("pseudoreplication", detected is not True, "donor-level inference must not use spot-level independent n", evidence=dict(pseudo)))
    statuses = {c["status"] for c in checks}
    status = "FAIL" if "FAIL" in statuses else "UNRESOLVED" if "UNRESOLVED" in statuses else "PASS"
    return result("M5", status, "evaluator and statistical protocol checks completed" if status == "PASS" else "evaluator/statistical evidence is incomplete or failed", checks=checks, evidence={"statistics": stats if isinstance(stats, dict) else None}, unresolved=[c["detail"] for c in checks if c["status"] == "UNRESOLVED"])


def _read_order_vector(case_root: Path, value: Any, key: Optional[str], label: str) -> np.ndarray:
    """Read one explicit order vector without allowing positional inference.

    Order-level spot and donor identities are evidence, not configuration
    labels.  They may therefore be supplied by a read-only array artifact or
    an explicit inline vector in a frozen in-memory configuration.  A missing
    vector is handled by the caller as unresolved; an invalid vector is a
    failed identity check.
    """
    if value is None:
        raise ValueError(f"{label} is required for every order")
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return np.asarray(value).reshape(-1)
    path = resolve_input(case_root, value)
    return np.asarray(read_array(path, key)).reshape(-1)


def _read_order_embedding(case_root: Path, value: Any, key: Optional[str]) -> np.ndarray:
    if value is None:
        raise ValueError("embedding is required for every order")
    if isinstance(value, (list, tuple, np.ndarray, pd.DataFrame)):
        return np.asarray(value)
    return np.asarray(read_array(resolve_input(case_root, value), key))


def _module_m6_order(case_root: Path, cfg: Mapping[str, Any], metadata: Optional[pd.DataFrame] = None) -> ModuleResult:
    """Check OSI only after explicit spot and donor identities are reconciled.

    The previous implementation passed donor keys directly to OSI and could
    therefore compare rows from different spots when two order arms happened
    to have the same row count.  Every arm now carries explicit spot IDs;
    arms are reconciled to one canonical spot order, donor mappings are
    compared by spot ID (and, when available, against frozen metadata), and
    only then is the donor-keyed OSI calculation allowed to run.
    """
    strict_contract = bool(cfg.get("require_perturbation_contract", False))
    try:
        perturbation = perturbation_contract(cfg.get("perturbation_contract"), required=strict_contract)
    except ValueError as exc:
        return result(
            "M6",
            "FAIL" if cfg.get("perturbation_contract") is not None else "UNRESOLVED",
            "perturbation identity is invalid or unavailable",
            checks=[_check("perturbation_contract", False if cfg.get("perturbation_contract") is not None else None, str(exc))],
            unresolved=[] if cfg.get("perturbation_contract") is not None else [str(exc)],
            repair_actions=["declare one canonical perturbation axis, unit, semantics and frozen fields before comparing order arms"],
        )
    contract_check = _check(
        "perturbation_contract",
        True if perturbation["status"] == "PASS" else None,
        "canonical perturbation contract validated" if perturbation["status"] == "PASS" else "legacy run did not declare a perturbation taxonomy",
        contract=perturbation,
    )
    orders = cfg.get("orders")
    required = int(cfg.get("osi_orders_required", 4))
    if not isinstance(orders, (list, dict)):
        return result("M6", "UNRESOLVED", "order/robustness evidence is unavailable", checks=[contract_check], evidence={"perturbation_contract": perturbation}, unresolved=["orders with embedding and donor_keys are required"])
    items = [{"name": str(k), **(v if isinstance(v, dict) else {"embedding": v})} for k, v in orders.items()] if isinstance(orders, dict) else [dict(item) for item in orders if isinstance(item, dict)]
    if len(items) < required:
        return result("M6", "UNRESOLVED", f"only {len(items)} order arms supplied; {required} are required", checks=[_check("order_count", None, f"need {required} order arms", observed=len(items), required=required)], unresolved=["insufficient frozen order arms"])
    names = [str(item.get("name")) for item in items]
    if len(set(names)) != len(names):
        return result("M6", "FAIL", "order/robustness evidence contains duplicate arm identities", checks=[_check("order_names_unique", False, "order arm names must be unique")], repair_actions=["assign one frozen identity to each order arm and rerun"])
    from invest_st.metrics.osi import osi
    latents: Dict[str, np.ndarray] = {}
    keys: Dict[str, np.ndarray] = {}
    spot_ids: Dict[str, np.ndarray] = {}
    checks: List[Dict[str, Any]] = [contract_check]
    unresolved: List[str] = []
    hard_fail = False
    canonical_spots: Optional[np.ndarray] = None
    canonical_name: Optional[str] = None
    try:
        for item in items:
            name = str(item.get("name"))
            emb_value = item.get("embedding", item.get("latent"))
            # Geometry may legitimately be defined at section/sample level
            # while inference is clustered at donor level (for example four
            # DLPFC sections per donor).  Keep the legacy donor_keys spelling
            # as a fallback, but do not force these two statistical roles to
            # be the same unit.
            key_value = item.get("geometry_keys", item.get("entity_keys", item.get("donor_keys", item.get("keys"))))
            spot_value = item.get("spot_ids", item.get("spot_id"))
            try:
                latent = _read_order_embedding(case_root, emb_value, item.get("embedding_key"))
                donor_keys = _read_order_vector(
                    case_root,
                    key_value,
                    item.get("geometry_keys_key", item.get("entity_keys_key", item.get("donor_keys_key"))),
                    f"order {name} geometry_keys",
                )
                order_spots = _read_order_vector(case_root, spot_value, item.get("spot_ids_key"), f"order {name} spot_ids")
            except FileNotFoundError as exc:
                unresolved.append(str(exc))
                checks.append(_check(f"order:{name}:evidence", None, str(exc)))
                continue
            except ValueError as exc:
                if str(exc).endswith("is required for every order"):
                    unresolved.append(str(exc))
                    checks.append(_check(f"order:{name}:identity", None, str(exc)))
                else:
                    hard_fail = True
                    checks.append(_check(f"order:{name}:identity", False, str(exc)))
                continue
            except (KeyError, TypeError) as exc:
                hard_fail = True
                checks.append(_check(f"order:{name}:identity", False, str(exc)))
                continue
            latent = np.asarray(latent)
            if latent.ndim != 2:
                hard_fail = True
                checks.append(_check(f"order:{name}:embedding_shape", False, "embedding must be two-dimensional"))
                continue
            order_spots = order_spots.astype(str)
            donor_keys = donor_keys.astype(str)
            if len(latent) != len(order_spots) or len(latent) != len(donor_keys):
                hard_fail = True
                checks.append(_check(f"order:{name}:row_identity", False, "embedding, spot_ids and donor_keys row counts must match"))
                continue
            if len(order_spots) == 0 or any(not value for value in order_spots):
                hard_fail = True
                checks.append(_check(f"order:{name}:spot_ids", False, "spot_ids must be non-empty"))
                continue
            if len(set(order_spots)) != len(order_spots):
                hard_fail = True
                checks.append(_check(f"order:{name}:spot_ids_unique", False, "spot_ids must be unique within each order"))
                continue
            if any(not value for value in donor_keys):
                hard_fail = True
                checks.append(_check(f"order:{name}:donor_keys", False, "donor_keys must be non-empty for every spot"))
                continue
            if canonical_spots is None:
                # Sorting makes the reconciliation deterministic and avoids
                # giving the first listed order special row-order authority.
                canonical_spots = np.asarray(sorted(set(order_spots), key=str), dtype=str)
                canonical_name = name
            elif set(order_spots) != set(canonical_spots):
                hard_fail = True
                missing = sorted(set(canonical_spots) - set(order_spots), key=str)[:5]
                extra = sorted(set(order_spots) - set(canonical_spots), key=str)[:5]
                checks.append(_check(f"order:{name}:spot_id_set", False, "all order arms must contain the identical explicit spot-ID set", missing=missing, extra=extra))
                continue
            pos = {spot: i for i, spot in enumerate(order_spots)}
            order = np.asarray([pos[spot] for spot in canonical_spots], dtype=int)
            spot_ids[name] = order_spots[order]
            latents[name] = latent[order]
            keys[name] = donor_keys[order]
            checks.append(_check(f"order:{name}:spot_id_set", True, "explicit spot IDs reconciled to canonical order", n_spots=len(order_spots)))

        if canonical_spots is None or unresolved:
            checks.append(_check("order_identity_evidence", None, "one or more order identity artifacts are unavailable"))
        elif not hard_fail:
            checks.append(_check("order_identity_evidence", True, "all order arms expose the same explicit spot-ID set", canonical_order=canonical_name, n_spots=len(canonical_spots)))

        # Donor alignment is checked *after* spot reconciliation.  Matching
        # row counts or donor-key sets is not sufficient when rows are
        # permuted or a spot was substituted.
        if not hard_fail and not unresolved and len(keys) == len(items):
            ref_name = str(items[0].get("name"))
            ref_keys = keys[ref_name]
            donor_consistent = True
            for name, order_keys in keys.items():
                same = np.array_equal(order_keys, ref_keys)
                donor_consistent &= same
                checks.append(_check(f"donor_alignment:{name}", same, "donor key for every canonical spot matches the reference order"))
            if not donor_consistent:
                hard_fail = True
            if metadata is not None:
                sid_col = str(cfg.get("spot_id", "spot_id"))
                did_col = str(cfg.get("osi_group_id", cfg.get("donor_id", "donor_id")))
                if sid_col not in metadata.columns or did_col not in metadata.columns:
                    unresolved.append("metadata spot/donor columns are unavailable for donor-key alignment")
                    checks.append(_check("donor_alignment:metadata", None, "metadata spot/donor columns are unavailable"))
                else:
                    meta_spots = metadata[sid_col].astype(str).to_numpy()
                    meta_donors = metadata[did_col].astype(str).to_numpy()
                    if len(set(meta_spots)) != len(meta_spots):
                        hard_fail = True
                        checks.append(_check("donor_alignment:metadata_unique", False, "metadata spot IDs must be unique"))
                    else:
                        meta_map = dict(zip(meta_spots, meta_donors))
                        unknown = sorted(set(canonical_spots) - set(meta_map), key=str)
                        if unknown:
                            hard_fail = True
                            checks.append(_check("donor_alignment:metadata_spots", False, "order spot IDs are absent from metadata", unknown=unknown[:5]))
                        else:
                            expected = np.asarray([meta_map[spot] for spot in canonical_spots], dtype=str)
                            same = np.array_equal(expected, ref_keys)
                            hard_fail |= not same
                            checks.append(_check("donor_alignment:metadata", same, "donor keys agree with frozen metadata by spot ID"))
        elif not hard_fail:
            checks.append(_check("donor_alignment", None, "donor alignment could not be verified because order identity evidence is incomplete"))
            unresolved.append("donor alignment could not be verified")

        if hard_fail:
            return result("M6", "FAIL", "order/robustness identity checks failed before OSI", checks=checks, unresolved=unresolved, repair_actions=["repair frozen spot-ID and donor-key mappings, then rerun all order arms"])
        if unresolved or len(latents) != len(items):
            return result("M6", "UNRESOLVED", "order/robustness identity evidence is incomplete", checks=checks, unresolved=unresolved or ["not all order arms were loaded"])
        outcome = osi(latents, keys, margin=float(cfg.get("osi_margin", 0.05)))
    except Exception as exc:
        return result("M6", "UNRESOLVED", "order-sensitivity check could not run", checks=[_check("osi", None, str(exc))], unresolved=[str(exc)])
    passed = outcome.get("pass") is True
    margin = outcome.get("margin")
    from .order_warning import assess_order_warning

    warning = assess_order_warning(outcome.get("osi"), margin, cfg.get("osi_warning_calibration"))
    checks.append(
        _check(
            "osi",
            passed,
            "donor-keyed OSI against frozen margin",
            osi=outcome.get("osi"),
            margin=margin,
            n_orders=outcome.get("n_orders"),
            order_warning=warning["status"],
            order_decision=warning["decision"],
        )
    )
    status = "PASS" if passed else "FAIL"
    evidence = dict(outcome)
    evidence["order_warning"] = warning
    evidence["order_decision"] = warning["decision"]
    evidence["geometry_group_column"] = str(cfg.get("osi_group_id", cfg.get("donor_id", "donor_id")))
    evidence["perturbation_contract"] = perturbation
    evidence["canonical_spot_ids"] = canonical_spots.tolist() if canonical_spots is not None else None
    evidence["canonical_spot_id_sha256"] = hashlib.sha256("\n".join(canonical_spots.tolist()).encode("utf-8")).hexdigest() if canonical_spots is not None else None
    if passed and warning["decision"] == "WARNING":
        summary = "order invariance gate passed within the hard gate but above the frozen clean envelope (WARNING)"
    elif passed and warning["decision"] == "UNRESOLVED":
        summary = "order invariance gate passed; WARNING tier unresolved (no usable negative-control calibration)"
    else:
        summary = "order invariance gate passed" if passed else "order invariance gate failed"
    return result("M6", status, summary, checks=checks, evidence=evidence, repair_actions=[] if passed else ["repair order-sensitive processing and rerun all frozen order arms"])


def _module_m6(case_root: Path, cfg: Mapping[str, Any], metadata: Optional[pd.DataFrame] = None) -> ModuleResult:
    order = _module_m6_order(case_root, cfg, metadata)
    if not cfg.get('require_seed_robustness') and cfg.get('robustness_replicates') is None:
        return order
    checks = list(order.checks); evidence = dict(order.evidence); missing = list(order.unresolved)
    try:
        if metadata is None:
            raise FileNotFoundError('Seed audit metadata is unavailable')
        from .reference_guards import seed_guard
        seed = seed_guard(case_root, cfg, metadata)
        checks.append(_check('computed_seed_robustness', seed['status'] == 'PASS', 'Identity-bound all-pair seed consistency against recomputed numerical calibration'))
        evidence['seed_robustness'] = seed
        statuses = [order.status, seed['status']]
    except FileNotFoundError as exc:
        checks.append(_check('computed_seed_robustness', None, str(exc))); missing.append(str(exc))
        statuses = [order.status, 'UNRESOLVED']
    except (ValueError, KeyError, TypeError) as exc:
        checks.append(_check('computed_seed_robustness', False, str(exc), **error_fields(exc)))
        statuses = [order.status, 'FAIL']
    status = 'FAIL' if 'FAIL' in statuses else 'UNRESOLVED' if 'UNRESOLVED' in statuses else 'PASS'
    return result('M6', status, 'Order and required seed consistency checked separately', checks=checks,
                  evidence=evidence, unresolved=missing, repair_actions=[] if status=='PASS' else ['Repair failed or missing order/seed evidence without changing frozen action rules'])


def _module_m7(case_root: Path, cfg: Mapping[str, Any], metadata: Optional[pd.DataFrame], load_error: Optional[str]) -> ModuleResult:
    spec = cfg.get("embedding")
    if metadata is None or not isinstance(spec, dict) or not spec.get("path"):
        return result("M7", "UNRESOLVED", "three-axis/cross-method evidence is unavailable", unresolved=[load_error or "embedding with explicit spot IDs is required"])
    try:
        latent, aligned, emb_path = _load_embedding(case_root, spec, metadata, cfg)
    except FileNotFoundError as exc:
        return result("M7", "UNRESOLVED", "embedding identity could not be established", unresolved=[str(exc)])
    except (ValueError, KeyError, TypeError) as exc:
        # An explicitly supplied but malformed/mismatched identity is a hard
        # failure; only unavailable evidence is unresolved.
        return result("M7", "FAIL", "embedding identity is invalid or mismatched", checks=[_check("embedding_identity", False, str(exc), **error_fields(exc))], repair_actions=["repair embedding spot-ID sidecar and rerun the three-axis audit"])
    except Exception as exc:
        return result("M7", "UNRESOLVED", "embedding identity could not be established", unresolved=[str(exc)])
    if cfg.get('three_axis_reference') is not None or cfg.get('require_three_axis_reference'):
        try:
            from .reference_guards import reference_guard
            computed = reference_guard(case_root, cfg, aligned, latent)
            checks = [_check(f'reference_relative:{key}', key not in computed['failed_metrics'],
                             'Measured index compared with frozen reference and independently recomputed numerical tolerance',
                             measured=computed['scores'][key], reference=computed['reference_scores'][key], tolerance=computed['rules'][key])
                      for key in computed['scores']]
            return result('M7', computed['status'], 'Computed reference-relative three-axis checks',
                          checks=checks, evidence=computed,
                          repair_actions=[] if computed['status']=='PASS' else ['Investigate measured reference-relative deterioration; do not treat reference as biological truth'])
        except FileNotFoundError as exc:
            return result('M7', 'UNRESOLVED', 'Frozen reference/calibration evidence unavailable', unresolved=[str(exc)])
        except (ValueError, KeyError, TypeError) as exc:
            return result('M7', 'FAIL', 'Invalid reference/calibration evidence', checks=[_check('computed_reference_gate', False, str(exc), **error_fields(exc))])
    layer_col = spec.get("layer_column", cfg.get("layer_column"))
    group_col = spec.get("group_column", cfg.get("group_column"))
    legacy = _legacy_config(cfg)
    checks: List[Dict[str, Any]] = []
    evidence: Dict[str, Any] = {"embedding": relpath_or_name(emb_path, case_root), "n_spots": int(len(latent)), "n_dimensions": int(latent.shape[1])}
    if not layer_col or not group_col or layer_col not in aligned.columns or group_col not in aligned.columns:
        return result("M7", "UNRESOLVED", "batch/spatial strata columns are unavailable", checks=[_check("strata_columns", None, "layer_column and group_column are required")], evidence=evidence, unresolved=["layer_column and group_column are required"])
    try:
        from invest_st.metrics import batch_axis_metrics, spatial_axis_metrics
        batch = batch_axis_metrics(latent, aligned[legacy["technical_batch"]].to_numpy(), aligned[layer_col].to_numpy(), aligned[group_col].to_numpy(), aligned[legacy["donor_id"]].to_numpy(), k_ilisi=legacy["k_ilisi"], k_graph=legacy["k_graph"], max_per_stratum=legacy["max_per_stratum"])
        spatial = spatial_axis_metrics(latent, aligned[layer_col].to_numpy(), k=min(8, max(2, aligned[layer_col].nunique())), seed=int(cfg.get("seed", 0)))
        evidence["batch"] = batch
        evidence["spatial"] = spatial
        checks.append(_check("batch_axis", True, "reused invest_st conditional batch-axis metrics"))
        checks.append(_check("spatial_axis", True, "reused invest_st spatial-axis metrics"))
    except Exception as exc:
        checks.append(_check("batch_spatial_metrics", None, str(exc)))
    disease = cfg.get("disease_axis_metrics")
    if disease is None:
        checks.append(_check("disease_axis", None, "donor-level disease-axis endpoint is not supplied by the current invest_st substrate"))
    else:
        disease_status = str(disease.get("status", "")) if isinstance(disease, Mapping) else ""
        disease_mode = str(disease.get("mode", "")) if isinstance(disease, Mapping) else ""
        if disease_mode == "donor_batch_matched_separation_auc_v1":
            if disease_status:
                checks.append(_check("disease_axis", False, "computed disease-axis mode cannot include a self-reported status", evidence=disease))
            else:
                try:
                    from .disease_metric_v1 import donor_batch_separation_auc
                    measured = donor_batch_separation_auc(
                        latent, aligned,
                        donor_col=legacy["donor_id"],
                        condition_col=legacy["disease_or_condition"],
                        batch_col=legacy["technical_batch"],
                    )
                    evidence["disease"] = measured
                    checks.append(_check("disease_axis_measurement", True, "no-fit batch-matched donor readout computed", metric_id=measured["metric_id"]))
                    checks.append(_check("disease_axis_preservation", None, "descriptive separation alone cannot certify preservation without a frozen reference and action rule"))
                except (ValueError, KeyError, TypeError) as exc:
                    checks.append(_check("disease_axis_measurement", False, f"computed donor readout is invalid: {exc}"))
        elif disease_status == "PASS":
            # A status in the same run configuration is self-asserted.  It is
            # not a computed donor-level endpoint and must not clear M7.
            checks.append(_check("disease_axis", None, "inline PASS is untrusted; an independently computed donor-level disease endpoint is required", evidence=disease))
        elif disease_status == "NOT_APPLICABLE" and bool(disease.get("reason")):
            condition_col = str(cfg.get("disease_or_condition", "disease_or_condition"))
            condition = aligned[condition_col] if condition_col in aligned else None
            no_contrast = condition is not None and condition.notna().all() and condition.astype(str).nunique() == 1
            checks.append(_check("disease_axis_applicability", bool(no_contrast), "non-applicability requires exactly one observed nonmissing condition", evidence=disease, observed_condition_levels=int(condition.nunique()) if condition is not None else None))
        else:
            checks.append(_check("disease_axis", False, "disease-axis evidence is invalid or an applicability reason is missing", evidence=disease))
        if "disease" not in evidence:
            evidence["disease"] = disease
    reversal = cfg.get("global_conditional_reversal", cfg.get("selection_reversal", cfg.get("global_vs_conditional")))
    if reversal is not None:
        if isinstance(reversal, Mapping):
            detected = reversal.get("reversed", reversal.get("selection_reversed"))
            if detected is None:
                global_best = reversal.get("global_best", reversal.get("global_winner"))
                conditional_best = reversal.get("conditional_best", reversal.get("conditional_winner"))
                if global_best is not None and conditional_best is not None:
                    detected = str(global_best) != str(conditional_best)
            checks.append(_check("global_conditional_reversal", detected is not True, "global and conditional rankings must not be conflated", evidence=dict(reversal)))
            evidence["global_conditional_reversal"] = dict(reversal)
        else:
            checks.append(_check("global_conditional_reversal", False, "global_conditional_reversal evidence must be a mapping"))
    statuses = {c["status"] for c in checks}
    status = "FAIL" if "FAIL" in statuses else "UNRESOLVED" if "UNRESOLVED" in statuses else "PASS"
    return result("M7", status, "three-axis checks completed" if status == "PASS" else "three-axis evidence is incomplete", checks=checks, evidence=evidence, unresolved=[c["detail"] for c in checks if c["status"] == "UNRESOLVED"])


def _module_m8(case_root: Path, cfg: Mapping[str, Any], config_path: Optional[Path] = None) -> Tuple[ModuleResult, Dict[str, Any]]:
    """Review claims against frozen external evidence.

    A claim's ``classification`` and ``evidence_refs`` in the run config are
    self-asserted inputs.  They are retained for traceability but can never
    establish M8 PASS.  PASS requires a separate JSON evidence artifact that
    declares a frozen independent or rule-based review, has an explicit
    protocol identity, and contains a review record for every submitted claim.
    The runtime recomputes each claim result from that artifact rather than
    trusting a self-reported status.
    """
    claims = cfg.get("claims")
    if claims is None:
        review = {"schema_version": SCHEMA_VERSION, "claims": [], "note": "no claim package supplied; claim governance is not applicable to this case"}
        return result("M8", "NOT_APPLICABLE", "no claims were supplied for review"), review
    if not isinstance(claims, list):
        return result("M8", "UNRESOLVED", "claim package has invalid shape", unresolved=["claims must be a list"]), {"schema_version": SCHEMA_VERSION, "claims": [], "note": "invalid claims shape"}
    if not claims:
        return result("M8", "NOT_APPLICABLE", "claim package contains no claims for review"), {"schema_version": SCHEMA_VERSION, "claims": [], "note": "no claim opportunities supplied"}

    # Accept the canonical key plus descriptive aliases used by staged
    # callers, but require every form to point to a separate artifact.  Inline
    # review records would recreate the self-assertion false-PASS risk.
    review_spec = None
    for key in ("claim_review_evidence", "review_evidence", "independent_review"):
        if cfg.get(key) is not None:
            review_spec = cfg.get(key)
            break
    if review_spec is None:
        review = {"schema_version": SCHEMA_VERSION, "claims": [], "note": "self-asserted claim labels/evidence are not review evidence"}
        return result("M8", "UNRESOLVED", "independent or rule-based claim review evidence is required", unresolved=["claim_review_evidence.path with frozen independent/rule_based review is required"]), review
    if isinstance(review_spec, (str, Path)):
        review_path = resolve_input(case_root, review_spec)
        spec_meta: Dict[str, Any] = {}
    elif isinstance(review_spec, Mapping):
        raw_path = review_spec.get("path", review_spec.get("file", review_spec.get("artifact")))
        if not raw_path:
            review = {"schema_version": SCHEMA_VERSION, "claims": [], "note": "review evidence must be an external artifact path"}
            return result("M8", "UNRESOLVED", "claim review evidence path is missing", unresolved=["inline review evidence is not accepted"]), review
        review_path = resolve_input(case_root, raw_path)
        spec_meta = dict(review_spec)
    else:
        review = {"schema_version": SCHEMA_VERSION, "claims": [], "note": "invalid review evidence specification"}
        return result("M8", "UNRESOLVED", "claim review evidence specification is invalid", unresolved=["review evidence must be a path or mapping with path"]), review

    checks: List[Dict[str, Any]] = []
    if config_path is not None and review_path.resolve() == config_path.resolve():
        checks.append(_check("review_evidence_separation", False, "review evidence must be separate from the self-asserted run config"))
        return result("M8", "FAIL", "claim review evidence is not independent from the run config", checks=checks, repair_actions=["provide a separate frozen independent or rule-based review artifact"]), {"schema_version": SCHEMA_VERSION, "claims": [], "note": "review evidence aliases the config"}
    if not review_path.exists():
        checks.append(_check("review_evidence_exists", None, f"review evidence artifact does not exist: {relpath_or_name(review_path, case_root)}"))
        return result("M8", "UNRESOLVED", "frozen claim review evidence is unavailable", checks=checks, unresolved=[checks[-1]["detail"]]), {"schema_version": SCHEMA_VERSION, "claims": [], "note": "review evidence unavailable"}
    try:
        raw_review = load_json_or_yaml(review_path)
    except Exception as exc:
        checks.append(_check("review_evidence_parse", None, f"review evidence could not be parsed: {exc}"))
        return result("M8", "UNRESOLVED", "frozen claim review evidence could not be parsed", checks=checks, unresolved=[str(exc)]), {"schema_version": SCHEMA_VERSION, "claims": [], "note": "review evidence parse failure"}
    if not isinstance(raw_review, Mapping):
        checks.append(_check("review_evidence_schema", None, "review evidence artifact must be a mapping"))
        return result("M8", "UNRESOLVED", "frozen claim review evidence has invalid shape", checks=checks, unresolved=[checks[-1]["detail"]]), {"schema_version": SCHEMA_VERSION, "claims": [], "note": "invalid review evidence shape"}

    # Metadata may be declared either beside ``claims`` or under ``review``.
    review_meta = raw_review.get("review") if isinstance(raw_review.get("review"), Mapping) else raw_review
    review_entries = raw_review.get("claims")
    if not isinstance(review_entries, list) and isinstance(review_meta, Mapping):
        review_entries = review_meta.get("claims")
    kind = spec_meta.get("kind", spec_meta.get("review_type", spec_meta.get("source_type", review_meta.get("kind", review_meta.get("review_type", review_meta.get("source_type"))))))
    kind = str(kind).lower() if kind is not None else ""
    frozen = spec_meta.get("frozen", review_meta.get("frozen"))
    protocol = spec_meta.get("protocol_version", review_meta.get("protocol_version"))
    checks.append(_check("review_evidence_kind", kind in ("independent", "rule_based", "rule-based"), "review evidence declares an independent or rule-based source", observed=kind))
    if kind == "rule-based":
        kind = "rule_based"
    checks.append(_check("review_evidence_frozen", frozen is True, "review evidence is explicitly frozen", observed=frozen))
    checks.append(_check("review_evidence_protocol", bool(protocol), "review evidence declares a protocol/ruleset version", observed=protocol))
    if kind == "independent":
        reviewer_ok = bool(spec_meta.get("reviewer_id", review_meta.get("reviewer_id", spec_meta.get("reviewer", review_meta.get("reviewer"))))) or bool(spec_meta.get("adjudicator_id", review_meta.get("adjudicator_id"))) or spec_meta.get("independence_attestation", review_meta.get("independence_attestation")) is True or spec_meta.get("independent", review_meta.get("independent")) is True
        checks.append(_check("reviewer_independence", reviewer_ok, "independent review has reviewer identity or explicit independence attestation"))
    elif kind == "rule_based":
        rules_ok = bool(spec_meta.get("rule_engine", review_meta.get("rule_engine", spec_meta.get("ruleset", review_meta.get("ruleset"))))) or bool(spec_meta.get("rule_version", review_meta.get("rule_version"))) or bool(spec_meta.get("ruleset_sha256", review_meta.get("ruleset_sha256")))
        checks.append(_check("review_ruleset_identity", rules_ok, "rule-based review has an explicit ruleset identity"))
    else:
        checks.append(_check("review_source_identity", None, "review source type is unknown"))
    expected_review_hash = spec_meta.get("sha256", spec_meta.get("artifact_sha256", review_meta.get("sha256", review_meta.get("artifact_sha256"))))
    if expected_review_hash:
        observed_hash = sha256_file(review_path)
        checks.append(_check("review_evidence_hash", expected_review_hash == observed_hash, "frozen review artifact hash matches the declared identity", expected=expected_review_hash, observed=observed_hash))
    else:
        checks.append(_check("review_evidence_hash", None, "review artifact hash was not frozen in the run config"))
    if not isinstance(review_entries, list):
        checks.append(_check("review_claims", None, "review evidence must contain a claims list"))
        return result("M8", "UNRESOLVED", "frozen review evidence has no claim-level records", checks=checks, unresolved=[checks[-1]["detail"]]), {"schema_version": SCHEMA_VERSION, "claims": [], "evidence_artifact": file_identity(review_path, case_root)}

    config_ids = [str(item.get("claim_id", i)) if isinstance(item, Mapping) else str(i) for i, item in enumerate(claims)]
    review_by_id: Dict[str, Mapping[str, Any]] = {}
    for item in review_entries:
        if isinstance(item, Mapping) and "claim_id" in item:
            cid = str(item["claim_id"])
            if cid in review_by_id:
                checks.append(_check(f"review_claim:{cid}:unique", False, "review evidence contains duplicate claim IDs"))
            review_by_id[cid] = item
    reviewed: List[Dict[str, Any]] = []
    for i, claim in enumerate(claims):
        cid = str(claim.get("claim_id", i)) if isinstance(claim, Mapping) else str(i)
        external = review_by_id.get(cid)
        if external is None:
            checks.append(_check(f"claim:{cid}", None, "claim has no independent/rule-based review record"))
            reviewed.append({"claim_id": cid, "status": "UNRESOLVED", "reason": "no external review record"})
            continue
        classification = external.get("classification")
        refs = external.get("evidence_refs")
        valid_class = classification in ("allowed", "forbidden", "uncertain")
        refs_ok = isinstance(refs, list) and len(refs) > 0
        blocked_ok = classification != "forbidden" or external.get("blocked") is True
        text_value = str(claim.get("text", "")).lower() if isinstance(claim, Mapping) else ""
        exploratory_overreach = bool(claim.get("exploratory_as_confirmatory", False)) if isinstance(claim, Mapping) else False
        exploratory_overreach = exploratory_overreach or ("exploratory" in text_value and "confirmatory" in text_value)
        claim_status = "FAIL" if exploratory_overreach else "PASS" if valid_class and refs_ok and blocked_ok else "FAIL" if valid_class and refs_ok and not blocked_ok else "UNRESOLVED"
        detail = "exploratory evidence cannot be presented as confirmatory" if exploratory_overreach else "external review classifies and links the claim; forbidden claims are blocked" if claim_status == "PASS" else "external review failed to block a forbidden claim" if not blocked_ok else "external review classification/evidence links are incomplete"
        checks.append(_check(f"claim:{cid}", claim_status == "PASS" if claim_status != "UNRESOLVED" else None, detail, classification=classification, evidence_refs=refs, source="external_review"))
        reviewed.append({"claim_id": cid, "text": claim.get("text") if isinstance(claim, Mapping) else None, "classification": classification, "evidence_refs": refs, "blocked": bool(external.get("blocked", False)), "status": claim_status, "reason": detail, "source": "external_review"})
    if set(review_by_id) != set(config_ids):
        checks.append(_check("review_claim_set", False, "review evidence claim IDs must exactly match the submitted claim IDs", missing=sorted(set(config_ids) - set(review_by_id)), extra=sorted(set(review_by_id) - set(config_ids))))
    statuses = {c["status"] for c in checks}
    status = "FAIL" if "FAIL" in statuses else "UNRESOLVED" if "UNRESOLVED" in statuses else "PASS"
    review = {"schema_version": SCHEMA_VERSION, "claims": reviewed, "summary": {"n_claims": len(reviewed), "n_forbidden": sum(1 for x in reviewed if x.get("classification") == "forbidden")}, "evidence_artifact": file_identity(review_path, case_root), "evidence_source": kind, "protocol_version": protocol}
    return result("M8", status, "claim governance checks completed from frozen external review" if status == "PASS" else "claim governance evidence is incomplete or unsafe", checks=checks, evidence={"n_claims": len(reviewed), "evidence_artifact": file_identity(review_path, case_root), "evidence_source": kind}, unresolved=[c["detail"] for c in checks if c["status"] == "UNRESOLVED"], repair_actions=[] if status == "PASS" else ["obtain a separate frozen independent or rule-based claim review and rerun"]), review


def _decision(module_results: Mapping[str, ModuleResult]) -> Tuple[str, List[str], List[str]]:
    failed = [mid for mid, value in module_results.items() if mid != "M9" and value.status == "FAIL"]
    unresolved = [mid for mid, value in module_results.items() if mid != "M9" and value.status == "UNRESOLVED"]
    if failed:
        return "REJECT", [f"hard module failure: {mid}" for mid in failed], unresolved
    if unresolved:
        return "ABSTAIN", [f"required evidence unresolved: {mid}" for mid in unresolved], unresolved
    m3 = module_results.get("M3")
    if m3 and m3.evidence.get("class") == "PARTIALLY_IDENTIFIABLE":
        return "ACCEPT_WITH_SCOPE", ["estimand is restricted to the supported common-support subset"], []
    m7 = module_results.get('M7')
    if m7 and m7.evidence.get('scope') == 'reference_relative_only':
        return 'ACCEPT_WITH_SCOPE', ['Measured indices and consistency gates pass relative to the frozen reference; biological/clinical validity is not certified'], []
    return "ACCEPT", ["all applicable modules passed"], []


def _write_outputs(output_dir: Path, case_root: Path, cfg: Mapping[str, Any], config_path: Optional[Path], metadata_path: Optional[Path], inputs: Mapping[str, Any], modules: Mapping[str, ModuleResult], claim_review: Mapping[str, Any], started_at: str, decision_status: str, decision_reasons: Sequence[str], unresolved: Sequence[str]) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path = output_dir / "findings.json"
    decision_path = output_dir / "decision.json"
    claim_path = output_dir / "claim_review.json"
    provenance_path = output_dir / "provenance.json"
    manifest_path = output_dir / "run_manifest.json"
    cells_path = output_dir / "cells.json"
    comparisons_path = output_dir / "comparisons.json"
    inventory_path = output_dir / "evidence_inventory.json"
    repairs_path = output_dir / "repair_actions.json"
    metrics_path = output_dir / "metrics.csv"
    canonical_paths = (findings_path, decision_path, claim_path, provenance_path, manifest_path,
                       cells_path, comparisons_path, inventory_path, repairs_path, metrics_path)
    existing = [path.name for path in canonical_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing audit evidence in "
            f"{output_dir}: {', '.join(existing)}; choose a new versioned output directory"
        )
    findings = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "case_root": str(case_root),
        "modules": [modules[item["id"]].as_dict() for item in MODULES],
        "diagnostics": diagnose(modules),
        "status_counts": {status: sum(1 for value in modules.values() if value.status == status) for status in ("PASS", "FAIL", "NOT_APPLICABLE", "UNRESOLVED")},
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "decision": decision_status,
        "reasons": list(decision_reasons),
        "unresolved_modules": list(unresolved),
        "non_compensatory": True,
        "modules": {mid: modules[mid].status for mid in sorted(modules)},
        "repair_actions": sorted({action for value in modules.values() for action in value.repair_actions}),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "read_only": True,
        "source_mutation_policy": "source data, manuscript, and historical artifacts are never written by STIA",
        "case_root": str(case_root),
        "config": file_identity(config_path, case_root) if config_path and config_path.exists() else {"path": None, "sha256": None, "size_bytes": None},
        "metadata": file_identity(metadata_path, case_root) if metadata_path and metadata_path.exists() else {"path": None, "sha256": None, "size_bytes": None},
        "declared_inputs": dict(inputs),
        "lineage": dict(cfg.get("provenance")) if isinstance(cfg.get("provenance"), Mapping) else None,
        "source_code_identity": cfg.get("source_code", cfg.get("code_identity")),
        "seed_identity": cfg.get("seeds", (cfg.get("provenance") or {}).get("seeds") if isinstance(cfg.get("provenance"), Mapping) else None),
        "order_identity": cfg.get("orders", (cfg.get("provenance") or {}).get("orders") if isinstance(cfg.get("provenance"), Mapping) else None),
        "environment": environment_identity(),
    }
    # Descriptive baseline surface and framework metrics are emitted for every
    # run, even when no benchmark evidence is available.
    tool_identity = {"name": TOOL_NAME, "version": TOOL_VERSION}
    cells = {"schema_version": SCHEMA_VERSION, "tool": tool_identity, "baseline_surface": list(BASELINE_SURFACE), "cells": baseline_cells(cfg)}
    comparisons = {"schema_version": SCHEMA_VERSION, "tool": tool_identity, "comparisons": baseline_comparisons(cells["cells"]), "threshold_policy": "uncalibrated thresholds remain PROVISIONAL"}
    quality = framework_quality_metrics(modules, cells=cells["cells"], config=cfg)
    metric_rows = quality + [
        {
            "schema_version": SCHEMA_VERSION,
            "metric_id": f"baseline:{cell['baseline_id']}:status",
            "value": cell.get("status"),
            "unit": "status",
            "threshold": None,
            "threshold_status": cell.get("threshold_status", "PROVISIONAL"),
            "status": "DESCRIPTIVE_ONLY",
            "metric_role": "supplementary",
            "calibration_identity": cell.get("calibration_identity"),
        }
        for cell in cells["cells"]
    ]
    evidence_inventory = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool_identity,
        "inputs": dict(inputs),
        "modules": {
            module_id: {
                "status": value.status,
                "checks": value.checks,
                "evidence": value.evidence,
            }
            for module_id, value in modules.items()
        },
        "regression_corpus": {
            "schema_version": SCHEMA_VERSION,
            "records": corpus_records(),
            "validation": validate_corpus(),
            "execution": execute_regression_corpus(),
        },
    }
    repair_rows = [
        {"schema_version": SCHEMA_VERSION, "module_id": module_id, "status": value.status, "action": action}
        for module_id, value in modules.items()
        for action in value.repair_actions
    ]
    repairs = {"schema_version": SCHEMA_VERSION, "tool": tool_identity, "actions": repair_rows, "non_compensatory": True}

    canonical_write(findings_path, findings)
    canonical_write(decision_path, decision)
    canonical_write(claim_path, dict(claim_review))
    canonical_write(provenance_path, provenance)
    canonical_write(cells_path, cells)
    canonical_write(comparisons_path, comparisons)
    canonical_write(inventory_path, evidence_inventory)
    canonical_write(repairs_path, repairs)
    canonical_csv_write(metrics_path, metric_rows, ("schema_version", "metric_id", "value", "unit", "threshold", "threshold_status", "status", "metric_role", "calibration_identity"))
    artifacts = {}
    for path in (findings_path, decision_path, claim_path, provenance_path, cells_path, comparisons_path, inventory_path, repairs_path, metrics_path):
        artifacts[path.name] = {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    lineage = cfg.get("provenance") if isinstance(cfg.get("provenance"), Mapping) else {}
    run_id = str(lineage.get("run_id")) if isinstance(lineage, Mapping) and lineage.get("run_id") else f"{started_at.replace(':', '').replace('-', '')}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": _now(),
        "case_root": str(case_root),
        "config_path": relpath_or_name(config_path, case_root) if config_path else None,
        "output_directory": relpath_or_name(output_dir, case_root),
        "read_only": True,
        "inputs": dict(inputs),
        "artifacts": artifacts,
        "self_sha256": None,
        "bundle_files": list(CANONICAL_BUNDLE_FILES),
        "threshold_policy": "uncalibrated thresholds remain PROVISIONAL",
    }
    manifest["self_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "self_sha256"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    canonical_write(manifest_path, manifest)
    return {p.name: p for p in (findings_path, decision_path, claim_path, provenance_path, manifest_path)}


class AuditRun:
    """Result handle returned by :func:`audit`."""

    def __init__(self, output_dir: Path, paths: Mapping[str, Path], decision: str, modules: Mapping[str, ModuleResult]):
        self.output_dir = output_dir
        self.paths = dict(paths)
        self.all_paths = {
            name: output_dir / name
            for name in CANONICAL_BUNDLE_FILES
            if (output_dir / name).exists()
        }
        self.decision = decision
        self.modules = dict(modules)

    def as_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision, "output_dir": str(self.output_dir), "outputs": {name: str(path) for name, path in self.all_paths.items()}, "modules": {mid: value.status for mid, value in self.modules.items()}}


def audit(case_root: str | Path, config: str | Path | Mapping[str, Any], output: str | Path | None = None, output_dir: str | Path | None = None) -> AuditRun:
    """Run a versioned, read-only STIA audit.

    ``config`` may be a frozen YAML/JSON path or an in-memory mapping.  The
    output directory is the only location created or written.  All five
    canonical output files are emitted even when input evidence is missing.
    """
    root = Path(case_root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"case-root does not exist or is not a directory: {root}")
    out_value = output if output is not None else output_dir
    out = (root / "stia_audit") if out_value is None else (Path(out_value) if Path(out_value).is_absolute() else root / Path(out_value))
    out = out.resolve()
    started = _now()
    cfg: Dict[str, Any] = {}
    config_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    metadata: Optional[pd.DataFrame] = None
    load_error: Optional[str] = None
    try:
        cfg, config_path = _load_config(root, config)
    except Exception as exc:
        load_error = f"config load failed: {exc}"
    try:
        metadata_path = _metadata_path(root, cfg)
        if metadata_path is not None and metadata_path.exists():
            metadata = load_table(metadata_path)
    except Exception as exc:
        load_error = f"metadata load failed: {exc}"
    m1, inputs = _module_m1(root, cfg, config_path, metadata_path, out, load_error)
    modules: Dict[str, ModuleResult] = {"M1": m1}
    modules["M2"] = _module_m2(metadata, cfg, load_error)
    modules["M3"] = _module_m3(metadata, cfg, load_error)
    modules["M4"] = _module_m4(root, cfg)
    modules["M5"] = _module_m5(cfg)
    modules["M6"] = _module_m6(root, cfg, metadata)
    modules["M7"] = _module_m7(root, cfg, metadata, load_error)
    modules["M8"], claim_review = _module_m8(root, cfg, config_path)
    decision_status, reasons, unresolved = _decision(modules)
    modules["M9"] = result("M9", "FAIL" if decision_status == "REJECT" else "UNRESOLVED" if decision_status == "ABSTAIN" else "PASS", f"final decision is {decision_status}", checks=[_check("non_compensatory_decision", decision_status in DECISIONS, "decision derived from module gates without averaging")], unresolved=[f"decision={decision_status}"] if decision_status == "ABSTAIN" else [], repair_actions=["repair failed module(s) and rerun the complete audit"] if decision_status == "REJECT" else [])
    paths = _write_outputs(out, root, cfg, config_path, metadata_path, inputs, modules, claim_review, started, decision_status, reasons, unresolved)
    return AuditRun(out, paths, decision_status, modules)
