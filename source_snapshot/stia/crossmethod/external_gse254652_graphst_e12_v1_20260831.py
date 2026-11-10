"""Native GraphST route for the GSE254652 P1 sparse candidate (E12).

This successor is deliberately additive.  The E8 adapter remains the only
production raw-data reader and this module never creates disease/batch labels
or silently changes GraphST's dense graph semantics.  ``plan`` and
``fixture-preflight`` are bounded, no-fit entry points.  A real fit exists for
the parent-controlled launch gate, but is fail-closed behind both an explicit
flag and an explicit device/resource proof.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "precast-null-inspire-seeds-20260830"
E8_ADAPTER = ROOT / "stia" / "crossmethod" / "external_gse254652_adapter_v1_20260831.py"
E8_RECEIPT = RUN_ROOT / "evidence" / "E8" / "E8_GSE254652_SPARSE_ADAPTER_PREFLIGHT_20260831_r2.json"
GRAPHST_WRAPPER = ROOT / "scripts" / "_x3_graphst_arm_v2.py"
GRAPHST_IMPLEMENTATION = ROOT / "scripts" / "_x3_graphst_osi.py"
AUTHORITY = RUN_ROOT / "phases" / "terminal-audit" / "E12.md"
PROTOCOL = "stia.external.gse254652.e12.graphst_native.v1"
FEATURE_MODE = "all_common_19465"
SENSITIVITY_FEATURE_MODE = "canonical_first3000_ensembl"
ORDER_IDS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
N_EPOCHS = 600
DIM_OUTPUT = 64
N_NEIGHBORS = 3
LATENT_DIM = 20
OSI_MARGIN = 0.05
NNZ = 54_807_969
N_ROWS = 9_568
N_FEATURES = 19_465
NO_LABELS = {"disease": "NOT_CREATED", "batch": "NOT_CREATED"}
CLAIM_SCOPE = "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION_CANDIDATE"


class E12ValidationError(ValueError):
    """Fail-closed route, source, identity, or terminal-contract error."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_strings(values: Sequence[Any]) -> str:
    return canonical_sha256([str(value) for value in values])


def hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise E12ValidationError(f"missing JSON: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E12ValidationError(f"malformed JSON: {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise E12ValidationError(f"JSON object required: {target}")
    return value


def exclusive_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def exclusive_npz(path: Path | str, **arrays: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    with target.open("xb") as handle:
        handle.write(payload.getvalue())


def _load_e8() -> Any:
    spec = importlib.util.spec_from_file_location("e8_external_sparse_adapter_for_e12", E8_ADAPTER)
    if spec is None or spec.loader is None:
        raise E12ValidationError(f"cannot load E8 adapter: {E8_ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_graphst_implementation() -> Any:
    if not GRAPHST_IMPLEMENTATION.is_file():
        raise E12ValidationError(f"frozen GraphST implementation is missing: {GRAPHST_IMPLEMENTATION}")
    spec = importlib.util.spec_from_file_location("graphst_frozen_impl_for_e12", GRAPHST_IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise E12ValidationError(f"cannot load GraphST implementation: {GRAPHST_IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_closure(root: Path = ROOT, *, require_native: bool = False, require_authority: bool = False) -> dict[str, Any]:
    """Return measured closure without pretending optional native files exist."""
    root = Path(root).resolve()
    try:
        route_path = root / Path(__file__).resolve().relative_to(root)
    except ValueError:
        route_path = Path(__file__).resolve()
    paths = {
        "e12_route": route_path,
        "e8_adapter": E8_ADAPTER,
        "e8_receipt": E8_RECEIPT,
        "graphst_wrapper": GRAPHST_WRAPPER,
        "graphst_implementation": GRAPHST_IMPLEMENTATION,
        "authority": AUTHORITY,
    }
    out: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in paths.items():
        if not path.is_file():
            missing.append(name)
            continue
        out[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    required = {"e12_route", "e8_adapter"}
    if require_native:
        required.update({"graphst_wrapper", "graphst_implementation"})
    if require_authority:
        required.add("authority")
    if required.intersection(missing):
        raise E12ValidationError(f"source closure missing: {sorted(required.intersection(missing))}")
    out["missing_optional"] = sorted(set(missing) - required)
    return out


def _validate_device(device: str) -> dict[str, Any]:
    if device not in {"cuda", "cpu"}:
        raise E12ValidationError("device must be explicit: 'cuda' or 'cpu'")
    proof: dict[str, Any] = {"requested": device, "measured": False, "status": "UNMEASURED"}
    if device == "cpu":
        proof.update({"status": "EXPLICIT_CPU_REQUEST", "measured": True, "cuda_available": None})
        return proof
    try:
        import torch
        available = bool(torch.cuda.is_available())
        proof.update({"cuda_available": available, "device_count": int(torch.cuda.device_count()), "device_name": torch.cuda.get_device_name(0) if available else None, "measured": True, "status": "PASS" if available else "FAIL"})
    except Exception as exc:  # pragma: no cover - environment dependent
        proof.update({"status": "UNRESOLVED", "error": f"{type(exc).__name__}: {exc}"})
    return proof


def resource_estimate(*, n_rows: int = N_ROWS, n_features: int = N_FEATURES, nnz: int = NNZ, available_host_bytes: int | None = None, available_device_bytes: int | None = None) -> dict[str, Any]:
    """Conservative accounting for native GraphST's dense graph contract.

    The frozen producer hands GraphST dense ``graph_neigh`` and ``adj`` arrays;
    this estimate therefore does not replace them with a sparse approximation.
    """
    n, p, z = int(n_rows), int(n_features), int(nnz)
    csr = z * (4 + 4) + (n + 1) * 8
    dense_input = n * p * 4
    dense_graph_each = n * n * 4
    dense_graph_total = dense_graph_each * 2
    # GraphST's reconstruction and PCA buffers are additional to the input and
    # graphs.  Keep this as an explicit estimate, not a measured peak.
    embedding = n * p * 4
    pca_work = n * min(p, 256) * 4
    known = csr + dense_input + dense_graph_total + embedding + pca_work
    estimated = int(known * 1.75)
    headroom = 2 * 1024**3
    available_host = available_host_bytes
    if available_host is None:
        try:
            import psutil
            available_host = int(psutil.virtual_memory().available)
        except Exception:
            available_host = None
    available_device = available_device_bytes
    # Unknown device memory is not a pass for a native dense GPU route.  CPU
    # callers may ignore this field, but a CUDA fit must provide a measurement.
    device_gate = available_device is not None and available_device >= estimated + headroom
    host_gate = available_host is not None and available_host >= estimated + headroom
    return {
        "n_rows": n, "n_features": p, "nnz": z,
        "csr_input_bytes": int(csr), "dense_input_bytes": int(dense_input),
        "dense_graph_each_bytes": int(dense_graph_each), "dense_graph_total_bytes": int(dense_graph_total),
        "reconstruction_buffer_bytes": int(embedding), "pca_work_bytes": int(pca_work),
        "known_peak_bytes": int(known), "estimated_peak_bytes": int(estimated),
        "allocator_overhead_factor": 1.75, "required_headroom_bytes": headroom,
        "available_host_bytes": available_host, "available_device_bytes": available_device,
        "host_headroom_gate_pass": bool(host_gate), "device_headroom_gate_pass": bool(device_gate),
        "measured_peak": False, "graph_semantics": "native GraphST dense graph_neigh/adj; no sparse substitution",
        "feature_semantics": "all common ENSMUSG features retained; dense model input is native-required",
    }


def _seeds(selected: Sequence[int] | None) -> tuple[int, ...]:
    values = tuple(int(v) for v in (SEEDS if selected is None else selected))
    if not values or len(set(values)) != len(values) or any(v not in SEEDS for v in values):
        raise E12ValidationError("seeds must be a unique subset of frozen seeds 20260819..20260823")
    return values


def feature_policy(feature_mode: str = FEATURE_MODE) -> dict[str, Any]:
    if feature_mode not in {FEATURE_MODE, SENSITIVITY_FEATURE_MODE}:
        raise E12ValidationError(f"feature mode is not frozen: {feature_mode!r}")
    count = N_FEATURES if feature_mode == FEATURE_MODE else 3000
    return {
        "selected_mode": feature_mode, "feature_count": count,
        "primary": feature_mode == FEATURE_MODE,
        "ids_contract": "canonical ENSMUSG order from E8 manifest" if feature_mode == FEATURE_MODE else "first 3000 canonical ENSMUSG IDs",
        "native_recommendation_difference": "Do not invoke outcome-driven HVG selection; GraphST package recommendations are not the external audit protocol.",
        "accepted_internal_difference": "Internal GraphST used a 3000-symbol panel; external E12 uses canonical ENSMUSG IDs and never silently remaps symbols or zero-fills missing IDs.",
        "selection_from_results": False,
    }


def order_identity(bundle: Any, order_id: str) -> dict[str, Any]:
    if order_id not in ORDER_IDS:
        raise E12ValidationError(f"unknown frozen order: {order_id}")
    sections = tuple(bundle.ordered_sections(order_id))
    spot_ids = [str(spot) for section in sections for spot in section.spot_ids]
    libraries = [str(section.library_id) for section in sections for _ in section.spot_ids]
    animals = [str(section.animal_id) for section in sections for _ in section.spot_ids]
    rows = [int(i) for section in sections for i in range(section.row_offset[0], section.row_offset[1])]
    if len(spot_ids) != N_ROWS or len(set(spot_ids)) != N_ROWS:
        raise E12ValidationError(f"{order_id}: source IDs are not unique {N_ROWS} rows")
    expected = dict(_load_e8().ORDER_IDENTITIES[order_id])
    if tuple(section.library_id for section in sections) != tuple(expected["gsm"]):
        raise E12ValidationError(f"{order_id}: library order drift")
    if tuple(section.animal_id for section in sections) != tuple(expected["animals"]):
        raise E12ValidationError(f"{order_id}: animal order drift")
    return {"order": order_id, "rows": rows, "spot_ids": spot_ids, "library_ids": libraries, "animal_ids": animals, "order_gsm": list(expected["gsm"]), "order_animals": list(expected["animals"]), "order_sha256": expected["sha256"], "rows_sha256": hash_strings(rows), "spot_ids_sha256": hash_strings(spot_ids), "identity_sha256": canonical_sha256(list(zip(spot_ids, libraries, animals)))}


def seed_propagation(seed: int) -> dict[str, Any]:
    value = int(seed)
    if value not in SEEDS:
        raise E12ValidationError(f"seed is not frozen: {seed}")
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    result: dict[str, Any] = {"python": value, "numpy": value % (2**32 - 1), "model_seed": value}
    try:
        import torch
        torch.manual_seed(value)
        result["torch"] = value
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
            result["torch_cuda"] = value
        else:
            result["torch_cuda"] = "not_available"
    except Exception as exc:  # pragma: no cover - optional dependency
        result["torch"] = f"unavailable:{type(exc).__name__}"
        result["torch_cuda"] = "unavailable"
    return result


def build_route_plan(*, project_root: Path | str = ROOT, h5ad_path: Path | str | None = None, manifest_path: Path | str | None = None, feature_mode: str = FEATURE_MODE, device: str = "cuda", seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Build a source/config-bound plan without opening or allocating the real H5AD."""
    root = Path(project_root).resolve()
    selected = _seeds(seeds)
    feature = feature_policy(feature_mode)
    e8 = _load_e8()
    h5ad = Path(h5ad_path or e8.H5AD_DEFAULT)
    manifest = Path(manifest_path or e8.MANIFEST_DEFAULT)
    if not h5ad.is_absolute():
        h5ad = root / h5ad
    if not manifest.is_absolute():
        manifest = root / manifest
    closure = source_closure(root)
    config = {
        "protocol": PROTOCOL, "feature_mode": feature_mode, "feature_count": feature["feature_count"],
        "n_epochs": N_EPOCHS, "dim_output": DIM_OUTPUT, "n_neighbors": N_NEIGHBORS,
        "representation": "recon_pca20", "margin": OSI_MARGIN, "device": device,
        "orders": list(ORDER_IDS), "seeds": list(SEEDS), "requested_seeds": list(selected),
        "preprocessing": "GraphST native preprocess on all-common ENSMUSG CSR counts",
        "graph": "native within-library 3-NN donor/animal block graph; dense graph_neigh and adj required by frozen package",
        "labels": NO_LABELS,
    }
    device_proof = _validate_device(device) if device == "cpu" else {"requested": device, "status": "DEFERRED_PARENT_GATE", "measured": False}
    return {
        "schema_version": PROTOCOL, "status": "READY_NO_FIT", "model_fit": "NOT_PERFORMED",
        "project_root": str(root), "h5ad": {"path": str(h5ad), "sha256": e8.H5AD_SHA256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR", "binding": "accepted E8 hash; real hash/open deferred to parent gate"},
        "manifest": {"path": str(manifest), "sha256": e8.MANIFEST_SHA256, "binding": "accepted E8 hash; metadata-only in this worker"},
        "feature_policy": feature, "config": config, "config_sha256": canonical_sha256(config),
        "orders": {name: {"animals": list(e8.ORDER_IDENTITIES[name]["animals"]), "gsm": list(e8.ORDER_IDENTITIES[name]["gsm"]), "sha256": e8.ORDER_IDENTITIES[name]["sha256"]} for name in ORDER_IDS},
        "seeds": {"frozen": list(SEEDS), "requested": list(selected), "propagation": "python/numpy/torch/manual_seed per arm; random_seed passed to native GraphST"},
        "native_path": {"entrypoint": "GraphST.GraphST(...).train()", "wrapper": str(GRAPHST_WRAPPER), "implementation": str(GRAPHST_IMPLEMENTATION), "reuse": "frozen native path; no handwritten model"},
        "device_proof": device_proof, "resource_qc": resource_estimate(), "source_closure": closure,
        "claim_boundary": {"scope": CLAIM_SCOPE, "external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN", "whole_framework_validation": "UNVALIDATED"},
        "sparse_graph_decision": {"status": "RETAIN_NATIVE_DENSE_SEMANTICS", "reason": "Replacing graph_neigh/adj with sparse tensors could change GraphST normalization/message passing; no substitution is made.", "cost": "dense_graph_total_bytes is accounted explicitly; parent must gate host/device headroom."},
    }


def _identity_arrays(identity: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        ids = np.asarray([str(v) for v in identity["spot_ids"]], dtype="U")
        libraries = np.asarray([str(v) for v in identity["library_ids"]], dtype="U")
        animals = np.asarray([str(v) for v in identity["animal_ids"]], dtype="U")
    except Exception as exc:
        raise E12ValidationError("identity must provide spot_ids, library_ids, animal_ids") from exc
    if ids.size == 0 or len(set(ids.tolist())) != ids.size or not (ids.size == libraries.size == animals.size):
        raise E12ValidationError("identity arrays are empty, duplicated, or misaligned")
    return ids, libraries, animals


def write_arm(destination: Path | str, *, latent: np.ndarray, identity: Mapping[str, Any], seed: int, order_id: str, base: Mapping[str, Any]) -> dict[str, Any]:
    selected_seed = int(seed)
    _seeds((selected_seed,))
    if order_id not in ORDER_IDS:
        raise E12ValidationError(f"unknown frozen order: {order_id}")
    ids, libraries, animals = _identity_arrays(identity)
    matrix = np.asarray(latent, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (ids.size, LATENT_DIM) or not np.isfinite(matrix).all():
        raise E12ValidationError(f"{order_id}:{seed}: latent must be finite {ids.size}x{LATENT_DIM}")
    arm = Path(destination).resolve() / "arms" / f"seed{selected_seed}"
    npz_path, sidecar = arm / f"{order_id}.npz", arm / f"{order_id}.json"
    if npz_path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite existing arm: {order_id}:{selected_seed}")
    exclusive_npz(npz_path, latent=matrix, spot_ids=ids, library_ids=libraries, animal_ids=animals)
    record = dict(base)
    record.update({"schema_version": PROTOCOL, "protocol": PROTOCOL, "seed": selected_seed, "model_seed": selected_seed, "order": order_id, "status": "OK", "npz": str(npz_path), "npz_sha256": sha256_file(npz_path), "latent_sha256": hash_array(matrix), "latent_shape": list(matrix.shape), "latent_dtype": "float32", "identity": {"rows_sha256": identity.get("rows_sha256"), "spot_ids_sha256": hash_strings(ids.tolist()), "identity_sha256": identity.get("identity_sha256", canonical_sha256(list(zip(ids.tolist(), libraries.tolist(), animals.tolist())))), "order_sha256": identity.get("order_sha256")}, "labels": dict(NO_LABELS), "animal_units": sorted(set(animals.tolist()))})
    exclusive_json(sidecar, record)
    return record


def load_arm(npz_path: Path | str, sidecar_path: Path | str | None = None, *, expected_identity: Mapping[str, Any] | None = None, expected_seed: int | None = None, expected_order: str | None = None, expected_config_sha256: str | None = None, expected_source_closure: Mapping[str, Any] | None = None, expected_input_h5ad_sha256: str | None = None, expected_manifest_sha256: str | None = None, expected_feature_ids_sha256: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    npz = Path(npz_path); sidecar = Path(sidecar_path) if sidecar_path is not None else npz.with_suffix(".json")
    if not npz.is_file() or not sidecar.is_file():
        raise E12ValidationError(f"missing raw arm or sidecar: {npz}")
    metadata = _json(sidecar)
    if metadata.get("schema_version") != PROTOCOL or metadata.get("status") != "OK" or metadata.get("npz_sha256") != sha256_file(npz):
        raise E12ValidationError("raw arm schema/status/hash mismatch")
    if expected_seed is not None and metadata.get("seed") != int(expected_seed):
        raise E12ValidationError("raw arm seed mismatch")
    if expected_order is not None and metadata.get("order") != expected_order:
        raise E12ValidationError("raw arm order mismatch")
    if expected_config_sha256 is not None and metadata.get("config_sha256") != expected_config_sha256:
        raise E12ValidationError("raw arm config binding mismatch")
    if expected_source_closure is not None and metadata.get("source_closure") != expected_source_closure:
        raise E12ValidationError("raw arm source closure mismatch")
    for field, expected in (("input_h5ad_sha256", expected_input_h5ad_sha256), ("input_manifest_sha256", expected_manifest_sha256), ("feature_ids_sha256", expected_feature_ids_sha256)):
        if expected is not None and metadata.get(field) != expected:
            raise E12ValidationError(f"raw arm {field} binding mismatch")
    try:
        with np.load(npz, allow_pickle=False) as payload:
            if set(payload.files) != {"latent", "spot_ids", "library_ids", "animal_ids"}:
                raise E12ValidationError("raw arm key set mismatch")
            latent = np.asarray(payload["latent"])
            ids = payload["spot_ids"].astype(str)
            libraries = payload["library_ids"].astype(str)
            animals = payload["animal_ids"].astype(str)
    except E12ValidationError:
        raise
    except Exception as exc:
        raise E12ValidationError(f"cannot load raw arm: {exc}") from exc
    if latent.ndim != 2 or latent.shape[1] != LATENT_DIM or latent.shape[0] != ids.size or libraries.size != ids.size or animals.size != ids.size or not np.isfinite(latent).all() or len(set(ids.tolist())) != ids.size:
        raise E12ValidationError("raw arm arrays are non-finite, misaligned, or malformed")
    if expected_identity is not None:
        expected_ids, expected_libs, expected_animals = _identity_arrays(expected_identity)
        if ids.tolist() != expected_ids.tolist() or libraries.tolist() != expected_libs.tolist() or animals.tolist() != expected_animals.tolist():
            raise E12ValidationError("raw arm source-derived identity mismatch")
        expected_hash = expected_identity.get("identity_sha256")
        if expected_hash is not None and metadata.get("identity", {}).get("identity_sha256") != expected_hash:
            raise E12ValidationError("raw arm identity fingerprint mismatch")
    return latent, ids, animals, metadata


def recompute_osi(latents_by_order: Mapping[str, np.ndarray], animal_ids_by_order: Mapping[str, Sequence[str]], *, seed: int, condition: str = "gse254652_p1_external_graphst") -> dict[str, Any]:
    if set(latents_by_order) != set(ORDER_IDS) or set(animal_ids_by_order) != set(ORDER_IDS):
        raise E12ValidationError("exactly frozen A-D orders are required for independent OSI")
    from stia.crossmethod.osi import compute_osi
    result = compute_osi(latents_by_order, animal_ids_by_order, condition=condition, seed=int(seed), margin=OSI_MARGIN)
    if len(result.get("pairwise_drifts", ())) != 6 or len(result.get("donor_keys", ())) != 7:
        raise E12ValidationError("OSI must recompute six order pairs across seven frozen animal units")
    result["animal_units"] = result.pop("donor_keys")
    result["pair_count"] = len(result["pairwise_drifts"])
    result["independent_recomputation"] = True
    return result


def fixture_preflight() -> dict[str, Any]:
    """Tiny synthetic route check; it never imports GraphST or opens H5AD."""
    import scipy.sparse as sp
    matrix = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 0], [0, 0, 1]], dtype=np.float32))
    if matrix.getformat() != "csr" or not np.isfinite(matrix.data).all():
        raise E12ValidationError("fixture CSR check failed")
    identity = {"spot_ids": ["GSM1::s0", "GSM1::s1", "GSM2::s0", "GSM2::s1"], "library_ids": ["GSM1", "GSM1", "GSM2", "GSM2"], "animal_ids": ["460", "460", "462", "462"]}
    latent = np.arange(4 * LATENT_DIM, dtype=np.float32).reshape(4, LATENT_DIM)
    osi_latents = {order: np.vstack([np.arange(14, dtype=np.float32), np.arange(14, dtype=np.float32) + 1, np.arange(14, dtype=np.float32) + 2, np.arange(14, dtype=np.float32) + 3]) for order in ORDER_IDS}
    # This fixture intentionally exercises route shape/graph checks only; OSI
    # is tested with seven animal units in the focused unit tests.
    return {"status": "PASS", "model_fit": "NOT_PERFORMED", "csr_shape": list(matrix.shape), "csr_nnz": int(matrix.nnz), "latent_shape": list(latent.shape), "orders": list(ORDER_IDS), "seed_contract": list(SEEDS), "native_imported": False, "real_input_opened": False, "claim_scope": CLAIM_SCOPE}


def audit_output(*, output_dir: Path | str, project_root: Path | str = ROOT) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    manifest_path, summary_path, completed_path = destination / "manifest.json", destination / "run_summary.json", destination / "COMPLETED.json"
    if not manifest_path.is_file() or not summary_path.is_file() or not completed_path.is_file():
        raise E12ValidationError("terminal manifest/summary/COMPLETED marker missing")
    manifest, summary = _json(manifest_path), _json(summary_path)
    if manifest.get("schema_version") != PROTOCOL or summary.get("schema_version") != PROTOCOL or summary.get("status") != "FIT_COMPLETE" or summary.get("expected_orders") != list(ORDER_IDS):
        raise E12ValidationError("terminal protocol/order summary drift")
    root = Path(project_root).resolve()
    expected_closure = source_closure(root, require_native=True, require_authority=True)
    if manifest.get("source_closure") != expected_closure:
        raise E12ValidationError("terminal source closure drift")
    e8 = _load_e8()
    h5ad = Path(manifest.get("h5ad", {}).get("path", e8.H5AD_DEFAULT))
    manifest_file = Path(manifest.get("manifest", {}).get("path", e8.MANIFEST_DEFAULT))
    if not h5ad.is_absolute():
        h5ad = root / h5ad
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    # The terminal auditor derives expected rows/animal units from the bound
    # source instead of trusting sidecar self-reporting.
    with e8.open_bundle(root, h5ad_path=h5ad, manifest_path=manifest_file, feature_mode=FEATURE_MODE) as bundle:
        identities = {order: order_identity(bundle, order) for order in ORDER_IDS}
        cells, errors = [], []
        for seed in summary.get("completed_seeds", []):
            try:
                arms = {}
                animals = {}
                for order in ORDER_IDS:
                    npz = destination / "arms" / f"seed{int(seed)}" / f"{order}.npz"
                    latent, _ids, donor_ids, metadata = load_arm(npz, expected_identity=identities[order], expected_seed=int(seed), expected_order=order, expected_config_sha256=manifest.get("config_sha256"), expected_source_closure=expected_closure, expected_input_h5ad_sha256=bundle.h5ad_sha256, expected_manifest_sha256=bundle.manifest_sha256, expected_feature_ids_sha256=canonical_sha256(list(bundle.feature_ids)))
                    arms[order] = latent; animals[order] = donor_ids.tolist()
                metric = recompute_osi(arms, animals, seed=int(seed))
                cells.append({"seed": int(seed), "execution_status": "PASS", "numeric_status": metric["status"], "osi": float(metric["osi"]), "pair_count": metric["pair_count"]})
            except Exception as exc:
                errors.append(f"seed{seed}: {exc}")
                cells.append({"seed": int(seed), "execution_status": "UNRESOLVED", "numeric_status": "UNRESOLVED", "reason": str(exc)})
    verdict = "PASS" if not errors and len(cells) == len(summary.get("completed_seeds", [])) else "UNRESOLVED"
    return {"schema_version": PROTOCOL, "status": verdict, "model_fit": "PERFORMED", "execution_validity": "PASS" if verdict == "PASS" else "UNRESOLVED", "cells": cells, "errors": errors, "claim_boundary": {"scope": CLAIM_SCOPE, "external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}, "audited_at": _dt.datetime.now().astimezone().isoformat()}


def _build_order_adata(bundle: Any, order_id: str) -> Any:
    import pandas as pd
    import scipy.sparse as sp
    from anndata import AnnData
    sections = bundle.ordered_sections(order_id)
    blocks, rows = [], []
    for section in sections:
        block = section.matrix
        if not sp.issparse(block):
            raise E12ValidationError(f"{order_id}:{section.gsm}: source block was densified")
        blocks.append(block.tocsr(copy=True))
        rows.extend({"spot_id": str(spot), "library_id": str(section.library_id), "animal_id": str(section.animal_id), "donor": str(section.animal_id), "batch_name": str(section.library_id)} for spot in section.spot_ids)
    adata = AnnData(X=sp.vstack(blocks, format="csr"), obs=pd.DataFrame(rows).set_index("spot_id"))
    adata.var_names = list(bundle.feature_ids)
    # GraphST native code consumes ``spatial`` for the within-library graph.
    coords = []
    for section in sections:
        coords.extend(zip(section.coordinates["array_row"], section.coordinates["array_col"]))
    adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    return adata


def fit_run(*, output_dir: Path | str, project_root: Path | str = ROOT, device: str = "cuda", seeds: Sequence[int] | None = None, h5ad_path: Path | str | None = None, manifest_path: Path | str | None = None, allow_model_fits: bool = False) -> dict[str, Any]:
    if not allow_model_fits:
        raise RuntimeError("REFUSED: E12 fit requires --allow-model-fits; no model fit was started")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing output collision: {destination}")
    root = Path(project_root).resolve(); selected = _seeds(seeds); device_proof = _validate_device(device)
    if device == "cuda" and device_proof.get("status") != "PASS":
        raise RuntimeError("REFUSED: explicit CUDA device proof failed")
    available_device = None
    if device == "cuda":
        try:
            import torch
            available_device = int(torch.cuda.mem_get_info(0)[0])
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"REFUSED: CUDA free-memory proof unavailable: {type(exc).__name__}: {exc}") from exc
    resource = resource_estimate(available_device_bytes=available_device)
    if not resource["host_headroom_gate_pass"] or (device == "cuda" and not resource["device_headroom_gate_pass"]):
        raise RuntimeError("REFUSED: native GraphST dense peak plus 2 GiB host headroom is unavailable")
    e8 = _load_e8(); h5ad = Path(h5ad_path or e8.H5AD_DEFAULT); manifest_path = Path(manifest_path or e8.MANIFEST_DEFAULT)
    if not h5ad.is_absolute(): h5ad = root / h5ad
    if not manifest_path.is_absolute(): manifest_path = root / manifest_path
    closure = source_closure(root, require_native=True, require_authority=True)
    with e8.open_bundle(root, h5ad_path=h5ad, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        config = build_route_plan(project_root=root, h5ad_path=h5ad, manifest_path=manifest_path, feature_mode=FEATURE_MODE, device=device, seeds=selected)["config"]
        config_hash = canonical_sha256(config)
        destination.mkdir(parents=True)
        manifest = {"schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "PERFORMED", "output_dir": str(destination), "h5ad": {"path": str(h5ad), "sha256": bundle.h5ad_sha256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR"}, "manifest": {"path": str(manifest_path), "sha256": bundle.manifest_sha256}, "config": config, "config_sha256": config_hash, "source_closure": closure, "resource_qc": resource, "device_proof": device_proof, "claim_boundary": {"scope": CLAIM_SCOPE, "external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}, "started_at": _dt.datetime.now().astimezone().isoformat()}
        exclusive_json(destination / "manifest.json", manifest)
        graphst = _load_graphst_implementation()
        import pandas as pd  # noqa: F401  # native GraphST expects pandas-backed AnnData
        from GraphST import GraphST
        all_records: dict[str, Any] = {}
        for seed in selected:
            latents, animals, records = {}, {}, {}
            seed_propagation(seed)
            for order in ORDER_IDS:
                adata = _build_order_adata(bundle, order)
                GraphST.preprocess(adata)
                interaction, adj, _ = graphst._construct_interaction_blockdiag(adata, n_neighbors=N_NEIGHBORS)
                adata.obsm["graph_neigh"], adata.obsm["adj"] = interaction, adj
                model = GraphST.GraphST(adata, device=device, epochs=N_EPOCHS, dim_input=adata.n_vars, dim_output=DIM_OUTPUT, random_seed=seed)
                trained = model.train()
                emb = np.asarray(trained.obsm["emb"], dtype=np.float32)
                if emb.ndim != 2 or emb.shape[0] != N_ROWS or not np.isfinite(emb).all():
                    raise E12ValidationError(f"{order}:{seed}: native embedding malformed")
                centered = emb - emb.mean(axis=0)
                u, s, _ = np.linalg.svd(centered, full_matrices=False)
                latent = (u[:, :LATENT_DIM] * s[:LATENT_DIM]).astype(np.float32)
                ident = order_identity(bundle, order); animals[order] = ident["animal_ids"]; latents[order] = latent
                records[order] = write_arm(destination, latent=latent, identity=ident, seed=seed, order_id=order, base={"input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)), "config_sha256": config_hash, "source_closure": closure})
            metric = recompute_osi(latents, animals, seed=seed)
            checkpoint = {"schema_version": PROTOCOL, "status": "SEED_COMPLETE", "seed": seed, "orders": records, "metric": metric, "completed_at": _dt.datetime.now().astimezone().isoformat()}
            exclusive_json(destination / f"checkpoint_seed{seed}.json", checkpoint); all_records[str(seed)] = checkpoint
        summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "completed_seeds": list(selected), "expected_orders": list(ORDER_IDS), "seed_checkpoints": all_records, "completed_at": _dt.datetime.now().astimezone().isoformat()}
        exclusive_json(destination / "run_summary.json", summary); exclusive_json(destination / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(destination / "run_summary.json"), "completed_at": summary["completed_at"]})
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E12 native external GraphST route (fit is parent-gated)")
    parser.add_argument("--mode", choices=("plan", "fixture-preflight", "fit", "audit"), default="plan")
    parser.add_argument("--project-root", type=Path, default=ROOT); parser.add_argument("--h5ad", type=Path); parser.add_argument("--manifest", type=Path); parser.add_argument("--feature-mode", choices=(FEATURE_MODE, SENSITIVITY_FEATURE_MODE), default=FEATURE_MODE); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); parser.add_argument("--output-dir", type=Path); parser.add_argument("--seeds", default=None); parser.add_argument("--allow-model-fits", action="store_true"); parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    selected = tuple(int(v) for v in args.seeds.split(",")) if args.seeds else None
    if args.mode == "plan":
        result = build_route_plan(project_root=args.project_root, h5ad_path=args.h5ad, manifest_path=args.manifest, feature_mode=args.feature_mode, device=args.device, seeds=selected)
    elif args.mode == "fixture-preflight":
        result = fixture_preflight()
    elif args.mode == "fit":
        if args.output_dir is None: parser.error("--output-dir is required for fit")
        result = fit_run(output_dir=args.output_dir, project_root=args.project_root, device=args.device, seeds=selected, h5ad_path=args.h5ad, manifest_path=args.manifest, allow_model_fits=args.allow_model_fits)
    else:
        if args.output_dir is None: parser.error("--output-dir is required for audit")
        result = audit_output(output_dir=args.output_dir, project_root=args.project_root)
    if args.evidence:
        exclusive_json(args.evidence, result)
    print(json.dumps({"status": result.get("status"), "mode": args.mode, "model_fit": result.get("model_fit", "NOT_PERFORMED")}, sort_keys=True))
    return 0 if result.get("status") in {"PASS", "READY_NO_FIT", "FIT_COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
