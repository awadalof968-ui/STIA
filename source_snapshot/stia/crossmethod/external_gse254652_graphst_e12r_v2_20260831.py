"""E12R repair: strict native GraphST integration and terminal auditing.

The E12 route remains frozen.  This additive successor closes its empty-run
terminality hole, proves that native GraphST actually consumes all requested
features in a tiny CPU smoke, and accounts host/device/full-SVD buffers
separately.  It never trains the accepted external candidate.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import inspect
import io
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stia.crossmethod import external_gse254652_graphst_e12_v1_20260831 as e12


ROOT = e12.ROOT
RUN_ROOT = e12.RUN_ROOT
E8_ADAPTER = e12.E8_ADAPTER
E8_RECEIPT = e12.E8_RECEIPT
E12_SOURCE = ROOT / "stia" / "crossmethod" / "external_gse254652_graphst_e12_v1_20260831.py"
GRAPHST_WRAPPER = e12.GRAPHST_WRAPPER
GRAPHST_IMPLEMENTATION = e12.GRAPHST_IMPLEMENTATION
AUTHORITY = RUN_ROOT / "phases" / "terminal-audit" / "E12R.md"
PROTOCOL = "stia.external.gse254652.e12r.graphst_native.v2"
FEATURE_MODE = e12.FEATURE_MODE
ORDER_IDS = e12.ORDER_IDS
SEEDS = e12.SEEDS
N_EPOCHS = e12.N_EPOCHS
DIM_OUTPUT = e12.DIM_OUTPUT
N_NEIGHBORS = e12.N_NEIGHBORS
LATENT_DIM = e12.LATENT_DIM
OSI_MARGIN = e12.OSI_MARGIN
N_ROWS = e12.N_ROWS
N_FEATURES = e12.N_FEATURES
NNZ = e12.NNZ
CLAIM_SCOPE = e12.CLAIM_SCOPE
NO_LABELS = dict(e12.NO_LABELS)


class E12RValidationError(ValueError):
    """Fail-closed source, native, artifact, and terminal-contract error."""


sha256_file = e12.sha256_file
canonical_sha256 = e12.canonical_sha256
hash_strings = e12.hash_strings
hash_array = e12.hash_array
exclusive_json = e12.exclusive_json
exclusive_npz = e12.exclusive_npz


def _json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise E12RValidationError(f"missing JSON: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E12RValidationError(f"malformed JSON: {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise E12RValidationError(f"JSON object required: {target}")
    return value


def source_closure(root: Path = ROOT, *, require_native: bool = False, require_authority: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = {
        "e12r_route": Path(__file__).resolve(),
        "e12_route": E12_SOURCE,
        "e8_adapter": E8_ADAPTER,
        "e8_receipt": E8_RECEIPT,
        "graphst_wrapper": GRAPHST_WRAPPER,
        "graphst_implementation": GRAPHST_IMPLEMENTATION,
        "authority": AUTHORITY,
    }
    required = {"e12r_route", "e12_route", "e8_adapter"}
    if require_native:
        required.update({"graphst_wrapper", "graphst_implementation"})
    if require_authority:
        required.add("authority")
    out: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in paths.items():
        if not path.is_file():
            missing.append(name)
            continue
        out[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if required.intersection(missing):
        raise E12RValidationError(f"source closure missing: {sorted(required.intersection(missing))}")
    out["missing_optional"] = sorted(set(missing) - required)
    return out


def native_api_snapshot() -> dict[str, Any]:
    """Inspect the installed native GraphST API without constructing a model."""
    try:
        import GraphST as package
        from GraphST import GraphST as graphst_module
        from GraphST import preprocess
        cls = graphst_module.GraphST
        train = cls.train
        deps = {name: str(importlib.metadata.version(name)) for name in ("GraphST", "torch", "numpy", "scipy", "scanpy", "anndata")}
        class_path = Path(inspect.getfile(cls)).resolve()
        preprocess_path = Path(inspect.getfile(preprocess)).resolve()
        return {
            "status": "PASS", "package_path": str(Path(package.__file__).resolve()),
            "class_path": str(class_path), "class_sha256": sha256_file(class_path),
            "preprocess_path": str(preprocess_path), "preprocess_sha256": sha256_file(preprocess_path),
            "constructor_signature": str(inspect.signature(cls)),
            "train_signature": str(inspect.signature(train)),
            "preprocess_signature": str(inspect.signature(preprocess)),
            "default_epochs": int(inspect.signature(cls).parameters["epochs"].default),
            "default_dim_output": int(inspect.signature(cls).parameters["dim_output"].default),
            "default_hvg_behavior": "GraphST constructor invokes package preprocess() when highly_variable is absent; E12R sets all features highly_variable after native-compatible normalization to prevent silent 3000-gene reduction.",
            "dependencies": deps,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": "UNRESOLVED", "error": f"{type(exc).__name__}: {exc}"}


def resource_estimate(*, n_rows: int = N_ROWS, n_features: int = N_FEATURES, nnz: int = NNZ, data_itemsize: int = 4, index_itemsize: int = 4, indptr_itemsize: int = 8, available_host_bytes: int | None = None, available_device_bytes: int | None = None) -> dict[str, Any]:
    """Separate raw CSR, native dense, model, full-SVD and device budgets."""
    n, p, z = int(n_rows), int(n_features), int(nnz)
    csr_data = z * int(data_itemsize)
    csr_indices = z * int(index_itemsize)
    csr_indptr = (n + 1) * int(indptr_itemsize)
    dense_x = n * p * 4
    dense_graph_each = n * n * 4
    graph_raw = dense_graph_each * 2
    graph_normalization = dense_graph_each * 2
    feature_device = dense_x * 2  # features + permutation augmentation
    graph_device = dense_graph_each * 2  # graph_neigh + normalized adj
    hidden_device = n * DIM_OUTPUT * 4 * 3
    reconstruction_device = dense_x
    full_rank = min(n, p)
    svd_u = n * full_rank * 4
    svd_vt = full_rank * p * 4
    svd_s = full_rank * 4
    svd_workspace = 2 * (svd_u + svd_vt)
    host_known = csr_data + csr_indices + csr_indptr + dense_x + graph_raw + graph_normalization + svd_u + svd_vt + svd_s + svd_workspace
    device_known = feature_device + graph_device + hidden_device + reconstruction_device
    host_estimated = int(host_known * 1.5)
    device_estimated = int(device_known * 1.5)
    headroom = 2 * 1024**3
    if available_host_bytes is None:
        try:
            import psutil
            available_host_bytes = int(psutil.virtual_memory().available)
        except Exception:
            available_host_bytes = None
    host_gate = available_host_bytes is not None and available_host_bytes >= host_estimated + headroom
    device_gate = available_device_bytes is not None and available_device_bytes >= device_estimated + headroom
    return {
        "n_rows": n, "n_features": p, "nnz": z,
        "csr_data_bytes": int(csr_data), "csr_indices_bytes": int(csr_indices), "csr_indptr_bytes": int(csr_indptr), "csr_total_bytes": int(csr_data + csr_indices + csr_indptr), "csr_dtypes": {"data": f"itemsize={data_itemsize}", "indices": f"itemsize={index_itemsize}", "indptr": f"itemsize={indptr_itemsize}"},
        "dense_input_host_bytes": int(dense_x), "dense_graph_each_host_bytes": int(dense_graph_each), "dense_graph_raw_host_bytes": int(graph_raw), "dense_graph_normalization_host_bytes": int(graph_normalization),
        "feature_device_bytes": int(feature_device), "graph_device_bytes": int(graph_device), "hidden_device_bytes": int(hidden_device), "reconstruction_device_bytes": int(reconstruction_device),
        "full_svd_u_host_bytes": int(svd_u), "full_svd_vt_host_bytes": int(svd_vt), "full_svd_s_host_bytes": int(svd_s), "full_svd_lapack_workspace_bytes": int(svd_workspace),
        "host_known_peak_bytes": int(host_known), "host_estimated_peak_bytes": host_estimated, "device_known_peak_bytes": int(device_known), "device_estimated_peak_bytes": device_estimated,
        "allocator_overhead_factor": 1.5, "required_headroom_bytes": headroom, "available_host_bytes": available_host_bytes, "available_device_bytes": available_device_bytes,
        "host_headroom_gate_pass": bool(host_gate), "device_headroom_gate_pass": bool(device_gate), "measured_peak": False,
        "phases": {"preprocess": "CSR->native normalization/log/scale; package native preprocess/HVG deliberately not called", "graph": "dense graph_neigh+adj native contract", "fit": "whole-graph GraphST tensors/Adam/autograd", "reconstruction": "native emb plus dense reconstruction", "full_svd": "np.linalg.svd(full_matrices=False) on reconstructed n x p; LAPACK workspace conservative"},
    }


def _seed_all(seed: int) -> dict[str, Any]:
    value = int(seed)
    if value not in SEEDS:
        raise E12RValidationError(f"seed is not frozen: {seed}")
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    result: dict[str, Any] = {"python": value, "numpy": value % (2**32 - 1), "model_seed": value}
    try:
        import torch
        torch.manual_seed(value)
        result["torch"] = value
        result["torch_cuda"] = value if torch.cuda.is_available() else "not_available"
    except Exception as exc:
        result["torch"] = f"unavailable:{type(exc).__name__}"
        result["torch_cuda"] = "unavailable"
    return result


def native_preprocess_all_features(adata: Any) -> Any:
    """Use GraphST's native normalization/log/scale recipe without HVG loss."""
    import numpy as _np
    import scanpy as sc
    import scipy.sparse as sp

    if not sp.issparse(adata.X):
        raise E12RValidationError("native external route requires sparse CSR input before preprocessing")
    if getattr(adata.X, "format", None) != "csr":
        adata.X = adata.X.tocsr()
    original_features = int(adata.n_vars)
    original_names = list(map(str, adata.var_names))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    # GraphST.get_feature slices by this flag.  Marking every canonical input
    # feature true preserves the E12 all-common feature contract; no HVG call.
    adata.var["highly_variable"] = _np.ones(original_features, dtype=bool)
    if int(adata.n_vars) != original_features or list(map(str, adata.var_names)) != original_names:
        raise E12RValidationError("native preprocessing changed feature axis")
    values = adata.X.data if sp.issparse(adata.X) else _np.asarray(adata.X)
    if not _np.isfinite(values).all():
        raise E12RValidationError("native-compatible preprocessing produced non-finite values")
    return adata


def native_graph(adata: Any) -> Any:
    """Reuse the frozen producer's within-animal block graph construction."""
    frozen = e12._load_graphst_implementation()
    interaction, adj, _ = frozen._construct_interaction_blockdiag(adata, n_neighbors=N_NEIGHBORS)
    if interaction.shape != (adata.n_obs, adata.n_obs) or adj.shape != interaction.shape:
        raise E12RValidationError("native graph shape mismatch")
    adata.obsm["graph_neigh"], adata.obsm["adj"] = interaction, adj
    return adata


def native_train_recon_pca20(adata: Any, *, seed: int, epochs: int = N_EPOCHS, device: str = "cpu") -> tuple[np.ndarray, dict[str, Any]]:
    """Run the actual GraphST class on a caller-supplied tiny/parent-gated AnnData."""
    _seed_all(seed)
    if device not in {"cpu", "cuda"}:
        raise E12RValidationError("device must be explicit")
    import torch
    from GraphST import GraphST as graphst_module
    torch_device = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise E12RValidationError("CUDA requested but unavailable")
    if int(adata.n_vars) != len(adata.var_names) or not np.all(np.asarray(adata.var["highly_variable"], dtype=bool)):
        raise E12RValidationError("all requested native features are not marked consumed")
    native_graph(adata)
    model = graphst_module.GraphST(adata, device=torch_device, epochs=int(epochs), dim_input=int(adata.n_vars), dim_output=DIM_OUTPUT, random_seed=int(seed))
    effective_features = int(model.features.shape[1])
    if effective_features != int(adata.n_vars):
        raise E12RValidationError(f"native GraphST silently reduced features: {effective_features} != {adata.n_vars}")
    trained = model.train()
    reconstruction = np.asarray(trained.obsm["emb"], dtype=np.float32)
    if reconstruction.shape != (adata.n_obs, adata.n_vars) or not np.isfinite(reconstruction).all():
        raise E12RValidationError("native reconstruction shape/finite failure")
    centered = reconstruction - reconstruction.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if u.shape[1] < LATENT_DIM:
        raise E12RValidationError("tiny native smoke needs at least 20 PCA components")
    latent = (u[:, :LATENT_DIM] * s[:LATENT_DIM]).astype(np.float32)
    return latent, {"effective_feature_count": effective_features, "input_feature_count": int(adata.n_vars), "epochs": int(epochs), "dim_output": DIM_OUTPUT, "representation": "recon_pca20", "seed": int(seed), "device": str(torch_device), "native_train": True}


def write_arm(destination: Path | str, *, latent: np.ndarray, identity: Mapping[str, Any], seed: int, order_id: str, base: Mapping[str, Any]) -> dict[str, Any]:
    if int(seed) not in SEEDS or order_id not in ORDER_IDS:
        raise E12RValidationError("seed/order is not frozen")
    try:
        ids = np.asarray([str(v) for v in identity["spot_ids"]], dtype="U")
        libraries = np.asarray([str(v) for v in identity["library_ids"]], dtype="U")
        animals = np.asarray([str(v) for v in identity["animal_ids"]], dtype="U")
    except Exception as exc:
        raise E12RValidationError("identity arrays are required") from exc
    matrix = np.asarray(latent, dtype=np.float32)
    if ids.size == 0 or len(set(ids.tolist())) != ids.size or not (ids.size == libraries.size == animals.size) or matrix.shape != (ids.size, LATENT_DIM) or not np.isfinite(matrix).all():
        raise E12RValidationError("raw arm identity/latent shape or finite check failed")
    arm = Path(destination).resolve() / "arms" / f"seed{int(seed)}"
    npz_path, sidecar = arm / f"{order_id}.npz", arm / f"{order_id}.json"
    if npz_path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing arm overwrite: {order_id}:{seed}")
    exclusive_npz(npz_path, latent=matrix, spot_ids=ids, library_ids=libraries, animal_ids=animals)
    record = dict(base)
    record.update({"schema_version": PROTOCOL, "protocol": PROTOCOL, "status": "OK", "seed": int(seed), "model_seed": int(seed), "order": order_id, "npz": str(npz_path), "npz_sha256": sha256_file(npz_path), "latent_sha256": hash_array(matrix), "latent_shape": list(matrix.shape), "latent_dtype": "float32", "identity": {"rows_sha256": identity.get("rows_sha256"), "spot_ids_sha256": hash_strings(ids.tolist()), "identity_sha256": identity.get("identity_sha256", canonical_sha256(list(zip(ids.tolist(), libraries.tolist(), animals.tolist())))), "order_sha256": identity.get("order_sha256")}, "animal_units": sorted(set(animals.tolist())), "labels": dict(NO_LABELS)})
    exclusive_json(sidecar, record)
    return record


def load_arm(npz_path: Path | str, *, expected_identity: Mapping[str, Any] | None = None, expected_seed: int | None = None, expected_order: str | None = None, expected_config_sha256: str | None = None, expected_source_closure: Mapping[str, Any] | None = None, expected_input_h5ad_sha256: str | None = None, expected_manifest_sha256: str | None = None, expected_feature_ids_sha256: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    npz = Path(npz_path); sidecar = npz.with_suffix(".json")
    if not npz.is_file() or not sidecar.is_file():
        raise E12RValidationError(f"missing arm/sidecar: {npz}")
    metadata = _json(sidecar)
    # E12R is a strict auditor for both its repaired protocol and the
    # already-run E12 producer protocol.  Compatibility is limited to the
    # immutable protocol identifier; all bindings and raw-array checks below
    # remain mandatory.
    if metadata.get("schema_version") not in {PROTOCOL, e12.PROTOCOL} or metadata.get("status") != "OK" or metadata.get("npz_sha256") != sha256_file(npz):
        raise E12RValidationError("arm schema/status/hash mismatch")
    checks = (("seed", expected_seed), ("order", expected_order), ("config_sha256", expected_config_sha256), ("source_closure", expected_source_closure), ("input_h5ad_sha256", expected_input_h5ad_sha256), ("input_manifest_sha256", expected_manifest_sha256), ("feature_ids_sha256", expected_feature_ids_sha256))
    for field, expected in checks:
        if expected is not None and metadata.get(field) != expected:
            raise E12RValidationError(f"arm {field} binding mismatch")
    try:
        with np.load(npz, allow_pickle=False) as payload:
            if set(payload.files) != {"latent", "spot_ids", "library_ids", "animal_ids"}:
                raise E12RValidationError("arm key set mismatch")
            latent = np.asarray(payload["latent"]); ids = payload["spot_ids"].astype(str); libraries = payload["library_ids"].astype(str); animals = payload["animal_ids"].astype(str)
    except E12RValidationError:
        raise
    except Exception as exc:
        raise E12RValidationError(f"raw arm reload failed: {exc}") from exc
    if latent.ndim != 2 or latent.shape != (ids.size, LATENT_DIM) or libraries.size != ids.size or animals.size != ids.size or len(set(ids.tolist())) != ids.size or not np.isfinite(latent).all():
        raise E12RValidationError("arm arrays are malformed/non-finite")
    if expected_identity is not None:
        exp_ids = [str(v) for v in expected_identity["spot_ids"]]; exp_lib = [str(v) for v in expected_identity["library_ids"]]; exp_animals = [str(v) for v in expected_identity["animal_ids"]]
        if ids.tolist() != exp_ids or libraries.tolist() != exp_lib or animals.tolist() != exp_animals:
            raise E12RValidationError("source-derived arm identity mismatch")
        if expected_identity.get("identity_sha256") is not None and metadata.get("identity", {}).get("identity_sha256") != expected_identity["identity_sha256"]:
            raise E12RValidationError("source-derived identity fingerprint mismatch")
    return latent, ids, animals, metadata


def recompute_osi(latents: Mapping[str, np.ndarray], animals: Mapping[str, Sequence[str]], *, seed: int, condition: str = "gse254652_p1_external_graphst") -> dict[str, Any]:
    if set(latents) != set(ORDER_IDS) or set(animals) != set(ORDER_IDS):
        raise E12RValidationError("exactly frozen A-D orders are required")
    from stia.crossmethod.osi import compute_osi
    result = compute_osi(latents, animals, condition=condition, seed=int(seed), margin=OSI_MARGIN)
    if len(result.get("pairwise_drifts", ())) != 6 or len(result.get("donor_keys", ())) != 7:
        raise E12RValidationError("OSI must contain six pairs and seven animal units")
    result["animal_units"] = result.pop("donor_keys"); result["pair_count"] = 6; result["independent_recomputation"] = True
    return result


def validate_terminal_contract(output_dir: Path | str) -> dict[str, Any]:
    """Validate terminal markers/counts before any expensive raw/source audit."""
    destination = Path(output_dir).resolve(); errors: list[str] = []
    try:
        manifest = _json(destination / "manifest.json")
    except Exception as exc:
        return {"status": "UNRESOLVED", "execution_validity": "UNRESOLVED", "errors": [str(exc)], "cells": []}
    if manifest.get("schema_version") not in {PROTOCOL, e12.PROTOCOL} or manifest.get("status") != "FIT_STARTED" or manifest.get("model_fit") != "PERFORMED":
        errors.append("manifest schema/status/model_fit drift")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or manifest.get("config_sha256") != canonical_sha256(config):
        errors.append("manifest config hash/structure mismatch")
        config = {}
    expected_config = {
        "protocol": manifest.get("schema_version"), "feature_mode": FEATURE_MODE, "feature_count": N_FEATURES,
        "n_epochs": N_EPOCHS, "dim_output": DIM_OUTPUT, "n_neighbors": N_NEIGHBORS,
        "representation": "recon_pca20", "margin": OSI_MARGIN,
        "orders": list(ORDER_IDS), "seeds": list(SEEDS),
        "preprocessing": "GraphST native preprocess on all-common ENSMUSG CSR counts",
        "graph": "native within-library 3-NN donor/animal block graph; dense graph_neigh and adj required by frozen package",
        "labels": NO_LABELS,
    }
    if isinstance(config, Mapping):
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                errors.append(f"config semantic drift: {key}")
    requested = config.get("requested_seeds")
    try:
        requested_tuple = tuple(int(v) for v in requested)
    except Exception:
        requested_tuple = ()
        errors.append("requested_seeds missing or malformed")
    if not requested_tuple or len(set(requested_tuple)) != len(requested_tuple) or any(v not in SEEDS for v in requested_tuple):
        errors.append("requested_seeds must be a nonempty unique frozen subset")
    try:
        summary = _json(destination / "run_summary.json")
    except Exception as exc:
        return {"status": "UNRESOLVED", "execution_validity": "UNRESOLVED", "errors": errors + [str(exc)], "cells": []}
    completed = summary.get("completed_seeds")
    try:
        completed_tuple = tuple(int(v) for v in completed)
    except Exception:
        completed_tuple = ()
        errors.append("completed_seeds missing or malformed")
    if summary.get("schema_version") not in {PROTOCOL, e12.PROTOCOL} or summary.get("status") != "FIT_COMPLETE" or summary.get("expected_orders") != list(ORDER_IDS):
        errors.append("summary schema/status/order drift")
    if len(completed_tuple) != len(set(completed_tuple)) or any(v not in SEEDS for v in completed_tuple):
        errors.append("completed_seeds contains duplicate/foreign seed")
    if completed_tuple != requested_tuple:
        errors.append("completed_seeds does not exactly equal manifest requested_seeds")
    marker_path = destination / "COMPLETED.json"
    try:
        marker = _json(marker_path)
        if marker.get("schema_version") not in {PROTOCOL, e12.PROTOCOL} or marker.get("status") != "COMPLETE" or marker.get("summary_sha256") != sha256_file(destination / "run_summary.json"):
            errors.append("COMPLETED marker status/summary hash mismatch")
    except Exception as exc:
        errors.append(str(exc))
    checkpoints = summary.get("seed_checkpoints")
    if not isinstance(checkpoints, Mapping):
        errors.append("seed_checkpoints missing")
        checkpoints = {}
    cells = []
    for seed in requested_tuple:
        cp_path = destination / f"checkpoint_seed{seed}.json"
        try:
            checkpoint = _json(cp_path)
            if checkpoint.get("schema_version") not in {PROTOCOL, e12.PROTOCOL} or checkpoint.get("status") != "SEED_COMPLETE" or checkpoint.get("seed") != seed or checkpoints.get(str(seed)) != checkpoint:
                raise E12RValidationError("checkpoint linkage/schema mismatch")
            if checkpoint.get("orders") is not None and set(checkpoint.get("orders", {})) != set(ORDER_IDS):
                raise E12RValidationError("checkpoint order set mismatch")
            cells.append({"seed": seed, "checkpoint": "PASS"})
        except Exception as exc:
            errors.append(f"seed{seed} checkpoint: {exc}"); cells.append({"seed": seed, "checkpoint": "UNRESOLVED", "reason": str(exc)})
    if not requested_tuple:
        errors.append("empty requested seed set cannot be terminal PASS")
    if errors:
        partial = bool(requested_tuple and completed_tuple and set(completed_tuple).issubset(set(requested_tuple)) and set(completed_tuple) != set(requested_tuple))
        return {"status": "PARTIAL" if partial else "UNRESOLVED", "execution_validity": "UNRESOLVED", "errors": errors, "cells": cells, "requested_seeds": list(requested_tuple), "completed_seeds": list(completed_tuple)}
    return {"status": "PASS", "execution_validity": "PASS", "errors": [], "cells": cells, "requested_seeds": list(requested_tuple), "completed_seeds": list(completed_tuple), "manifest": manifest, "summary": summary}


def audit_output(*, output_dir: Path | str, project_root: Path | str = ROOT) -> dict[str, Any]:
    contract = validate_terminal_contract(output_dir)
    if contract["status"] != "PASS":
        return {"schema_version": PROTOCOL, "status": contract["status"], "execution_validity": "UNRESOLVED", "errors": contract["errors"], "cells": contract["cells"], "claim_boundary": {"scope": CLAIM_SCOPE, "external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}, "audited_at": _dt.datetime.now().astimezone().isoformat()}
    manifest = contract["manifest"]; summary = contract["summary"]; root = Path(project_root).resolve(); closure = source_closure(root, require_native=True, require_authority=True)
    saved_closure = manifest.get("source_closure")
    # The fit was produced by E12, while E12R is an additive auditor.  Compare
    # immutable hashes rather than serialized Windows path strings (which can
    # differ in console Unicode normalization), and explicitly verify the
    # original E12 authority document used by the fit.
    closure_drift = []
    if not isinstance(saved_closure, Mapping):
        closure_drift.append("saved source closure missing")
    else:
        for key, saved in saved_closure.items():
            if key == "missing_optional":
                continue
            expected_hash = saved.get("sha256") if isinstance(saved, Mapping) else None
            if key == "authority":
                actual_path = RUN_ROOT / "phases" / "terminal-audit" / "E12.md"
            else:
                actual_path = closure.get(key, {}).get("path") if isinstance(closure.get(key), Mapping) else None
            if not actual_path or not Path(actual_path).is_file() or sha256_file(Path(actual_path)) != expected_hash:
                closure_drift.append(key)
    if closure_drift:
        return {"schema_version": PROTOCOL, "status": "UNRESOLVED", "execution_validity": "UNRESOLVED", "errors": ["terminal source closure drift: " + ",".join(closure_drift)], "cells": contract["cells"]}
    closure_for_arms = saved_closure
    e8 = e12._load_e8(); h5ad = Path(manifest.get("h5ad", {}).get("path", e8.H5AD_DEFAULT)); manifest_file = Path(manifest.get("manifest", {}).get("path", e8.MANIFEST_DEFAULT))
    if not h5ad.is_absolute(): h5ad = root / h5ad
    if not manifest_file.is_absolute(): manifest_file = root / manifest_file
    errors: list[str] = []; cells: list[dict[str, Any]] = []
    with e8.open_bundle(root, h5ad_path=h5ad, manifest_path=manifest_file, feature_mode=FEATURE_MODE) as bundle:
        identities = {order: e12.order_identity(bundle, order) for order in ORDER_IDS}; feature_hash = canonical_sha256(list(bundle.feature_ids))
        for seed in contract["requested_seeds"]:
            latents, animal_ids = {}, {}
            try:
                for order in ORDER_IDS:
                    npz = Path(output_dir).resolve() / "arms" / f"seed{seed}" / f"{order}.npz"
                    latent, _ids, animals, metadata = load_arm(npz, expected_identity=identities[order], expected_seed=seed, expected_order=order, expected_config_sha256=manifest.get("config_sha256"), expected_source_closure=closure_for_arms, expected_input_h5ad_sha256=bundle.h5ad_sha256, expected_manifest_sha256=bundle.manifest_sha256, expected_feature_ids_sha256=feature_hash)
                    checkpoint_record = summary["seed_checkpoints"][str(seed)]["orders"][order]
                    if checkpoint_record.get("status") != "OK" or checkpoint_record.get("npz_sha256") != metadata.get("npz_sha256") or Path(str(checkpoint_record.get("npz"))).resolve() != npz.resolve():
                        raise E12RValidationError(f"checkpoint→sidecar linkage mismatch: {order}:{seed}")
                    latents[order] = latent; animal_ids[order] = animals.tolist()
                metric = recompute_osi(latents, animal_ids, seed=seed)
                cells.append({"seed": seed, "execution_status": "PASS", "numeric_status": metric["status"], "osi": float(metric["osi"]), "pair_count": metric["pair_count"]})
            except Exception as exc:
                errors.append(f"seed{seed}: {exc}"); cells.append({"seed": seed, "execution_status": "UNRESOLVED", "numeric_status": "UNRESOLVED", "reason": str(exc)})
    verdict = "PASS" if not errors and all(cell["execution_status"] == "PASS" for cell in cells) else "UNRESOLVED"
    return {"schema_version": PROTOCOL, "status": verdict, "execution_validity": "PASS" if verdict == "PASS" else "UNRESOLVED", "errors": errors, "cells": cells, "claim_boundary": {"scope": CLAIM_SCOPE, "external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}, "audited_at": _dt.datetime.now().astimezone().isoformat()}


def build_tiny_adata(order: str = "A_original", *, n_per_animal: int = 4, n_features: int = 24) -> Any:
    """Make a real sparse AnnData fixture for native GraphST smoke only."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    e8 = e12._load_e8()
    if order not in ORDER_IDS:
        raise E12RValidationError(f"unknown order: {order}")
    animals = ["460", "462", "463", "708", "709", "710", "713"]
    ordering = list(e8.ORDER_IDENTITIES[order]["animals"])
    rows: list[dict[str, Any]] = []; values: list[list[float]] = []; coords: list[list[float]] = []
    for animal_index, animal in enumerate(ordering):
        for local in range(n_per_animal):
            rows.append({"spot_id": f"tiny::{animal}::{local}", "animal_id": animal, "donor": animal, "library_id": f"tiny-{animal}"})
            row = np.zeros(n_features, dtype=np.float32); row[(animal_index + local) % n_features] = 5 + animal_index; row[(animal_index * 3 + local + 1) % n_features] = 2
            values.append(row.tolist()); coords.append([float(animal_index * 10 + local), float(local)])
    matrix = sp.csr_matrix(np.asarray(values, dtype=np.float32))
    adata = ad.AnnData(X=matrix, obs=pd.DataFrame(rows).set_index("spot_id"))
    adata.var_names = [f"ENSMUSG_TINY_{i:05d}" for i in range(n_features)]
    adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    return adata


def native_fixture_smoke(*, seed: int = 20260819, epochs: int = 2) -> dict[str, Any]:
    """Actual two-order CPU native GraphST smoke; dimensions are fixture-only."""
    import torch
    if torch.cuda.is_available():
        # The smoke is intentionally CPU-only even on CUDA hosts.
        torch.cuda.empty_cache()
    outputs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="e12r-native-smoke-") as td:
        for order in ORDER_IDS[:2]:
            adata = native_preprocess_all_features(build_tiny_adata(order))
            latent, details = native_train_recon_pca20(adata, seed=seed, epochs=epochs, device="cpu")
            identity = {"spot_ids": list(map(str, adata.obs_names)), "library_ids": adata.obs["library_id"].astype(str).tolist(), "animal_ids": adata.obs["animal_id"].astype(str).tolist()}
            record = write_arm(td, latent=latent, identity=identity, seed=seed, order_id=order, base={"config_sha256": "fixture-config", "source_closure": {"fixture": True}, "input_h5ad_sha256": "fixture-input", "input_manifest_sha256": "fixture-manifest", "feature_ids_sha256": canonical_sha256(list(map(str, adata.var_names)))})
            reloaded, _ids, _animals, _meta = load_arm(record["npz"], expected_seed=seed, expected_order=order, expected_config_sha256="fixture-config", expected_source_closure={"fixture": True}, expected_input_h5ad_sha256="fixture-input", expected_manifest_sha256="fixture-manifest", expected_feature_ids_sha256=canonical_sha256(list(map(str, adata.var_names))))
            outputs[order] = {"latent_shape": list(latent.shape), "input_feature_count": details["input_feature_count"], "effective_feature_count": details["effective_feature_count"], "native_train": details["native_train"], "seed": details["seed"], "order": order, "raw_reload_shape": list(reloaded.shape), "raw_reload_finite": bool(np.isfinite(reloaded).all())}
    return {"status": "PASS", "model_fit": "FIXTURE_ONLY", "orders": outputs, "requested_seed": int(seed), "production_epochs": N_EPOCHS, "production_feature_count": N_FEATURES, "cuda_used": False, "external_input_opened": False, "raw_write_reload": "PASS"}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="E12R native GraphST repair (no external fit)")
    parser.add_argument("--mode", choices=("native-api", "fixture-smoke", "audit"), default="native-api")
    parser.add_argument("--project-root", type=Path, default=ROOT); parser.add_argument("--output-dir", type=Path); parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "native-api":
        result = {"status": "PASS" if native_api_snapshot().get("status") == "PASS" else "UNRESOLVED", "model_fit": "NOT_PERFORMED", "native_api": native_api_snapshot(), "resource_qc": resource_estimate()}
    elif args.mode == "fixture-smoke":
        result = native_fixture_smoke()
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for audit")
        result = audit_output(output_dir=args.output_dir, project_root=args.project_root)
    if args.evidence:
        exclusive_json(args.evidence, result)
    print(json.dumps({"status": result.get("status"), "mode": args.mode, "model_fit": result.get("model_fit", "NOT_PERFORMED")}, sort_keys=True))
    return 0 if result.get("status") in {"PASS", "READY_NO_FIT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
