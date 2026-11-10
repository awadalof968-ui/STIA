"""E9R repair: source-bound external PCA30 with a real terminal auditor.

E9's output and source remain immutable.  This successor fixes the backed
AnnData view-of-view path by slicing the backed CSR matrix once and constructing
an in-memory sparse AnnData before invoking the accepted log1p-CPM/full-SVD
functions.  The fit path remains parent-gated; preflight and audit do not fit.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stia.crossmethod import external_gse254652_adapter_v1_20260831 as e8
from stia.crossmethod import external_pca_e9_20260831 as e9


PROJECT_DEFAULT = e8.PROJECT_DEFAULT
H5AD_DEFAULT = e8.H5AD_DEFAULT
MANIFEST_DEFAULT = e8.MANIFEST_DEFAULT
PCA_DIM = 30
FEATURE_MODE = "all_common_19465"
PROTOCOL_VERSION = "stia.external.gse254652.e9r.pca_four_order.v1"
ORDER_IDS = tuple(e8.ORDER_IDENTITIES)
NO_LABELS = {"disease": "NOT_CREATED", "batch": "NOT_CREATED", "donor": "NOT_CREATED"}
REQUIRED_HEADROOM_BYTES = 2 * 1024**3
PARENT_MIN_FREE_BYTES = 10 * 1024**3
PCA_SOURCE_REL = Path("scripts/_x3_pca_harmony_osi.py")
PCA_R3_REL = Path("scripts/_x3_pca_strict_e2_r3_20260830.py")
E9_SOURCE_REL = Path("stia/crossmethod/external_pca_e9_20260831.py")
METRIC_REL = Path("stia/crossmethod/osi.py")
FROZEN_CONTRACT_REL = Path("stia/crossmethod/frozen_contract.py")
PLAN_REL = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/evidence/E9R/E9R_EXTERNAL_PCA_PLAN_20260831.json")


class E9RValidationError(ValueError):
    """Raised when source, input, order, artifact, or metric identity drifts."""


def sha256_file(path: Path | str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _canonical(value: Any) -> str:
    return e8.canonical_sha256(value)


def available_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def resource_estimate(*, n_rows: int, n_features: int, nnz: int, available: int | None = None) -> dict[str, Any]:
    """Conservative full-SVD peak accounting, without allocating an array."""
    n = int(n_rows); p = int(n_features); z = int(nnz); item4 = np.dtype(np.float32).itemsize; item8 = np.dtype(np.float64).itemsize
    rank = min(n, p)
    csr = z * (np.dtype(np.int64).itemsize + np.dtype(np.int32).itemsize) + (n + 1) * np.dtype(np.int64).itemsize
    original_dense_int64 = n * p * item8
    dense_f32 = n * p * item4
    svd_u = n * rank * item4
    svd_vt = rank * p * item4
    latent = n * PCA_DIM * item4
    # These terms are simultaneous/conservative accounting for the source
    # X.toarray() conversion, Xs/log1p temporaries, centered Xc, and SVD
    # arrays.  LAPACK workspace is explicitly unknown and reserved separately.
    known = csr + original_dense_int64 + dense_f32 + (2 * dense_f32) + dense_f32 + svd_u + svd_vt + latent
    lapack_unknown = 2 * (svd_u + svd_vt)
    estimated = known + lapack_unknown
    avail = available_bytes() if available is None else int(available)
    return {
        "input_csr_bytes": int(csr),
        "original_dense_int64_bytes": int(original_dense_int64),
        "preprocessing_dense_float32_bytes": int(dense_f32),
        "preprocessing_temporary_float32_bytes": int(2 * dense_f32),
        "centered_matrix_float32_bytes": int(dense_f32),
        "svd_u_float32_bytes": int(svd_u),
        "svd_vt_float32_bytes": int(svd_vt),
        "latent_float32_bytes": int(latent),
        "lapack_workspace_bytes": int(lapack_unknown),
        "lapack_workspace_status": "UNKNOWN_CONSERVATIVE_RESERVE_2X_U_PLUS_VT",
        "known_peak_bytes": int(known),
        "estimated_peak_bytes": int(estimated),
        "required_headroom_bytes": REQUIRED_HEADROOM_BYTES,
        "parent_min_free_bytes": PARENT_MIN_FREE_BYTES,
        "available_bytes": avail,
        "headroom_gate_pass": bool(avail is not None and avail >= estimated + REQUIRED_HEADROOM_BYTES),
        "parent_min_free_gate_pass": bool(avail is not None and avail >= PARENT_MIN_FREE_BYTES),
        "solver": "numpy.linalg.svd(full_matrices=False)",
        "densification": "fit-only; preflight/audit never allocate dense input",
    }


def _source_identities(root: Path, *, require_plan: bool = False) -> dict[str, dict[str, str]]:
    paths = {
        "e9r_runner": root / Path(__file__).relative_to(root),
        "e9_runner": root / E9_SOURCE_REL,
        "e8_adapter": root / "stia/crossmethod/external_gse254652_adapter_v1_20260831.py",
        "pca_source": root / PCA_SOURCE_REL,
        "pca_r3_runner": root / PCA_R3_REL,
        "metric": root / METRIC_REL,
        "frozen_contract": root / FROZEN_CONTRACT_REL,
        "plan": root / PLAN_REL,
    }
    result: dict[str, dict[str, str]] = {}
    for label, path in paths.items():
        if not path.is_file():
            if label == "plan" and not require_plan:
                continue
            raise E9RValidationError(f"missing source identity {label}: {path}")
        result[label] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _bundle_order_identity(bundle: e8.ExternalBundle, order_id: str) -> dict[str, Any]:
    # E9's identity derivation is reused, while this successor independently
    # checks the resulting row permutation and all actual key arrays.
    identity = e9.order_identity(bundle, order_id)
    rows = e9.order_rows(bundle, order_id)
    if identity["n_rows"] != 9568 or rows.size != 9568:
        raise E9RValidationError(f"{order_id}: expected 9568 source rows")
    if len(identity["library_keys"]) != len(identity["spot_ids"]) or len(identity["animal_keys"]) != len(identity["spot_ids"]):
        raise E9RValidationError(f"{order_id}: identity arrays are misaligned")
    return identity


def materialize_order_sparse(bundle: e8.ExternalBundle, order_id: str, *, max_rows: int | None = None) -> tuple[Any, np.ndarray]:
    """Slice backed CSR once, then copy that row-bound slice into memory."""
    try:
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover
        raise E9RValidationError("scipy is required for sparse materialization") from exc
    rows = e9.order_rows(bundle, order_id)
    if max_rows is not None:
        if int(max_rows) <= 0 or int(max_rows) > rows.size:
            raise E9RValidationError("max_rows must be within the order")
        rows = rows[: int(max_rows)]
    # Crucially, index the backed source X directly.  Do not create an
    # AnnData view and then index that view (the E9 defect was view-of-view).
    block = bundle._adata.X[rows, :]
    if not sp.issparse(block):
        raise E9RValidationError(f"{order_id}: backed row slice was densified")
    return block.tocsr(copy=True), rows


def prepare_order_matrix(bundle: e8.ExternalBundle, order_id: str, pca_module: Any, *, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Use the accepted preprocessing on a correctly materialized sparse order."""
    from anndata import AnnData

    sparse_block, rows = materialize_order_sparse(bundle, order_id, max_rows=max_rows)
    memory_adata = AnnData(X=sparse_block)
    memory_adata.var_names = list(bundle.feature_ids)
    matrix = np.asarray(pca_module.panel_subset_log1p_cpm(memory_adata, list(bundle.feature_ids)), dtype=np.float32)
    if matrix.shape != (rows.size, len(bundle.feature_ids)) or not np.isfinite(matrix).all():
        raise E9RValidationError(f"{order_id}: preprocessing output shape/finite check failed")
    return matrix, rows


def _load_pca_source(root: Path) -> tuple[Any, Path, str]:
    path = root / PCA_SOURCE_REL
    spec = importlib.util.spec_from_file_location("e9r_bound_pca_source", path)
    if spec is None or spec.loader is None:
        raise E9RValidationError(f"cannot import PCA source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path, sha256_file(path)


def _exclusive_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing output collision: {path}")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _write_order(output_dir: Path, order_id: str, latent: np.ndarray, identity: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    ids = np.asarray(identity["spot_ids"], dtype="U"); libraries = np.asarray(identity["library_keys"], dtype="U"); animals = np.asarray(identity["animal_keys"], dtype="U"); z = np.asarray(latent)
    if z.shape != (len(ids), PCA_DIM) or len(ids) != len(libraries) or len(ids) != len(animals) or len(set(ids.tolist())) != len(ids) or not np.isfinite(z).all():
        raise E9RValidationError(f"{order_id}: latent/key shape or finiteness mismatch")
    npz = output_dir / "orders" / f"{order_id}.npz"; sidecar = npz.with_suffix(".json")
    _exclusive_npz(npz, latent=z, spot_ids=ids, library_keys=libraries, animal_keys=animals)
    record = dict(metadata)
    record.update({"order_id": order_id, "npz": str(npz), "npz_sha256": sha256_file(npz), "latent_shape": list(z.shape), "latent_dtype": str(z.dtype), "identity": {"rows_sha256": identity["rows_sha256"], "spot_ids_sha256": identity["spot_ids_sha256"], "n_rows": identity["n_rows"], "order_gsm": identity["order_gsm"], "order_sha256": identity["order_sha256"]}})
    _exclusive_json(sidecar, record)
    return record


def _preflight_manifest(bundle: e8.ExternalBundle, root: Path, *, requested_seed: int) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_VERSION,
        "status": "READY_NO_FIT",
        "model_fit": "NOT_PERFORMED",
        "feature_mode": FEATURE_MODE,
        "feature_count": len(bundle.feature_ids),
        "feature_ids_sha256": _canonical(list(bundle.feature_ids)),
        "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [9568, 19465], "sparse_format": "CSR"},
        "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256},
        "source_identities": _source_identities(root),
        "pca_protocol": {"dimension": PCA_DIM, "preprocess": "panel_subset_log1p_cpm", "fit": "fit_pca", "solver": "centered numpy.linalg.svd full_matrices=False", "model_seed": None, "requested_seed": int(requested_seed), "replicates_are_stochastic": False},
        "orders": {name: {key: value for key, value in _bundle_order_identity(bundle, name).items() if key not in {"spot_ids", "library_keys", "animal_keys"}} for name in ORDER_IDS},
        "labels": dict(NO_LABELS),
        "resource_qc": resource_estimate(n_rows=9568, n_features=len(bundle.feature_ids), nnz=54807969),
        "claim_boundary": {"model_fit": "NOT_PERFORMED", "external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN", "exposure": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION_CANDIDATE"},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "observed_at": _datetime.datetime.now().astimezone().isoformat(),
    }


def preflight(project_root: Path | str = PROJECT_DEFAULT, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT, requested_seed: int = 20260819) -> dict[str, Any]:
    root = Path(project_root).resolve()
    with e8.open_bundle(root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        return _preflight_manifest(bundle, root, requested_seed=requested_seed)


def fit_bundle(project_root: Path | str = PROJECT_DEFAULT, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT, output_dir: Path | str, requested_seed: int = 20260819, allow_model_fits: bool = False) -> dict[str, Any]:
    if not allow_model_fits:
        raise RuntimeError("REFUSED: E9R fit requires --allow-model-fits; no model fit was started")
    root = Path(project_root).resolve(); destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to reuse output namespace: {destination}")
    with e8.open_bundle(root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        manifest = _preflight_manifest(bundle, root, requested_seed=requested_seed)
        manifest["source_identities"] = _source_identities(root, require_plan=True)
        resource = manifest["resource_qc"]
        if not resource["headroom_gate_pass"]:
            raise RuntimeError("REFUSED: E9R estimated peak plus 2 GiB headroom is unavailable")
        pca, source_path, source_sha = _load_pca_source(root)
        destination.mkdir(parents=True)
        manifest.update({"status": "FIT_STARTED", "model_fit": "STARTED", "fit_source_sha256": source_sha, "output_dir": str(destination.resolve())})
        _exclusive_json(destination / "manifest.json", manifest)
        records: dict[str, Any] = {}
        metadata_base = {"schema_version": PROTOCOL_VERSION, "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "feature_ids_sha256": _canonical(list(bundle.feature_ids)), "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "source_identities": manifest["source_identities"], "requested_seed": int(requested_seed), "model_seed": None, "seed_behavior": "deterministic_full_svd_seed_unused", "labels": dict(NO_LABELS), "pca_dim": PCA_DIM, "model_fit": "PERFORMED"}
        for order_id in ORDER_IDS:
            identity = _bundle_order_identity(bundle, order_id)
            matrix, _rows = prepare_order_matrix(bundle, order_id, pca)
            latent = np.asarray(pca.fit_pca(matrix, int(requested_seed)))
            records[order_id] = _write_order(destination, order_id, latent, identity, metadata_base)
        summary = {"schema_version": PROTOCOL_VERSION, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "requested_seed": int(requested_seed), "terminal": {"expected_orders": list(ORDER_IDS), "completed_orders": list(records), "n_completed_orders": len(records)}, "orders": records}
        _exclusive_json(destination / "run_summary.json", summary)
        return summary


def _canonical_align(latent: np.ndarray, ids: Sequence[str], keys: Sequence[str], canonical_ids: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    ids_list = [str(v) for v in ids]; canonical = [str(v) for v in canonical_ids]
    if len(ids_list) != len(set(ids_list)) or set(ids_list) != set(canonical):
        raise E9RValidationError("order IDs do not form the canonical source set")
    index = {value: i for i, value in enumerate(ids_list)}
    positions = [index[value] for value in canonical]
    matrix = np.asarray(latent)[positions]
    aligned_keys = [str(keys[i]) for i in positions]
    if matrix.shape[0] != len(canonical):
        raise E9RValidationError("canonical alignment shape mismatch")
    return matrix, aligned_keys


def recompute_metric(latents: Mapping[str, np.ndarray], animal_keys: Mapping[str, Sequence[str]], *, requested_seed: int) -> dict[str, Any]:
    """Recompute the frozen six-pair OSI metric keyed by actual animal IDs."""
    from stia.crossmethod.osi import compute_osi

    result = compute_osi(latents, animal_keys, condition="gse254652_p1_external", seed=int(requested_seed), margin=0.05)
    result["metric_key_semantics"] = "actual E6 animal_id values; not fabricated donor labels"
    result["pair_count"] = len(result["pairwise_drifts"])
    if result["pair_count"] != 6:
        raise E9RValidationError("frozen four-order metric must contain six pairwise drifts")
    return result


def _check_terminal_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != PROTOCOL_VERSION or summary.get("status") != "FIT_COMPLETE" or summary.get("model_fit") != "PERFORMED":
        raise E9RValidationError("terminal summary is missing FIT_COMPLETE/model-fit fields")
    terminal = summary.get("terminal")
    if not isinstance(terminal, Mapping) or tuple(terminal.get("expected_orders", ())) != ORDER_IDS or tuple(terminal.get("completed_orders", ())) != ORDER_IDS or terminal.get("n_completed_orders") != 4:
        raise E9RValidationError("terminal order completeness is not exactly A/B/C/D")


def audit_output(project_root: Path | str, output_dir: Path | str, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT) -> dict[str, Any]:
    root = Path(project_root).resolve(); destination = Path(output_dir)
    manifest_file = destination / "manifest.json"; summary_file = destination / "run_summary.json"
    if not manifest_file.is_file() or not summary_file.is_file():
        raise E9RValidationError("E9R terminal manifest or summary is missing")
    saved = json.loads(manifest_file.read_text(encoding="utf-8")); summary = json.loads(summary_file.read_text(encoding="utf-8")); _check_terminal_summary(summary)
    if saved.get("schema_version") != PROTOCOL_VERSION or saved.get("feature_mode") != FEATURE_MODE or saved.get("feature_count") != 19465 or saved.get("pca_protocol", {}).get("dimension") != PCA_DIM or saved.get("labels") != NO_LABELS:
        raise E9RValidationError("terminal protocol/feature/label contract drift")
    current_sources = _source_identities(root, require_plan=True)
    if saved.get("source_identities") != current_sources:
        raise E9RValidationError("source/plan closure hash binding drift")
    with e8.open_bundle(root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        if saved.get("h5ad", {}).get("sha256") != bundle.h5ad_sha256 or saved.get("manifest", {}).get("sha256") != bundle.manifest_sha256 or saved.get("feature_ids_sha256") != _canonical(list(bundle.feature_ids)):
            raise E9RValidationError("terminal input/feature binding drift")
        identities = {order_id: _bundle_order_identity(bundle, order_id) for order_id in ORDER_IDS}
        canonical_ids = identities["A_original"]["spot_ids"]
        latents: dict[str, np.ndarray] = {}; animals: dict[str, list[str]] = {}; arm_results: dict[str, Any] = {}
        for order_id in ORDER_IDS:
            npz = destination / "orders" / f"{order_id}.npz"; sidecar = npz.with_suffix(".json")
            if not npz.is_file() or not sidecar.is_file():
                raise E9RValidationError(f"terminal arm missing: {order_id}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("npz_sha256") != sha256_file(npz) or metadata.get("source_identities") != saved.get("source_identities") or metadata.get("model_seed") is not None or metadata.get("labels") != NO_LABELS or metadata.get("feature_mode") != FEATURE_MODE or metadata.get("feature_count") != 19465 or metadata.get("pca_dim") != PCA_DIM or int(metadata.get("requested_seed", -1)) != int(saved.get("pca_protocol", {}).get("requested_seed", -2)):
                raise E9RValidationError(f"{order_id}: sidecar provenance/hash/label drift")
            with np.load(npz, allow_pickle=False) as payload:
                if set(payload.files) != {"latent", "spot_ids", "library_keys", "animal_keys"}:
                    raise E9RValidationError(f"{order_id}: raw array key set drift")
                latent = np.asarray(payload["latent"]); ids = [str(v) for v in payload["spot_ids"].tolist()]; libs = [str(v) for v in payload["library_keys"].tolist()]; an = [str(v) for v in payload["animal_keys"].tolist()]
            identity = identities[order_id]
            if latent.shape != (9568, PCA_DIM) or ids != identity["spot_ids"] or libs != identity["library_keys"] or an != identity["animal_keys"] or metadata.get("npz_sha256") != sha256_file(npz):
                raise E9RValidationError(f"{order_id}: raw source identity mismatch")
            aligned, aligned_animals = _canonical_align(latent, ids, an, canonical_ids)
            latents[order_id] = aligned; animals[order_id] = aligned_animals
            arm_results[order_id] = {"status": "PASS", "n_rows": int(latent.shape[0]), "latent_shape": list(latent.shape), "npz_sha256": sha256_file(npz), "spot_ids_sha256": _canonical(ids), "library_keys_sha256": _canonical(libs), "animal_keys_sha256": _canonical(an)}
        metric = recompute_metric(latents, animals, requested_seed=int(saved.get("pca_protocol", {}).get("requested_seed", 20260819)))
    return {"schema_version": PROTOCOL_VERSION, "status": "PASS", "model_fit": "AUDITED_ONLY", "terminal_complete": True, "orders": arm_results, "metric": metric, "source_closure_verified": True, "canonical_row_alignment": True}


def write_receipt(receipt: Mapping[str, Any], output_dir: Path | str, *, filename: str = "E9R_EXTERNAL_PCA_PREFLIGHT_20260831.json") -> Path:
    path = Path(output_dir) / filename; _exclusive_json(path, receipt); return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E9R repaired external PCA30 preparation/audit")
    parser.add_argument("--mode", choices=("preflight", "fit", "audit"), default="preflight")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--h5ad", type=Path, default=H5AD_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--requested-seed", type=int, default=20260819)
    parser.add_argument("--allow-model-fits", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        receipt = preflight(args.project_root, h5ad_path=args.h5ad, manifest_path=args.manifest, requested_seed=args.requested_seed); output = write_receipt(receipt, args.output_dir) if args.output_dir else None
        print(json.dumps({"status": receipt["status"], "model_fit": receipt["model_fit"], "orders": list(receipt["orders"]), "output": str(output) if output else None}, ensure_ascii=False, sort_keys=True)); return 0
    if args.output_dir is None:
        parser.error("--output-dir is required with --mode fit/audit")
    if args.mode == "fit":
        result = fit_bundle(args.project_root, h5ad_path=args.h5ad, manifest_path=args.manifest, output_dir=args.output_dir, requested_seed=args.requested_seed, allow_model_fits=args.allow_model_fits)
    else:
        result = audit_output(args.project_root, args.output_dir, h5ad_path=args.h5ad, manifest_path=args.manifest)
    print(json.dumps({"status": result["status"], "model_fit": result["model_fit"], "orders": list(result["orders"]), "osi": result.get("metric", {}).get("osi")}, ensure_ascii=False, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
