"""E9 external PCA30 four-order baseline preparation.

This module is a thin, source-bound successor over the E8 sparse external
adapter.  ``preflight`` and ``audit`` never densify or fit.  The guarded
``fit`` path is available only to the parent launch gate and reuses the
accepted ``_x3_pca_harmony_osi.py`` log1p-CPM and deterministic full-SVD
functions.  No disease, batch, donor, or outcome labels are manufactured.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stia.crossmethod import external_gse254652_adapter_v1_20260831 as e8


PROJECT_DEFAULT = e8.PROJECT_DEFAULT
H5AD_DEFAULT = e8.H5AD_DEFAULT
MANIFEST_DEFAULT = e8.MANIFEST_DEFAULT
PCA_DIM = 30
FEATURE_MODE = "all_common_19465"
PROTOCOL_VERSION = "stia.external.gse254652.e9.pca_four_order.v1"
PCA_SOURCE_REL = Path("scripts/_x3_pca_harmony_osi.py")
PCA_R3_REL = Path("scripts/_x3_pca_strict_e2_r3_20260830.py")
ORDER_IDS = tuple(e8.ORDER_IDENTITIES)
NO_LABELS = {"disease": "NOT_CREATED", "batch": "NOT_CREATED", "donor": "NOT_CREATED"}
REQUIRED_HEADROOM_BYTES = 2 * 1024**3


class E9ValidationError(ValueError):
    """Raised when source/order/output identity cannot be proven."""


def sha256_file(path: Path | str) -> str:
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


def _canonical_sha(value: Any) -> str:
    return e8.canonical_sha256(value)


def _available_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def resource_estimate(*, n_rows: int, n_features: int, nnz: int, available_bytes: int | None = None) -> dict[str, Any]:
    """Estimate the accepted dense full-SVD working set without allocating it."""
    dense_bytes = int(n_rows) * int(n_features) * np.dtype(np.float32).itemsize
    # One dense matrix, a centered working copy, the SVD work array, and the
    # source CSR payload.  The estimate is intentionally conservative.
    csr_bytes = int(nnz) * (np.dtype(np.int64).itemsize + np.dtype(np.int32).itemsize) + (int(n_rows) + 1) * np.dtype(np.int64).itemsize
    estimated_peak = dense_bytes * 3 + csr_bytes
    available = _available_bytes() if available_bytes is None else int(available_bytes)
    gate = available is not None and available >= estimated_peak + REQUIRED_HEADROOM_BYTES
    return {
        "dense_float32_bytes": dense_bytes,
        "csr_bytes_estimate": csr_bytes,
        "estimated_peak_bytes": estimated_peak,
        "required_headroom_bytes": REQUIRED_HEADROOM_BYTES,
        "available_bytes": available,
        "headroom_gate_pass": bool(gate),
        "solver": "numpy.linalg.svd(full_matrices=False)",
        "densification_allowed": False,
    }


def _source_identities(project_root: Path) -> dict[str, dict[str, str]]:
    root = Path(project_root).resolve()
    e8_path = root / "stia" / "crossmethod" / "external_gse254652_adapter_v1_20260831.py"
    pca_path = root / PCA_SOURCE_REL
    r3_path = root / PCA_R3_REL
    for label, path in (("e8_adapter", e8_path), ("pca_source", pca_path), ("pca_r3_runner", r3_path)):
        if not path.is_file():
            raise E9ValidationError(f"missing source identity {label}: {path}")
    return {
        "e8_adapter": {"path": str(e8_path), "sha256": sha256_file(e8_path)},
        "pca_source": {"path": str(pca_path), "sha256": sha256_file(pca_path)},
        "pca_r3_runner": {"path": str(r3_path), "sha256": sha256_file(r3_path)},
    }


def order_rows(bundle: e8.ExternalBundle, order_id: str) -> np.ndarray:
    """Return the explicit source-row permutation for one frozen E7 order."""
    if order_id not in ORDER_IDS:
        raise E9ValidationError(f"unknown E7 order: {order_id}")
    expected = tuple(bundle.orders[order_id]["gsm"])
    by_gsm = {section.gsm: section for section in bundle.sections}
    if set(by_gsm) != set(expected) or len(by_gsm) != len(expected):
        raise E9ValidationError(f"{order_id}: section identity set drift")
    rows: list[int] = []
    ids: list[str] = []
    for gsm in expected:
        section = by_gsm[gsm]
        start, end = map(int, section.row_offset)
        if end <= start or section.n_rows != end - start:
            raise E9ValidationError(f"{order_id}:{gsm}: invalid half-open offset")
        rows.extend(range(start, end))
        ids.extend(section.spot_ids)
    if len(rows) != sum(section.n_rows for section in bundle.sections) or len(set(rows)) != len(rows):
        raise E9ValidationError(f"{order_id}: row permutation is not a bijection")
    expected_hash = str(bundle.orders[order_id]["sha256"])
    measured_hash = _canonical_sha(list(expected))
    if measured_hash != expected_hash:
        raise E9ValidationError(f"{order_id}: frozen order hash drift")
    # IDs are checked here, before any guarded fit, so an order/row mismatch
    # cannot be hidden by a later PCA call.
    if len(ids) != len(set(ids)) or any(not item.startswith(gsm + "::") for item, gsm in zip(ids, [g for g in expected for _ in range(by_gsm[g].n_rows)])):
        raise E9ValidationError(f"{order_id}: namespaced spot-ID alignment drift")
    return np.asarray(rows, dtype=np.int64)


def order_identity(bundle: e8.ExternalBundle, order_id: str) -> dict[str, Any]:
    rows = order_rows(bundle, order_id)
    by_gsm = {section.gsm: section for section in bundle.sections}
    ids: list[str] = []
    animals: list[str] = []
    libraries: list[str] = []
    for gsm in bundle.orders[order_id]["gsm"]:
        section = by_gsm[gsm]
        ids.extend(section.spot_ids)
        animals.extend([section.animal_id] * section.n_rows)
        libraries.extend([section.gsm] * section.n_rows)
    return {
        "order_id": order_id,
        "rows_sha256": _canonical_sha(rows.tolist()),
        "spot_ids_sha256": _canonical_sha(ids),
        "spot_ids": ids,
        "library_keys": libraries,
        "animal_keys": animals,
        "n_rows": int(rows.size),
        "order_gsm": list(bundle.orders[order_id]["gsm"]),
        "order_sha256": str(bundle.orders[order_id]["sha256"]),
    }


def _manifest(bundle: e8.ExternalBundle, project_root: Path, *, requested_seed: int) -> dict[str, Any]:
    source = _source_identities(project_root)
    memory = resource_estimate(n_rows=9568, n_features=len(bundle.feature_ids), nnz=54807969)
    return {
        "schema_version": PROTOCOL_VERSION,
        "status": "READY_NO_FIT",
        "feature_mode": FEATURE_MODE,
        "feature_count": len(bundle.feature_ids),
        "feature_ids_sha256": _canonical_sha(list(bundle.feature_ids)),
        "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [9568, 19465], "sparse_format": "CSR"},
        "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256},
        "source_identities": source,
        "pca_protocol": {"dimension": PCA_DIM, "preprocess_entrypoint": "panel_subset_log1p_cpm", "fit_entrypoint": "fit_pca", "solver": "centered numpy.linalg.svd full_matrices=False", "model_seed": None, "requested_seed": int(requested_seed), "replicates_are_stochastic": False},
        "orders": {name: {key: value for key, value in order_identity(bundle, name).items() if key not in {"spot_ids", "library_keys", "animal_keys"}} for name in ORDER_IDS},
        "labels": dict(NO_LABELS),
        "resource_qc": memory,
        "claim_boundary": {"model_fit": "NOT_PERFORMED", "external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN", "exposure": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION_CANDIDATE"},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "observed_at": _datetime.datetime.now().astimezone().isoformat(),
    }


def preflight(project_root: Path | str = PROJECT_DEFAULT, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT, requested_seed: int = 20260819) -> dict[str, Any]:
    """Read-only E8-bound input/order/resource check; never densifies or fits."""
    root = Path(project_root).resolve()
    with e8.open_bundle(root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        receipt = _manifest(bundle, root, requested_seed=requested_seed)
        receipt["orders"] = {name: order_identity(bundle, name) for name in ORDER_IDS}
        receipt["model_fit"] = "NOT_PERFORMED"
        receipt["status"] = "READY_NO_FIT"
        return receipt


def _load_pca_source(project_root: Path) -> tuple[Any, Path, str]:
    path = project_root / PCA_SOURCE_REL
    measured = sha256_file(path)
    spec = importlib.util.spec_from_file_location("e9_bound_pca_source", path)
    if spec is None or spec.loader is None:
        raise E9ValidationError(f"cannot import PCA source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path, measured


def _exclusive_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing output collision: {path}")
    # Passing an xb handle gives exclusive-create semantics while retaining
    # numpy's compressed NPZ writer.
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def write_order_artifact(output_dir: Path | str, *, order_id: str, latent: np.ndarray, identity: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    if order_id not in ORDER_IDS:
        raise E9ValidationError(f"unknown order: {order_id}")
    z = np.asarray(latent)
    ids = np.asarray(identity["spot_ids"], dtype="U")
    libraries = np.asarray(identity["library_keys"], dtype="U")
    animals = np.asarray(identity["animal_keys"], dtype="U")
    if z.ndim != 2 or z.shape != (len(ids), PCA_DIM) or len(set(ids.tolist())) != len(ids) or not np.isfinite(z).all():
        raise E9ValidationError(f"{order_id}: latent/identity shape or finiteness mismatch")
    destination = Path(output_dir)
    npz = destination / "orders" / f"{order_id}.npz"
    sidecar = npz.with_suffix(".json")
    _exclusive_npz(npz, latent=z, spot_ids=ids, library_keys=libraries, animal_keys=animals)
    record = dict(metadata)
    record.update({"order_id": order_id, "npz": str(npz), "npz_sha256": sha256_file(npz), "latent_shape": list(z.shape), "latent_dtype": str(z.dtype), "identity": {key: identity[key] for key in ("rows_sha256", "spot_ids_sha256", "n_rows", "order_gsm", "order_sha256")}})
    _exclusive_json(sidecar, record)
    return record


def fit_bundle(project_root: Path | str = PROJECT_DEFAULT, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT, output_dir: Path | str, requested_seed: int = 20260819, allow_model_fits: bool = False) -> dict[str, Any]:
    """Parent-gated four-order fit using the accepted full-SVD PCA source."""
    if not allow_model_fits:
        raise RuntimeError("REFUSED: E9 fit requires --allow-model-fits; no model fit was started")
    root = Path(project_root).resolve()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to reuse output namespace: {destination}")
    with e8.open_bundle(root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        manifest = _manifest(bundle, root, requested_seed=requested_seed)
        if not manifest["resource_qc"]["headroom_gate_pass"]:
            raise RuntimeError("REFUSED: host resource headroom gate failed before PCA")
        pca, source_path, source_sha = _load_pca_source(root)
        destination.mkdir(parents=True)
        manifest.update({"status": "FIT_STARTED", "model_fit": "STARTED", "fit_source_sha256": source_sha, "output_dir": str(destination.resolve())})
        _exclusive_json(destination / "manifest.json", manifest)
        records: dict[str, Any] = {}
        for order_id in ORDER_IDS:
            identity = order_identity(bundle, order_id)
            rows = order_rows(bundle, order_id)
            view = bundle._adata[rows, :]
            matrix = np.asarray(pca.panel_subset_log1p_cpm(view, list(bundle.feature_ids)), dtype=np.float32)
            latent = np.asarray(pca.fit_pca(matrix, int(requested_seed)))
            metadata = {"schema_version": PROTOCOL_VERSION, "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "feature_ids_sha256": _canonical_sha(list(bundle.feature_ids)), "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "source_identities": manifest["source_identities"], "requested_seed": int(requested_seed), "model_seed": None, "seed_behavior": "deterministic_full_svd_seed_unused", "labels": dict(NO_LABELS), "pca_dim": PCA_DIM, "model_fit": "PERFORMED"}
            records[order_id] = write_order_artifact(destination, order_id=order_id, latent=latent, identity=identity, metadata=metadata)
        terminal = {"schema_version": PROTOCOL_VERSION, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "requested_seed": int(requested_seed), "orders": records}
        _exclusive_json(destination / "run_summary.json", terminal)
        return terminal


def audit_output(project_root: Path | str, output_dir: Path | str, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT) -> dict[str, Any]:
    """Independent source/ID/hash audit of a completed E9 output namespace."""
    root = Path(project_root).resolve(); destination = Path(output_dir)
    manifest_path_out = destination / "manifest.json"
    if not manifest_path_out.is_file():
        raise E9ValidationError(f"missing E9 output manifest: {manifest_path_out}")
    saved = json.loads(manifest_path_out.read_text(encoding="utf-8"))
    if saved.get("schema_version") != PROTOCOL_VERSION or saved.get("feature_mode") != FEATURE_MODE or saved.get("feature_count") != 19465:
        raise E9ValidationError("E9 output manifest protocol/feature drift")
    with e8.open_bundle(root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        if saved.get("h5ad", {}).get("sha256") != bundle.h5ad_sha256 or saved.get("manifest", {}).get("sha256") != bundle.manifest_sha256:
            raise E9ValidationError("E9 output input binding drift")
        results: dict[str, Any] = {}
        for order_id in ORDER_IDS:
            npz = destination / "orders" / f"{order_id}.npz"; sidecar = npz.with_suffix(".json")
            if not npz.is_file() or not sidecar.is_file():
                raise E9ValidationError(f"missing E9 order artifact: {order_id}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("npz_sha256") != sha256_file(npz):
                raise E9ValidationError(f"{order_id}: NPZ hash drift")
            with np.load(npz, allow_pickle=False) as loaded:
                latent = np.asarray(loaded["latent"]); ids = [str(v) for v in loaded["spot_ids"].tolist()]; libs = [str(v) for v in loaded["library_keys"].tolist()]; animals = [str(v) for v in loaded["animal_keys"].tolist()]
            identity = order_identity(bundle, order_id)
            if latent.shape != (9568, PCA_DIM) or not np.isfinite(latent).all() or ids != identity["spot_ids"] or libs != identity["library_keys"] or animals != identity["animal_keys"]:
                raise E9ValidationError(f"{order_id}: latent or source identity mismatch")
            if metadata.get("model_seed") is not None or metadata.get("labels") != NO_LABELS:
                raise E9ValidationError(f"{order_id}: forbidden seed/label metadata")
            results[order_id] = {"status": "PASS", "n_rows": int(latent.shape[0]), "latent_shape": list(latent.shape), "npz_sha256": sha256_file(npz), "spot_ids_sha256": identity["spot_ids_sha256"]}
    return {"schema_version": PROTOCOL_VERSION, "status": "PASS", "model_fit": "AUDITED_ONLY", "orders": results}


def write_preflight(receipt: Mapping[str, Any], output_dir: Path | str, *, filename: str = "E9_EXTERNAL_PCA_PREFLIGHT_20260831.json") -> Path:
    path = Path(output_dir) / filename
    _exclusive_json(path, receipt)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E9 external PCA30 four-order preparation")
    parser.add_argument("--mode", choices=("preflight", "fit", "audit"), default="preflight")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--h5ad", type=Path, default=H5AD_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--requested-seed", type=int, default=20260819)
    parser.add_argument("--allow-model-fits", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        receipt = preflight(args.project_root, h5ad_path=args.h5ad, manifest_path=args.manifest, requested_seed=args.requested_seed)
        output = write_preflight(receipt, args.output_dir) if args.output_dir else None
        print(json.dumps({"status": receipt["status"], "model_fit": receipt["model_fit"], "orders": list(receipt["orders"]), "output": str(output) if output else None}, ensure_ascii=False, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required with --mode fit/audit")
    if args.mode == "fit":
        result = fit_bundle(args.project_root, h5ad_path=args.h5ad, manifest_path=args.manifest, output_dir=args.output_dir, requested_seed=args.requested_seed, allow_model_fits=args.allow_model_fits)
    else:
        result = audit_output(args.project_root, args.output_dir, h5ad_path=args.h5ad, manifest_path=args.manifest)
    print(json.dumps({"status": result["status"], "model_fit": result["model_fit"], "orders": list(result["orders"])}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
