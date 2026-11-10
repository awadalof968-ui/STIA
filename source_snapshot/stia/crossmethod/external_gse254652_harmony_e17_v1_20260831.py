"""Source-bound GSE254652 Harmony route (E17).

The route is deliberately additive: it consumes the accepted sparse E8 bundle,
uses library identity only as an explicitly non-causal Harmony proxy, and never
creates disease or physical-batch labels.  ``preflight`` and ``native-smoke``
are no-fit paths; production fitting requires ``--allow-model-fits`` and an
absent output directory.  The H4 runtime repair is reused verbatim so the
known Windows MKL/Lloyd probe failure is handled without changing Harmony's
theta/sigma/iteration/seed contract.
"""
from __future__ import annotations

import argparse, datetime as _dt, hashlib, importlib.util, json, os, platform, random, sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "precast-null-inspire-seeds-20260830"
E8_REL = Path("stia/crossmethod/external_gse254652_adapter_v1_20260831.py")
H4_REL = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/repair/H4_harmony_strict_20260830.py")
H4_AUTH_REL = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/phases/terminal-audit/H4.md")
METRIC_REL = Path("stia/crossmethod/osi.py")
PLAN_REL = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/PLAN.md")
PROTOCOL = "stia.external.gse254652.harmony.e17.v1"
FEATURE_MODE = "all_common_19465"
N_ROWS, N_FEATURES, PCA_DIM = 9568, 19465, 30
SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
ORDERS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
THETA, SIGMA, HARMONY_ITERS = 2.0, 0.1, 20
MARGIN = 0.05
HEADROOM = 2 * 1024**3

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np


class HarmonyE17Error(ValueError):
    pass


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise HarmonyE17Error(f"missing JSON: {p}")
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HarmonyE17Error(f"JSON object required: {p}")
    return value


def _exclusive_json(path: Path | str, value: Mapping[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    except FileExistsError as exc:
        raise HarmonyE17Error(f"refusing overwrite: {p}") from exc


def _load_e8(root: Path = ROOT) -> Any:
    path = root / E8_REL
    spec = importlib.util.spec_from_file_location("e17_e8_adapter", path)
    if spec is None or spec.loader is None:
        raise HarmonyE17Error(f"cannot import E8 adapter: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def _load_h4(root: Path = ROOT) -> Any:
    path = root / H4_REL
    spec = importlib.util.spec_from_file_location("e17_h4_repair", path)
    if spec is None or spec.loader is None:
        raise HarmonyE17Error(f"cannot import H4 repair: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def source_closure(root: Path | str = ROOT) -> dict[str, dict[str, str]]:
    root = Path(root).resolve()
    paths = {"e17_runner": Path(__file__).resolve(), "e8_adapter": root / E8_REL,
             "h4_repair": root / H4_REL, "h4_authority": root / H4_AUTH_REL,
             "osi_metric": root / METRIC_REL, "plan": root / PLAN_REL}
    out = {}
    for name, path in paths.items():
        if not path.is_file():
            raise HarmonyE17Error(f"source closure member missing: {path}")
        out[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return out


def dependency_snapshot() -> dict[str, Any]:
    """Capture real installed Harmony dependencies and baseline probe state."""
    import importlib.metadata, harmonypy, harmonypy.harmony as harmony, sklearn, threadpoolctl
    try:
        probe = {"status": "PASS", "info": threadpoolctl.ThreadpoolController().info()}
    except Exception as exc:
        probe = {"status": "ERROR", "exception_type": type(exc).__name__, "message": str(exc), "known_h4_probe_error": "0xc06d007f" in str(exc).lower()}
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "harmonypy_version": importlib.metadata.version("harmonypy"), "sklearn_version": sklearn.__version__,
            "threadpoolctl_version": threadpoolctl.__version__, "harmonypy_path": str(Path(harmonypy.__file__).resolve()),
            "harmony_core_path": str(Path(harmony.__file__).resolve()), "threadpool_probe_before_patch": probe,
            "backend": "harmonypy.run_harmony; H4 exact WinError guard + sklearn Elkan fallback"}


def resource_estimate(n_rows: int = N_ROWS, n_features: int = N_FEATURES, available_bytes: int | None = None) -> dict[str, Any]:
    # Fit-only dense log1p-CPM + PCA/Harmony workspaces; preflight never allocates these.
    dense = int(n_rows * n_features * 4)
    pca = int(n_rows * PCA_DIM * 4)
    estimated = int(dense * 2 + pca * 8 + 256 * 1024**2)
    if available_bytes is None:
        try:
            import psutil; available_bytes = int(psutil.virtual_memory().available)
        except Exception: available_bytes = None
    return {"dense_input_bytes": dense, "pca_workspace_bytes": pca * 8, "estimated_peak_bytes": estimated,
            "required_free_bytes": estimated + HEADROOM, "available_bytes": available_bytes,
            "headroom_bytes": HEADROOM, "fit_only": True,
            "gate_pass": bool(available_bytes is not None and available_bytes >= estimated + HEADROOM)}


def _seed_all(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1))


def _materialize(bundle: Any, order: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if order not in ORDERS: raise HarmonyE17Error(f"unknown order: {order}")
    sections = bundle.ordered_sections(order)
    import scipy.sparse as sp
    blocks = [section.matrix for section in sections]
    X = sp.vstack(blocks, format="csr")
    ids = np.asarray([v for section in sections for v in section.spot_ids], dtype="U")
    libs = np.asarray([section.gsm for section in sections for _ in section.spot_ids], dtype="U")
    animals = np.asarray([section.animal_id for section in sections for _ in section.spot_ids], dtype="U")
    if X.shape != (N_ROWS, N_FEATURES) or ids.size != N_ROWS or len(set(ids.tolist())) != N_ROWS:
        raise HarmonyE17Error(f"{order}: sparse source/order shape or ID contract failed")
    return X, ids, libs, animals, np.asarray([section.gsm for section in sections], dtype="U")


def _log1p_cpm_dense(X: Any) -> np.ndarray:
    import scipy.sparse as sp
    if not sp.issparse(X): raise HarmonyE17Error("fit input must remain sparse until explicit fit conversion")
    X = X.tocsr(copy=True).astype(np.float32)
    totals = np.asarray(X.sum(axis=1)).ravel(); scale = np.divide(1e4, np.maximum(totals, 1e-9))
    X = X.multiply(scale[:, None]); X.data = np.log1p(X.data); X.data = np.clip(X.data, 0, 50)
    dense = np.asarray(X.toarray(), dtype=np.float32)
    if dense.shape != (X.shape[0], X.shape[1]) or not np.isfinite(dense).all(): raise HarmonyE17Error("normalization produced invalid dense matrix")
    return dense


def _fit_one(X: Any, libraries: Sequence[str], seed: int, *, h4: Any | None = None) -> np.ndarray:
    _seed_all(seed)
    h4 = h4 or _load_h4()
    h4.apply_runtime_repair()
    import pandas as pd
    from harmonypy import run_harmony
    matrix = _log1p_cpm_dense(X)
    if matrix.shape[0] < PCA_DIM + 1: dim = min(PCA_DIM, matrix.shape[0] - 1, matrix.shape[1])
    else: dim = PCA_DIM
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    z0 = (centered @ vt[:dim].T).T
    meta = pd.DataFrame({"library_id_proxy": np.asarray(libraries, dtype=str)})
    # harmonypy 0.0.9 has a scalar-sigma indexing defect when its automatic
    # ``nclust`` rounds to one (tiny smoke fixtures); production uses the
    # normal 100-cluster cap.  Explicitly enforce the same lower bound only.
    nclust = min(100, max(2, int(round(z0.shape[1] / 30.0))))
    result = run_harmony(z0, meta, vars_use="library_id_proxy", theta=THETA, sigma=SIGMA,
                         nclust=nclust, max_iter_harmony=HARMONY_ITERS, max_iter_kmeans=20,
                         random_state=int(seed), verbose=False)
    latent = np.asarray(result.Z_corr.T, dtype=np.float32)
    if latent.shape != (matrix.shape[0], dim) or not np.isfinite(latent).all(): raise HarmonyE17Error(f"Harmony latent shape/finite failure: {latent.shape}")
    return latent


def _tiny_bundle() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    import scipy.sparse as sp
    rng = np.random.default_rng(17); counts = sp.csr_matrix(rng.poisson(2, size=(28, 12)).astype(np.float32))
    libs = np.asarray(["GSM8047868"] * 4 + ["GSM8047869"] * 4 + ["GSM8047870"] * 4 + ["GSM8047879"] * 4 + ["GSM8047880"] * 4 + ["GSM8047881"] * 4 + ["GSM8047882"] * 4)
    ids = np.asarray([f"tiny::{i}" for i in range(28)]); animals = np.asarray([str(i) for i in range(7) for _ in range(4)])
    return counts, ids, libs, animals


def native_fixture_smoke(seed: int = 20260819) -> dict[str, Any]:
    X, ids, libs, animals = _tiny_bundle(); z = _fit_one(X, libs, seed)
    return {"status": "PASS", "model_fit": "FIXTURE_ONLY", "backend": "harmonypy", "solver_fallback": "H4 Elkan",
            "latent_shape": list(z.shape), "raw_ids_roundtrip": bool(len(set(ids.tolist())) == ids.size),
            "library_proxy": "library_id only; non-causal proxy, physical batch unknown", "external_input_opened": False,
            "seed": int(seed), "parameters": {"theta": THETA, "sigma": SIGMA, "max_iter_harmony": HARMONY_ITERS}}


def preflight(project_root: Path | str = ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve(); e8 = _load_e8(root)
    with e8.open_bundle(root, feature_mode=FEATURE_MODE) as bundle:
        # Derive order identities without materializing a dense matrix.
        order_receipts = {}
        for order in ORDERS:
            _x, ids, libs, animals, sections = _materialize(bundle, order)
            order_receipts[order] = {"n_rows": int(ids.size), "ordered_spot_ids_sha256": canonical_sha256(ids.tolist()),
                                     "library_ids_sha256": canonical_sha256(libs.tolist()), "animal_ids_sha256": canonical_sha256(animals.tolist()),
                                     "order_sha256": bundle.orders[order]["sha256"]}
        deps = dependency_snapshot()
        return {"schema_version": PROTOCOL, "status": "READY_NO_FIT", "model_fit": "NOT_PERFORMED",
                "dataset": "GSE254652_P1", "dataset_scope": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION_CANDIDATE",
                "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)),
                "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR"},
                "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256}, "orders": order_receipts,
                "seeds": list(SEEDS), "source_closure": source_closure(root), "dependencies": deps,
                "resource_qc": resource_estimate(), "labels": {"disease": "NOT_CREATED", "physical_batch": "UNKNOWN_NOT_CREATED", "harmony_proxy": "library_id"},
                "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN", "model_fit": "NOT_PERFORMED"},
                "next_action": "after fresh resource/process/dedup gate run one printed fit command; no manual source edits"}


def _write_arm(path: Path, latent: np.ndarray, ids: np.ndarray, libs: np.ndarray, animals: np.ndarray, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".json").exists(): raise FileExistsError(f"refusing arm overwrite: {path}")
    with path.open("xb") as f: np.savez_compressed(f, latent=latent, spot_ids=ids, library_ids=libs, animal_ids=animals)
    rec = dict(metadata); rec.update({"npz": str(path.resolve()), "npz_sha256": sha256_file(path), "latent_sha256": hashlib.sha256(np.asarray(latent).tobytes()).hexdigest(), "latent_shape": list(latent.shape), "raw_keys": ["latent", "spot_ids", "library_ids", "animal_ids"]})
    _exclusive_json(path.with_suffix(".json"), rec); return rec


def fit_bundle(*, project_root: Path | str = ROOT, output_dir: Path | str, seeds: Sequence[int] = SEEDS, allow_model_fits: bool = False) -> dict[str, Any]:
    if not allow_model_fits: raise RuntimeError("REFUSED: Harmony E17 fit requires --allow-model-fits; no model fit was started")
    destination = Path(output_dir).resolve()
    if destination.exists(): raise FileExistsError(f"refusing output collision: {destination}")
    root = Path(project_root).resolve(); selected = tuple(int(v) for v in seeds)
    if not selected or len(set(selected)) != len(selected) or any(v not in SEEDS for v in selected): raise HarmonyE17Error("seeds must be a unique nonempty frozen subset")
    budget = resource_estimate()
    if not budget["gate_pass"]: raise RuntimeError("REFUSED: Harmony E17 resource/headroom gate failed")
    e8 = _load_e8(root); h4 = _load_h4(root); deps = dependency_snapshot()
    destination.mkdir(parents=True)
    with e8.open_bundle(root, feature_mode=FEATURE_MODE) as bundle:
        manifest = {"schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "STARTED", "dataset": "GSE254652_P1", "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)), "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR"}, "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256}, "orders": list(ORDERS), "seeds": list(selected), "parameters": {"pca_dim": PCA_DIM, "theta": THETA, "sigma": SIGMA, "max_iter_harmony": HARMONY_ITERS, "batch_proxy": "library_id", "batch_proxy_causal": False}, "labels": {"disease": "NOT_CREATED", "physical_batch": "UNKNOWN_NOT_CREATED"}, "source_closure": source_closure(root), "dependencies": deps, "resource_qc": budget, "output_dir": str(destination), "started_at": _dt.datetime.now().astimezone().isoformat()}
        manifest["config_sha256"] = canonical_sha256({k: manifest[k] for k in ("parameters", "feature_mode", "feature_count", "orders", "seeds")}); _exclusive_json(destination / "manifest.json", manifest)
        records = {}
        for seed in selected:
            for order in ORDERS:
                X, ids, libs, animals, _ = _materialize(bundle, order); latent = _fit_one(X, libs, seed, h4=h4)
                records[f"{seed}:{order}"] = _write_arm(destination / "arms" / f"seed{seed}" / f"{order}.npz", latent, ids, libs, animals, {"schema_version": PROTOCOL, "status": "OK", "seed": seed, "order": order, "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "feature_ids_sha256": manifest["feature_ids_sha256"], "source_closure": manifest["source_closure"], "batch_proxy": "library_id", "batch_proxy_causal": False, "harmonypy_random_state": seed, "parameters": manifest["parameters"]})
        summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "requested_seeds": list(selected), "expected_orders": list(ORDERS), "completed_arms": len(records), "arms": records, "claim_boundary": manifest["labels"], "completed_at": _dt.datetime.now().astimezone().isoformat()}; _exclusive_json(destination / "run_summary.json", summary); _exclusive_json(destination / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(destination / "run_summary.json")}); return summary


def audit_output(*, output_dir: Path | str, project_root: Path | str = ROOT) -> dict[str, Any]:
    dest = Path(output_dir).resolve(); manifest = _json(dest / "manifest.json"); summary = _json(dest / "run_summary.json")
    if manifest.get("schema_version") != PROTOCOL or summary.get("status") != "FIT_COMPLETE": raise HarmonyE17Error("terminal schema/status mismatch")
    if manifest.get("source_closure") != source_closure(Path(project_root).resolve()): raise HarmonyE17Error("source closure drift")
    e8 = _load_e8(Path(project_root).resolve()); arms = []; errors = []
    with e8.open_bundle(Path(project_root).resolve(), feature_mode=FEATURE_MODE) as bundle:
        for seed in manifest.get("seeds", []):
            for order in ORDERS:
                try:
                    p = dest / "arms" / f"seed{int(seed)}" / f"{order}.npz"; meta = _json(p.with_suffix(".json"))
                    if meta.get("npz_sha256") != sha256_file(p) or meta.get("seed") != int(seed) or meta.get("order") != order: raise HarmonyE17Error("arm sidecar binding mismatch")
                    with np.load(p, allow_pickle=False) as z:
                        latent, ids, libs, animals = z["latent"], z["spot_ids"].astype(str), z["library_ids"].astype(str), z["animal_ids"].astype(str)
                    _, exp_ids, exp_libs, exp_animals, _ = _materialize(bundle, order)
                    if latent.shape != (N_ROWS, PCA_DIM) or ids.tolist() != exp_ids.tolist() or libs.tolist() != exp_libs.tolist() or animals.tolist() != exp_animals.tolist() or not np.isfinite(latent).all(): raise HarmonyE17Error("raw arm identity/latent mismatch")
                    arms.append({"seed": int(seed), "order": order, "execution_status": "PASS", "latent_shape": list(latent.shape), "npz_sha256": sha256_file(p)})
                except Exception as exc: errors.append(f"{seed}:{order}:{exc}")
    return {"schema_version": PROTOCOL, "status": "PASS" if not errors else "UNRESOLVED", "model_fit": "AUDITED_ONLY", "execution_validity": "PASS" if not errors else "UNRESOLVED", "n_arms": len(arms), "arms": arms, "errors": errors, "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--mode", choices=("preflight", "native-smoke", "fit", "audit"), default="preflight"); ap.add_argument("--project-root", type=Path, default=ROOT); ap.add_argument("--output-dir", type=Path); ap.add_argument("--allow-model-fits", action="store_true"); ap.add_argument("--evidence", type=Path); ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS)); args = ap.parse_args(argv)
    try:
        if args.mode == "preflight": result = preflight(args.project_root)
        elif args.mode == "native-smoke": result = native_fixture_smoke()
        elif args.mode == "fit":
            if args.output_dir is None: ap.error("--output-dir required for fit")
            result = fit_bundle(project_root=args.project_root, output_dir=args.output_dir, seeds=args.seeds, allow_model_fits=args.allow_model_fits)
        else:
            if args.output_dir is None: ap.error("--output-dir required for audit")
            result = audit_output(output_dir=args.output_dir, project_root=args.project_root)
        if args.evidence: _exclusive_json(args.evidence, result)
        print(json.dumps({"status": result.get("status"), "model_fit": result.get("model_fit"), "mode": args.mode}, sort_keys=True)); return 0 if result.get("status") in {"PASS", "READY_NO_FIT", "FIT_COMPLETE"} else 1
    except Exception as exc:
        print(f"E17_HARMONY_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
