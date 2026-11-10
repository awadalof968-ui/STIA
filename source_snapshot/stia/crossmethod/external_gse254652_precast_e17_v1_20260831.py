"""Source-bound GSE254652 PRECAST route (E17).

PRECAST is invoked through the installed R 4.5.3/PRECAST 1.9 stack recorded
by gap4, with ``LOCAL_RUNTIME_ROOT/x3_rlib45`` prepended to ``.libPaths``.  The Python
adapter reuses E8's sparse source/order identity and the accepted v16
donor-block/explicit ``output_row_index`` discipline.  Preflight and the
fixture smoke are fit-free with respect to the real dataset; a production fit
is impossible without the explicit model-fit flag and resource/duplicate gate.
"""
from __future__ import annotations

import argparse, datetime as _dt, hashlib, importlib.util, json, os, platform, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "precast-null-inspire-seeds-20260830"
E8_REL = Path("stia/crossmethod/external_gse254652_adapter_v1_20260831.py")
R_CORE_REL = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/launch/E17/external_gse254652_precast_e17_core_v1_20260831.R")
R_EXE = Path(r"LOCAL_RUNTIME_ROOT\R\R-4.5.3\bin\Rscript.exe")
R_LIB = Path(r"LOCAL_RUNTIME_ROOT\x3_rlib45")
PROTOCOL = "stia.external.gse254652.precast.e17.v1"
FEATURE_MODE = "all_common_19465"
N_ROWS, N_FEATURES, LATENT_DIM = 9568, 19465, 15
SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
ORDERS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
K, Q, KNN = 20, 15, 10
HEADROOM = 2 * 1024**3

if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np


class PrecastE17Error(ValueError):
    pass


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file(): raise PrecastE17Error(f"missing JSON: {p}")
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise PrecastE17Error(f"JSON object required: {p}")
    return value


def _exclusive_json(path: Path | str, value: Mapping[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("x", encoding="utf-8", newline="\n") as f: json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    except FileExistsError as exc: raise PrecastE17Error(f"refusing overwrite: {p}") from exc


def _load_e8(root: Path = ROOT) -> Any:
    path = root / E8_REL; spec = importlib.util.spec_from_file_location("e17_precast_e8_adapter", path)
    if spec is None or spec.loader is None: raise PrecastE17Error(f"cannot import E8 adapter: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def source_closure(root: Path | str = ROOT) -> dict[str, dict[str, str]]:
    root = Path(root).resolve(); plan = RUN_ROOT / "PLAN.md"
    paths = {"e17_runner": Path(__file__).resolve(), "e8_adapter": root / E8_REL, "r_core": root / R_CORE_REL,
             "precast_gap4_core": root / "scripts/_x3_stageC_precast_core_v16_20260826.R", "precast_gap4_artifact": root / "scripts/_x3_precast_osi_v16_gap_artifact_20260829.py", "plan": plan}
    out = {}
    for name, path in paths.items():
        if not path.is_file(): raise PrecastE17Error(f"source closure member missing: {path}")
        out[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return out


def _r_packages() -> dict[str, Any]:
    if not R_EXE.is_file(): return {"status": "NOT_READY", "blocker": f"Rscript missing: {R_EXE}"}
    if not R_LIB.is_dir(): return {"status": "NOT_READY", "blocker": f"recorded R library missing: {R_LIB}"}
    expr = ".libPaths(c('LOCAL_RUNTIME_ROOT/x3_rlib45',.libPaths())); cat(R.version.string,'\\n'); for(p in c('Matrix','Seurat','Biobase','PRECAST','reticulate')) {cat(p,as.character(packageVersion(p)),find.package(p),sep='\\t'); cat('\\n')}"
    try:
        proc = subprocess.run([str(R_EXE), "--vanilla", "-e", expr], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env={**os.environ, "R_LIBS_USER": str(R_LIB), "RETICULATE_PYTHON": str(Path(sys.executable).resolve())})
    except Exception as exc: return {"status": "NOT_READY", "blocker": f"R dependency probe failed: {type(exc).__name__}: {exc}"}
    if proc.returncode != 0: return {"status": "NOT_READY", "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:], "blocker": "recorded R library/package probe failed"}
    packages = {}; r_version = None
    for line in proc.stdout.splitlines():
        if line.startswith("R version"): r_version = line
        elif "\t" in line:
            p, version, path = line.split("\t", 2); packages[p] = {"version": version, "path": path}
    missing = [p for p in ("Matrix", "Seurat", "Biobase", "PRECAST", "reticulate") if p not in packages]
    return {"status": "PASS" if not missing else "NOT_READY", "r_version": r_version, "library_paths": [str(R_LIB), r"LOCAL_RUNTIME_ROOT\R\R-4.5.3\library"], "packages": packages, "missing": missing, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:], "blocker": "missing recorded PRECAST dependency" if missing else None}


def resource_estimate(n_rows: int = N_ROWS, n_features: int = N_FEATURES, available_bytes: int | None = None) -> dict[str, Any]:
    dense = int(n_rows * n_features * 4); r_copy = dense * 3; known = dense + r_copy + int(n_rows * 64 * 2); estimated = int(known * 1.25)
    if available_bytes is None:
        try:
            import psutil; available_bytes = int(psutil.virtual_memory().available)
        except Exception: available_bytes = None
    required = estimated + HEADROOM
    return {"dense_counts_bytes": dense, "known_host_bytes": known, "estimated_peak_bytes": estimated, "required_free_bytes": required, "headroom_bytes": HEADROOM, "available_bytes": available_bytes, "gate_pass": bool(available_bytes is not None and available_bytes >= required), "fit_only": True, "device": "host/R"}


def _materialize(bundle: Any, order: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if order not in ORDERS: raise PrecastE17Error(f"unknown order: {order}")
    sections = bundle.ordered_sections(order)
    import scipy.sparse as sp
    X = sp.vstack([section.matrix for section in sections], format="csr")
    ids = np.asarray([v for section in sections for v in section.spot_ids], dtype="U")
    libs = np.asarray([section.gsm for section in sections for _ in section.spot_ids], dtype="U")
    animals = np.asarray([section.animal_id for section in sections for _ in section.spot_ids], dtype="U")
    spatial = np.asarray([[section.coordinates["array_row"][i], section.coordinates["array_col"][i]] for section in sections for i in range(section.n_rows)], dtype=np.float32)
    donor_code = np.repeat(np.arange(len(sections), dtype=np.int32), [section.n_rows for section in sections])
    if X.shape != (N_ROWS, N_FEATURES) or ids.size != N_ROWS or len(set(ids.tolist())) != N_ROWS or not np.isfinite(X.data).all() or (X.data < 0).any(): raise PrecastE17Error(f"{order}: sparse source/order contract failed")
    return X, ids, libs, animals, spatial, donor_code


def _tiny_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # PRECAST's native fixed-number graph requires >4*k spots per section;
    # 50 spots/animal keeps this fixture tiny while exercising the real graph,
    # Seurat handoff, K/q fit, and explicit row-map output.
    n_per = 50; rng = np.random.default_rng(41); X = rng.poisson(50, size=(7 * n_per, 24)).astype(np.float32); X[X < 1] = 1
    libs = np.asarray([f"tiny-lib-{i}" for i in range(7) for _ in range(n_per)]); ids = np.asarray([f"tiny::{i}" for i in range(7 * n_per)]); animals = np.asarray([str(i) for i in range(7) for _ in range(n_per)]); spatial = np.asarray([[float(i), float(j)] for i in range(7) for j in range(n_per)], dtype=np.float32); code = np.repeat(np.arange(7, dtype=np.int32), n_per)
    return X, ids, libs, animals, spatial, code


def _invoke_r(input_npz: Path, output_npz: Path, seed: int, *, k: int = K, q: int = Q, timeout: int = 300) -> dict[str, Any]:
    if not R_EXE.is_file(): raise PrecastE17Error(f"Rscript missing: {R_EXE}")
    if not R_LIB.is_dir(): raise PrecastE17Error(f"recorded R library missing: {R_LIB}")
    env = dict(os.environ, R_LIBS_USER=str(R_LIB), RETICULATE_PYTHON=str(Path(sys.executable).resolve()), TMP=str(Path(tempfile.gettempdir())), TEMP=str(Path(tempfile.gettempdir()))); env["OMP_NUM_THREADS"] = "1"; env["OPENBLAS_NUM_THREADS"] = "1"
    proc = subprocess.run([str(R_EXE), "--vanilla", str(ROOT / R_CORE_REL), str(input_npz), str(output_npz), str(int(seed)), str(int(k)), str(int(q))], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    result = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:], "rscript": str(R_EXE), "r_library": str(R_LIB)}
    if proc.returncode != 0 or not output_npz.is_file(): raise PrecastE17Error(f"R PRECAST failed rc={proc.returncode}: {proc.stderr[-1200:]}")
    return result


def _save_input(path: Path, X: Any, spatial: np.ndarray, donor_code: np.ndarray, donor_levels: Sequence[str], feature_ids: Sequence[str]) -> None:
    import scipy.sparse as sp
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    dense = np.asarray(dense, dtype=np.float32)
    if not np.isfinite(dense).all() or (dense < 0).any() or not np.equal(dense, np.floor(dense)).all(): raise PrecastE17Error("PRECAST raw count input invalid")
    np.savez_compressed(path, X=dense, spatial=np.asarray(spatial, dtype=np.float32), donor_code=np.asarray(donor_code, dtype=np.int32), donor_levels=np.asarray(donor_levels, dtype="U"), donor_n=np.int32(len(donor_levels)), input_row_index=np.arange(dense.shape[0], dtype=np.int32), block_sizes_in=np.bincount(donor_code, minlength=len(donor_levels)).astype(np.int32), feature_ids=np.asarray(feature_ids, dtype="U"))


def native_fixture_smoke(seed: int = 20260819) -> dict[str, Any]:
    X, ids, libs, animals, spatial, donor_code = _tiny_fixture(); features = [f"ENSMUSGTINY{i:05d}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory(prefix="e17pc_") as td:
        d = Path(td); inp, out = d / "i.npz", d / "o.npz"; _save_input(inp, X, spatial, donor_code, sorted(set(animals.tolist()), key=int), features); invocation = _invoke_r(inp, out, seed, k=2, q=5, timeout=300)
        with np.load(out, allow_pickle=False) as z: latent = np.asarray(z["latent"]); row_map = np.asarray(z["output_row_index"]); in_rows = np.asarray(z["input_row_index"])
        if latent.shape != (350, 5) or not np.isfinite(latent).all() or row_map.tolist() != list(range(350)) or in_rows.tolist() != list(range(350)): raise PrecastE17Error("native PRECAST smoke output mapping/shape failed")
    return {"status": "PASS", "model_fit": "FIXTURE_ONLY", "backend": "native PRECAST R 1.9", "r_invocation": invocation, "latent_shape": list(latent.shape), "raw_mapping_roundtrip": "PASS", "production_input_opened": False, "seed": int(seed), "parameters": {"K": K, "q": Q, "knn_neigh": KNN}}


def preflight(project_root: Path | str = ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve(); e8 = _load_e8(root); deps = _r_packages()
    with e8.open_bundle(root, feature_mode=FEATURE_MODE) as bundle:
        orders = {}
        for order in ORDERS:
            _x, ids, libs, animals, spatial, code = _materialize(bundle, order); orders[order] = {"n_rows": int(ids.size), "ordered_spot_ids_sha256": canonical_sha256(ids.tolist()), "library_ids_sha256": canonical_sha256(libs.tolist()), "animal_ids_sha256": canonical_sha256(animals.tolist()), "spatial_sha256": hashlib.sha256(spatial.tobytes()).hexdigest(), "order_sha256": bundle.orders[order]["sha256"]}
        status = "READY_NO_FIT" if deps.get("status") == "PASS" else "NOT_READY_DEPENDENCY"
        return {"schema_version": PROTOCOL, "status": status, "model_fit": "NOT_PERFORMED", "dataset": "GSE254652_P1", "dataset_scope": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION_CANDIDATE", "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)), "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR"}, "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256}, "orders": orders, "seeds": list(SEEDS), "source_closure": source_closure(root), "dependencies": deps, "resource_qc": resource_estimate(), "labels": {"disease": "NOT_CREATED", "physical_batch": "UNKNOWN_NOT_CREATED"}, "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN", "model_fit": "NOT_PERFORMED"}, "blocker": deps.get("blocker"), "next_action": "run one printed fit command after fresh resource/process/dedup gate"}


def _write_arm(path: Path, latent: np.ndarray, ids: np.ndarray, libs: np.ndarray, animals: np.ndarray, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".json").exists(): raise FileExistsError(f"refusing arm overwrite: {path}")
    with path.open("xb") as f: np.savez_compressed(f, latent=np.asarray(latent, dtype=np.float32), spot_ids=ids, library_ids=libs, animal_ids=animals)
    rec = dict(metadata); rec.update({"npz": str(path.resolve()), "npz_sha256": sha256_file(path), "latent_sha256": hashlib.sha256(np.asarray(latent, dtype=np.float32).tobytes()).hexdigest(), "latent_shape": list(latent.shape), "raw_keys": ["latent", "spot_ids", "library_ids", "animal_ids"]}); _exclusive_json(path.with_suffix(".json"), rec); return rec


def fit_bundle(*, project_root: Path | str = ROOT, output_dir: Path | str, seeds: Sequence[int] = SEEDS, allow_model_fits: bool = False) -> dict[str, Any]:
    if not allow_model_fits: raise RuntimeError("REFUSED: PRECAST E17 fit requires --allow-model-fits; no model fit was started")
    dest = Path(output_dir).resolve()
    if dest.exists(): raise FileExistsError(f"refusing output collision: {dest}")
    selected = tuple(int(v) for v in seeds)
    if not selected or len(set(selected)) != len(selected) or any(v not in SEEDS for v in selected): raise PrecastE17Error("seeds must be a unique nonempty frozen subset")
    deps = _r_packages()
    if deps.get("status") != "PASS": raise RuntimeError(f"REFUSED: recorded PRECAST dependency gate failed: {deps.get('blocker')}")
    budget = resource_estimate()
    if not budget["gate_pass"]: raise RuntimeError("REFUSED: PRECAST E17 host-memory gate failed")
    root = Path(project_root).resolve(); e8 = _load_e8(root); dest.mkdir(parents=True)
    with e8.open_bundle(root, feature_mode=FEATURE_MODE) as bundle:
        manifest = {"schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "STARTED", "dataset": "GSE254652_P1", "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)), "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR"}, "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256}, "orders": list(ORDERS), "seeds": list(selected), "parameters": {"K": K, "q": Q, "knn_neigh": KNN, "coreNum": 1, "coreNum_int": 1, "beta_grid": 1, "maxIter_ICM": 6, "maxIter": 30, "feature_mode": FEATURE_MODE, "preprocessing": "native PRECAST/Seurat raw-count normalization", "row_map": "explicit output_row_index"}, "labels": {"disease": "NOT_CREATED", "physical_batch": "UNKNOWN_NOT_CREATED"}, "source_closure": source_closure(root), "dependencies": deps, "resource_qc": budget, "output_dir": str(dest), "started_at": _dt.datetime.now().astimezone().isoformat()}; manifest["config_sha256"] = canonical_sha256(manifest["parameters"]); _exclusive_json(dest / "manifest.json", manifest)
        records, errors = {}, []
        with tempfile.TemporaryDirectory(prefix="e17pc_") as td:
            for seed in selected:
                for order in ORDERS:
                    try:
                        X, ids, libs, animals, spatial, code = _materialize(bundle, order); inp, out = Path(td) / f"i_{seed}_{order}.npz", Path(td) / f"o_{seed}_{order}.npz"; _save_input(inp, X, spatial, code, [str(v) for v in bundle.orders[order]["animals"]], bundle.feature_ids); invocation = _invoke_r(inp, out, seed)
                        with np.load(out, allow_pickle=False) as z: latent = np.asarray(z["latent"]); row_map = np.asarray(z["output_row_index"])
                        if row_map.tolist() != list(range(N_ROWS)) or latent.shape != (N_ROWS, LATENT_DIM) or not np.isfinite(latent).all(): raise PrecastE17Error("R output mapping/latent contract failed")
                        records[f"{seed}:{order}"] = _write_arm(dest / "arms" / f"seed{seed}" / f"{order}.npz", latent, ids, libs, animals, {"schema_version": PROTOCOL, "status": "OK", "seed": seed, "order": order, "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "feature_ids_sha256": manifest["feature_ids_sha256"], "source_closure": manifest["source_closure"], "r_invocation": invocation, "parameters": manifest["parameters"]})
                    except Exception as exc: errors.append(f"{seed}:{order}:{type(exc).__name__}: {exc}")
        summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE" if not errors else "FIT_PARTIAL", "model_fit": "PERFORMED", "requested_seeds": list(selected), "expected_orders": list(ORDERS), "completed_arms": len(records), "expected_arms": len(selected) * len(ORDERS), "errors": errors, "arms": records, "completed_at": _dt.datetime.now().astimezone().isoformat()}; _exclusive_json(dest / "run_summary.json", summary)
        if not errors: _exclusive_json(dest / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(dest / "run_summary.json")})
        return summary


def audit_output(*, output_dir: Path | str, project_root: Path | str = ROOT) -> dict[str, Any]:
    dest = Path(output_dir).resolve(); manifest = _json(dest / "manifest.json"); summary = _json(dest / "run_summary.json")
    if manifest.get("schema_version") != PROTOCOL or summary.get("status") != "FIT_COMPLETE": raise PrecastE17Error("terminal PRECAST schema/status mismatch")
    if manifest.get("source_closure") != source_closure(Path(project_root).resolve()): raise PrecastE17Error("source closure drift")
    e8 = _load_e8(Path(project_root).resolve()); arms, errors = [], []
    with e8.open_bundle(Path(project_root).resolve(), feature_mode=FEATURE_MODE) as bundle:
        for seed in manifest.get("seeds", []):
            for order in ORDERS:
                try:
                    p = dest / "arms" / f"seed{int(seed)}" / f"{order}.npz"; meta = _json(p.with_suffix(".json"));
                    if meta.get("npz_sha256") != sha256_file(p) or meta.get("seed") != int(seed) or meta.get("order") != order: raise PrecastE17Error("arm sidecar binding mismatch")
                    with np.load(p, allow_pickle=False) as z: latent, ids, libs, animals = z["latent"], z["spot_ids"].astype(str), z["library_ids"].astype(str), z["animal_ids"].astype(str)
                    _, exp_ids, exp_libs, exp_animals, _, _ = _materialize(bundle, order)
                    if latent.shape != (N_ROWS, LATENT_DIM) or ids.tolist() != exp_ids.tolist() or libs.tolist() != exp_libs.tolist() or animals.tolist() != exp_animals.tolist() or not np.isfinite(latent).all(): raise PrecastE17Error("raw arm identity/latent mismatch")
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
    except Exception as exc: print(f"E17_PRECAST_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
