"""E17T additive PRECAST production-control repair.

This successor deliberately leaves E17 frozen.  It gives real production R
calls a 7,200-second recorded timeout (the tiny fixture keeps a 300-second
bound), persists every R input/output/stdout/stderr file under the versioned
output namespace, and fail-stops before the next arm after the first error.
"""
from __future__ import annotations

import argparse, datetime as _dt, hashlib, importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "precast-null-inspire-seeds-20260830"
E17_REL = Path("stia/crossmethod/external_gse254652_precast_e17_v1_20260831.py")
R_CORE_REL = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/launch/E17/external_gse254652_precast_e17_core_v1_20260831.R")
PROTOCOL = "stia.external.gse254652.precast.e17r2.v1"
SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
ORDERS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
N_ROWS, N_FEATURES, LATENT_DIM = 9568, 19465, 15
K, Q, KNN = 20, 15, 10
TINY_TIMEOUT_SECONDS, PRODUCTION_TIMEOUT_SECONDS = 300, 7200

if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np


class PrecastE17R2Error(ValueError):
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
    if not p.is_file(): raise PrecastE17R2Error(f"missing JSON: {p}")
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise PrecastE17R2Error(f"JSON object required: {p}")
    return value


def _exclusive_json(path: Path | str, value: Mapping[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("x", encoding="utf-8", newline="\n") as f: json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    except FileExistsError as exc: raise PrecastE17R2Error(f"refusing overwrite: {p}") from exc


def _load_e17(root: Path = ROOT) -> Any:
    path = root / E17_REL; spec = importlib.util.spec_from_file_location("e17_precast_frozen", path)
    if spec is None or spec.loader is None: raise PrecastE17R2Error(f"cannot import E17 route: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def source_closure(root: Path | str = ROOT) -> dict[str, dict[str, str]]:
    root = Path(root).resolve(); paths = {"e17r2_runner": Path(__file__).resolve(), "e17_route": root / E17_REL, "r_core": root / R_CORE_REL, "e8_adapter": root / "stia/crossmethod/external_gse254652_adapter_v1_20260831.py", "precast_gap4_core": root / "scripts/_x3_stageC_precast_core_v16_20260826.R", "precast_gap4_artifact": root / "scripts/_x3_precast_osi_v16_gap_artifact_20260829.py", "plan": RUN_ROOT / "PLAN.md"}
    out = {}
    for name, path in paths.items():
        if not path.is_file(): raise PrecastE17R2Error(f"source closure member missing: {path}")
        out[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return out


def _r_packages() -> dict[str, Any]:
    """Use the exact gap4 Rscript/library probe; do not use default requireNamespace."""
    e17 = _load_e17(); return e17._r_packages()


def resource_estimate() -> dict[str, Any]:
    return _load_e17().resource_estimate()


def _materialize(bundle: Any, order: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_e17()._materialize(bundle, order)


def _save_input(path: Path, X: Any, spatial: np.ndarray, donor_code: np.ndarray, donor_levels: Sequence[str], feature_ids: Sequence[str]) -> None:
    _load_e17()._save_input(path, X, spatial, donor_code, donor_levels, feature_ids)


def invoke_r(input_npz: Path, output_npz: Path, seed: int, *, k: int = K, q: int = Q, timeout_seconds: int | None = None, evidence_dir: Path | None = None) -> dict[str, Any]:
    """Run R once, preserving complete subprocess evidence and timeout status."""
    e17 = _load_e17(); timeout = PRODUCTION_TIMEOUT_SECONDS if timeout_seconds is None else int(timeout_seconds)
    if timeout <= 0: raise PrecastE17R2Error("timeout_seconds must be positive")
    if not e17.R_EXE.is_file(): raise PrecastE17R2Error(f"Rscript missing: {e17.R_EXE}")
    if not e17.R_LIB.is_dir(): raise PrecastE17R2Error(f"recorded R library missing: {e17.R_LIB}")
    env = dict(os.environ, R_LIBS_USER=str(e17.R_LIB), RETICULATE_PYTHON=str(Path(sys.executable).resolve()), TMP=str(Path(tempfile.gettempdir())), TEMP=str(Path(tempfile.gettempdir())), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [str(e17.R_EXE), "--vanilla", str(ROOT / R_CORE_REL), str(input_npz), str(output_npz), str(int(seed)), str(int(k)), str(int(q))]
    status, returncode, stdout, stderr = "COMPLETED", None, "", ""
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
        returncode, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
        if returncode != 0 or not output_npz.is_file(): status = "FAILED"
    except subprocess.TimeoutExpired as exc:
        status = "TIMEOUT"; returncode = None
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    record = {"status": status, "returncode": returncode, "timeout_seconds": timeout, "command": command, "rscript": str(e17.R_EXE), "r_library": str(e17.R_LIB), "stdout_bytes": len(stdout.encode()), "stderr_bytes": len(stderr.encode())}
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (evidence_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        record.update({"input_npz": str(input_npz.resolve()), "output_npz": str(output_npz.resolve()), "stdout": str((evidence_dir / "stdout.log").resolve()), "stderr": str((evidence_dir / "stderr.log").resolve())})
    if status != "COMPLETED":
        failure = PrecastE17R2Error(f"R PRECAST {status.lower()} rc={returncode}; evidence={evidence_dir}")
        setattr(failure, "invocation", record)
        raise failure
    return record


def _write_failure(dest: Path, manifest: Mapping[str, Any], *, seed: int, order: str, error: BaseException, invocation: Mapping[str, Any] | None, completed: Sequence[str]) -> None:
    failure = {"schema_version": PROTOCOL, "status": "FIT_FAILED", "failed_seed": int(seed), "failed_order": order, "error_type": type(error).__name__, "error": str(error), "invocation": dict(invocation or {}), "completed_arms": list(completed), "next_arm_started": False, "manifest_sha256": sha256_file(dest / "manifest.json"), "recorded_at": _dt.datetime.now().astimezone().isoformat()}
    _exclusive_json(dest / "run_failure.json", failure)
    _exclusive_json(dest / "run_summary.json", {"schema_version": PROTOCOL, "status": "FIT_FAILED", "model_fit": "PARTIAL_PRESERVED", "completed_arms": len(completed), "failed_seed": int(seed), "failed_order": order, "next_arm_started": False, "failure": failure})


def guarded_arm_sequence(*, destination: Path, bundle: Any, selected: Sequence[int], manifest: Mapping[str, Any], invoke: Callable[..., Mapping[str, Any]], e17: Any) -> dict[str, Any]:
    """Persist one arm's R evidence and stop immediately on first failure."""
    records: dict[str, Any] = {}; completed: list[str] = []
    for seed in selected:
        for order in ORDERS:
            key = f"{int(seed)}:{order}"; arm_evidence = destination / "r_evidence" / f"seed{int(seed)}" / order; inp = arm_evidence / "input.npz"; out = arm_evidence / "output.npz"
            X, ids, libs, animals, spatial, code = _materialize(bundle, order)
            _save_input(inp, X, spatial, code, [str(v) for v in bundle.orders[order]["animals"]], bundle.feature_ids)
            invocation: Mapping[str, Any] | None = None
            try:
                invocation = invoke(inp, out, int(seed), k=K, q=Q, timeout_seconds=PRODUCTION_TIMEOUT_SECONDS, evidence_dir=arm_evidence)
                with np.load(out, allow_pickle=False) as z: latent = np.asarray(z["latent"]); row_map = np.asarray(z["output_row_index"])
                if row_map.tolist() != list(range(N_ROWS)) or latent.shape != (N_ROWS, LATENT_DIM) or not np.isfinite(latent).all(): raise PrecastE17R2Error("R output mapping/latent contract failed")
                record = e17._write_arm(destination / "arms" / f"seed{int(seed)}" / f"{order}.npz", latent, ids, libs, animals, {"schema_version": PROTOCOL, "status": "OK", "seed": int(seed), "order": order, "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "feature_ids_sha256": manifest["feature_ids_sha256"], "source_closure": manifest["source_closure"], "r_invocation": invocation, "parameters": manifest["parameters"]})
                records[key] = record; completed.append(key)
            except Exception as exc:
                _write_failure(destination, manifest, seed=int(seed), order=order, error=exc, invocation=getattr(exc, "invocation", invocation), completed=completed)
                raise
    return records


def fit_bundle(*, project_root: Path | str = ROOT, output_dir: Path | str, seeds: Sequence[int] = SEEDS, allow_model_fits: bool = False) -> dict[str, Any]:
    if not allow_model_fits: raise RuntimeError("REFUSED: PRECAST E17R2 fit requires --allow-model-fits; no model fit was started")
    dest = Path(output_dir).resolve()
    if dest.exists(): raise FileExistsError(f"refusing output collision: {dest}")
    selected = tuple(int(v) for v in seeds)
    if not selected or len(set(selected)) != len(selected) or any(v not in SEEDS for v in selected): raise PrecastE17R2Error("seeds must be a unique nonempty frozen subset")
    deps = _r_packages()
    if deps.get("status") != "PASS": raise RuntimeError(f"REFUSED: recorded PRECAST dependency gate failed: {deps.get('blocker')}")
    budget = resource_estimate()
    if not budget.get("gate_pass"): raise RuntimeError("REFUSED: PRECAST E17R2 host-memory gate failed")
    root = Path(project_root).resolve(); e17 = _load_e17(root); e8 = e17._load_e8(root); dest.mkdir(parents=True)
    with e8.open_bundle(root, feature_mode="all_common_19465") as bundle:
        manifest = {"schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "STARTED", "dataset": "GSE254652_P1", "feature_mode": "all_common_19465", "feature_count": len(bundle.feature_ids), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)), "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR"}, "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256}, "orders": list(ORDERS), "seeds": list(selected), "parameters": {"K": K, "q": Q, "knn_neigh": KNN, "coreNum": 1, "coreNum_int": 1, "beta_grid": 1, "maxIter_ICM": 6, "maxIter": 30, "r_timeout_seconds": PRODUCTION_TIMEOUT_SECONDS, "r_failure_policy": "fail_stop_before_next_arm", "raw_r_evidence": "persisted under r_evidence/seed/order", "row_map": "explicit output_row_index"}, "labels": {"disease": "NOT_CREATED", "physical_batch": "UNKNOWN_NOT_CREATED"}, "source_closure": source_closure(root), "dependencies": deps, "resource_qc": budget, "output_dir": str(dest), "started_at": _dt.datetime.now().astimezone().isoformat()}; manifest["config_sha256"] = canonical_sha256(manifest["parameters"]); _exclusive_json(dest / "manifest.json", manifest)
        try:
            records = guarded_arm_sequence(destination=dest, bundle=bundle, selected=selected, manifest=manifest, invoke=invoke_r, e17=e17)
        except Exception:
            raise
        summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "requested_seeds": list(selected), "expected_orders": list(ORDERS), "completed_arms": len(records), "expected_arms": len(selected) * len(ORDERS), "arms": records, "completed_at": _dt.datetime.now().astimezone().isoformat()}; _exclusive_json(dest / "run_summary.json", summary); _exclusive_json(dest / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(dest / "run_summary.json")}); return summary


def native_fixture_smoke(seed: int = 20260819) -> dict[str, Any]:
    e17 = _load_e17(); X, ids, libs, animals, spatial, code = e17._tiny_fixture(); features = [f"ENSMUSGTINY{i:05d}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory(prefix="e17tpc_") as td:
        d = Path(td); inp, out = d / "i.npz", d / "o.npz"; _save_input(inp, X, spatial, code, sorted(set(animals.tolist()), key=int), features); invocation = invoke_r(inp, out, seed, k=2, q=5, timeout_seconds=TINY_TIMEOUT_SECONDS, evidence_dir=d / "r_evidence")
        with np.load(out, allow_pickle=False) as z: latent, row_map, in_rows = np.asarray(z["latent"]), np.asarray(z["output_row_index"]), np.asarray(z["input_row_index"])
    if latent.shape != (350, 5) or not np.isfinite(latent).all() or row_map.tolist() != list(range(350)) or in_rows.tolist() != list(range(350)): raise PrecastE17R2Error("native PRECAST smoke output mapping/shape failed")
    return {"status": "PASS", "model_fit": "FIXTURE_ONLY", "entrypoint": "E17R2 invoke_r subprocess", "backend": "native PRECAST R 1.9", "r_invocation": invocation, "latent_shape": list(latent.shape), "raw_mapping_roundtrip": "PASS", "production_input_opened": False, "tiny_timeout_seconds": TINY_TIMEOUT_SECONDS, "production_timeout_seconds": PRODUCTION_TIMEOUT_SECONDS}


def audit_output(*, output_dir: Path | str, project_root: Path | str = ROOT) -> dict[str, Any]:
    """Check terminal or preserved-failure output without opening the H5AD."""
    dest = Path(output_dir).resolve(); manifest = _json(dest / "manifest.json"); summary = _json(dest / "run_summary.json")
    if manifest.get("schema_version") != PROTOCOL: raise PrecastE17R2Error("E17R2 manifest schema mismatch")
    if manifest.get("source_closure") != source_closure(Path(project_root).resolve()): raise PrecastE17R2Error("E17R2 source closure drift")
    expected = int(len(manifest.get("seeds", ())) * len(ORDERS)); completed = int(summary.get("completed_arms", 0)); errors = []
    for key, record in (summary.get("arms") or {}).items():
        npz = Path(record.get("npz", "")); sidecar = npz.with_suffix(".json")
        if not npz.is_file() or not sidecar.is_file() or record.get("npz_sha256") != sha256_file(npz): errors.append(f"{key}: arm/sidecar/hash missing")
        seed, order = key.split(":", 1)
        evidence = dest / "r_evidence" / f"seed{int(seed)}" / order
        for name in ("input.npz", "output.npz", "stdout.log", "stderr.log"):
            if not (evidence / name).is_file(): errors.append(f"{key}: missing preserved R evidence {name}")
    failure = dest / "run_failure.json"
    if summary.get("status") == "FIT_FAILED":
        if not failure.is_file(): errors.append("FIT_FAILED summary lacks run_failure.json")
        return {"schema_version": PROTOCOL, "status": "FIT_FAILED" if not errors else "UNRESOLVED", "model_fit": "AUDITED_FAILURE", "completed_arms": completed, "expected_arms": expected, "errors": errors, "next_arm_started": False}
    if summary.get("status") != "FIT_COMPLETE" or completed != expected or not (dest / "COMPLETED.json").is_file(): errors.append("terminal completion/count marker mismatch")
    return {"schema_version": PROTOCOL, "status": "PASS" if not errors else "UNRESOLVED", "model_fit": "AUDITED_ONLY", "completed_arms": completed, "expected_arms": expected, "errors": errors, "preserved_r_evidence": not errors}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--mode", choices=("native-smoke", "fit", "audit"), default="native-smoke"); ap.add_argument("--project-root", type=Path, default=ROOT); ap.add_argument("--output-dir", type=Path); ap.add_argument("--allow-model-fits", action="store_true"); ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS)); args = ap.parse_args(argv)
    try:
        if args.mode == "native-smoke": result = native_fixture_smoke()
        elif args.mode == "fit":
            if args.output_dir is None: ap.error("--output-dir required for fit")
            result = fit_bundle(project_root=args.project_root, output_dir=args.output_dir, seeds=args.seeds, allow_model_fits=args.allow_model_fits)
        else:
            if args.output_dir is None: ap.error("--output-dir required for audit")
            result = audit_output(project_root=args.project_root, output_dir=args.output_dir)
        print(json.dumps({"status": result.get("status"), "model_fit": result.get("model_fit"), "mode": args.mode}, sort_keys=True)); return 0 if result.get("status") in {"PASS", "FIT_COMPLETE"} else 1
    except Exception as exc: print(f"E17T_PRECAST_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
