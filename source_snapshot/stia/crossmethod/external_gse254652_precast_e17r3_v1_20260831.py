"""E17U PRECAST directory-preparation successor.

E17T proved the timeout/fail-stop/evidence contract but exposed a missing
parent-directory creation before the first production input save.  This
additive successor creates each arm evidence directory before *any* input
materialization/save and records preparation failures durably.  Production
dimensions remain 9,568×19,465 with q=15; fixture dimensions are accepted only
through the explicit ``fixture_only`` test path.
"""
from __future__ import annotations

import argparse, datetime as _dt, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "precast-null-inspire-seeds-20260830"
E17T_REL = Path("stia/crossmethod/external_gse254652_precast_e17r2_v1_20260831.py")
E17_REL = Path("stia/crossmethod/external_gse254652_precast_e17_v1_20260831.py")
R_CORE_REL = Path("Lunacy/runs/precast-null-inspire-seeds-20260830/launch/E17/external_gse254652_precast_e17_core_v1_20260831.R")
PROTOCOL = "stia.external.gse254652.precast.e17u.v1"
ORDERS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
N_ROWS, N_FEATURES, LATENT_DIM = 9568, 19465, 15
K, Q, KNN = 20, 15, 10
PRODUCTION_TIMEOUT_SECONDS = 7200

if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


class PrecastE17UError(ValueError):
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
    if not p.is_file(): raise PrecastE17UError(f"missing JSON: {p}")
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise PrecastE17UError(f"JSON object required: {p}")
    return value


def _exclusive_json(path: Path | str, value: Mapping[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("x", encoding="utf-8", newline="\n") as f: json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    except FileExistsError as exc: raise PrecastE17UError(f"refusing overwrite: {p}") from exc


def _load_e17t(root: Path = ROOT) -> Any:
    path = root / E17T_REL; spec = importlib.util.spec_from_file_location("e17u_precast_e17t", path)
    if spec is None or spec.loader is None: raise PrecastE17UError(f"cannot import E17T route: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def _load_e17(root: Path = ROOT) -> Any:
    path = root / E17_REL; spec = importlib.util.spec_from_file_location("e17u_precast_e17", path)
    if spec is None or spec.loader is None: raise PrecastE17UError(f"cannot import E17 route: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def source_closure(root: Path | str = ROOT) -> dict[str, dict[str, str]]:
    root = Path(root).resolve(); paths = {"e17u_runner": Path(__file__).resolve(), "e17t_runner": root / E17T_REL, "e17_route": root / E17_REL, "r_core": root / R_CORE_REL, "e8_adapter": root / "stia/crossmethod/external_gse254652_adapter_v1_20260831.py", "precast_gap4_core": root / "scripts/_x3_stageC_precast_core_v16_20260826.R", "precast_gap4_artifact": root / "scripts/_x3_precast_osi_v16_gap_artifact_20260829.py", "plan": RUN_ROOT / "PLAN.md"}
    out = {}
    for name, path in paths.items():
        if not path.is_file(): raise PrecastE17UError(f"source closure member missing: {path}")
        out[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return out


def invoke_r(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    return _load_e17t().invoke_r(*args, **kwargs)


def _write_failure(dest: Path, manifest: Mapping[str, Any], *, seed: int, order: str, error: BaseException, invocation: Mapping[str, Any] | None, completed: Sequence[str], evidence_dir: Path) -> None:
    failure = {"schema_version": PROTOCOL, "status": "FIT_FAILED", "failed_seed": int(seed), "failed_order": order, "error_type": type(error).__name__, "error": str(error), "invocation": dict(invocation or {}), "evidence_dir": str(evidence_dir.resolve()), "completed_arms": list(completed), "next_arm_started": False, "manifest_sha256": sha256_file(dest / "manifest.json"), "recorded_at": _dt.datetime.now().astimezone().isoformat()}
    _exclusive_json(dest / "run_failure.json", failure)
    _exclusive_json(dest / "run_summary.json", {"schema_version": PROTOCOL, "status": "FIT_FAILED", "model_fit": "PARTIAL_PRESERVED", "completed_arms": len(completed), "failed_seed": int(seed), "failed_order": order, "next_arm_started": False, "failure": failure})


def guarded_arm_sequence(*, destination: Path, bundle: Any, selected: Sequence[int], manifest: Mapping[str, Any], invoke: Callable[..., Mapping[str, Any]], e17: Any, fixture_only: bool = False) -> dict[str, Any]:
    """Run arms serially; prepare evidence directory before input save."""
    records: dict[str, Any] = {}; completed: list[str] = []
    for seed in selected:
        for order in ORDERS:
            key = f"{int(seed)}:{order}"; evidence_dir = destination / "r_evidence" / f"seed{int(seed)}" / order
            # This mkdir is intentionally before _materialize/_save_input: the
            # production failure was FileNotFoundError at the first input save.
            evidence_dir.mkdir(parents=True, exist_ok=True)
            inp, out = evidence_dir / "input.npz", evidence_dir / "output.npz"; invocation: Mapping[str, Any] | None = None
            try:
                X, ids, libs, animals, spatial, code = e17._materialize(bundle, order)
                e17._save_input(inp, X, spatial, code, [str(v) for v in bundle.orders[order]["animals"]], bundle.feature_ids)
                q = 5 if fixture_only else Q; invocation = invoke(inp, out, int(seed), k=2 if fixture_only else K, q=q, timeout_seconds=300 if fixture_only else PRODUCTION_TIMEOUT_SECONDS, evidence_dir=evidence_dir)
                with __import__("numpy").load(out, allow_pickle=False) as z: latent, row_map = __import__("numpy").asarray(z["latent"]), __import__("numpy").asarray(z["output_row_index"])
                expected_rows = int(X.shape[0]); expected_dim = int(q)
                if row_map.tolist() != list(range(expected_rows)) or latent.shape != (expected_rows, expected_dim) or not __import__("numpy").isfinite(latent).all(): raise PrecastE17UError("R output mapping/latent contract failed")
                record = e17._write_arm(destination / "arms" / f"seed{int(seed)}" / f"{order}.npz", latent, ids, libs, animals, {"schema_version": PROTOCOL, "status": "OK", "seed": int(seed), "order": order, "fixture_only": bool(fixture_only), "input_h5ad_sha256": getattr(bundle, "h5ad_sha256", "fixture"), "input_manifest_sha256": getattr(bundle, "manifest_sha256", "fixture"), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)), "source_closure": manifest["source_closure"], "r_invocation": invocation, "parameters": manifest["parameters"]})
                records[key] = record; completed.append(key)
            except Exception as exc:
                _write_failure(destination, manifest, seed=int(seed), order=order, error=exc, invocation=getattr(exc, "invocation", invocation), completed=completed, evidence_dir=evidence_dir)
                raise
    return records


def fit_bundle(*, project_root: Path | str = ROOT, output_dir: Path | str, seeds: Sequence[int] = SEEDS, allow_model_fits: bool = False) -> dict[str, Any]:
    """Run the production route with E17U's durable preparation/fail-stop boundary."""
    if not allow_model_fits:
        raise RuntimeError("REFUSED: PRECAST E17U fit requires --allow-model-fits; no model fit was started")
    dest = Path(output_dir).resolve()
    if dest.exists():
        raise FileExistsError(f"refusing output collision: {dest}")
    selected = tuple(int(v) for v in seeds)
    if not selected or len(set(selected)) != len(selected) or any(v not in SEEDS for v in selected):
        raise PrecastE17UError("seeds must be a unique nonempty frozen subset")
    root = Path(project_root).resolve()
    e17 = _load_e17(root)
    deps = e17._r_packages()
    if deps.get("status") != "PASS":
        raise RuntimeError(f"REFUSED: recorded PRECAST dependency gate failed: {deps.get('blocker')}")
    budget = e17.resource_estimate()
    if not budget.get("gate_pass"):
        raise RuntimeError("REFUSED: PRECAST E17U host-memory gate failed")
    e8 = e17._load_e8(root)
    dest.mkdir(parents=True)
    with e8.open_bundle(root, feature_mode="all_common_19465") as bundle:
        manifest = {
            "schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "STARTED",
            "dataset": "GSE254652_P1", "feature_mode": "all_common_19465",
            "feature_count": len(bundle.feature_ids), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)),
            "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [N_ROWS, N_FEATURES], "sparse_format": "CSR"},
            "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256},
            "orders": list(ORDERS), "seeds": list(selected),
            "parameters": {"K": K, "q": Q, "knn_neigh": KNN, "coreNum": 1, "coreNum_int": 1, "beta_grid": 1, "maxIter_ICM": 6, "maxIter": 30, "r_timeout_seconds": PRODUCTION_TIMEOUT_SECONDS, "r_failure_policy": "fail_stop_before_next_arm", "raw_r_evidence": "persisted under r_evidence/seed/order", "row_map": "explicit output_row_index"},
            "labels": {"disease": "NOT_CREATED", "physical_batch": "UNKNOWN_NOT_CREATED"},
            "source_closure": source_closure(root), "dependencies": deps, "resource_qc": budget,
            "output_dir": str(dest), "started_at": _dt.datetime.now().astimezone().isoformat(),
        }
        manifest["config_sha256"] = canonical_sha256(manifest["parameters"])
        _exclusive_json(dest / "manifest.json", manifest)
        records = guarded_arm_sequence(destination=dest, bundle=bundle, selected=selected, manifest=manifest, invoke=invoke_r, e17=e17)
    summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "requested_seeds": list(selected), "expected_orders": list(ORDERS), "completed_arms": len(records), "expected_arms": len(selected) * len(ORDERS), "arms": records, "completed_at": _dt.datetime.now().astimezone().isoformat()}
    _exclusive_json(dest / "run_summary.json", summary)
    _exclusive_json(dest / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(dest / "run_summary.json")})
    return summary


def audit_output(*, output_dir: Path | str, project_root: Path | str = ROOT) -> dict[str, Any]:
    """Audit terminal or preserved-failure markers without reopening the H5AD."""
    dest = Path(output_dir).resolve(); manifest = _json(dest / "manifest.json"); summary = _json(dest / "run_summary.json")
    if manifest.get("schema_version") != PROTOCOL:
        raise PrecastE17UError("E17U manifest schema mismatch")
    if manifest.get("source_closure") != source_closure(Path(project_root).resolve()):
        raise PrecastE17UError("E17U source closure drift")
    expected = int(len(manifest.get("seeds", ())) * len(ORDERS)); status = summary.get("status")
    if status == "FIT_FAILED":
        return {"schema_version": PROTOCOL, "status": "FIT_FAILED" if (dest / "run_failure.json").is_file() and not summary.get("next_arm_started", True) else "UNRESOLVED", "model_fit": "AUDITED_FAILURE", "completed_arms": int(summary.get("completed_arms", 0)), "expected_arms": expected, "next_arm_started": bool(summary.get("next_arm_started", True))}
    if status != "FIT_COMPLETE" or int(summary.get("completed_arms", 0)) != expected or not (dest / "COMPLETED.json").is_file():
        return {"schema_version": PROTOCOL, "status": "UNRESOLVED", "model_fit": "AUDITED_ONLY", "completed_arms": int(summary.get("completed_arms", 0)), "expected_arms": expected}
    return {"schema_version": PROTOCOL, "status": "PASS", "model_fit": "AUDITED_ONLY", "completed_arms": expected, "expected_arms": expected, "raw_r_evidence": (dest / "r_evidence").is_dir()}


def native_fixture_smoke(seed: int = 20260819) -> dict[str, Any]:
    e17 = _load_e17(); X, ids, libs, animals, spatial, code = e17._tiny_fixture(); features = [f"ENSMUSGTINY{i:05d}" for i in range(X.shape[1])]
    class Bundle: pass
    bundle = Bundle(); bundle.feature_ids = tuple(features); bundle.h5ad_sha256 = "fixture"; bundle.manifest_sha256 = "fixture"; bundle.orders = {order: {"animals": sorted(set(animals.tolist()), key=int)} for order in ORDERS}
    with tempfile.TemporaryDirectory(prefix="e17upc_") as td:
        dest = Path(td) / "absent"; dest.mkdir(); manifest = {"source_closure": source_closure(), "feature_ids_sha256": canonical_sha256(features), "parameters": {"K": 2, "q": 5, "knn_neigh": KNN, "fixture_only": True}}; _exclusive_json(dest / "manifest.json", manifest)
        original = e17._materialize
        try:
            e17._materialize = lambda _bundle, _order: (X, ids, libs, animals, spatial, code)
            records = guarded_arm_sequence(destination=dest, bundle=bundle, selected=[seed], manifest=manifest, invoke=invoke_r, e17=e17, fixture_only=True)
        finally: e17._materialize = original
    return {"status": "PASS", "model_fit": "FIXTURE_ONLY", "entrypoint": "guarded_arm_sequence with real _save_input + invoke_r", "completed_arms": len(records), "first_arm_evidence_created_before_save": True, "raw_mapping": "PASS", "production_dimensions_untouched": [N_ROWS, N_FEATURES], "production_timeout_seconds": PRODUCTION_TIMEOUT_SECONDS}


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
        print(json.dumps({"status": result.get("status"), "model_fit": result.get("model_fit"), "mode": args.mode}, sort_keys=True)); return 0 if result.get("status") in {"PASS", "FIT_COMPLETE", "FIT_FAILED"} else 1
    except Exception as exc: print(f"E17U_PRECAST_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
