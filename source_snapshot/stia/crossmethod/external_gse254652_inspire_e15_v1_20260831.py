"""Source-bound INSPIRE external route for GSE254652 P1 (E15).

The route is additive and parent-gated.  It freezes one pooled 3,000-feature
panel from the first arm, reuses that panel for all four frozen section orders,
and persists latent/shared/gamma/full readouts with source-derived spot and
animal identities.  ``fit`` refuses to run without ``--allow-model-fits``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(r"LOCAL_INPUT_ROOT\Downloads\INSPIRE-main\INSPIRE-main\INSPIRE")
RUN_ROOT = ROOT / "Lunacy/runs/precast-null-inspire-seeds-20260830"
H5AD_DEFAULT = ROOT / "data/external/gse254652_p1_sparse_20260830_r1/gse254652_p1_sparse_20260830_r1.h5ad"
MANIFEST_DEFAULT = ROOT / "data/external/gse254652_p1_sparse_20260830_r1/gse254652_p1_sparse_20260830_r1_manifest.json"
PROTOCOL = "stia.external.gse254652.e15.inspire_four_order.v1"
FEATURE_COUNT = 3000
LATENT_DIM = 32
N_STEPS = 400
N_FACTORS = 20
SEED_SET = (20260819, 20260820, 20260821, 20260822, 20260823)
ORDER_IDS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
OSI_MARGIN = 0.05
NO_LABELS = {"disease": "NOT_CREATED", "batch": "NOT_CREATED"}
# A process-unique cache avoids numba's non-atomic Windows index update when
# a prior failed attempt left the same package cache behind.
E15_TMP = RUN_ROOT / "evidence" / "E15" / f".runtime_tmp_{os.getpid()}"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from stia.crossmethod import external_gse254652_adapter_v1_20260831 as e8
from stia.crossmethod.osi import compute_osi


class E15Error(ValueError):
    pass


def _configure_runtime_environment() -> None:
    """Pin a writable temporary directory before importing scanpy/numba.

    On this host ``TEMP=LOCAL_RUNTIME_ROOT`` makes numba's import-time cache probe
    block indefinitely.  The directory is run-local and additive; it never
    replaces scientific outputs or historical evidence.
    """
    E15_TMP.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(E15_TMP)
    os.environ["TMP"] = str(E15_TMP)
    os.environ["NUMBA_CACHE_DIR"] = str(E15_TMP)
    # Scanpy's Seurat-v3 HVG helper is numerically identical with numba JIT
    # disabled, while avoiding the broken Windows cache write path observed in
    # E15 r1-r4.  Record this environment in the manifest for provenance.
    os.environ["NUMBA_DISABLE_JIT"] = "1"


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def hash_array(value: Any) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def source_closure() -> dict[str, Any]:
    members = {
        "route": Path(__file__).resolve(),
        "external_adapter": Path(e8.__file__).resolve(),
        "osi": (ROOT / "stia/crossmethod/osi.py").resolve(),
        "frozen_contract": (ROOT / "stia/crossmethod/frozen_contract.py").resolve(),
        # Resolve the vendored package directly.  Importing INSPIRE pulls in
        # scanpy/umap and can take minutes on a cold Windows environment;
        # planning and provenance must stay lightweight and side-effect free.
        "inspire_init": (ROOT.parent / "INSPIRE" / "__init__.py").resolve(),
        "inspire_utils": (ROOT.parent / "INSPIRE" / "utils.py").resolve(),
        "inspire_model": (ROOT.parent / "INSPIRE" / "model.py").resolve(),
        "inspire_networks": (ROOT.parent / "INSPIRE" / "networks.py").resolve(),
    }
    out = {}
    for key, path in members.items():
        if not path.is_file():
            raise E15Error(f"source closure member missing: {path}")
        out[key] = {"path": str(path), "sha256": sha256_file(path)}
    return out


def seed_all(seed: int) -> dict[str, Any]:
    if int(seed) not in SEED_SET:
        raise E15Error(f"seed is not frozen: {seed}")
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1))
    import torch
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return {"python": int(seed), "numpy": int(seed) % (2**32 - 1), "torch": int(seed), "torch_cuda": int(seed) if torch.cuda.is_available() else "not_available"}


def resource_estimate() -> dict[str, Any]:
    import psutil
    import torch
    n, p = 9568, FEATURE_COUNT
    dense_x = n * p * 4
    graph = n * n * 4
    known_host = dense_x + graph + 2 * graph + 2 * dense_x
    known_device = dense_x * 2 + graph * 2 + n * LATENT_DIM * 4 * 3
    return {
        "n_rows": n, "n_features": p, "known_host_bytes": int(known_host), "known_device_bytes": int(known_device),
        "estimated_peak_bytes": int(max(known_host, known_device) * 1.5),
        "available_host_bytes": int(psutil.virtual_memory().available),
        "available_device_bytes": int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "headroom_bytes": 2 * 1024**3,
        "measured_peak": False,
    }


def _make_sections(bundle: Any, order_id: str) -> tuple[list[Any], list[str], list[str], list[str]]:
    import anndata as ad
    import pandas as pd
    ordered = bundle.ordered_sections(order_id)
    sections, spot_ids, donors, libraries = [], [], [], []
    for section in ordered:
        matrix = section.matrix.copy()
        ids = list(map(str, section.spot_ids))
        obs = pd.DataFrame({
            "canonical_spot_id": ids,
            "library_id": [section.gsm] * section.n_rows,
            "animal_id": [section.animal_id] * section.n_rows,
            "donor": [section.animal_id] * section.n_rows,
            "batch": [section.gsm] * section.n_rows,
        }, index=ids)
        spatial = np.column_stack([np.asarray(section.coordinates["array_col"], dtype=float), np.asarray(section.coordinates["array_row"], dtype=float)])
        a = ad.AnnData(X=matrix, obs=obs, var=__import__("pandas").DataFrame(index=list(bundle.feature_ids)))
        a.obsm["spatial"] = spatial
        sections.append(a); spot_ids.extend(ids); donors.extend([section.animal_id] * section.n_rows); libraries.extend([section.gsm] * section.n_rows)
    return sections, spot_ids, donors, libraries


def _pca32(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < LATENT_DIM or not np.isfinite(matrix).all():
        raise E15Error(f"PCA32 input invalid: {matrix.shape}")
    centered = matrix - matrix.mean(axis=0)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    if left.shape[1] < LATENT_DIM:
        raise E15Error("PCA32 rank below 32")
    return np.asarray(left[:, :LATENT_DIM] * singular[:LATENT_DIM], dtype=np.float32)


def _prepare_sections(sections: list[Any], panel: Sequence[str] | None = None) -> tuple[list[Any], Any, list[str]]:
    _configure_runtime_environment()
    # INSPIRE/model.py imports UMAP only for an unused visualization helper.
    # Importing the full UMAP→PyNNDescent stack triggers a Windows numba cache
    # race before any model code runs, so provide a minimal import-only module.
    # No UMAP routine is called by this route.
    import types
    sys.modules.setdefault("umap", types.ModuleType("umap"))
    import INSPIRE
    from scripts.tech_batch_v3_model import align_panel
    st, full = INSPIRE.utils.preprocess(sections, num_hvgs=FEATURE_COUNT, min_genes_qc=1, min_cells_qc=1, spot_size=100, hvg_strategy="pooled")
    selected = list(map(str, st[0].var_names))
    if panel is not None:
        if set(selected) != set(panel):
            raise E15Error("pooled HVG panel differs across order arms; refusing implicit panel drift")
        selected = list(panel)
    st = [align_panel(a, selected) for a in st]
    full = align_panel(full, selected)
    if any(a.n_vars != FEATURE_COUNT for a in st) or full.n_vars != FEATURE_COUNT:
        raise E15Error("fixed panel dimension mismatch")
    return st, full, selected


def fit_one_order(sections: list[Any], *, panel: Sequence[str], seed: int) -> dict[str, Any]:
    import torch
    import INSPIRE
    seed_all(seed)
    st, full, selected = _prepare_sections(sections, panel=panel)
    st = INSPIRE.utils.build_graph_GAT(st, rad_coef=1.1)
    model = INSPIRE.model.Model_GAT(st, n_spatial_factors=N_FACTORS, n_training_steps=N_STEPS, seed=int(seed))
    model.train(record_final_loss=True)
    full_eval, basis_df = model.eval(full.copy())
    with torch.no_grad():
        Zs, betas, _alphas, _recons, basis_val, gammas = model.net(model.adj_matrix_dict, model.node_feats_dict, model.slice_label_dict)
    latent = np.concatenate([z.detach().cpu().numpy() for z in Zs.values()], axis=0).astype(np.float32)
    b_full = np.concatenate([b.detach().cpu().numpy() for b in betas.values()], axis=0)
    mu = basis_df.to_numpy(dtype=np.float64)
    gamma_np = gammas.detach().cpu().numpy()
    slice_of = np.repeat(np.arange(len(st)), [a.n_obs for a in st])
    log_lam = np.log(np.maximum(b_full @ mu, 0.0) + 1e-6)
    readouts = {"latent": latent, "shared": _pca32(log_lam), "gamma": _pca32(gamma_np[slice_of]), "full": _pca32(log_lam + gamma_np[slice_of])}
    if any(v.shape != (sum(a.n_obs for a in st), LATENT_DIM) or not np.isfinite(v).all() for v in readouts.values()):
        raise E15Error("readout shape/finite failure")
    del model, st, full, full_eval
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"readouts": readouts, "panel": selected, "loss_history": getattr(model, "loss_val", []) if False else [], "n_rows": int(latent.shape[0])}


def fit_run(*, output_dir: Path | str, project_root: Path | str = ROOT, seed: int = 20260819, allow_model_fits: bool = False, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT) -> dict[str, Any]:
    if not allow_model_fits:
        raise E15Error("REFUSED: E15 fit requires --allow-model-fits")
    if seed not in SEED_SET:
        raise E15Error("seed is not frozen")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing output collision: {destination}")
    destination.mkdir(parents=True)
    _configure_runtime_environment()
    with e8.open_bundle(project_root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode="all_common_19465") as bundle:
        closure = source_closure()
        config = {"protocol": PROTOCOL, "feature_count": FEATURE_COUNT, "n_steps": N_STEPS, "n_factors": N_FACTORS, "representation": "latent/shared/gamma/full PCA32", "orders": list(ORDER_IDS), "requested_seed": int(seed), "labels": NO_LABELS, "panel_selection": "pooled HVG from A_original then frozen by exact gene set", "preprocessing": "INSPIRE.utils.preprocess + pooled HVG + source spatial coordinates", "numba_disable_jit": True, "numba_cache_workaround": "E15 r1-r4 cache-write failures; Python fallback for scanpy HVG"}
        manifest = {"schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "PERFORMED", "output_dir": str(destination), "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [9568, 19465]}, "input_manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256}, "config": config, "config_sha256": canonical_sha256(config), "source_closure": closure, "orders": {k: dict(v) for k, v in bundle.orders.items()}, "resource_qc": resource_estimate(), "seed_provenance": seed_all(seed), "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}, "started_at": _dt.datetime.now().astimezone().isoformat()}
        write_exclusive(destination / "manifest.json", manifest)
        panel = None; checkpoints = {}
        for order in ORDER_IDS:
            sections, ids, donors, libraries = _make_sections(bundle, order)
            result = fit_one_order(sections, panel=panel or [], seed=int(seed)) if panel else None
            if result is None:
                import types
                sys.modules.setdefault("umap", types.ModuleType("umap"))
                import INSPIRE
                st, full, panel = _prepare_sections(sections, panel=None)
                st = INSPIRE.utils.build_graph_GAT(st, rad_coef=1.1)
                import torch
                seed_all(seed); model = INSPIRE.model.Model_GAT(st, n_spatial_factors=N_FACTORS, n_training_steps=N_STEPS, seed=int(seed)); model.train(record_final_loss=True); full_eval, basis_df = model.eval(full.copy())
                with torch.no_grad(): Zs, betas, _a, _r, _bv, gammas = model.net(model.adj_matrix_dict, model.node_feats_dict, model.slice_label_dict)
                latent=np.concatenate([z.detach().cpu().numpy() for z in Zs.values()],0).astype(np.float32); b_full=np.concatenate([b.detach().cpu().numpy() for b in betas.values()],0); mu=basis_df.to_numpy(dtype=np.float64); g=gammas.detach().cpu().numpy(); so=np.repeat(np.arange(len(st)),[a.n_obs for a in st]); ll=np.log(np.maximum(b_full@mu,0.0)+1e-6); readouts={"latent":latent,"shared":_pca32(ll),"gamma":_pca32(g[so]),"full":_pca32(ll+g[so])}; result={"readouts":readouts,"panel":panel,"n_rows":int(latent.shape[0])}; del model, st, full, full_eval
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            arm_dir = destination / "arms" / f"seed{seed}"; arm_dir.mkdir(parents=True, exist_ok=True); npz = arm_dir / f"{order}.npz"; side = arm_dir / f"{order}.json"
            arrays = {**result["readouts"], "canonical_spot_ids": np.asarray(ids, dtype="U"), "donor_keys": np.asarray(donors, dtype="U"), "library_ids": np.asarray(libraries, dtype="U")}; np.savez_compressed(npz, **arrays)
            sidecar = {"schema_version": PROTOCOL, "status": "OK", "seed": int(seed), "order": order, "npz": str(npz), "npz_sha256": sha256_file(npz), "readout_shapes": {k: list(v.shape) for k,v in result["readouts"].items()}, "panel_sha256": canonical_sha256(list(panel)), "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "source_closure": closure, "labels": NO_LABELS}
            write_exclusive(side, sidecar); checkpoints[order] = sidecar
        summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "requested_seed": int(seed), "completed_orders": list(ORDER_IDS), "checkpoints": checkpoints, "completed_at": _dt.datetime.now().astimezone().isoformat()}; write_exclusive(destination / "run_summary.json", summary); write_exclusive(destination / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(destination / "run_summary.json"), "completed_at": summary["completed_at"]}); return summary


def audit_run(output_dir: Path | str) -> dict[str, Any]:
    destination = Path(output_dir).resolve(); errors=[]; latents={}; donors={}; ids_by={}
    try: manifest=json.loads((destination/"manifest.json").read_text(encoding="utf-8")); summary=json.loads((destination/"run_summary.json").read_text(encoding="utf-8")); marker=json.loads((destination/"COMPLETED.json").read_text(encoding="utf-8"))
    except Exception as exc: return {"schema_version":PROTOCOL,"status":"UNRESOLVED","errors":[str(exc)],"cells":[]}
    if manifest.get("status")!="FIT_STARTED" or summary.get("status")!="FIT_COMPLETE" or summary.get("completed_orders")!=list(ORDER_IDS): errors.append("terminal markers/order set invalid")
    if marker.get("status")!="COMPLETE" or marker.get("summary_sha256")!=sha256_file(destination/"run_summary.json"): errors.append("completion marker hash invalid")
    with e8.open_bundle(ROOT, h5ad_path=manifest["h5ad"]["path"], manifest_path=manifest["input_manifest"]["path"], feature_mode="all_common_19465") as bundle:
        for order in ORDER_IDS:
            npz=destination/"arms"/f"seed{manifest['config']['requested_seed']}"/f"{order}.npz"; side=npz.with_suffix('.json')
            try:
                sm=json.loads(side.read_text(encoding='utf-8')); assert sm.get('npz_sha256')==sha256_file(npz)
                with np.load(npz,allow_pickle=False) as q:
                    assert set(q.files)=={'latent','shared','gamma','full','canonical_spot_ids','donor_keys','library_ids'}
                    for k in ('latent','shared','gamma','full'): assert q[k].shape==(9568,LATENT_DIM) and np.isfinite(q[k]).all()
                    ids=q['canonical_spot_ids'].astype(str).tolist(); dr=q['donor_keys'].astype(str).tolist()
                assert len(ids)==len(set(ids))==9568 and len(dr)==9568 and len(set(dr))==7
                latents[order]=q['latent']; donors[order]=dr; ids_by[order]=ids
            except Exception as exc: errors.append(f'{order}: {exc}')
    if ids_by:
        base=set(ids_by.get(ORDER_IDS[0],[]));
        if any(set(ids_by.get(o,[]))!=base for o in ORDER_IDS): errors.append('canonical spot ID sets differ')
    metric=compute_osi(latents,donors,condition='gse254652_p1_external_inspire',seed=int(manifest.get('config',{}).get('requested_seed',20260819)),margin=OSI_MARGIN) if len(latents)==4 else None
    return {"schema_version":PROTOCOL,"status":"PASS" if not errors else "UNRESOLVED","execution_validity":"PASS" if not errors else "UNRESOLVED","errors":errors,"cells":[{"seed":int(manifest.get('config',{}).get('requested_seed',20260819)),"execution_status":"PASS" if not errors else "UNRESOLVED","numeric_status":metric.get('status') if metric else "UNRESOLVED","osi":float(metric['osi']) if metric else None,"pair_count":6 if metric else None}],"metric":metric,"claim_boundary":{"external_validation":"NOT_PERFORMED","disease_or_batch_causal_claim":"FORBIDDEN"},"audited_at":_dt.datetime.now().astimezone().isoformat()}


def main(argv: Sequence[str] | None=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('plan', 'fit', 'audit'), default='plan')
    ap.add_argument('--project-root', type=Path, default=ROOT)
    ap.add_argument('--h5ad', type=Path, default=H5AD_DEFAULT)
    ap.add_argument('--manifest', type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--seed', type=int, default=20260819)
    ap.add_argument('--allow-model-fits', action='store_true')
    ap.add_argument('--evidence', type=Path)
    args = ap.parse_args(argv)
    if args.mode == 'plan':
        result = {
            "schema_version": PROTOCOL, "status": "READY_NO_FIT", "model_fit": "NOT_PERFORMED",
            "dataset": "GSE254652_P1", "feature_count": FEATURE_COUNT, "orders": list(ORDER_IDS),
            "seed_contract": list(SEED_SET), "resource_qc": resource_estimate(),
            "source_closure": source_closure(), "labels": NO_LABELS,
        }
    elif args.mode == 'fit':
        if args.output_dir is None:
            ap.error('--output-dir required')
        result = fit_run(output_dir=args.output_dir, project_root=args.project_root, seed=args.seed,
                         allow_model_fits=args.allow_model_fits, h5ad_path=args.h5ad,
                         manifest_path=args.manifest)
    else:
        if args.output_dir is None:
            ap.error('--output-dir required')
        result = audit_run(args.output_dir)
    if args.evidence: write_exclusive(args.evidence,result)
    print(json.dumps({"status":result.get('status'),"mode":args.mode,"model_fit":result.get('model_fit','NOT_PERFORMED')},ensure_ascii=False)); return 0 if result.get('status') in {'READY_NO_FIT','FIT_COMPLETE','PASS'} else 1

if __name__=='__main__': raise SystemExit(main())
