"""Explicit CPU device route for the unchanged E10R native STAligner protocol.

E10R remains immutable.  This additive successor fixes only device selection:
the native ``train_STAligner`` receives ``torch.device('cpu')`` and the
process explicitly uses four Torch/BLAS threads.  All genes, graph edges,
epochs, seeds, orders, raw-arm layout, provenance and OSI semantics remain
unchanged.  No fallback to CUDA and no silent protocol reduction is allowed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "precast-null-inspire-seeds-20260830"
AUTHORITY = RUN_ROOT / "phases" / "terminal-audit" / "E10C.md"
PROTOCOL = "stia.external.gse254652.e10c.staligner_cpu_four_order.v1"
THREADS = 4
CPU_MIN_FREE = 12 * 1024**3
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from stia.crossmethod import external_gse254652_staligner_e10r_v1_20260831 as e10r


class E10CValidationError(ValueError):
    """Fail-closed CPU route/source/artifact error."""


def _json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise E10CValidationError(f"missing JSON: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E10CValidationError(f"JSON object required: {target}")
    return value


def exclusive_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")


def sha256_file(path: Path | str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_cpu_threads() -> dict[str, Any]:
    """Set and measure the process-local four-thread CPU contract."""
    before = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    for name in before:
        os.environ[name] = str(THREADS)
    import torch
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(THREADS)
    except RuntimeError:
        # A caller may have initialized Torch before this route; effective
        # intra-op threads are still checked and no alternate backend is used.
        pass
    effective = {"torch_num_threads": int(torch.get_num_threads()), "torch_num_interop_threads": int(torch.get_num_interop_threads()), "blas_env": {name: os.environ.get(name) for name in before}, "previous_blas_env": before}
    if effective["torch_num_threads"] != THREADS:
        raise E10CValidationError("Torch thread contract is not exactly four")
    return effective


def source_closure(root: Path = ROOT) -> dict[str, dict[str, str]]:
    paths = e10r.source_closure(root)
    paths["e10c_runner"] = {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))}
    if not AUTHORITY.is_file():
        raise E10CValidationError(f"missing E10C authority: {AUTHORITY}")
    paths["e10c_authority"] = {"path": str(AUTHORITY.resolve()), "sha256": sha256_file(AUTHORITY)}
    return paths


def cpu_resource_estimate(*, n_rows: int, n_features: int, nnz: int, available_bytes: int | None = None) -> dict[str, Any]:
    """Conservative host budget; CPU keeps every working tensor host-resident."""
    base = e10r.resource_estimate(n_rows=n_rows, n_features=n_features, nnz=nnz, available_bytes=available_bytes)
    # E10R's known peak includes dense X, decoder output, edge messages,
    # optimizer and autograd.  CPU execution additionally retains the source
    # CSR/AnnData and transient native KNN intermediates on the host.  Keep a
    # conservative 1.75x allocator/thread-runtime factor, plus the mandated
    # 2 GiB post-estimate reserve and a 12 GiB absolute floor.
    host_known = int(base["known_peak_bytes"] + base["input_csr_bytes"] + base["graph_edge_bytes"])
    estimated = int(host_known * 1.75)
    required = max(CPU_MIN_FREE, estimated + 2 * 1024**3)
    if available_bytes is None:
        try:
            import psutil
            available_bytes = int(psutil.virtual_memory().available)
        except Exception:
            available_bytes = None
    return {"protocol": PROTOCOL, "host_resident_csr_ann_data_bytes": int(base["input_csr_bytes"]), "host_resident_dense_input_bytes": int(base["dense_input_bytes"]), "host_resident_decoder_node_output_bytes": int(base["decoder_node_output_bytes"]), "host_resident_attention_edge_message_bytes": int(base["attention_edge_message_bytes"]), "host_resident_model_parameter_bytes": int(base["model_parameter_bytes"]), "host_resident_optimizer_state_bytes": int(base["optimizer_state_bytes"]), "graph_edge_count": int(base["graph_edge_count"]), "host_known_peak_bytes": host_known, "allocator_thread_runtime_factor": 1.75, "estimated_cpu_peak_bytes": estimated, "required_free_bytes": required, "required_free_gib": required / 1024**3, "minimum_free_bytes": CPU_MIN_FREE, "available_bytes": available_bytes, "cpu_gate_pass": bool(available_bytes is not None and int(available_bytes) >= required), "training_vs_inference": {"training": "whole-graph autograd + Adam state + all decoder/attention activations", "inference": "whole-graph forward without Adam/autograd; still dense decoder output"}, "batching": "whole_graph; native train_STAligner has no minibatch sampler", "device": "cpu", "effective_threads": THREADS, "cuda_fallback": False, "placement_note": "CPU route keeps CSR/AnnData/todense/native tensors host-resident; CUDA identity is not claimed"}


def _fit_native_cpu(adata: Any, seed: int, *, train_fn: Any | None = None, n_epochs: int = e10r.N_EPOCHS) -> np.ndarray:
    settings = configure_cpu_threads()
    import torch
    if train_fn is None:
        from STAligner import train_STAligner
        train_fn = train_STAligner
    device = torch.device("cpu")
    trained = train_fn(adata, hidden_dims=list(e10r.HIDDEN_DIMS), n_epochs=int(n_epochs), lr=0.001, random_seed=int(seed), knn_neigh=e10r.KNN_NEIGH, device=device, verbose=False)
    if getattr(trained, "obsm", None) is None or "STAligner" not in trained.obsm:
        raise E10CValidationError("native CPU train did not return STAligner latent")
    latent = np.asarray(trained.obsm["STAligner"], dtype=np.float32)
    if latent.ndim != 2 or latent.shape != (adata.n_obs, e10r.LATENT_DIM) or not np.isfinite(latent).all():
        raise E10CValidationError(f"native CPU latent shape/finite failure: {latent.shape}")
    return latent


def fit_cpu_run(*, project_root: Path | str = ROOT, output_dir: Path | str, seeds: Sequence[int] = e10r.SEEDS, allow_model_fits: bool = False) -> dict[str, Any]:
    if not allow_model_fits:
        raise RuntimeError("REFUSED: E10C CPU fit requires --allow-model-fits; no model fit was started")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing output collision: {destination}")
    root = Path(project_root).resolve(); e8 = e10r._load_e8(); receipt = _json(e10r.E8_RECEIPT)
    selected = tuple(int(v) for v in seeds)
    if not selected or any(v not in e10r.SEEDS for v in selected) or len(set(selected)) != len(selected):
        raise E10CValidationError("seeds must be a unique subset of the frozen five seeds")
    threads = configure_cpu_threads(); budget = cpu_resource_estimate(n_rows=9568, n_features=19465, nnz=54807969)
    if not budget["cpu_gate_pass"]:
        raise RuntimeError("REFUSED: E10C CPU host-memory gate failed; wait for resource headroom")
    native = e10r._native_snapshot(); closure = source_closure(root)
    destination.mkdir(parents=True)
    with e8.open_bundle(root, h5ad_path=Path(receipt["h5ad"]["path"]), manifest_path=Path(receipt["manifest"]["path"]), feature_mode=e10r.FEATURE_MODE) as bundle:
        identities = {order: e10r.order_identity(bundle, order) for order in e10r.ORDER_IDS}
        config = {"protocol": PROTOCOL, "parent_protocol": e10r.PROTOCOL, "feature_mode": e10r.FEATURE_MODE, "feature_count": len(bundle.feature_ids), "hidden_dims": list(e10r.HIDDEN_DIMS), "n_epochs": e10r.N_EPOCHS, "pretrain_epochs": 500, "alignment_epochs": 500, "lr": 0.001, "knn_neigh": e10r.KNN_NEIGH, "orders": list(e10r.ORDER_IDS), "seeds": list(e10r.SEEDS), "requested_seeds": list(selected), "preprocessing": "sparse log1p-CPM", "graph": "native Cal_Spatial_Net per-library KNN k=10 block diagonal", "backend": "native STAligner", "device": "cpu", "threads": threads, "labels": e10r.NO_LABELS, "cuda_fallback": False}
        manifest = {"schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "STARTED", "output_dir": str(destination), "config": config, "feature_ids_sha256": e10r.canonical_sha256(list(bundle.feature_ids)), "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [9568, 19465], "sparse_format": "CSR"}, "input_manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256}, "source_closure": closure, "native_api": native, "resource_qc": budget, "orders": {name: {key: value for key, value in identity.items() if key not in {"spot_ids", "library_ids", "animal_ids", "rows"}} for name, identity in identities.items()}, "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN", "cuda_numerical_identity": "NOT_CLAIMED"}, "started_at": _dt.datetime.now().astimezone().isoformat()}
        e10r.exclusive_json(destination / "manifest.json", manifest); all_records: dict[str, Any] = {}
        for seed in selected:
            latents: dict[str, np.ndarray] = {}; donors: dict[str, np.ndarray] = {}; records: dict[str, Any] = {}
            for order in e10r.ORDER_IDS:
                adata = e10r.build_order_adata(bundle, order); e10r.build_native_graph(adata, k=e10r.KNN_NEIGH); latent = _fit_native_cpu(adata, seed)
                ident = identities[order]; latents[order] = latent; donors[order] = np.asarray(ident["animal_ids"], dtype="U")
                records[order] = e10r._write_arm(destination, latent=latent, identity=ident, seed=seed, order_id=order, base={"schema_version": PROTOCOL, "protocol": PROTOCOL, "feature_mode": e10r.FEATURE_MODE, "feature_count": len(bundle.feature_ids), "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "source_closure": closure, "backend": "native STAligner", "device": "cpu", "torch_num_threads": THREADS, "model_seed": int(seed), "labels": e10r.NO_LABELS})
            metric = e10r._metric(latents, donors, seed); checkpoint = {"schema_version": PROTOCOL, "status": "SEED_COMPLETE", "seed": int(seed), "backend": "cpu", "device": "cpu", "orders": records, "metric": metric, "completed_at": _dt.datetime.now().astimezone().isoformat()}; e10r.exclusive_json(destination / f"checkpoint_seed{seed}.json", checkpoint); all_records[str(seed)] = checkpoint
        summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "backend": "cpu", "device": "cpu", "torch_num_threads": THREADS, "requested_seeds": list(selected), "expected_orders": list(e10r.ORDER_IDS), "completed_seeds": list(selected), "seed_checkpoints": all_records, "terminal": {"expected_order_count": len(selected) * len(e10r.ORDER_IDS), "completion_marker": "COMPLETED.json"}, "completed_at": _dt.datetime.now().astimezone().isoformat()}; e10r.exclusive_json(destination / "run_summary.json", summary); e10r.exclusive_json(destination / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(destination / "run_summary.json"), "completed_at": summary["completed_at"]}); return summary


def real_input_preflight_cpu(*, project_root: Path | str = ROOT) -> dict[str, Any]:
    """Run the production sparse/graph traversal only after the CPU gate."""
    threads = configure_cpu_threads()
    budget = cpu_resource_estimate(n_rows=9568, n_features=19465, nnz=54807969)
    if not budget["cpu_gate_pass"]:
        raise RuntimeError("REFUSED: E10C CPU host-memory gate failed; no real preflight was started")
    result = e10r.real_input_preflight(project_root=project_root)
    result.update({"protocol": PROTOCOL, "device": "cpu", "threads": threads, "cpu_resource_qc": budget})
    return result


def audit_cpu_output(*, project_root: Path | str = ROOT, output_dir: Path | str) -> dict[str, Any]:
    destination = Path(output_dir).resolve(); manifest_file = destination / "manifest.json"; summary_file = destination / "run_summary.json"; complete = destination / "COMPLETED.json"
    if not manifest_file.is_file() or not summary_file.is_file() or not complete.is_file(): raise E10CValidationError("CPU terminal manifest/summary/completion marker missing")
    saved = _json(manifest_file); summary = _json(summary_file)
    if saved.get("schema_version") != PROTOCOL or saved.get("config", {}).get("device") != "cpu" or saved.get("config", {}).get("feature_count") != 19465 or summary.get("status") != "FIT_COMPLETE": raise E10CValidationError("CPU protocol/device/terminal contract drift")
    root = Path(project_root).resolve(); closure = source_closure(root)
    if saved.get("source_closure") != closure: raise E10CValidationError("CPU source closure drift")
    e8 = e10r._load_e8(); receipt = _json(e10r.E8_RECEIPT); cells = []; errors = []
    with e8.open_bundle(root, h5ad_path=Path(receipt["h5ad"]["path"]), manifest_path=Path(receipt["manifest"]["path"]), feature_mode=e10r.FEATURE_MODE) as bundle:
        identities = {order: e10r.order_identity(bundle, order) for order in e10r.ORDER_IDS}
        for seed in summary.get("completed_seeds", []):
            try:
                latents: dict[str, np.ndarray] = {}; donors: dict[str, np.ndarray] = {}
                for order in e10r.ORDER_IDS:
                    npz = destination / "arms" / f"seed{int(seed)}" / f"{order}.npz"; sidecar = npz.with_suffix(".json")
                    if not npz.is_file() or not sidecar.is_file(): raise E10CValidationError(f"missing arm {order}:{seed}")
                    latent, donor = e10r._check_arm(npz, sidecar, identities[order], int(seed), order, closure); meta = _json(sidecar)
                    if meta.get("device") != "cpu" or meta.get("torch_num_threads") != THREADS: raise E10CValidationError(f"device metadata drift {order}:{seed}")
                    latents[order] = latent; donors[order] = donor
                metric = e10r._metric(latents, donors, int(seed)); cells.append({"seed": int(seed), "execution_status": "PASS", "numeric_status": str(metric["status"]), "osi": float(metric["osi"]), "pair_count": len(metric["pairwise_drifts"])})
            except Exception as exc:
                errors.append(f"seed{seed}: {exc}"); cells.append({"seed": int(seed), "execution_status": "UNRESOLVED", "numeric_status": "UNRESOLVED", "reason": str(exc)})
    return {"schema_version": PROTOCOL, "status": "PASS" if not errors else "UNRESOLVED", "execution_validity": "PASS" if not errors else "UNRESOLVED", "backend": "cpu", "device": "cpu", "cells": cells, "errors": errors, "source_closure_verified": not errors, "claim_boundary": {"external_validation": "NOT_PERFORMED", "cuda_numerical_identity": "NOT_CLAIMED"}}


def fixture_cpu_smoke() -> dict[str, Any]:
    import scipy.sparse as sp
    from anndata import AnnData
    class FakeTrain:
        def __init__(self): self.kwargs = None
        def __call__(self, adata: Any, **kwargs: Any) -> Any:
            self.kwargs = kwargs; adata.obsm["STAligner"] = np.zeros((adata.n_obs, e10r.LATENT_DIM), dtype=np.float32); return adata
    fake = FakeTrain(); adata = AnnData(X=sp.csr_matrix(np.ones((6, 4), dtype=np.float32)), obs={"batch_name": ["l1"] * 3 + ["l2"] * 3}); adata.obsm["spatial"] = np.arange(12, dtype=np.float32).reshape(6, 2); e10r.build_native_graph(adata, k=2, graph_builder=lambda obj, **_: obj.uns.__setitem__("adj", sp.eye(obj.n_obs, format="csr"))); _fit_native_cpu(adata, 20260819, train_fn=fake, n_epochs=3)
    return {"status": "PASS", "device": str(fake.kwargs["device"]), "torch_num_threads": THREADS, "n_epochs": fake.kwargs["n_epochs"], "production_n_epochs": e10r.N_EPOCHS, "model_fit": "FIXTURE_ONLY", "cuda_fallback": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E10C native STAligner CPU route")
    parser.add_argument("--mode", choices=("native-smoke", "fixture", "preflight", "fit", "audit"), default="native-smoke"); parser.add_argument("--project-root", type=Path, default=ROOT); parser.add_argument("--output-dir", type=Path); parser.add_argument("--evidence", type=Path); parser.add_argument("--seed", type=int); parser.add_argument("--seeds", default=None); parser.add_argument("--allow-model-fits", action="store_true"); parser.add_argument("--allow-real-preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "native-smoke": result = {"status": "PASS", "model_fit": "NOT_PERFORMED", "native_api": e10r._native_snapshot(), "source_closure": source_closure(Path(args.project_root)), "threads": configure_cpu_threads(), "cpu_resource_qc": cpu_resource_estimate(n_rows=9568, n_features=19465, nnz=54807969)}
    elif args.mode == "fixture": result = fixture_cpu_smoke()
    elif args.mode == "preflight":
        if not args.allow_real_preflight: raise SystemExit("REFUSED: E10C real preflight requires --allow-real-preflight")
        result = real_input_preflight_cpu(project_root=args.project_root)
    elif args.output_dir is None: parser.error("--output-dir is required for fit/audit")
    elif args.mode == "fit":
        selected = tuple(int(v) for v in (args.seeds.split(",") if args.seeds else ([args.seed] if args.seed is not None else e10r.SEEDS))); result = fit_cpu_run(project_root=args.project_root, output_dir=args.output_dir, seeds=selected, allow_model_fits=args.allow_model_fits)
    else: result = audit_cpu_output(project_root=args.project_root, output_dir=args.output_dir)
    if args.evidence: exclusive_json(args.evidence, result)
    print(json.dumps({"status": result["status"], "mode": args.mode, "device": result.get("device", "cpu"), "model_fit": result.get("model_fit", "NOT_PERFORMED")}, sort_keys=True)); return 0 if result["status"] in {"PASS", "FIT_COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
