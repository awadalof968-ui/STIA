"""Executable, source-bound native STAligner runner for E10R.

The E8 adapter is the only raw-data reader.  This successor adds the missing
fit lifecycle to E10's readiness receipt: sparse log1p-CPM preparation,
per-library native KNN graphs, explicit seed/order arms, immutable checkpoints,
and an independent raw-arm/OSI auditor.  Fitting is fail-closed behind
``allow_model_fits`` and is intentionally not run by the E10R worker.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "precast-null-inspire-seeds-20260830"
E8_ADAPTER = ROOT / "stia" / "crossmethod" / "external_gse254652_adapter_v1_20260831.py"
E8_RECEIPT = RUN_ROOT / "evidence" / "E8" / "E8_GSE254652_SPARSE_ADAPTER_PREFLIGHT_20260831_r2.json"
H3_RESOURCE = RUN_ROOT / "evidence" / "H3" / "staligner_api_resource_20260830.json"
AUTHORITY = RUN_ROOT / "phases" / "terminal-audit" / "E10R.md"
METRIC = ROOT / "stia" / "crossmethod" / "osi.py"
CONTRACT = ROOT / "stia" / "crossmethod" / "frozen_contract.py"
PROTOCOL = "stia.external.gse254652.e10r.staligner_four_order.v1"
FEATURE_MODE = "all_common_19465"
ORDER_IDS = ("A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1")
SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
LATENT_DIM = 30
HIDDEN_DIMS = (512, 30)
N_EPOCHS = 1000
KNN_NEIGH = 10
NO_LABELS = {"disease": "NOT_CREATED", "batch": "NOT_CREATED", "physical_batch": "UNKNOWN"}


class E10RValidationError(ValueError):
    """Fail-closed source, identity, artifact, or terminal-contract error."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise E10RValidationError(f"missing JSON: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E10RValidationError(f"malformed JSON: {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise E10RValidationError(f"JSON object required: {target}")
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
    spec = importlib.util.spec_from_file_location("e8_external_sparse_adapter_for_e10r", E8_ADAPTER)
    if spec is None or spec.loader is None:
        raise E10RValidationError(f"cannot load E8 adapter: {E8_ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_snapshot() -> dict[str, Any]:
    expected = _json(H3_RESOURCE).get("native_api")
    if not isinstance(expected, Mapping):
        raise E10RValidationError("H3 native API resource missing")
    try:
        import STAligner
        from STAligner import train_STAligner
    except Exception as exc:  # pragma: no cover - environment dependent
        raise E10RValidationError(f"native STAligner import failed: {exc}") from exc
    train_path = Path(inspect.getfile(train_STAligner)).resolve()
    signature = str(inspect.signature(train_STAligner))
    if train_path != Path(str(expected["train_module_path"])).resolve() or sha256_file(train_path) != str(expected["train_module_sha256"]):
        raise E10RValidationError("native train_STAligner path/hash drift")
    if signature != str(expected["train_signature"]):
        raise E10RValidationError("native train_STAligner signature drift")
    deps: dict[str, str] = {}
    for distribution in ("torch", "torch_geometric", "numpy", "scipy", "anndata", "pandas"):
        deps[distribution] = str(importlib.metadata.version(distribution))
        expected_v = expected.get("dependency_versions", {}).get(distribution)
        if expected_v is not None and deps[distribution] != str(expected_v):
            raise E10RValidationError(f"dependency version drift: {distribution}={deps[distribution]}")
    return {"status": "PASS", "distribution": expected.get("distribution"), "distribution_version": expected.get("distribution_version"), "train_module_path": str(train_path), "train_module_sha256": sha256_file(train_path), "train_signature": signature, "dependency_versions": deps, "package_path": str(Path(STAligner.__file__).resolve()), "h3_resource": str(H3_RESOURCE), "h3_resource_sha256": sha256_file(H3_RESOURCE)}


def source_closure(root: Path = ROOT, *, require_authority: bool = False) -> dict[str, dict[str, str]]:
    paths = {"e10r_runner": root / Path(__file__).relative_to(root), "e8_adapter": E8_ADAPTER, "e8_receipt": E8_RECEIPT, "native_resource": H3_RESOURCE, "metric": METRIC, "frozen_contract": CONTRACT, "authority": AUTHORITY}
    out: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            if name == "authority" and not require_authority:
                continue
            raise E10RValidationError(f"missing source closure file {name}: {path}")
        out[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return out


def resource_estimate(*, n_rows: int, n_features: int, nnz: int, available_bytes: int | None = None) -> dict[str, Any]:
    """Conservative host/GPU accounting for native dense STAligner training."""
    n, p, z = int(n_rows), int(n_features), int(nnz)
    csr = z * (4 + 4) + (n + 1) * 8
    dense_x = n * p * 4
    hidden = n * HIDDEN_DIMS[0] * 4
    latent = n * LATENT_DIM * 4
    # Cal_Spatial_Net emits k directed neighbors plus one self-loop per node;
    # native STAligner consumes the resulting whole graph (there is no
    # minibatch sampler).  Attention message tensors are edge x C, with the
    # largest attention output C=512.  The final decoder conv4 has
    # ``attention=False`` and therefore allocates node x p, not edge x p.
    graph_edges = n * (KNN_NEIGH + 1)
    graph = graph_edges * 2 * 8 + n * 8
    attention_edge_message = graph_edges * HIDDEN_DIMS[0] * 4
    decoder_node_output = n * p * 4
    model_params = (p * HIDDEN_DIMS[0] + HIDDEN_DIMS[0] * HIDDEN_DIMS[1] + HIDDEN_DIMS[0] + HIDDEN_DIMS[1]) * 4
    optimizer = model_params * 2
    activations = (dense_x + hidden + latent) * 3 + attention_edge_message + decoder_node_output
    known = csr + dense_x + graph + model_params + optimizer + activations
    estimated = int(known * 1.5)  # allocator, autograd, CUDA and framework overhead
    inference_known = csr + graph + model_params + dense_x + hidden + latent + attention_edge_message + decoder_node_output
    if available_bytes is None:
        try:
            import psutil
            available_bytes = int(psutil.virtual_memory().available)
        except Exception:
            available_bytes = None
    return {"input_csr_bytes": int(csr), "dense_input_bytes": int(dense_x), "graph_edge_count": int(graph_edges), "graph_edge_bytes": int(graph), "attention_edge_message_bytes": int(attention_edge_message), "attention_message_out_features": HIDDEN_DIMS[0], "decoder_node_output_bytes": int(decoder_node_output), "decoder_edge_output_features": 0, "decoder_edge_semantics": "conv4 attention=False returns node x 19465 before propagate; no edge x 19465 allocation", "activation_bytes": int(activations), "model_parameter_bytes": int(model_params), "optimizer_state_bytes": int(optimizer), "known_peak_bytes": int(known), "estimated_peak_bytes": estimated, "inference_known_peak_bytes": int(inference_known), "inference_estimated_peak_bytes": int(inference_known * 1.5), "training_vs_inference": "training includes autograd and Adam state; inference omits optimizer/autograd", "batching": "whole_graph", "minibatch_supported": False, "allocator_framework_overhead_factor": 1.5, "available_bytes": available_bytes, "headroom_gate_pass": bool(available_bytes is not None and available_bytes >= estimated + 2 * 1024**3), "gpu_required": True, "native_dense_conversion": "STAligner.train_STAligner calls adata.X.todense()", "fit_only": True}


def sparse_log1p_cpm(matrix: Any) -> Any:
    """Apply accepted log1p-CPM semantics while retaining CSR storage."""
    import scipy.sparse as sp
    if not sp.isspmatrix_csr(matrix):
        matrix = matrix.tocsr(copy=True)
    else:
        matrix = matrix.copy()
    matrix = matrix.astype(np.float32, copy=False)
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    counts = np.diff(matrix.indptr)
    repeated = np.repeat(totals, counts)
    valid = repeated > 0
    values = matrix.data
    values[valid] = np.log1p(values[valid] / repeated[valid] * 1.0e4)
    values[~valid] = 0.0
    np.clip(values, 0.0, 50.0, out=values)
    matrix.eliminate_zeros()
    return matrix


def order_identity(bundle: Any, order_id: str) -> dict[str, Any]:
    sections = bundle.ordered_sections(order_id)
    ids = [spot for section in sections for spot in section.spot_ids]
    libraries = [section.library_id for section in sections for _ in section.spot_ids]
    animals = [section.animal_id for section in sections for _ in section.spot_ids]
    if len(ids) != 9568 or len(ids) != len(set(ids)):
        raise E10RValidationError(f"{order_id}: source IDs are not unique 9568 rows")
    rows = [int(i) for section in sections for i in range(section.row_offset[0], section.row_offset[1])]
    return {"order": order_id, "spot_ids": ids, "library_ids": libraries, "animal_ids": animals, "rows": rows, "order_gsm": [section.gsm for section in sections], "order_animals": [section.animal_id for section in sections], "rows_sha256": canonical_sha256(rows), "spot_ids_sha256": canonical_sha256(ids), "identity_sha256": canonical_sha256(list(zip(ids, libraries, animals)))}


def build_order_adata(bundle: Any, order_id: str, *, sparse_transform: bool = True) -> Any:
    """Materialize one source order and attach only slice partition metadata."""
    import pandas as pd
    import scipy.sparse as sp
    from anndata import AnnData
    sections = bundle.ordered_sections(order_id)
    blocks = []
    obs_rows: list[dict[str, Any]] = []
    coords: list[list[float]] = []
    for section in sections:
        block = section.matrix
        if not sp.issparse(block):
            raise E10RValidationError(f"{order_id}:{section.gsm} source block was densified")
        blocks.append(block.tocsr(copy=True))
        for i, spot in enumerate(section.spot_ids):
            obs_rows.append({"spot_id": str(spot), "library_id": str(section.library_id), "animal_id": str(section.animal_id), "batch_name": str(section.library_id)})
            coords.append([float(section.coordinates["array_row"][i]), float(section.coordinates["array_col"][i])])
    matrix = sp.vstack(blocks, format="csr")
    if sparse_transform:
        matrix = sparse_log1p_cpm(matrix)
    obs = pd.DataFrame(obs_rows).set_index("spot_id")
    adata = AnnData(X=matrix, obs=obs)
    adata.var_names = list(bundle.feature_ids)
    adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    return adata


def build_native_graph(adata: Any, *, k: int = KNN_NEIGH, graph_builder: Any | None = None) -> Any:
    """Build native STAligner per-library KNN blocks and global edgeList."""
    import scipy.sparse as sp
    if graph_builder is None:
        from STAligner import Cal_Spatial_Net
        graph_builder = Cal_Spatial_Net
    batches = np.asarray(adata.obs["batch_name"].astype(str).values)
    blocks = []
    n_total = int(adata.n_obs)
    for batch in list(dict.fromkeys(batches.tolist())):
        idx = np.flatnonzero(batches == batch)
        if idx.size < 2:
            raise E10RValidationError(f"library {batch} has fewer than two spots")
        sub = adata[idx].copy()
        local_k = min(int(k), int(idx.size - 1))
        graph_builder(sub, k_cutoff=local_k, max_neigh=local_k + 1, model="KNN", verbose=False)
        adj = sub.uns.get("adj")
        if adj is None or not sp.issparse(adj):
            raise E10RValidationError(f"library {batch} native graph did not return sparse adj")
        rows, cols = adj.nonzero()
        blocks.append(sp.coo_matrix((np.ones(rows.size, dtype=np.float32), (idx[rows], idx[cols])), shape=(n_total, n_total)))
    adjacency = sum(blocks, sp.coo_matrix((n_total, n_total))).tocsr()
    cross = 0
    for batch in list(dict.fromkeys(batches.tolist())):
        mask = batches == batch
        cross += int(adjacency[mask][:, ~mask].nnz)
    if cross:
        raise E10RValidationError(f"native graph contains {cross} cross-library edges")
    adata.uns["edgeList"] = adjacency.nonzero()
    return adata


def _seed_all(seed: int) -> dict[str, Any]:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1))
    out: dict[str, Any] = {"python": int(seed), "numpy": int(seed) % (2**32 - 1)}
    try:
        import torch
        torch.manual_seed(int(seed)); out["torch"] = int(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed)); out["torch_cuda"] = int(seed)
        else: out["torch_cuda"] = "not_available"
    except Exception as exc:  # pragma: no cover - optional test fixture
        out["torch"] = f"unavailable:{type(exc).__name__}"; out["torch_cuda"] = "unavailable"
    return out


def _fit_native(adata: Any, seed: int) -> np.ndarray:
    _seed_all(seed)
    import torch
    from STAligner import train_STAligner
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained = train_STAligner(adata, hidden_dims=list(HIDDEN_DIMS), n_epochs=N_EPOCHS, lr=0.001, random_seed=int(seed), knn_neigh=KNN_NEIGH, device=device, verbose=False)
    latent = np.asarray(trained.obsm["STAligner"], dtype=np.float32)
    if latent.ndim != 2 or latent.shape != (adata.n_obs, LATENT_DIM) or not np.isfinite(latent).all():
        raise E10RValidationError(f"native latent shape/finite failure: {latent.shape}")
    return latent


def _write_arm(destination: Path, *, latent: np.ndarray, identity: Mapping[str, Any], seed: int, order_id: str, base: Mapping[str, Any]) -> dict[str, Any]:
    arm = destination / "arms" / f"seed{int(seed)}"; npz = arm / f"{order_id}.npz"; sidecar = arm / f"{order_id}.json"
    ids = np.asarray(identity["spot_ids"], dtype="U"); libraries = np.asarray(identity["library_ids"], dtype="U"); animals = np.asarray(identity["animal_ids"], dtype="U")
    if latent.shape != (ids.size, LATENT_DIM) or len(set(ids.tolist())) != ids.size or not np.isfinite(latent).all():
        raise E10RValidationError(f"{order_id}:{seed}: latent/identity shape failure")
    exclusive_npz(npz, latent=np.asarray(latent, dtype=np.float32), spot_ids=ids, library_ids=libraries, donor_keys=animals)
    record = dict(base); record.update({"seed": int(seed), "order": order_id, "npz": str(npz), "npz_sha256": sha256_file(npz), "latent_shape": list(latent.shape), "latent_dtype": "float32", "identity": {key: identity[key] for key in ("rows_sha256", "spot_ids_sha256", "identity_sha256", "order_gsm", "order_animals")}})
    exclusive_json(sidecar, record)
    return record


def _metric(latents: Mapping[str, np.ndarray], donors: Mapping[str, Sequence[str]], seed: int) -> dict[str, Any]:
    from stia.crossmethod.osi import compute_osi
    result = compute_osi(latents, donors, condition="gse254652_p1_external_staligner", seed=int(seed), margin=0.05)
    if len(result.get("pairwise_drifts", ())) != 6:
        raise E10RValidationError("frozen OSI requires six pairwise order contrasts")
    return result


def fit_run(*, project_root: Path | str = ROOT, output_dir: Path | str, seeds: Sequence[int] = SEEDS, h5ad_path: Path | str | None = None, manifest_path: Path | str | None = None, allow_model_fits: bool = False) -> dict[str, Any]:
    if not allow_model_fits:
        raise RuntimeError("REFUSED: E10R fit requires --allow-model-fits; no model fit was started")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing output collision: {destination}")
    root = Path(project_root).resolve(); e8 = _load_e8(); receipt = _json(E8_RECEIPT)
    h5ad = Path(h5ad_path or receipt["h5ad"]["path"]).resolve(); manifest_path = Path(manifest_path or receipt["manifest"]["path"]).resolve()
    closure = source_closure(root, require_authority=True); native = _native_snapshot()
    selected = tuple(int(v) for v in seeds)
    if not selected or any(v not in SEEDS for v in selected) or len(set(selected)) != len(selected):
        raise E10RValidationError("seeds must be a unique subset of the frozen five seeds")
    destination.mkdir(parents=True)
    with e8.open_bundle(root, h5ad_path=h5ad, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        identities = {order: order_identity(bundle, order) for order in ORDER_IDS}
        resource = resource_estimate(n_rows=9568, n_features=len(bundle.feature_ids), nnz=54807969)
        if not resource["headroom_gate_pass"]:
            raise RuntimeError("REFUSED: E10R estimated native STAligner peak plus 2 GiB headroom unavailable")
        config = {"protocol": PROTOCOL, "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "hidden_dims": list(HIDDEN_DIMS), "n_epochs": N_EPOCHS, "pretrain_epochs": 500, "alignment_epochs": 500, "lr": 0.001, "knn_neigh": KNN_NEIGH, "orders": list(ORDER_IDS), "seeds": list(SEEDS), "requested_seeds": list(selected), "preprocessing": "log1p-CPM on sparse CSR counts", "graph": "native Cal_Spatial_Net per-library KNN k=10, block diagonal", "labels": NO_LABELS}
        manifest = {"schema_version": PROTOCOL, "status": "FIT_STARTED", "model_fit": "STARTED", "output_dir": str(destination), "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)), "h5ad": {"path": str(h5ad), "sha256": bundle.h5ad_sha256, "shape": [9568, 19465], "sparse_format": "CSR"}, "input_manifest": {"path": str(manifest_path), "sha256": bundle.manifest_sha256}, "source_closure": closure, "native_api": native, "config": config, "orders": {name: {k: value for k, value in identity.items() if k not in {"spot_ids", "library_ids", "animal_ids", "rows"}} for name, identity in identities.items()}, "resource_qc": resource, "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN", "physical_batch_axis": "UNKNOWN"}, "started_at": _dt.datetime.now().astimezone().isoformat()}
        exclusive_json(destination / "manifest.json", manifest)
        all_records: dict[str, Any] = {}
        for seed in selected:
            latents: dict[str, np.ndarray] = {}; donors: dict[str, np.ndarray] = {}; records: dict[str, Any] = {}
            for order in ORDER_IDS:
                adata = build_order_adata(bundle, order)
                adata = build_native_graph(adata, k=KNN_NEIGH)
                latent = _fit_native(adata, seed)
                ident = identities[order]; donors[order] = np.asarray(ident["animal_ids"], dtype="U"); latents[order] = latent
                records[order] = _write_arm(destination, latent=latent, identity=ident, seed=seed, order_id=order, base={"schema_version": PROTOCOL, "protocol": PROTOCOL, "feature_mode": FEATURE_MODE, "feature_count": len(bundle.feature_ids), "input_h5ad_sha256": bundle.h5ad_sha256, "input_manifest_sha256": bundle.manifest_sha256, "source_closure": closure, "seed": int(seed), "model_seed": int(seed), "labels": NO_LABELS})
            metric = _metric(latents, donors, seed)
            checkpoint = {"schema_version": PROTOCOL, "status": "SEED_COMPLETE", "seed": int(seed), "orders": records, "metric": metric, "completed_at": _dt.datetime.now().astimezone().isoformat()}
            exclusive_json(destination / f"checkpoint_seed{seed}.json", checkpoint)
            all_records[str(seed)] = checkpoint
        summary = {"schema_version": PROTOCOL, "status": "FIT_COMPLETE", "model_fit": "PERFORMED", "requested_seeds": list(selected), "expected_orders": list(ORDER_IDS), "completed_seeds": [int(v) for v in selected], "seed_checkpoints": all_records, "terminal": {"expected_seed_count": len(selected), "expected_order_count": len(selected) * len(ORDER_IDS), "completion_marker": "COMPLETED.json"}, "completed_at": _dt.datetime.now().astimezone().isoformat()}
        exclusive_json(destination / "run_summary.json", summary)
        exclusive_json(destination / "COMPLETED.json", {"schema_version": PROTOCOL, "status": "COMPLETE", "summary_sha256": sha256_file(destination / "run_summary.json"), "completed_at": summary["completed_at"]})
        return summary


def _check_arm(npz: Path, sidecar: Path, identity: Mapping[str, Any], seed: int, order: str, closure: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    metadata = _json(sidecar)
    if metadata.get("seed") != int(seed) or metadata.get("order") != order or metadata.get("npz_sha256") != sha256_file(npz) or metadata.get("source_closure") != closure:
        raise E10RValidationError(f"arm provenance mismatch: {order}:{seed}")
    with np.load(npz, allow_pickle=False) as payload:
        expected_keys = {"latent", "spot_ids", "library_ids", "donor_keys"}
        if set(payload.files) != expected_keys:
            raise E10RValidationError(f"arm keys mismatch: {order}:{seed}")
        latent = np.asarray(payload["latent"]); ids = payload["spot_ids"].astype(str).tolist(); libraries = payload["library_ids"].astype(str).tolist(); donors = payload["donor_keys"].astype(str).tolist()
    if latent.shape != (len(ids), LATENT_DIM) or latent.dtype.kind not in "fc" or not np.isfinite(latent).all():
        raise E10RValidationError(f"arm latent failure: {order}:{seed}")
    for field, actual, key in (("spot_ids", ids, "spot_ids"), ("library_ids", libraries, "library_ids"), ("donor_keys", donors, "animal_ids")):
        if actual != list(identity[key]):
            raise E10RValidationError(f"source-derived {field} mismatch: {order}:{seed}")
    if metadata.get("identity", {}).get("identity_sha256") != identity["identity_sha256"]:
        raise E10RValidationError(f"identity fingerprint mismatch: {order}:{seed}")
    return latent, np.asarray(donors, dtype="U")


def audit_output(*, project_root: Path | str = ROOT, output_dir: Path | str, h5ad_path: Path | str | None = None, manifest_path: Path | str | None = None) -> dict[str, Any]:
    destination = Path(output_dir).resolve(); root = Path(project_root).resolve(); manifest_file = destination / "manifest.json"; summary_file = destination / "run_summary.json"; completed = destination / "COMPLETED.json"
    if not manifest_file.is_file() or not summary_file.is_file() or not completed.is_file():
        raise E10RValidationError("terminal manifest/summary/completion marker missing")
    saved = _json(manifest_file); summary = _json(summary_file)
    if saved.get("schema_version") != PROTOCOL or saved.get("status") != "FIT_STARTED" or summary.get("status") != "FIT_COMPLETE" or summary.get("expected_orders") != list(ORDER_IDS):
        raise E10RValidationError("terminal protocol/status/order contract drift")
    closure = source_closure(root, require_authority=True)
    if saved.get("source_closure") != closure:
        raise E10RValidationError("source closure drift")
    e8 = _load_e8(); receipt = _json(E8_RECEIPT); h5ad = Path(h5ad_path or receipt["h5ad"]["path"]).resolve(); manifest_path = Path(manifest_path or receipt["manifest"]["path"]).resolve()
    with e8.open_bundle(root, h5ad_path=h5ad, manifest_path=manifest_path, feature_mode=FEATURE_MODE) as bundle:
        if saved.get("h5ad", {}).get("sha256") != bundle.h5ad_sha256 or saved.get("input_manifest", {}).get("sha256") != bundle.manifest_sha256:
            raise E10RValidationError("terminal input hash drift")
        identities = {order: order_identity(bundle, order) for order in ORDER_IDS}
        cells = []; errors = []
        for seed in summary.get("completed_seeds", []):
            latents: dict[str, np.ndarray] = {}; donors: dict[str, np.ndarray] = {}
            try:
                for order in ORDER_IDS:
                    npz = destination / "arms" / f"seed{int(seed)}" / f"{order}.npz"; sidecar = npz.with_suffix(".json")
                    if not npz.is_file() or not sidecar.is_file(): raise E10RValidationError(f"missing arm {order}:{seed}")
                    latents[order], donors[order] = _check_arm(npz, sidecar, identities[order], int(seed), order, closure)
                metric = _metric(latents, donors, int(seed)); cells.append({"seed": int(seed), "execution_status": "PASS", "numeric_status": str(metric["status"]), "osi": float(metric["osi"]), "pair_count": len(metric["pairwise_drifts"])})
            except Exception as exc:
                errors.append(f"seed{seed}: {exc}"); cells.append({"seed": int(seed), "execution_status": "UNRESOLVED", "numeric_status": "UNRESOLVED", "reason": str(exc)})
    verdict = "PASS" if not errors and len(cells) == len(summary.get("completed_seeds", [])) else "UNRESOLVED"
    return {"schema_version": PROTOCOL, "status": verdict, "execution_validity": "PASS" if verdict == "PASS" else "UNRESOLVED", "output_dir": str(destination), "cells": cells, "errors": errors, "source_closure_verified": not errors, "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}, "audited_at": _dt.datetime.now().astimezone().isoformat()}


def fixture_graph_preflight() -> dict[str, Any]:
    """Bounded graph/preprocessing path used by tests; no real input or fit."""
    import scipy.sparse as sp
    from anndata import AnnData
    class FakeGraph:
        def __call__(self, adata: Any, **_: Any) -> None:
            n = adata.n_obs; rows = np.arange(n); cols = (rows + 1) % n
            adata.uns["adj"] = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(n, n)) + sp.eye(n)
    X = sp.csr_matrix(np.arange(24, dtype=np.float32).reshape(6, 4)); adata = AnnData(X=sparse_log1p_cpm(X), obs={"batch_name": ["l1"] * 3 + ["l2"] * 3}); adata.obsm["spatial"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    build_native_graph(adata, k=2, graph_builder=FakeGraph())
    return {"status": "PASS", "shape": list(adata.shape), "sparse_format": adata.X.getformat(), "edge_count": int(len(adata.uns["edgeList"][0])), "cross_library_edges": 0, "model_fit": "NOT_PERFORMED"}


def real_input_preflight(*, project_root: Path | str = ROOT, h5ad_path: Path | str | None = None, manifest_path: Path | str | None = None) -> dict[str, Any]:
    """Traverse the production sparse/preprocessing/graph path, never train.

    This is intentionally a separately gated command: native ``Cal_Spatial_Net``
    densifies each library while constructing ``adj`` and the full path can
    consume several GiB.  E10R did not invoke it while the external PCA lane
    was live; the parent may run it after a fresh resource census.
    """
    root = Path(project_root).resolve(); e8 = _load_e8(); receipt = _json(E8_RECEIPT)
    h5ad = Path(h5ad_path or receipt["h5ad"]["path"]).resolve(); manifest = Path(manifest_path or receipt["manifest"]["path"]).resolve()
    native = _native_snapshot(); orders: dict[str, Any] = {}
    with e8.open_bundle(root, h5ad_path=h5ad, manifest_path=manifest, feature_mode=FEATURE_MODE) as bundle:
        for order in ORDER_IDS:
            adata = build_order_adata(bundle, order, sparse_transform=True)
            before = adata.X.getformat()
            build_native_graph(adata, k=KNN_NEIGH)
            orders[order] = {"shape": [int(v) for v in adata.shape], "sparse_format_after_preprocess": before, "edge_count": int(len(adata.uns["edgeList"][0])), "n_libraries": int(adata.obs["batch_name"].nunique()), "cross_library_edges": 0}
    return {"status": "PASS", "model_fit": "NOT_PERFORMED", "orders": orders, "native_api": native, "resource_qc": resource_estimate(n_rows=9568, n_features=19465, nnz=54807969), "claim_boundary": {"external_validation": "NOT_PERFORMED", "disease_or_batch_causal_claim": "FORBIDDEN"}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E10R native external STAligner fit/audit")
    parser.add_argument("--mode", choices=("fit", "audit", "fixture-preflight", "native-smoke", "real-preflight"), default="native-smoke")
    parser.add_argument("--project-root", type=Path, default=ROOT); parser.add_argument("--output-dir", type=Path); parser.add_argument("--evidence", type=Path); parser.add_argument("--seed", type=int); parser.add_argument("--seeds", default=None); parser.add_argument("--allow-model-fits", action="store_true"); parser.add_argument("--allow-real-preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "fixture-preflight":
        result = fixture_graph_preflight()
    elif args.mode == "real-preflight":
        if not args.allow_real_preflight:
            raise SystemExit("REFUSED: real-preflight requires --allow-real-preflight; no model fit is started")
        result = real_input_preflight(project_root=args.project_root)
    elif args.mode == "native-smoke":
        result = {"status": "PASS", "model_fit": "NOT_PERFORMED", "native_api": _native_snapshot(), "source_closure": source_closure(Path(args.project_root)), "resource_qc": resource_estimate(n_rows=9568, n_features=19465, nnz=54807969), "claim_boundary": {"model_fit": "NOT_PERFORMED"}}
    elif args.output_dir is None:
        parser.error("--output-dir is required for fit/audit")
    elif args.mode == "fit":
        selected = tuple(int(v) for v in (args.seeds.split(",") if args.seeds else ([args.seed] if args.seed is not None else SEEDS)))
        result = fit_run(project_root=args.project_root, output_dir=args.output_dir, seeds=selected, allow_model_fits=args.allow_model_fits)
    else:
        result = audit_output(project_root=args.project_root, output_dir=args.output_dir)
    if args.evidence:
        exclusive_json(args.evidence, result)
    print(json.dumps({"status": result["status"], "mode": args.mode, "model_fit": result.get("model_fit", result.get("claim_boundary", {}).get("model_fit", "NOT_PERFORMED"))}, sort_keys=True))
    return 0 if result["status"] in {"PASS", "FIT_COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
