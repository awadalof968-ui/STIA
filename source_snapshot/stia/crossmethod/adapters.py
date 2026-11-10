"""Frozen A-D runners that reuse the existing project adapter functions.

The project scripts are loaded lazily from an explicit project root.  This
keeps smoke tests independent of heavyweight model environments while making
the production path auditable: the output records the exact source script
hash and protocol parameters.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .frozen_contract import (
    FROZEN_COHORT,
    FROZEN_SEEDS,
    METHODS,
    METHOD_PROTOCOLS,
    OSI_PROTOCOL_VERSION,
    ORDER_INDICES,
    PROTOCOL_VERSION,
    IdentityError,
    load_frozen_sidecar,
    load_panel,
    obs_names,
    obs_values,
    order_rows,
    project_paths,
    validate_frozen_identity,
)
from .osi import compute_osi


@contextmanager
def _legacy_import_stdout_guard():
    """Isolate legacy adapters that replace ``sys.stdout`` at import time."""
    original = sys.stdout
    sink = io.BytesIO()
    isolated = io.TextIOWrapper(sink, encoding="utf-8")
    try:
        sys.stdout = isolated
        yield
    finally:
        sys.stdout = original
        try:
            isolated.detach()
        except Exception:
            pass


def load_project_adapter(project_root: Path | str, method: str) -> tuple[Any, Path, str]:
    if method not in METHODS:
        raise ValueError(f"unsupported C1 method {method!r}")
    root = Path(project_root)
    source = root / METHOD_PROTOCOLS[method]["source_adapter"]
    if not source.is_file():
        raise IdentityError(f"existing project adapter is missing: {source}")
    name = f"stia_c1_existing_{method.lower()}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise IdentityError(f"cannot load existing adapter: {source}")
    module = importlib.util.module_from_spec(spec)
    with _legacy_import_stdout_guard():
        spec.loader.exec_module(module)
    return module, source, hashlib.sha256(source.read_bytes()).hexdigest()


def _matrix(adata: Any) -> np.ndarray:
    value = adata.X
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def _panel_matrix(adata: Any, panel_genes: Sequence[str]) -> np.ndarray:
    names = [str(value) for value in list(adata.var_names)]
    lookup = {name: index for index, name in enumerate(names)}
    x = _matrix(adata)
    out = np.zeros((x.shape[0], len(panel_genes)), dtype=np.float32)
    for j, gene in enumerate(panel_genes):
        index = lookup.get(str(gene))
        if index is not None:
            out[:, j] = x[:, index]
    totals = np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return np.clip(np.log1p(out / totals * 1e4), 0, 50).astype(np.float32)


def _adata_with_panel(adata: Any, panel_genes: Sequence[str], row_indices: Sequence[int]) -> Any:
    """Make the same panel/log1p-CPM input used by existing project adapters."""
    try:
        import anndata as ad
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover - only production path
        raise RuntimeError("anndata and scipy are required for a model fit") from exc
    rows = list(row_indices)
    matrix = _panel_matrix(adata, panel_genes)[rows]
    obs = adata.obs.iloc[rows].copy() if hasattr(adata.obs, "iloc") else adata.obs[rows]
    names = np.asarray(obs_names(adata), dtype=str)[rows]
    spatial = np.asarray(adata.obsm["spatial"])[rows]
    result = ad.AnnData(X=sp.csr_matrix(matrix), obs=obs, obsm={"spatial": spatial})
    result.var_names = list(panel_genes)
    result.obs_names = names
    return result


def validate_case(
    project_root: Path | str | None = None, *, cohort: str = FROZEN_COHORT, strict: bool = True,
) -> dict[str, Any]:
    if cohort != FROZEN_COHORT:
        raise IdentityError(f"C1 is frozen to cohort {FROZEN_COHORT!r}; got {cohort!r}")
    paths = project_paths(project_root)
    sidecar = load_frozen_sidecar(project_root, cohort)
    panel, panel_sha = load_panel(project_root)
    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover - external environment
        raise RuntimeError("anndata is required for frozen input validation") from exc
    if not paths["cohort_h5ad"].is_file():
        raise IdentityError(f"missing frozen cohort H5AD: {paths['cohort_h5ad']}")
    adata = ad.read_h5ad(paths["cohort_h5ad"])
    identity = validate_frozen_identity(
        adata, sidecar=sidecar, panel_genes=panel, panel_sha256=panel_sha, strict=strict
    )
    identity.update({
        "cohort": cohort,
        "sidecar_sha256": hashlib.sha256(paths["cohort_sidecar"].read_bytes()).hexdigest(),
        "h5ad_path": str(paths["cohort_h5ad"]),
        "panel_path": str(paths["panel"]),
    })
    return identity


def fit_one_order(
    method: str,
    adata: Any,
    panel_genes: Sequence[str],
    arm: str,
    seed: int,
    *,
    project_root: Path | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit exactly one method/order; returns latent and donor keys in row order."""
    module, _, _ = load_project_adapter(project_root, method)
    rows = order_rows(adata, arm, require_frozen=True)
    prepared = _adata_with_panel(adata, panel_genes, rows)
    donors = np.asarray(obs_values(adata, "donor"), dtype=str)[rows]
    if method == "PCA":
        latent = module.fit_pca(np.asarray(prepared.X.toarray()), seed)
    elif method == "Harmony":
        batch = np.asarray(obs_values(adata, "proc_batch"), dtype=str)[rows]
        latent = module.fit_harmony(np.asarray(prepared.X.toarray()), batch, seed)
    elif method == "STAligner":
        prepared = module.build_graph(prepared, k=METHOD_PROTOCOLS[method]["parameters"]["knn_neigh"])
        latent = module.train_one(prepared, seed)
    else:  # pragma: no cover - guarded above
        raise ValueError(method)
    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim != 2 or latent.shape[0] != len(rows) or not np.isfinite(latent).all():
        raise RuntimeError(f"{method} adapter returned invalid latent shape {latent.shape}")
    return latent, donors


def run_frozen_osi(
    method: str,
    *,
    project_root: Path | str | None = None,
    cohort: str = FROZEN_COHORT,
    seeds: Sequence[int] = FROZEN_SEEDS,
    output_dir: Path | str | None = None,
    allow_model_fits: bool = False,
) -> dict[str, Any]:
    """Run the frozen A-D contract, only when explicitly enabled.

    C1 never invokes this with ``allow_model_fits=True``.  The explicit guard
    prevents a validation/smoke command from silently launching a 20-fit
    batch.  Each incomplete seed is UNRESOLVED; partial results are retained.
    """
    if method not in METHODS:
        raise ValueError(f"unsupported C1 method {method!r}")
    if not allow_model_fits:
        raise RuntimeError("model fits are guarded; pass allow_model_fits=True in the parent-scheduled run")
    if tuple(seeds) != FROZEN_SEEDS:
        raise IdentityError("C1 locked target requires exactly the five frozen seeds")
    root = Path(project_root) if project_root is not None else project_paths()["root"]
    validation = validate_case(root, cohort=cohort, strict=True)
    panel, panel_sha = load_panel(root)
    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anndata is required for model fits") from exc
    adata = ad.read_h5ad(project_paths(root)["cohort_h5ad"])
    source_module, source_path, source_sha = load_project_adapter(root, method)
    del source_module
    rows = []
    for seed in seeds:
        latents: dict[str, np.ndarray] = {}
        donors: dict[str, np.ndarray] = {}
        errors = {}
        started = time.time()
        for arm in ORDER_INDICES:
            try:
                z, d = fit_one_order(method, adata, panel, arm, int(seed), project_root=root)
                latents[arm], donors[arm] = z, d
            except Exception as exc:  # retain first failure; no implicit PASS
                errors[arm] = f"{type(exc).__name__}: {exc}"
        if errors:
            row = {
                "condition": cohort,
                "seed": int(seed),
                "status": "UNRESOLVED",
                "orders_expected": list(ORDER_INDICES),
                "orders_ok": sorted(latents),
                "errors": errors,
            }
        else:
            row = compute_osi(latents, donors, condition=cohort, seed=int(seed))
        row["runtime_seconds"] = round(time.time() - started, 3)
        rows.append(row)
    output = {
        "document": "C1 frozen A-D cross-method OSI output",
        "protocol_version": OSI_PROTOCOL_VERSION,
        "adapter_contract_version": PROTOCOL_VERSION,
        "method": method,
        "cohort": cohort,
        "seeds": list(seeds),
        "margin": 0.05,
        "method_protocol": METHOD_PROTOCOLS[method],
        "source_adapter": str(source_path),
        "source_adapter_sha256": source_sha,
        "panel_sha256": panel_sha,
        "identity": validation,
        "rows": rows,
        "summary": {
            "n_total": len(rows),
            "n_pass": sum(row.get("status") == "PASS" for row in rows),
            "n_fail": sum(row.get("status") == "FAIL" for row in rows),
            "n_unresolved": sum(row.get("status") == "UNRESOLVED" for row in rows),
            "verdict": "UNRESOLVED" if any(row.get("status") == "UNRESOLVED" for row in rows) else "COMPLETE",
        },
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / f"{method.lower()}_{cohort}_c1.json"
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return output
