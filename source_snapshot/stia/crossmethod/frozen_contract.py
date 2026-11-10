"""Frozen cross-method identity and order contract for C1.

This module is deliberately dependency-light.  It describes the identities
that a future model run must prove before fitting anything; it does not import
anndata, torch, harmonypy, or STAligner at import time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "C1.0.0"
OSI_PROTOCOL_VERSION = "INSPIRE-E11-AD-1.0.0"
OSI_MARGIN = 0.05
PANEL_SHA256 = "c35736c8bf64c7dccb55735387a54d13641ac6214cf0f8149846cb0738c4c334"
FROZEN_COHORT = "balanced_orthogonal_diffuse_ds10"
FROZEN_SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
FROZEN_DONORS = ("AD0", "AD1", "AD2", "AD3", "AD4", "CT0", "CT1", "CT2", "CT3", "CT4")
FROZEN_BATCHES = tuple(f"d{i}" for i in range(10))
FROZEN_SPOTS_PER_DONOR = 1500
FROZEN_N_SPOTS = 15000
FROZEN_PANEL_SIZE = 3000

ORDER_INDICES = {
    "A_original": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    "B_interleaved": (2, 7, 0, 9, 4, 6, 1, 8, 3, 5),
    "C_hash_batch_v1": (9, 5, 1, 3, 2, 0, 4, 7, 8, 6),
    "D_hash_batch_v1": (4, 3, 6, 8, 0, 9, 2, 5, 7, 1),
}
ORDER_BATCHES = {
    arm: tuple(FROZEN_BATCHES[i] for i in indices)
    for arm, indices in ORDER_INDICES.items()
}

REQUIRED_OBS_FIELDS = ("donor", "batch", "proc_batch", "group", "Layer")
METHODS = ("PCA", "Harmony", "STAligner")
METHOD_PROTOCOLS = {
    "PCA": {
        "protocol_id": "PCA30_LOG1P_CPM_PANEL3000_C1_v1",
        "source_adapter": "scripts/_x3_pca_harmony_osi.py",
        "representation": "PCA(30)",
        "model_seed": 0,
        "parameters": {"pca_dim": 30, "preprocessing": "log1p-CPM", "panel_size": 3000},
    },
    "Harmony": {
        "protocol_id": "HARMONY_PCA30_LOG1P_CPM_THETA2_SIGMA0.1_ITER20_C1_v1",
        "source_adapter": "scripts/_x3_pca_harmony_osi.py",
        "representation": "Harmony-corrected PCA(30)",
        "model_seed": 0,
        "parameters": {
            "pca_dim": 30,
            "preprocessing": "log1p-CPM",
            "theta": 2.0,
            "sigma": 0.1,
            "max_iter_harmony": 20,
            "batch_key": "proc_batch",
            "panel_size": 3000,
        },
    },
    "STAligner": {
        "protocol_id": "STALIGNER_V2_BLOCKDIAG_20260813_C1_AD_v1",
        "source_adapter": "scripts/_x3_staligner_osi.py",
        "representation": "STAligner native latent",
        "model_seed": 0,
        "parameters": {
            "hidden_dims": [512, 30],
            "n_epochs": 1000,
            "knn_neigh": 10,
            "graph": "per-slice KNN block-diagonal",
            "preprocessing": "log1p-CPM",
            "panel_size": 3000,
        },
    },
}


class IdentityError(ValueError):
    """Raised when a frozen input cannot be proven comparable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def default_project_root() -> Path:
    return Path(r"LOCAL_INPUT_ROOT\Downloads\INSPIRE-main\INSPIRE-main\INSPIRE")


def project_paths(project_root: Path | str | None = None) -> dict[str, Path]:
    root = Path(project_root) if project_root is not None else default_project_root()
    frozen = root / "results" / "external" / "x3" / "frozen_stageB"
    return {
        "root": root,
        "frozen_dir": frozen,
        "cohort_h5ad": frozen / f"{FROZEN_COHORT}.h5ad",
        "cohort_sidecar": frozen / f"{FROZEN_COHORT}.json",
        "panel": root / "results" / "tech_batch" / "panel_genes.json",
    }


def load_frozen_sidecar(project_root: Path | str | None = None, cohort: str = FROZEN_COHORT) -> dict[str, Any]:
    paths = project_paths(project_root)
    sidecar = paths["frozen_dir"] / f"{cohort}.json"
    if not sidecar.is_file():
        raise IdentityError(f"missing frozen cohort sidecar: {sidecar}")
    try:
        info = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed external sidecar
        raise IdentityError(f"invalid frozen cohort sidecar: {sidecar}: {exc}") from exc
    expected = {
        "scenario": "balanced_orthogonal",
        "geometry": "diffuse",
        "data_seed": 10,
        "n_spots": FROZEN_N_SPOTS,
        "n_donors": len(FROZEN_DONORS),
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise IdentityError(f"frozen sidecar {key}={info.get(key)!r}; expected {value!r}")
    if tuple(info.get("donors", ())) != FROZEN_DONORS:
        raise IdentityError("frozen sidecar donor identity mismatch")
    if not isinstance(info.get("input_hash"), str) or len(info["input_hash"]) != 64:
        raise IdentityError("frozen sidecar input_hash is missing or malformed")
    return info


def load_panel(project_root: Path | str | None = None) -> tuple[list[str], str]:
    path = project_paths(project_root)["panel"]
    if not path.is_file():
        raise IdentityError(f"missing frozen feature panel: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        genes = list(payload["genes"])
    except Exception as exc:  # pragma: no cover - malformed external panel
        raise IdentityError(f"invalid feature panel: {path}: {exc}") from exc
    actual = sha256_file(path)
    if actual != PANEL_SHA256:
        raise IdentityError(f"feature panel hash mismatch: {actual} != {PANEL_SHA256}")
    if len(genes) != FROZEN_PANEL_SIZE or len(set(genes)) != len(genes):
        raise IdentityError("feature panel must contain 3000 unique ordered genes")
    return genes, actual


def _values(column: Any) -> list[str]:
    """Convert pandas-like or fixture columns to strings without pandas."""
    if hasattr(column, "to_numpy"):
        column = column.to_numpy()
    if hasattr(column, "values"):
        column = column.values
    return [str(item) for item in list(column)]


def obs_values(adata: Any, field: str) -> list[str]:
    try:
        return _values(adata.obs[field])
    except Exception as exc:
        raise IdentityError(f"missing obs field {field!r}") from exc


def obs_names(adata: Any) -> list[str]:
    value = getattr(adata, "obs_names", None)
    if value is None:
        raise IdentityError("adata.obs_names is required")
    return _values(value)


def _stable_rows(names: Sequence[str], columns: Mapping[str, Sequence[str]]) -> list[list[str]]:
    rows = []
    for i, name in enumerate(names):
        rows.append([str(name)] + [str(columns[field][i]) for field in sorted(columns)])
    return sorted(rows, key=lambda row: row[0])


def order_rows(adata: Any, arm: str, *, require_frozen: bool = True) -> list[int]:
    """Return exact donor-block row indices for one frozen A-D arm.

    Within-donor order is preserved.  The arm is checked against first-seen
    batch labels, so a caller cannot accidentally substitute batch-sorted or
    dynamic/rotating orders.
    """
    if arm not in ORDER_INDICES:
        raise IdentityError(f"unknown frozen order arm: {arm}")
    donors = obs_values(adata, "donor")
    batches = obs_values(adata, "batch")
    expected_donors = FROZEN_DONORS if require_frozen else tuple(dict.fromkeys(donors))
    if len(expected_donors) != len(ORDER_INDICES[arm]):
        raise IdentityError("order contract requires exactly ten donor blocks")
    blocks: list[list[int]] = []
    for donor_index in ORDER_INDICES[arm]:
        donor = expected_donors[donor_index]
        idx = [i for i, value in enumerate(donors) if value == donor]
        if not idx:
            raise IdentityError(f"{arm}: empty donor block {donor}")
        blocks.append(idx)
    result = [item for block in blocks for item in block]
    first_seen: list[str] = []
    for row in result:
        if batches[row] not in first_seen:
            first_seen.append(batches[row])
    expected_batches = ORDER_BATCHES[arm]
    if tuple(first_seen) != expected_batches:
        raise IdentityError(f"{arm}: first-seen batch order {first_seen!r} != {list(expected_batches)!r}")
    if len(set(result)) != len(result) or len(result) != len(donors):
        raise IdentityError(f"{arm}: row permutation does not retain every spot exactly once")
    return result


def validate_frozen_identity(
    adata: Any,
    *,
    sidecar: Mapping[str, Any] | None = None,
    panel_genes: Sequence[str] | None = None,
    panel_sha256: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate labels/spot identity and return hashes used in provenance.

    ``strict=False`` is only for bounded synthetic smoke fixtures; it retains
    all structural checks and never authorizes a real model run.
    """
    names = obs_names(adata)
    if len(set(names)) != len(names):
        raise IdentityError("spot IDs are not unique")
    columns = {field: obs_values(adata, field) for field in REQUIRED_OBS_FIELDS}
    if any(len(values) != len(names) for values in columns.values()):
        raise IdentityError("spot identity columns have inconsistent lengths")
    donors = columns["donor"]
    batches = columns["batch"]
    if strict:
        if len(names) != FROZEN_N_SPOTS:
            raise IdentityError(f"n_spots={len(names)}; expected {FROZEN_N_SPOTS}")
        if set(donors) != set(FROZEN_DONORS):
            raise IdentityError("donor labels do not match frozen ten-donor identity")
        if set(batches) != set(FROZEN_BATCHES):
            raise IdentityError("batch labels do not match frozen d0..d9 identity")
        counts = {donor: donors.count(donor) for donor in FROZEN_DONORS}
        if any(value != FROZEN_SPOTS_PER_DONOR for value in counts.values()):
            raise IdentityError(f"donor spot counts do not match 1500 each: {counts}")
        if tuple(columns["group"][i] for i in range(len(names)) if donors[i].startswith("AD")) and any(
            columns["group"][i] != "AD" for i in range(len(names)) if donors[i].startswith("AD")
        ):
            raise IdentityError("AD donor/group identity mismatch")
        if any(columns["group"][i] != "CT" for i in range(len(names)) if donors[i].startswith("CT")):
            raise IdentityError("CT donor/group identity mismatch")
    if panel_genes is not None:
        if len(panel_genes) != (FROZEN_PANEL_SIZE if strict else len(panel_genes)):
            raise IdentityError("panel gene count mismatch")
        if strict and panel_sha256 != PANEL_SHA256:
            raise IdentityError("panel hash is not the frozen 3000-gene hash")
    coordinates = getattr(adata, "obsm", {}).get("spatial") if hasattr(getattr(adata, "obsm", {}), "get") else None
    if coordinates is None:
        raise IdentityError("obsm['spatial'] is required")
    try:
        coordinate_rows = [[float(v) for v in row] for row in coordinates]
    except Exception as exc:
        raise IdentityError("spatial coordinates are not numeric") from exc
    if len(coordinate_rows) != len(names) or any(len(row) != 2 for row in coordinate_rows):
        raise IdentityError("spatial coordinates must be n_spots x 2")
    stable = _stable_rows(names, columns)
    result = {
        "identity_status": "PASS",
        "n_spots": len(names),
        "n_donors": len(set(donors)),
        "donors": sorted(set(donors)),
        "batches": sorted(set(batches)),
        "retained_ids_set_sha256": canonical_sha256(sorted(names)),
        "labels_set_sha256": canonical_sha256(stable),
        "spatial_set_sha256": canonical_sha256(sorted(zip(names, coordinate_rows))),
        "panel_sha256": panel_sha256,
        "sidecar_input_hash": sidecar.get("input_hash") if sidecar else None,
        "orders": {},
    }
    for arm in ORDER_INDICES:
        rows = order_rows(adata, arm, require_frozen=strict)
        ordered = [names[i] for i in rows]
        result["orders"][arm] = {
            "ordered_ids_sha256": canonical_sha256(ordered),
            "set_sha256": result["retained_ids_set_sha256"],
            "first_seen_batches": list(ORDER_BATCHES[arm]),
            "n_rows": len(rows),
            "identity_status": "PASS",
        }
    return result

