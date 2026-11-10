"""Sparse, source-bound adapter for the GSE254652 P1 candidate.

This module only validates and exposes the frozen raw-count H5AD.  It never
creates disease/batch labels, selects data-dependent features, densifies the
matrix, or invokes a model.  The public ``open_bundle`` path is pinned to the
E6R H5AD and manifest hashes; all downstream method adapters must consume the
returned seven library sections and explicit order identities.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_DEFAULT = Path(r"LOCAL_INPUT_ROOT\Downloads\INSPIRE-main\INSPIRE-main\INSPIRE")
H5AD_DEFAULT = Path("data/external/gse254652_p1_sparse_20260830_r1/gse254652_p1_sparse_20260830_r1.h5ad")
MANIFEST_DEFAULT = Path("data/external/gse254652_p1_sparse_20260830_r1/gse254652_p1_sparse_20260830_r1_manifest.json")
H5AD_SHA256 = "a4aaddd288803f8c36d508a62f6fa0e335d8517a2698c338021c282a8930de6d"
MANIFEST_SHA256 = "9caa1b5e33e8daf95345a6250783334abaf6b6b59714900bfa995545698c5f73"
SCHEMA_VERSION = "stia.external.gse254652.e8.sparse_adapter.v1"
RECEIPT_SCHEMA_VERSION = "stia.external.gse254652.e8.adapter_preflight.v1"
FEATURE_MODES = ("all_common_19465", "canonical_first3000_ensembl")
COORDINATE_FIELDS = ("array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres", "in_tissue")
LIBRARY_SPECS = (
    ("GSM8047868", "460", 0, 592),
    ("GSM8047869", "462", 592, 1229),
    ("GSM8047870", "463", 1229, 2062),
    ("GSM8047879", "708", 2062, 4096),
    ("GSM8047880", "709", 4096, 6006),
    ("GSM8047881", "710", 6006, 7928),
    ("GSM8047882", "713", 7928, 9568),
)
ORDER_IDENTITIES = {
    "A_original": {
        "gsm": ("GSM8047868", "GSM8047869", "GSM8047870", "GSM8047879", "GSM8047880", "GSM8047881", "GSM8047882"),
        "animals": ("460", "462", "463", "708", "709", "710", "713"),
        "sha256": "2fbf7e1813f6404e1d9c93432a1e1e3557b44f474b7535ab6a77320f56b3aec9",
    },
    "B_interleaved": {
        "gsm": ("GSM8047868", "GSM8047882", "GSM8047869", "GSM8047881", "GSM8047870", "GSM8047880", "GSM8047879"),
        "animals": ("460", "713", "462", "710", "463", "709", "708"),
        "sha256": "c9164e2b2291fe1009c3dd0123bad8172e9d5f6601d974c6c7a4ec8a292cdf96",
    },
    "C_hash_batch_v1": {
        "gsm": ("GSM8047868", "GSM8047880", "GSM8047879", "GSM8047869", "GSM8047881", "GSM8047870", "GSM8047882"),
        "animals": ("460", "709", "708", "462", "710", "463", "713"),
        "sha256": "d194695da1262e03a672fcee70948ddd40c5c70709702d490a91465b8d291363",
    },
    "D_hash_batch_v1": {
        "gsm": ("GSM8047882", "GSM8047869", "GSM8047868", "GSM8047879", "GSM8047870", "GSM8047880", "GSM8047881"),
        "animals": ("713", "462", "460", "708", "463", "709", "710"),
        "sha256": "c43feb1d5371565f3b05744b3d5c5669e905d81aa9d9d33159a55f994fc238f9",
    },
}


class AdapterValidationError(ValueError):
    """Raised when the accepted external candidate cannot be proven intact."""


def _resolve(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AdapterValidationError(f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdapterValidationError(f"malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterValidationError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_file_hash(path: Path | str, expected_sha256: str) -> str:
    """Verify one file against an explicit SHA256 and return the measured hash."""
    target = Path(path)
    measured = sha256_file(target)
    if measured.lower() != str(expected_sha256).lower():
        raise AdapterValidationError(f"SHA256 mismatch for {target}: {measured} != {expected_sha256}")
    return measured


def _feature_selector(feature_mode: str, n_features: int) -> tuple[slice, int]:
    if feature_mode not in FEATURE_MODES:
        raise AdapterValidationError(f"feature mode is not frozen: {feature_mode!r}")
    if n_features != 19465:
        raise AdapterValidationError(f"external feature count {n_features} != 19465")
    if feature_mode == "all_common_19465":
        return slice(None), n_features
    return slice(0, 3000), 3000


def _validate_orders(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    protocol = manifest.get("proposed_external_order_protocol")
    arms = protocol.get("arms") if isinstance(protocol, Mapping) else None
    if not isinstance(arms, Mapping):
        raise AdapterValidationError("proposed external order identities are missing")
    result: dict[str, dict[str, Any]] = {}
    for name, expected in ORDER_IDENTITIES.items():
        item = arms.get(name)
        if not isinstance(item, Mapping):
            raise AdapterValidationError(f"order identity missing: {name}")
        gsm = tuple(str(v) for v in item.get("ordered_gsm", ()))
        animals = tuple(str(v) for v in item.get("ordered_animal_ids", ()))
        measured_hash = canonical_sha256(list(gsm))
        if gsm != expected["gsm"] or animals != expected["animals"] or measured_hash != expected["sha256"] or str(item.get("order_sha256")) != expected["sha256"]:
            raise AdapterValidationError(f"order identity drift: {name}")
        result[name] = {"gsm": list(gsm), "animals": list(animals), "sha256": measured_hash}
    return result


def validate_manifest(manifest: Mapping[str, Any], *, h5ad_path: Path) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    """Validate E6R manifest structure and return features, libraries, orders."""
    if manifest.get("schema_version") != "inspire.external_dataset.gse254652.e6.sparse_materialization.v1":
        raise AdapterValidationError("manifest schema is not the accepted E6 sparse materialization")
    materialized = manifest.get("materialized_h5ad")
    if not isinstance(materialized, Mapping) or materialized.get("sha256") != H5AD_SHA256 or materialized.get("shape") != [9568, 19465] or materialized.get("sparse_format") != "CSR":
        raise AdapterValidationError("manifest H5AD identity/shape/sparse contract is invalid")
    declared_path = materialized.get("path")
    if not isinstance(declared_path, str) or Path(declared_path).resolve() != h5ad_path.resolve():
        raise AdapterValidationError("manifest H5AD path is not the accepted candidate")
    feature_space = manifest.get("feature_space")
    feature_ids = feature_space.get("common_feature_ids_order") if isinstance(feature_space, Mapping) else None
    if not isinstance(feature_ids, list) or len(feature_ids) != 19465 or len(set(feature_ids)) != 19465 or any(not isinstance(gene, str) or not gene.startswith("ENSMUSG") for gene in feature_ids):
        raise AdapterValidationError("manifest feature order is not the unique 19465 ENSMUSG contract")
    libraries = manifest.get("libraries")
    if not isinstance(libraries, list) or len(libraries) != len(LIBRARY_SPECS):
        raise AdapterValidationError("manifest must contain exactly seven libraries")
    expected_by_gsm = {gsm: (animal, start, end) for gsm, animal, start, end in LIBRARY_SPECS}
    normalized: list[dict[str, Any]] = []
    all_ids: list[str] = []
    for item in libraries:
        if not isinstance(item, Mapping) or item.get("gsm") not in expected_by_gsm:
            raise AdapterValidationError("manifest library identity is malformed")
        gsm = str(item["gsm"]); animal, start, end = expected_by_gsm[gsm]
        if str(item.get("animal_id")) != animal or item.get("shape") != [end - start, 19465] or item.get("global_spot_ids") is None:
            raise AdapterValidationError(f"manifest library contract drift: {gsm}")
        ids = item["global_spot_ids"]
        if not isinstance(ids, list) or len(ids) != end - start or len(set(ids)) != len(ids) or any(not isinstance(v, str) or not v.startswith(gsm + "::") for v in ids):
            raise AdapterValidationError(f"manifest IDs malformed: {gsm}")
        coordinates = item.get("coordinates")
        if not isinstance(coordinates, Mapping) or any(field not in coordinates for field in COORDINATE_FIELDS) or any(not isinstance(coordinates[field], list) or len(coordinates[field]) != end - start for field in COORDINATE_FIELDS):
            raise AdapterValidationError(f"manifest coordinates malformed: {gsm}")
        all_ids.extend(ids)
        normalized.append({"gsm": gsm, "animal_id": animal, "row_offset": [start, end], "spot_ids": tuple(ids), "coordinates": {field: tuple(coordinates[field]) for field in COORDINATE_FIELDS}})
    if len(set(all_ids)) != len(all_ids):
        raise AdapterValidationError("manifest global spot IDs are duplicated")
    if tuple(item["gsm"] for item in normalized) != tuple(spec[0] for spec in LIBRARY_SPECS):
        raise AdapterValidationError("manifest library order is not the frozen source order")
    return tuple(str(v) for v in feature_ids), tuple(normalized), _validate_orders(manifest)


def _obs_values(adata: Any, field: str) -> list[Any]:
    try:
        value = adata.obs[field]
    except Exception as exc:
        raise AdapterValidationError(f"H5AD is missing required field {field!r}") from exc
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


@dataclass
class SparseSection:
    """A row-bounded sparse view; ``matrix`` performs a CSR-only slice."""

    gsm: str
    animal_id: str
    row_offset: tuple[int, int]
    spot_ids: tuple[str, ...]
    coordinates: Mapping[str, tuple[Any, ...]]
    _matrix_source: Any
    _column_selector: slice

    @property
    def library_id(self) -> str:
        return self.gsm

    @property
    def n_rows(self) -> int:
        return self.row_offset[1] - self.row_offset[0]

    @property
    def matrix(self) -> Any:
        try:
            import scipy.sparse as sp
        except Exception as exc:  # pragma: no cover - production dependency
            raise AdapterValidationError("scipy is required for sparse sections") from exc
        block = self._matrix_source[self.row_offset[0] : self.row_offset[1], self._column_selector]
        if not sp.issparse(block):
            raise AdapterValidationError(f"{self.gsm} section was densified")
        return block.tocsr(copy=False)


@dataclass
class ExternalBundle:
    """Validated external candidate with seven source-order sparse sections."""

    schema_version: str
    feature_mode: str
    feature_ids: tuple[str, ...]
    sections: tuple[SparseSection, ...]
    orders: Mapping[str, Mapping[str, Any]]
    h5ad_path: Path
    manifest_path: Path
    h5ad_sha256: str
    manifest_sha256: str
    _adata: Any

    def ordered_sections(self, order_id: str) -> tuple[SparseSection, ...]:
        if order_id not in self.orders:
            raise AdapterValidationError(f"unknown frozen order identity: {order_id}")
        by_gsm = {section.gsm: section for section in self.sections}
        return tuple(by_gsm[gsm] for gsm in self.orders[order_id]["gsm"])

    def close(self) -> None:
        file_handle = getattr(self._adata, "file", None)
        close = getattr(file_handle, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "ExternalBundle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def open_bundle(project_root: Path | str = PROJECT_DEFAULT, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT, feature_mode: str = "all_common_19465") -> ExternalBundle:
    """Open the accepted H5AD as sparse sections after strict hash/identity checks."""
    root = Path(project_root); h5ad = _resolve(root, h5ad_path); manifest_file = _resolve(root, manifest_path)
    measured_h5ad = verify_file_hash(h5ad, H5AD_SHA256)
    measured_manifest = verify_file_hash(manifest_file, MANIFEST_SHA256)
    manifest = _json(manifest_file)
    feature_ids, library_rows, orders = validate_manifest(manifest, h5ad_path=h5ad)
    try:
        import anndata as ad
        import numpy as np
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover - production dependency
        raise AdapterValidationError("anndata and scipy are required for the sparse preflight") from exc
    try:
        adata = ad.read_h5ad(h5ad, backed="r")
    except Exception as exc:
        raise AdapterValidationError(f"cannot open accepted H5AD sparsely: {h5ad}: {exc}") from exc
    try:
        if tuple(adata.shape) != (9568, 19465) or getattr(adata.X, "format", None) != "csr":
            raise AdapterValidationError("H5AD shape or CSR format is not accepted")
        if list(map(str, adata.var_names)) != list(feature_ids):
            raise AdapterValidationError("H5AD feature order differs from manifest ENSMUSG order")
        spot_ids = list(map(str, adata.obs_names))
        if len(spot_ids) != len(set(spot_ids)):
            raise AdapterValidationError("H5AD namespaced observation IDs are duplicated")
        selector, feature_count = _feature_selector(feature_mode, adata.shape[1])
        sections: list[SparseSection] = []
        for row in library_rows:
            start, end = row["row_offset"]
            current_ids = tuple(spot_ids[start:end])
            if current_ids != row["spot_ids"]:
                raise AdapterValidationError(f"H5AD spot-ID order drift: {row['gsm']}")
            for field in COORDINATE_FIELDS:
                actual = tuple(_obs_values(adata, field)[start:end])
                if actual != row["coordinates"][field]:
                    raise AdapterValidationError(f"H5AD coordinate drift: {row['gsm']}:{field}")
            if tuple(str(v) for v in _obs_values(adata, "gsm")[start:end]) != (row["gsm"],) * (end - start):
                raise AdapterValidationError(f"H5AD GSM identity drift: {row['gsm']}")
            if tuple(str(v) for v in _obs_values(adata, "library_id")[start:end]) != (row["gsm"],) * (end - start):
                raise AdapterValidationError(f"H5AD library identity drift: {row['gsm']}")
            if tuple(str(v) for v in _obs_values(adata, "animal_id")[start:end]) != (row["animal_id"],) * (end - start):
                raise AdapterValidationError(f"H5AD animal identity drift: {row['gsm']}")
            section = SparseSection(row["gsm"], row["animal_id"], (start, end), current_ids, row["coordinates"], adata.X, selector)
            # One CSR-only probe proves the backing dataset cannot silently densify.
            probe = section.matrix
            if not sp.issparse(probe) or probe.shape != (end - start, feature_count):
                raise AdapterValidationError(f"H5AD sparse section shape/format invalid: {row['gsm']}")
            data = np.asarray(probe.data)
            if not np.isfinite(data).all() or (data < 0).any() or not np.equal(data, np.floor(data)).all():
                raise AdapterValidationError(f"H5AD raw counts are not finite nonnegative integers: {row['gsm']}")
            sections.append(section)
        return ExternalBundle(SCHEMA_VERSION, feature_mode, tuple(feature_ids[:feature_count]), tuple(sections), orders, h5ad, manifest_file, measured_h5ad, measured_manifest, adata)
    except Exception:
        file_handle = getattr(adata, "file", None)
        close = getattr(file_handle, "close", None)
        if callable(close):
            close()
        raise


def _section_receipt(section: SparseSection) -> dict[str, Any]:
    import numpy as np

    matrix = section.matrix
    data = np.asarray(matrix.data)
    count_ok = bool(np.isfinite(data).all() and (data >= 0).all() and np.equal(data, np.floor(data)).all())
    return {
        "gsm": section.gsm,
        "library_id": section.library_id,
        "animal_id": section.animal_id,
        "row_offset": list(section.row_offset),
        "n_rows": section.n_rows,
        "spot_ids_sha256": canonical_sha256(list(section.spot_ids)),
        "coordinate_columns": list(COORDINATE_FIELDS),
        "coordinate_rows_sha256": canonical_sha256([[section.coordinates[field][i] for field in COORDINATE_FIELDS] for i in range(section.n_rows)]),
        "sparse_format": "CSR",
        "dtype": str(matrix.dtype),
        "shape": [int(v) for v in matrix.shape],
        "nnz": int(matrix.nnz),
        "raw_count_nonnegative_integer": count_ok,
    }


def preflight(project_root: Path | str = PROJECT_DEFAULT, *, h5ad_path: Path | str = H5AD_DEFAULT, manifest_path: Path | str = MANIFEST_DEFAULT, feature_mode: str = "all_common_19465") -> dict[str, Any]:
    """Run a read-only sparse preflight and return a JSON-serializable receipt."""
    observed = _datetime.datetime.now().astimezone().isoformat()
    with open_bundle(project_root, h5ad_path=h5ad_path, manifest_path=manifest_path, feature_mode=feature_mode) as bundle:
        sections = [_section_receipt(section) for section in bundle.sections]
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "adapter_schema_version": bundle.schema_version,
            "status": "PASS_CONDITIONAL_RAW_SPARSE_ONLY",
            "observed_at": observed,
            "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "feature_mode": bundle.feature_mode,
            "feature_count": len(bundle.feature_ids),
            "feature_ids_sha256": canonical_sha256(list(bundle.feature_ids)),
            "h5ad": {"path": str(bundle.h5ad_path), "sha256": bundle.h5ad_sha256, "shape": [9568, 19465], "sparse_format": "CSR"},
            "manifest": {"path": str(bundle.manifest_path), "sha256": bundle.manifest_sha256},
            "sections": sections,
            "orders": {name: dict(value) for name, value in bundle.orders.items()},
            "labels": {"disease": "NOT_CREATED", "batch": "NOT_CREATED"},
            "claim_boundary": {"external_validation": "NOT_PERFORMED", "exposure": "PROJECT_EXPOSED_EXTERNAL_REPRODUCTION_CANDIDATE", "disease_or_batch_causal_claim": "FORBIDDEN", "model_fit": "NOT_PERFORMED"},
        }


def write_receipt(receipt: Mapping[str, Any], output_dir: Path | str, *, filename: str = "E8_GSE254652_SPARSE_ADAPTER_PREFLIGHT_20260831.json") -> Path:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); path = destination / filename
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only GSE254652 P1 sparse adapter preflight")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--h5ad", type=Path, default=H5AD_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="all_common_19465")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", default="E8_GSE254652_SPARSE_ADAPTER_PREFLIGHT_20260831.json")
    args = parser.parse_args(argv)
    receipt = preflight(args.project_root, h5ad_path=args.h5ad, manifest_path=args.manifest, feature_mode=args.feature_mode)
    output = write_receipt(receipt, args.output_dir, filename=args.filename) if args.output_dir else None
    print(json.dumps({"status": receipt["status"], "feature_mode": receipt["feature_mode"], "feature_count": receipt["feature_count"], "sections": len(receipt["sections"]), "orders": list(receipt["orders"]), "output": str(output) if output else None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
