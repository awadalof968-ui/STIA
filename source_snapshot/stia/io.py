# -*- coding: utf-8 -*-
"""Small, read-only input and canonical-output helpers for STIA."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .contracts import SCHEMA_VERSION


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode("utf-8")


def canonical_write(path: Path, value: Any) -> None:
    """Write canonical JSON without ever replacing existing evidence."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def canonical_csv_write(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Iterable[str]) -> None:
    """Write a deterministic CSV once, refusing to replace evidence."""
    names = list(fieldnames)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in names})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relpath_or_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_identity(path: Path, root: Path) -> Dict[str, Any]:
    p = path.resolve()
    return {
        "path": relpath_or_name(p, root),
        "sha256": sha256_file(p),
        "size_bytes": p.stat().st_size,
    }


def environment_identity() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "anndata", "yaml"):
        try:
            from importlib.metadata import version
            package_name = "scikit-learn" if name == "sklearn" else "PyYAML" if name == "yaml" else name
            versions[name] = version(package_name)
        except Exception:
            versions[name] = None
    return versions


def load_json_or_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        # JSON is valid YAML; this fallback keeps the tool usable in a minimal
        # environment while making non-JSON YAML an explicit configuration error.
        return json.loads(text)


def resolve_input(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_table(path: Path):
    import pandas as pd
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".h5ad", ".h5"):
        try:
            import anndata as ad
        except ImportError as exc:
            raise RuntimeError("anndata is required to read .h5ad inputs") from exc
        return ad.read_h5ad(path, backed="r").obs.reset_index()
    raise ValueError(f"unsupported metadata format: {path.suffix}")


def read_array(path: Path, key: Optional[str] = None):
    import numpy as np
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        if key is not None:
            if key not in loaded:
                raise KeyError(f"{key} absent from {path.name}")
            return loaded[key]
        names = list(loaded.files)
        if len(names) != 1:
            raise ValueError(f"array key required for multi-array npz: {path.name}")
        return loaded[names[0]]
    if suffix in (".csv", ".tsv"):
        import pandas as pd
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        return frame.to_numpy()
    raise ValueError(f"unsupported array format: {path.suffix}")


def validate_canonical_bundle(directory: Path | str) -> Dict[str, Any]:
    """Parse and validate the required STIA bundle without touching sources."""
    root = Path(directory)
    required = ("findings.json", "decision.json", "claim_review.json", "provenance.json", "run_manifest.json", "cells.json", "comparisons.json", "evidence_inventory.json", "repair_actions.json", "metrics.csv")
    missing = [name for name in required if not (root / name).is_file()]
    parsed: Dict[str, Any] = {}
    errors: list[str] = []
    for name in required:
        path = root / name
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            parsed[name] = value
            if value.get("schema_version") != SCHEMA_VERSION:
                errors.append(f"{name}: schema_version must be {SCHEMA_VERSION}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    metrics_path = root / "metrics.csv"
    if metrics_path.is_file():
        try:
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                names = reader.fieldnames or []
                expected = ["schema_version", "metric_id", "value", "unit", "threshold", "threshold_status", "status", "metric_role", "calibration_identity"]
                if names != expected:
                    errors.append("metrics.csv: canonical header mismatch")
        except Exception as exc:
            errors.append(f"metrics.csv: {exc}")
    return {"valid": not missing and not errors, "missing": missing, "errors": errors, "files": parsed}
