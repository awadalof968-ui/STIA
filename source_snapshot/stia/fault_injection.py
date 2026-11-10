# -*- coding: utf-8 -*-
"""Deterministic graded faults for prospective STIA framework ablation.

These helpers generate benchmark cases; they never fit a model or choose a
severity from observed outcomes.  Callers must freeze levels before scoring.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


FAULT_INJECTION_VERSION = "stia.fault_injection.v1.20260909"


def _matrix(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty 2-D matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _level(value: float, name: str, *, upper: float = 1.0) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > upper:
        raise ValueError(f"{name} must be between 0 and {upper}")
    return result


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{FAULT_INJECTION_VERSION}|{seed}|{value}".encode()).hexdigest()


def inject_row_identity_mismatch(
    entity_ids: Sequence[Any],
    fraction: float,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate a stable subset of identity labels while retaining the ID set."""
    ids = np.asarray([str(value) for value in entity_ids], dtype=str)
    if ids.size < 2 or np.unique(ids).size != ids.size:
        raise ValueError("entity_ids must contain at least two unique identities")
    fraction = _level(fraction, "fraction")
    count = 0 if fraction == 0 else max(2, int(round(ids.size * fraction)))
    count = min(count, ids.size)
    ranked = sorted(range(ids.size), key=lambda index: (_stable_rank(ids[index], int(seed)), ids[index]))
    selected = np.asarray(ranked[:count], dtype=np.int64)
    corrupted = np.array(ids, copy=True)
    if selected.size:
        corrupted[selected] = np.roll(ids[selected], 1)
    actual = float(np.mean(corrupted != ids))
    return corrupted, {
        "schema_version": FAULT_INJECTION_VERSION,
        "fault": "row_identity_mismatch",
        "requested_fraction": fraction,
        "changed_rows": int(np.count_nonzero(corrupted != ids)),
        "actual_fraction": actual,
        "seed": int(seed),
        "identity_set_preserved": set(corrupted.tolist()) == set(ids.tolist()),
    }


def interpolate_order_drift(reference: Any, alternative: Any, level: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend aligned reference and order-alternative representations."""
    left = _matrix(reference, "reference")
    right = _matrix(alternative, "alternative")
    if left.shape != right.shape:
        raise ValueError("reference and alternative shapes differ")
    level = _level(level, "level")
    result = left + level * (right - left)
    return result, {
        "schema_version": FAULT_INJECTION_VERSION,
        "fault": "order_drift_interpolation",
        "level": level,
        "shape": list(result.shape),
        "level_zero_is_exact_reference": level != 0.0 or np.array_equal(result, left),
        "level_one_is_exact_alternative": level != 1.0 or np.array_equal(result, right),
    }


def shrink_representation_variance(values: Any, multiplier: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Shrink all rows toward the feature mean; zero is exact collapse."""
    matrix = _matrix(values, "values")
    multiplier = _level(multiplier, "multiplier")
    center = matrix.mean(axis=0, keepdims=True)
    result = center + multiplier * (matrix - center)
    before = float(np.var(matrix, axis=0).sum())
    after = float(np.var(result, axis=0).sum())
    return result, {
        "schema_version": FAULT_INJECTION_VERSION,
        "fault": "representation_collapse",
        "variance_multiplier": multiplier,
        "total_variance_before": before,
        "total_variance_after": after,
        "exact_collapse": bool(multiplier == 0.0 and np.unique(result, axis=0).shape[0] == 1),
    }


__all__ = [
    "FAULT_INJECTION_VERSION",
    "inject_row_identity_mismatch",
    "interpolate_order_drift",
    "shrink_representation_variance",
]
