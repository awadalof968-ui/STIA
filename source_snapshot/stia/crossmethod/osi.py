"""Pure NumPy donor-keyed OSI implementation used by the C1 runner."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .frozen_contract import OSI_MARGIN


def donor_distance_matrix(latent: np.ndarray, donor_keys: Sequence[str]) -> tuple[list[str], np.ndarray]:
    z = np.asarray(latent, dtype=np.float64)
    keys = np.asarray([str(item) for item in donor_keys], dtype=object)
    if z.ndim != 2 or z.shape[0] != len(keys) or z.shape[1] == 0:
        raise ValueError("latent must be a finite n x d matrix aligned to donor keys")
    if not np.isfinite(z).all():
        raise ValueError("latent contains non-finite values")
    donors = sorted(set(keys.tolist()))
    if len(donors) < 2:
        raise ValueError("at least two donor keys are required")
    means = np.vstack([z[keys == donor].mean(axis=0) for donor in donors])
    delta = means[:, None, :] - means[None, :, :]
    matrix = np.sqrt(np.sum(delta * delta, axis=2))
    norm = float(np.linalg.norm(matrix))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("donor distance matrix is collapsed or non-finite")
    return donors, matrix / norm


def compute_osi(
    latents_by_arm: Mapping[str, np.ndarray], donor_keys_by_arm: Mapping[str, Sequence[str]], *,
    condition: str, seed: int, margin: float = OSI_MARGIN,
) -> dict:
    if margin != OSI_MARGIN:
        raise ValueError("OSI margin is frozen at 0.05")
    required = {"A_original", "B_interleaved", "C_hash_batch_v1", "D_hash_batch_v1"}
    if set(latents_by_arm) != required or set(donor_keys_by_arm) != required:
        raise ValueError("exactly the frozen A-D arms are required")
    matrices = {}
    donor_lists = {}
    for arm in sorted(required):
        donor_lists[arm], matrices[arm] = donor_distance_matrix(latents_by_arm[arm], donor_keys_by_arm[arm])
    if any(donor_lists[arm] != donor_lists["A_original"] for arm in required):
        raise ValueError("donor-key ordering differs across frozen arms")
    pairwise = []
    for left, right in combinations(sorted(required), 2):
        drift = float(np.linalg.norm(matrices[left] - matrices[right]))
        pairwise.append({"first_order_id": left, "second_order_id": right, "drift": drift})
    osi = max(item["drift"] for item in pairwise)
    failures = [
        {
            "scope": "order_pair",
            "condition": condition,
            "seed": int(seed),
            "first_order_id": item["first_order_id"],
            "second_order_id": item["second_order_id"],
            "reason": "osi_margin_exceeded",
        }
        for item in pairwise if item["drift"] > margin
    ]
    return {
        "row_id": f"{condition}:{int(seed)}",
        "condition": condition,
        "seed": int(seed),
        "donor_keys": donor_lists["A_original"],
        "order_ids": sorted(required),
        "normalized_distance_matrices": {arm: matrices[arm].tolist() for arm in sorted(required)},
        "pairwise_drifts": pairwise,
        "osi": osi,
        "epsilon": margin,
        "finite": bool(np.isfinite(osi)),
        "non_collapsed": True,
        "status": "PASS" if np.isfinite(osi) and osi <= margin else "FAIL",
        "failure_localization": failures,
    }

