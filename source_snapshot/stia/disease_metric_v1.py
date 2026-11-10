"""No-fit, donor-unit disease-axis descriptive readout.

This is not a predictive model, a test of disease-signal preservation, or a
clinical endpoint. It measures whether between-condition donor centroids are
farther apart than within-condition donor centroids in overlapping technical
batches. Donor pairs are correlated; no pair-level confidence interval or
significance test is emitted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


METRIC_ID = "donor_batch_matched_separation_auc_v1"


def donor_batch_separation_auc(
    embedding: Any,
    metadata: pd.DataFrame,
    *,
    donor_col: str = "donor_id",
    condition_col: str = "disease_or_condition",
    batch_col: str = "technical_batch",
) -> dict[str, Any]:
    """Return an equal-batch average of donor-centroid distance ranking AUCs.

    Within each eligible batch, at least two independent donors per condition
    are required. AUC compares all between-condition donor-pair distances to
    within-condition donor-pair distances, with half credit for ties. It is
    descriptive and intentionally has no learned threshold or fitted model.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be a DataFrame")
    required = (donor_col, condition_col, batch_col)
    if any(column not in metadata.columns for column in required):
        raise ValueError(f"missing donor, condition, or batch column: {required}")
    if metadata.loc[:, list(required)].isna().any().any():
        raise ValueError("missing donor, condition, or batch identity")
    if any(metadata[column].astype(str).str.len().eq(0).any() for column in required):
        raise ValueError("empty donor, condition, or batch identity")
    try:
        values = np.asarray(embedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding must be numeric") from exc
    if values.ndim != 2 or len(values) != len(metadata) or values.shape[1] == 0:
        raise ValueError("embedding must be 2D and row-aligned to metadata")
    if not np.isfinite(values).all():
        raise ValueError("embedding values must be finite")

    frame = metadata.loc[:, list(required)].reset_index(drop=True).astype(str).copy()
    if frame.groupby(donor_col, sort=True)[condition_col].nunique().gt(1).any():
        raise ValueError("donor has inconsistent condition labels")
    if frame[condition_col].nunique() < 2:
        raise ValueError("at least two condition levels are required")
    frame["_row"] = np.arange(len(frame))
    batch_scores: dict[str, dict[str, Any]] = {}
    for batch, batch_frame in frame.groupby(batch_col, sort=True):
        centroids = []
        donor_conditions = []
        for donor, donor_frame in batch_frame.groupby(donor_col, sort=True):
            centroids.append(values[donor_frame["_row"].to_numpy(dtype=int)].mean(axis=0))
            donor_conditions.append(str(donor_frame[condition_col].iloc[0]))
        counts = pd.Series(donor_conditions).value_counts()
        if len(counts) < 2 or counts.min() < 2:
            continue
        points = np.stack(centroids)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        upper = np.triu_indices(len(points), k=1)
        pair_distances = distances[upper]
        labels = np.asarray(donor_conditions)
        same = labels[upper[0]] == labels[upper[1]]
        within = pair_distances[same]
        between = pair_distances[~same]
        if not len(within) or not len(between):
            continue
        comparisons = between[:, None] - within[None, :]
        auc = float((np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)
        batch_scores[str(batch)] = {
            "score": auc,
            "n_donors": len(centroids),
            "n_within_condition_pairs": int(len(within)),
            "n_between_condition_pairs": int(len(between)),
            "donors_by_condition": {str(key): int(value) for key, value in sorted(counts.items())},
        }
    if not batch_scores:
        raise ValueError("no eligible batch has at least two donors per condition with condition overlap")
    return {
        "metric_id": METRIC_ID,
        "status": "MEASURED_DESCRIPTIVE",
        "statistical_unit": "donor",
        "score": float(np.mean([row["score"] for row in batch_scores.values()])),
        "aggregation": "equal_weight_across_eligible_batches",
        "eligible_batches": sorted(batch_scores),
        "n_donors": int(frame[donor_col].nunique()),
        "batch_scores": batch_scores,
        "no_fit": True,
        "inference": "none; donor pairs are dependent and no significance threshold is supplied",
        "claim_boundary": "descriptive separation, not disease-signal preservation, prediction, causality or clinical validity",
    }
