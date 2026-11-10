# -*- coding: utf-8 -*-
"""Downstream propagation consumer (task W3, preparation stage).

Implements the frozen downstream endpoints from
``DOWNSTREAM_PROTOCOL_20260907_r1.json`` so that sub-0.05 representation drift
can be scored as harmless numerical variation or as a change that reaches
downstream scientific structure:

* ``cluster_assignment_stability``  ARI and NMI of donor/sample cluster labels
  against the canonical order, for k in {2,3,4} and frozen clustering seeds;
* ``local_neighborhood_stability``  mean kNN Jaccard of donor/sample centroids
  against the canonical order;
* ``centroid_geometry_displacement`` normalised Frobenius displacement of the
  pairwise distance matrix against the canonical order.

Rules enforced here (see the task's analysis_rules):
  * the same frozen observations and exact identities are used across arms;
  * clustering seed variation is reported separately from order variation;
  * the biological unit is the entity passed in (donor or sample), never a spot.

Final scoring on INSPIRE r5 is gated: this module must not be pointed at the
live r5 result root until all 12 receipts and the independent terminal analysis
pass.  ``--self-test`` runs deterministic synthetic checks only.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

try:  # sklearn is present in the accepted env; keep a clear error otherwise.
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"sklearn is required for downstream propagation: {exc}")

DEFAULT_K_VALUES = (2, 3, 4)
DEFAULT_CLUSTER_SEEDS = (0, 1, 2, 3, 4)


def _as_float_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embedding must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding contains non-finite values")
    return matrix


def _centroids(latent: np.ndarray, entity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(entity_ids).astype(str)
    unique = np.asarray(sorted(np.unique(ids).tolist()), dtype=str)
    if unique.size == 0:
        raise ValueError("no entities present")
    centroids = np.vstack([latent[ids == entity].mean(axis=0) for entity in unique])
    return unique, centroids


def centroid_geometry_displacement(latent_a: Any, latent_b: Any, entity_ids: Any) -> Dict[str, float]:
    """Normalised Frobenius displacement between two arms' distance matrices."""
    a = _as_float_matrix(latent_a)
    b = _as_float_matrix(latent_b)
    ids = np.asarray(entity_ids).astype(str)
    if len(a) != len(b) or len(a) != len(ids):
        raise ValueError("arms and entity ids must share the same row count")
    _, ca = _centroids(a, ids)
    _, cb = _centroids(b, ids)
    da = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(axis=2))
    db = np.sqrt(((cb[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2))
    norm_a = float(np.linalg.norm(da))
    norm_b = float(np.linalg.norm(db))
    if norm_a <= 0 or norm_b <= 0:
        raise ValueError("degenerate geometry (zero pairwise-distance norm)")
    return {
        "raw_frobenius": float(np.linalg.norm(da - db)),
        "normalised_by_reference": float(np.linalg.norm(da - db) / norm_a),
        "relative_to_alternative": float(np.linalg.norm(da - db) / norm_b),
    }


def local_neighborhood_stability(latent_a: Any, latent_b: Any, entity_ids: Any, k: int = 10) -> float:
    """Mean kNN Jaccard of entity centroids between two arms."""
    a = _as_float_matrix(latent_a)
    b = _as_float_matrix(latent_b)
    ids = np.asarray(entity_ids).astype(str)
    _, ca = _centroids(a, ids)
    _, cb = _centroids(b, ids)
    if len(ca) <= k:
        raise ValueError(f"need more than k={k} entities for kNN stability")
    da = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(axis=2))
    db = np.sqrt(((cb[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    na = np.argsort(da, axis=1)[:, :k]
    nb = np.argsort(db, axis=1)[:, :k]
    jaccards = [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(na, nb)]
    return float(np.mean(jaccards))


def cluster_assignment_stability(
    latent_a: Any,
    latent_b: Any,
    entity_ids: Any,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    cluster_seeds: Sequence[int] = DEFAULT_CLUSTER_SEEDS,
    algorithm: str = "elkan",
) -> Dict[str, Dict[str, float]]:
    """ARI/NMI of cluster labels between two arms, per k and clustering seed.

    ``algorithm="elkan"`` is the pinned-host MKL compatibility path recorded in
    the frozen protocol; it is not a new scientific arm.
    """
    a = _as_float_matrix(latent_a)
    b = _as_float_matrix(latent_b)
    ids = np.asarray(entity_ids).astype(str)
    _, ca = _centroids(a, ids)
    _, cb = _centroids(b, ids)
    out: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        if ca.shape[0] <= k:
            continue
        aris, nmis = [], []
        for seed in cluster_seeds:
            la = KMeans(n_clusters=k, n_init=10, algorithm=algorithm, random_state=seed).fit(ca).labels_
            lb = KMeans(n_clusters=k, n_init=10, algorithm=algorithm, random_state=seed).fit(cb).labels_
            aris.append(adjusted_rand_score(la, lb))
            nmis.append(normalized_mutual_info_score(la, lb))
        out[f"k{k}"] = {
            "ari_mean": float(np.mean(aris)),
            "ari_min": float(np.min(aris)),
            "nmi_mean": float(np.mean(nmis)),
            "n_seeds": float(len(cluster_seeds)),
        }
    return out


def propagate(
    arms: Mapping[str, Any],
    entity_ids: Any,
    canonical: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    cluster_seeds: Sequence[int] = DEFAULT_CLUSTER_SEEDS,
    knn_k: int = 10,
) -> Dict[str, Any]:
    """Score every arm against the canonical arm on all three endpoints."""
    if canonical not in arms:
        raise ValueError(f"canonical arm {canonical!r} is absent")
    ids = np.asarray(entity_ids).astype(str)
    reference = arms[canonical]
    result: Dict[str, Any] = {"canonical": canonical, "k_values": list(k_values), "knn_k": knn_k, "arms": {}}
    for name, latent in arms.items():
        if name == canonical:
            continue
        entry: Dict[str, Any] = {}
        try:
            entry["centroid_geometry_displacement"] = centroid_geometry_displacement(reference, latent, ids)
            entry["local_neighborhood_stability_knn_jaccard"] = local_neighborhood_stability(reference, latent, ids, k=knn_k)
            entry["cluster_assignment_stability"] = cluster_assignment_stability(reference, latent, ids, k_values, cluster_seeds)
            entry["status"] = "SCORED"
        except ValueError as exc:
            entry["status"] = "UNRESOLVED"
            entry["reason"] = str(exc)
        result["arms"][name] = entry
    return result


def pairwise_arms(arms: Mapping[str, Any], entity_ids: Any, k_values: Sequence[int] = DEFAULT_K_VALUES, knn_k: int = 10) -> Dict[str, Any]:
    """All unordered arm pairs (used when no canonical arm is privileged)."""
    out: Dict[str, Any] = {}
    for a, b in combinations(sorted(arms), 2):
        out[f"{a}|{b}"] = {
            "centroid_geometry_displacement": centroid_geometry_displacement(arms[a], arms[b], entity_ids),
            "local_neighborhood_stability_knn_jaccard": local_neighborhood_stability(arms[a], arms[b], entity_ids, k=knn_k),
            "cluster_assignment_stability": cluster_assignment_stability(arms[a], arms[b], entity_ids, k_values),
        }
    return out


def self_test() -> int:
    """Deterministic synthetic checks; no real data is touched."""
    rng = np.random.default_rng(20260909)
    ids = np.repeat([f"d{i}" for i in range(12)], 20)
    base = np.repeat(rng.normal(size=(12, 6)), 20, axis=0) + rng.normal(scale=0.01, size=(240, 6))

    identical = propagate({"canonical": base, "same": base.copy()}, ids, "canonical")
    same = identical["arms"]["same"]
    ok = True
    ok &= abs(same["centroid_geometry_displacement"]["normalised_by_reference"]) < 1e-12
    ok &= abs(same["local_neighborhood_stability_knn_jaccard"] - 1.0) < 1e-12
    ok &= all(v["ari_mean"] == 1.0 for v in same["cluster_assignment_stability"].values())
    print("identical-input control:", "PASS" if ok else "FAIL", same["cluster_assignment_stability"])

    shifted = base.copy()
    shifted[:120] += 3.0
    moved = propagate({"canonical": base, "shifted": shifted}, ids, "canonical")["arms"]["shifted"]
    detected = moved["centroid_geometry_displacement"]["normalised_by_reference"] > 0.1
    print("perturbed-input detection:", "PASS" if detected else "FAIL", moved["centroid_geometry_displacement"])
    ok &= detected

    degenerate_ok = False
    try:
        centroid_geometry_displacement(np.zeros((4, 3)), np.zeros((4, 3)), np.asarray(["a", "a", "b", "b"]))
    except ValueError:
        degenerate_ok = True
    print("degenerate geometry rejected:", "PASS" if degenerate_ok else "FAIL")
    ok &= degenerate_ok

    print("SELF_TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="downstream propagation consumer (W3 preparation)")
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        return self_test()
    parser.error("only --self-test is enabled during preparation; final scoring is gated on r5 completion")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
