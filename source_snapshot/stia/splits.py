# -*- coding: utf-8 -*-
"""Deterministic donor-level split plans for STIA validation experiments.

Unsupervised A/B/C/D order audits intentionally do not call this module: all
arms must use the same complete frozen input.  These split plans are for
supervised probes, tuning, and external predictive validation where leakage
across donors would invalidate the estimate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .contracts import SCHEMA_VERSION, TOOL_NAME, TOOL_VERSION
from .io import canonical_bytes, canonical_write, environment_identity, load_table, sha256_file


MetadataInput = Union[pd.DataFrame, str, Path]
SPLIT_PROTOCOL_VERSION = "1.0.0"


@dataclass
class SplitPlan:
    """In-memory split result plus donor- and spot-level assignments."""

    mode: str
    manifest: Dict[str, Any]
    donor_assignments: pd.DataFrame
    spot_assignments: pd.DataFrame

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.manifest)


def _stable_frame_hash(frame: pd.DataFrame, columns: Tuple[str, ...]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare_metadata(
    metadata: MetadataInput,
    donor_col: str,
    condition_col: str,
    spot_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
        source: Dict[str, Any] = {"path": None, "sha256": None}
    else:
        path = Path(metadata).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"metadata does not exist: {path}")
        frame = load_table(path)
        source = {"path": str(path), "sha256": sha256_file(path)}
    required = (spot_col, donor_col, condition_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    if frame.loc[:, list(required)].isna().any().any():
        raise ValueError("spot, donor, and condition identifiers cannot be missing")
    clean = frame.copy()
    for column in required:
        clean[column] = clean[column].astype(str)
        if (clean[column].str.len() == 0).any():
            raise ValueError(f"metadata column contains empty identifiers: {column}")
    if clean[spot_col].duplicated().any():
        raise ValueError("spot identifiers must be unique")
    donor_condition_counts = clean.groupby(donor_col, sort=True)[condition_col].nunique()
    invalid = donor_condition_counts[donor_condition_counts != 1].index.tolist()
    if invalid:
        raise ValueError(f"each donor must map to exactly one condition; invalid donors: {invalid[:5]}")
    source["selected_columns_sha256"] = _stable_frame_hash(clean, required)
    source["n_rows"] = int(len(clean))
    source["n_donors"] = int(clean[donor_col].nunique())
    return clean, source


def _donor_table(frame: pd.DataFrame, donor_col: str, condition_col: str) -> pd.DataFrame:
    return (
        frame.loc[:, [donor_col, condition_col]]
        .drop_duplicates()
        .sort_values([condition_col, donor_col], kind="mergesort")
        .reset_index(drop=True)
    )


def _holdout_counts(n_donors: int, fractions: np.ndarray) -> np.ndarray:
    if n_donors < 3:
        raise ValueError("each condition needs at least three donors for train/validation/test coverage")
    target = fractions * n_donors
    counts = np.maximum(np.floor(target).astype(int), 1)
    while int(counts.sum()) < n_donors:
        index = int(np.argmax(target - counts))
        counts[index] += 1
    while int(counts.sum()) > n_donors:
        candidates = np.where(counts > 1)[0]
        if not len(candidates):
            raise RuntimeError("cannot satisfy positive holdout counts")
        index = int(candidates[np.argmax(counts[candidates] - target[candidates])])
        counts[index] -= 1
    return counts


def grouped_holdout_split(
    metadata: MetadataInput,
    *,
    donor_col: str = "donor_id",
    condition_col: str = "disease_or_condition",
    spot_col: str = "spot_id",
    seed: int = 20260823,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> SplitPlan:
    """Create one deterministic, condition-stratified donor holdout plan."""

    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train/validation/test fractions must be positive and sum to 1")
    frame, source = _prepare_metadata(metadata, donor_col, condition_col, spot_col)
    donors = _donor_table(frame, donor_col, condition_col)
    rng = np.random.default_rng(int(seed))
    records = []
    roles = ("train", "validation", "test")
    for condition in sorted(donors[condition_col].unique()):
        condition_donors = donors.loc[donors[condition_col] == condition, donor_col].to_numpy(copy=True)
        condition_donors = condition_donors[rng.permutation(len(condition_donors))]
        counts = _holdout_counts(len(condition_donors), fractions)
        start = 0
        for role, count in zip(roles, counts):
            for donor in condition_donors[start:start + int(count)]:
                records.append({"donor_id": str(donor), "condition": str(condition), "split": role})
            start += int(count)
    donor_assignments = pd.DataFrame.from_records(records).sort_values(["split", "condition", "donor_id"], kind="mergesort").reset_index(drop=True)
    role_map = donor_assignments.set_index("donor_id")["split"]
    spot_assignments = pd.DataFrame({
        "spot_id": frame[spot_col].astype(str),
        "donor_id": frame[donor_col].astype(str),
        "condition": frame[condition_col].astype(str),
        "split": frame[donor_col].astype(str).map(role_map),
    }).sort_values("spot_id", kind="mergesort").reset_index(drop=True)
    if spot_assignments["split"].isna().any():
        raise RuntimeError("not every spot inherited a donor split")
    role_sets = {role: set(donor_assignments.loc[donor_assignments["split"] == role, "donor_id"]) for role in roles}
    donor_disjoint = all(role_sets[left].isdisjoint(role_sets[right]) for i, left in enumerate(roles) for right in roles[i + 1:])
    expected_conditions = set(donors[condition_col].astype(str))
    condition_coverage = all(set(donor_assignments.loc[donor_assignments["split"] == role, "condition"]) == expected_conditions for role in roles)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "mode": "grouped_holdout",
        "seed": int(seed),
        "grouping_unit": "donor",
        "columns": {"spot": spot_col, "donor": donor_col, "condition": condition_col},
        "fractions": {"train": float(fractions[0]), "validation": float(fractions[1]), "test": float(fractions[2])},
        "source_metadata": source,
        "environment": environment_identity(),
        "counts": {
            role: {
                "donors": int((donor_assignments["split"] == role).sum()),
                "spots": int((spot_assignments["split"] == role).sum()),
            }
            for role in roles
        },
        "checks": {
            "donor_disjoint": donor_disjoint,
            "condition_coverage_in_every_split": condition_coverage,
            "all_spots_inherit_donor_split": bool(spot_assignments["split"].notna().all()),
        },
        "leakage_policy": {
            "train": "fit preprocessing, feature selection, and model parameters here",
            "validation": "select hyperparameters and thresholds only; never refit preprocessing on test data",
            "test": "locked outcome-blind final evaluation; open once after analysis is frozen",
        },
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError(f"invalid grouped holdout plan: {manifest['checks']}")
    return SplitPlan("grouped_holdout", manifest, donor_assignments, spot_assignments)


def grouped_nested_cv_split(
    metadata: MetadataInput,
    *,
    donor_col: str = "donor_id",
    condition_col: str = "disease_or_condition",
    spot_col: str = "spot_id",
    seed: int = 20260823,
    outer_folds: int = 5,
    inner_folds: int = 3,
) -> SplitPlan:
    """Create condition-stratified nested CV with donors as indivisible groups."""

    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be at least two")
    frame, source = _prepare_metadata(metadata, donor_col, condition_col, spot_col)
    donors = _donor_table(frame, donor_col, condition_col).rename(columns={donor_col: "donor_id", condition_col: "condition"})
    counts = donors["condition"].value_counts()
    if int(counts.min()) < outer_folds:
        raise ValueError(f"each condition needs at least outer_folds={outer_folds} donors; observed={counts.to_dict()}")
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=int(seed))
    records = []
    fold_summaries = []
    for outer_index, (development_index, test_index) in enumerate(outer.split(donors["donor_id"], donors["condition"])):
        development = donors.iloc[development_index].reset_index(drop=True)
        test = donors.iloc[test_index].reset_index(drop=True)
        development_counts = development["condition"].value_counts()
        if int(development_counts.min()) < inner_folds:
            raise ValueError(
                f"outer fold {outer_index} leaves fewer than inner_folds={inner_folds} donors "
                f"in a condition: {development_counts.to_dict()}"
            )
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=int(seed) + outer_index + 1)
        for inner_index, (train_index, validation_index) in enumerate(inner.split(development["donor_id"], development["condition"])):
            for role, table in (
                ("train", development.iloc[train_index]),
                ("validation", development.iloc[validation_index]),
            ):
                for row in table.itertuples(index=False):
                    records.append({
                        "outer_fold": int(outer_index),
                        "inner_fold": int(inner_index),
                        "donor_id": str(row.donor_id),
                        "condition": str(row.condition),
                        "split": role,
                    })
        # The locked outer test set is represented exactly once.  Repeating it
        # for every inner fold could make downstream summaries count the same
        # test observations multiple times.
        for row in test.itertuples(index=False):
            records.append({
                "outer_fold": int(outer_index),
                "inner_fold": -1,
                "donor_id": str(row.donor_id),
                "condition": str(row.condition),
                "split": "test",
            })
        fold_summaries.append({
            "outer_fold": int(outer_index),
            "development_donors": int(len(development)),
            "locked_test_donors": int(len(test)),
        })
    donor_assignments = pd.DataFrame.from_records(records).sort_values(
        ["outer_fold", "inner_fold", "split", "condition", "donor_id"], kind="mergesort"
    ).reset_index(drop=True)
    spot_base = frame.loc[:, [spot_col, donor_col, condition_col]].rename(
        columns={spot_col: "spot_id", donor_col: "donor_id", condition_col: "condition"}
    )
    spot_parts = []
    for (outer_index, inner_index), assignment in donor_assignments.groupby(["outer_fold", "inner_fold"], sort=True):
        part = spot_base.merge(assignment.loc[:, ["donor_id", "split"]], on="donor_id", how="inner", validate="many_to_one")
        part.insert(0, "inner_fold", int(inner_index))
        part.insert(0, "outer_fold", int(outer_index))
        spot_parts.append(part)
    spot_assignments = pd.concat(spot_parts, ignore_index=True).sort_values(
        ["outer_fold", "inner_fold", "spot_id"], kind="mergesort"
    ).reset_index(drop=True)
    disjoint = True
    coverage = True
    expected_conditions = set(donors["condition"])
    for outer_index, outer_assignment in donor_assignments.groupby("outer_fold", sort=True):
        test_assignment = outer_assignment.loc[outer_assignment["inner_fold"] == -1]
        test_set = set(test_assignment["donor_id"])
        coverage &= set(test_assignment["condition"]) == expected_conditions
        for _, assignment in outer_assignment.loc[outer_assignment["inner_fold"] >= 0].groupby("inner_fold", sort=True):
            train_set = set(assignment.loc[assignment["split"] == "train", "donor_id"])
            validation_set = set(assignment.loc[assignment["split"] == "validation", "donor_id"])
            disjoint &= train_set.isdisjoint(validation_set)
            disjoint &= train_set.isdisjoint(test_set)
            disjoint &= validation_set.isdisjoint(test_set)
            coverage &= set(assignment.loc[assignment["split"] == "train", "condition"]) == expected_conditions
            coverage &= set(assignment.loc[assignment["split"] == "validation", "condition"]) == expected_conditions
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "mode": "grouped_nested_cv",
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "grouping_unit": "donor",
        "columns": {"spot": spot_col, "donor": donor_col, "condition": condition_col},
        "source_metadata": source,
        "environment": environment_identity(),
        "folds": fold_summaries,
        "checks": {
            "donor_disjoint_within_every_fold": bool(disjoint),
            "condition_coverage_in_every_role": bool(coverage),
            "all_spots_inherit_donor_role": bool(spot_assignments["split"].notna().all()),
            "outer_test_excluded_from_inner_training_and_validation": bool(disjoint),
        },
        "leakage_policy": {
            "inner_train": "fit preprocessing, feature selection, and model parameters",
            "inner_validation": "select hyperparameters and thresholds",
            "outer_test": "evaluate only after the inner analysis is frozen; never use for preprocessing or selection",
        },
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError(f"invalid grouped nested-CV plan: {manifest['checks']}")
    return SplitPlan("grouped_nested_cv", manifest, donor_assignments, spot_assignments)


def write_split_plan(plan: SplitPlan, output_dir: Union[str, Path]) -> Mapping[str, Path]:
    """Write a split plan as three exclusive, versionable artifacts."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "split_manifest.json"
    donor_path = output / "donor_assignments.csv"
    spot_path = output / "spot_assignments.csv"
    paths = (manifest_path, donor_path, spot_path)
    existing = [path.name for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing split evidence in {output}: {', '.join(existing)}; "
            "choose a new versioned output directory"
        )
    plan.donor_assignments.to_csv(donor_path, index=False, lineterminator="\n", mode="x")
    plan.spot_assignments.to_csv(spot_path, index=False, lineterminator="\n", mode="x")
    manifest = dict(plan.manifest)
    manifest["artifacts"] = {
        donor_path.name: {"sha256": sha256_file(donor_path), "rows": int(len(plan.donor_assignments))},
        spot_path.name: {"sha256": sha256_file(spot_path), "rows": int(len(plan.spot_assignments))},
    }
    manifest["plan_sha256"] = hashlib.sha256(canonical_bytes(plan.manifest)).hexdigest()
    canonical_write(manifest_path, manifest)
    return {path.name: path for path in paths}
