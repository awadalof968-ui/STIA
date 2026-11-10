# -*- coding: utf-8 -*-
"""Stable-identity input canonicalization and reversible output restoration.

This is a method-agnostic mitigation for nuisance order.  It does not change
an integration model's objective and must not be presented as a repair for
method-internal stochasticity or non-deterministic kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np


ORDER_REPAIR_VERSION = "stia.order_repair.v1.20260909"
ORDER_REPAIR_AXES = ("group", "entity_within_group", "entity_global")


def _strings(values: Sequence[Any], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not value.strip() for value in result):
        raise ValueError(f"{name} contains an empty identity")
    return result


def _digest(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalOrderPlan:
    """One immutable row permutation plus its exact inverse."""

    axis: str
    entity_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    forward_indices: tuple[int, ...]
    inverse_indices: tuple[int, ...]

    def apply_rows(self, values: Any) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != len(self.entity_ids):
            raise ValueError("values first dimension must match the order plan")
        return np.array(array[np.asarray(self.forward_indices, dtype=np.int64)], copy=True)

    def restore_rows(self, canonical_values: Any) -> np.ndarray:
        array = np.asarray(canonical_values)
        if array.ndim == 0 or array.shape[0] != len(self.entity_ids):
            raise ValueError("canonical_values first dimension must match the order plan")
        return np.array(array[np.asarray(self.inverse_indices, dtype=np.int64)], copy=True)

    def receipt(self) -> dict[str, Any]:
        canonical_entities = tuple(self.entity_ids[index] for index in self.forward_indices)
        canonical_groups = tuple(self.group_ids[index] for index in self.forward_indices)
        return {
            "schema_version": ORDER_REPAIR_VERSION,
            "axis": self.axis,
            "row_count": len(self.entity_ids),
            "source_entity_order_sha256": _digest(self.entity_ids),
            "source_group_order_sha256": _digest(self.group_ids),
            "canonical_entity_order_sha256": _digest(canonical_entities),
            "canonical_group_order_sha256": _digest(canonical_groups),
            "permutation_is_bijection": sorted(self.forward_indices) == list(range(len(self.entity_ids))),
            "output_restoration_required": True,
            "claim_boundary": "Input canonicalization mitigates nuisance order only; it does not prove or repair method-internal stochastic mechanisms.",
        }


def build_canonical_order_plan(
    entity_ids: Sequence[Any],
    group_ids: Sequence[Any] | None = None,
    *,
    axis: str = "group",
) -> CanonicalOrderPlan:
    """Create a stable order without losing the exact caller order.

    ``group`` sorts group blocks but preserves row order inside each group.
    ``entity_within_group`` sorts both groups and entities within groups.
    ``entity_global`` ignores groups and sorts globally by entity identity.
    """
    if axis not in ORDER_REPAIR_AXES:
        raise ValueError(f"axis must be one of {ORDER_REPAIR_AXES}")
    entities = _strings(entity_ids, "entity_ids")
    if len(set(entities)) != len(entities):
        raise ValueError("entity_ids must be unique for reversible restoration")
    groups = _strings(group_ids if group_ids is not None else entities, "group_ids")
    if len(groups) != len(entities):
        raise ValueError("group_ids length must equal entity_ids length")
    indices = list(range(len(entities)))
    if axis == "group":
        unique_groups = sorted(set(groups))
        forward = [index for group in unique_groups for index in indices if groups[index] == group]
    elif axis == "entity_within_group":
        forward = sorted(indices, key=lambda index: (groups[index], entities[index]))
    else:
        forward = sorted(indices, key=lambda index: entities[index])
    inverse = np.argsort(np.asarray(forward, dtype=np.int64)).astype(int).tolist()
    plan = CanonicalOrderPlan(axis, entities, groups, tuple(forward), tuple(inverse))
    if sorted(plan.forward_indices) != indices or sorted(plan.inverse_indices) != indices:
        raise AssertionError("constructed order plan is not bijective")
    return plan


__all__ = [
    "ORDER_REPAIR_VERSION",
    "ORDER_REPAIR_AXES",
    "CanonicalOrderPlan",
    "build_canonical_order_plan",
]
