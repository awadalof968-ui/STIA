# -*- coding: utf-8 -*-
"""Canonical perturbation taxonomy for STIA audits.

The taxonomy prevents materially different experiments from being collapsed
under the ambiguous label ``order sensitivity``.  It is deliberately small
and data-oriented so it can be embedded unchanged in JSON/YAML provenance.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping


PERTURBATION_AXES = (
    "SAMPLE_SLICE_BATCH_ORDER",
    "SPOT_CELL_ROW_ORDER",
    "REFERENCE_TARGET_REVERSAL",
    "SEED_STOCHASTICITY",
    "PHYSICAL_ANATOMICAL_ORDER",
)

PERTURBATION_SEMANTICS = (
    "NUISANCE_PERMUTATION",
    "DIRECTED_ROLE_CHANGE",
    "STOCHASTIC_REPEAT",
    "INTENDED_PHYSICAL_SEQUENCE",
)

REQUIRED_FIXED_FIELDS = (
    "observations",
    "entity_identities",
    "hyperparameters",
    "random_seed",
    "intended_analysis",
)

_EXPECTED_SEMANTICS = {
    "SAMPLE_SLICE_BATCH_ORDER": "NUISANCE_PERMUTATION",
    "SPOT_CELL_ROW_ORDER": "NUISANCE_PERMUTATION",
    "REFERENCE_TARGET_REVERSAL": "DIRECTED_ROLE_CHANGE",
    "SEED_STOCHASTICITY": "STOCHASTIC_REPEAT",
    "PHYSICAL_ANATOMICAL_ORDER": "INTENDED_PHYSICAL_SEQUENCE",
}

_ALLOWED_UNITS = {
    "SAMPLE_SLICE_BATCH_ORDER": {"sample", "slice", "batch"},
    "SPOT_CELL_ROW_ORDER": {"spot", "cell", "row"},
    "REFERENCE_TARGET_REVERSAL": {"pair", "reference_target"},
    "SEED_STOCHASTICITY": {"run", "fit"},
    "PHYSICAL_ANATOMICAL_ORDER": {"slice", "section"},
}


def perturbation_contract(value: Any, *, required: bool = False) -> Dict[str, Any]:
    """Validate and normalize one perturbation contract.

    Missing contracts remain explicitly ``LEGACY_UNSPECIFIED`` unless a caller
    requests strict handling.  An explicitly supplied malformed contract is
    always rejected; it is never silently coerced into an order audit.
    """
    if value is None:
        if required:
            raise ValueError("perturbation_contract is required")
        return {
            "status": "LEGACY_UNSPECIFIED",
            "axis": None,
            "unit": None,
            "semantics": None,
            "fixed_fields": [],
            "auditable_as_unintended_order": None,
        }
    if not isinstance(value, Mapping):
        raise ValueError("perturbation_contract must be a mapping")

    axis = str(value.get("axis", "")).strip().upper()
    unit = str(value.get("unit", "")).strip().lower()
    semantics = str(value.get("semantics", "")).strip().upper()
    if axis not in PERTURBATION_AXES:
        raise ValueError(f"unknown perturbation axis: {axis or '<missing>'}")
    if semantics not in PERTURBATION_SEMANTICS:
        raise ValueError(f"unknown perturbation semantics: {semantics or '<missing>'}")
    expected = _EXPECTED_SEMANTICS[axis]
    if semantics != expected:
        raise ValueError(f"{axis} requires semantics {expected}, not {semantics}")
    if unit not in _ALLOWED_UNITS[axis]:
        allowed = ", ".join(sorted(_ALLOWED_UNITS[axis]))
        raise ValueError(f"{axis} requires unit in {{{allowed}}}, not {unit or '<missing>'}")

    raw_fixed = value.get("fixed_fields", ())
    if not isinstance(raw_fixed, (list, tuple, set)):
        raise ValueError("perturbation_contract.fixed_fields must be a list")
    fixed = sorted({str(item).strip() for item in raw_fixed if str(item).strip()})
    missing = []
    if semantics == "NUISANCE_PERMUTATION":
        missing = [item for item in REQUIRED_FIXED_FIELDS if item not in fixed]
        if missing:
            raise ValueError("nuisance-permutation contract is missing fixed_fields: " + ", ".join(missing))

    return {
        "status": "PASS",
        "axis": axis,
        "unit": unit,
        "semantics": semantics,
        "fixed_fields": fixed,
        "auditable_as_unintended_order": axis in {
            "SAMPLE_SLICE_BATCH_ORDER",
            "SPOT_CELL_ROW_ORDER",
        },
        "changes_intended_analysis_role": axis == "REFERENCE_TARGET_REVERSAL",
        "is_physical_sequence": axis == "PHYSICAL_ANATOMICAL_ORDER",
    }


__all__ = [
    "PERTURBATION_AXES",
    "PERTURBATION_SEMANTICS",
    "REQUIRED_FIXED_FIELDS",
    "perturbation_contract",
]
