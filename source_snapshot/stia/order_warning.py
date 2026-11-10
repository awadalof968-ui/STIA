# -*- coding: utf-8 -*-
"""Order-warning tier between the clean envelope and the frozen hard gate.

Implements W1/W2 of DUAL_THRESHOLD_DOWNSTREAM_EXTERNAL_VALIDATION_20260909_r1:

* the hard OSI gate (``osi >= margin``) is unchanged and never lowered;
* a separate WARNING tier marks OSI above a prospectively frozen negative-control
  envelope but below the hard gate;
* a missing, malformed or not-yet-calibrated envelope is ``UNRESOLVED`` and is
  never converted into a pass.

The module is additive: it does not change ``stia.contracts.STATUSES`` or any
existing public API.  ``stia.engine._module_m6`` records the assessment in the
M6 evidence and in the ``osi`` check so JSON/CSV exports keep
``PASS`` / ``WARNING`` / ``FAIL`` / ``UNRESOLVED`` distinct.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# Warning-tier outcomes (distinct from the module status vocabulary).
WARNING_STATUSES = ("NONE", "WARNING", "UNRESOLVED")
# Three-state order decision surfaced to consumers.
ORDER_DECISIONS = ("PASS", "WARNING", "FAIL", "UNRESOLVED")

_CALIBRATION_REQUIRED_FIELDS = ("family", "envelope_upper", "calibration_id", "envelope_source")


def assess_order_warning(
    osi_value: Optional[float],
    margin: Optional[float],
    calibration: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Assess one OSI value against the frozen hard gate and the warning envelope.

    Returns a dict with:
      ``status``          NONE | WARNING | UNRESOLVED
      ``decision``        PASS | WARNING | FAIL | UNRESOLVED
      ``envelope_upper``  the applied envelope (or None)
      ``calibration_id``  identity of the applied calibration (or None)
      ``envelope_source`` provenance of the envelope (or None)
      ``reasons``         list of human-readable reasons
    """
    out: Dict[str, Any] = {
        "status": "UNRESOLVED",
        "decision": "UNRESOLVED",
        "envelope_upper": None,
        "calibration_id": None,
        "envelope_source": None,
        "reasons": [],
    }
    if osi_value is None:
        out["reasons"].append("OSI is unavailable; no order decision can be made")
        return out
    try:
        osi_float = float(osi_value)
    except (TypeError, ValueError):
        out["reasons"].append("OSI is not numeric")
        return out
    if margin is None:
        out["reasons"].append("hard-fail margin is unavailable")
        return out
    try:
        margin_float = float(margin)
    except (TypeError, ValueError):
        out["reasons"].append("hard-fail margin is not numeric")
        return out

    # Hard gate has priority and is evaluated before any warning logic.
    if osi_float >= margin_float:
        out.update({"status": "NONE", "decision": "FAIL"})
        out["reasons"].append(f"OSI {osi_float:g} >= hard margin {margin_float:g}")
        return out

    if calibration is None:
        out["reasons"].append("no negative-control calibration supplied; WARNING tier unresolved")
        return out
    if not isinstance(calibration, Mapping):
        out["reasons"].append("calibration is malformed (expected a mapping)")
        return out

    missing = [field for field in _CALIBRATION_REQUIRED_FIELDS if field not in calibration]
    if missing:
        out["reasons"].append("calibration missing required fields: " + ", ".join(missing))
        return out

    family = str(calibration.get("family"))
    out["calibration_id"] = calibration.get("calibration_id")
    out["envelope_source"] = calibration.get("envelope_source")
    raw_env = calibration.get("envelope_upper")
    if raw_env is None:
        out["reasons"].append(f"{family} family envelope is not calibrated")
        return out
    try:
        envelope = float(raw_env)
    except (TypeError, ValueError):
        out["reasons"].append("calibration envelope_upper is not numeric")
        return out
    if not (0.0 < envelope < margin_float):
        out["reasons"].append(
            f"calibration envelope_upper {envelope:g} must lie strictly between 0 and the hard margin {margin_float:g}"
        )
        return out
    out["envelope_upper"] = envelope

    if osi_float <= envelope:
        out.update({"status": "NONE", "decision": "PASS"})
        out["reasons"].append(f"OSI {osi_float:g} within frozen clean envelope {envelope:g}")
        return out

    out.update({"status": "WARNING", "decision": "WARNING"})
    out["reasons"].append(
        f"OSI {osi_float:g} above frozen clean envelope {envelope:g} but below hard margin {margin_float:g}"
    )
    return out
