"""Command-line entry point for bounded C1 validation and parent-scheduled runs."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

from .adapters import run_frozen_osi, validate_case
from .frozen_contract import FROZEN_COHORT, FROZEN_SEEDS, METHODS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C1 frozen PCA/Harmony/STAligner A-D adapter runner")
    parser.add_argument("--mode", choices=("validate", "smoke", "run"), default="validate")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--cohort", default=FROZEN_COHORT)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-model-fits", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "smoke":
        suite = unittest.defaultTestLoader.loadTestsFromName(__package__ + ".test_frozen_adapters")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.mode == "validate":
        result = validate_case(args.project_root, cohort=args.cohort, strict=True)
        print(json.dumps({
            "status": result["identity_status"],
            "cohort": result["cohort"],
            "n_spots": result["n_spots"],
            "n_donors": result["n_donors"],
            "panel_sha256": result["panel_sha256"],
            "orders": {key: value["identity_status"] for key, value in result["orders"].items()},
        }, ensure_ascii=False, sort_keys=True))
        return 0
    if args.method is None:
        parser.error("--method is required with --mode run")
    if not args.allow_model_fits:
        print("REFUSED: --mode run requires --allow-model-fits; no model fit was started", file=sys.stderr)
        return 2
    result = run_frozen_osi(
        args.method,
        project_root=args.project_root,
        cohort=args.cohort,
        seeds=FROZEN_SEEDS,
        output_dir=args.output_dir,
        allow_model_fits=True,
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["summary"]["verdict"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

