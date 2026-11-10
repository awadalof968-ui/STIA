"""Verify slide-level aggregation, saved tests and replay counts in Online Resource 2."""

from collections import Counter, defaultdict
import csv
import importlib.util
import json
from math import isclose
from pathlib import Path
from statistics import mean

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parent
SPACE = ROOT / "verified_20261106"
REPLAY = ROOT / "stia138_880_replay"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_spatial():
    donors = rows(SPACE / "STAligner_physical_KNN_donor_seed.csv")
    grouped = defaultdict(list)
    for row in donors:
        grouped[(row["seed"], row["slide"], row["k"])].append(float(row["physical_recall"]))
    assert len(donors) == 48*3*4 and len(grouped) == 12*3*4
    assert len({(r["seed"], r["slide"], r["k"], r["donor"]) for r in donors}) == len(donors)
    for row in rows(SPACE / "STAligner_physical_KNN_slide_seed.csv"):
        key = (row["seed"], row["slide"], row["k"])
        assert isclose(float(row["physical_recall"]), mean(grouped[key]), abs_tol=1e-12)
    table = rows(SPACE / "Figure3_slide_units_verified.csv")
    measures = {(r["method"], r["slide"]): r for r in table}
    slides = sorted({slide for _, slide, _ in grouped})
    for slide in slides:
        expected = mean(mean(v) for (_, s, k), v in grouped.items() if (s, k) == (slide, "10"))
        assert isclose(float(measures["STAligner", slide]["spatial_lattice_overlap"]), expected, abs_tol=1e-12)
    return slides, measures


def check_inference(slides, measures):
    spec = importlib.util.spec_from_file_location(
        "inference", ROOT / "source_snapshot/stia/inference_optimization_v1.py"
    )
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    comparisons = read(SPACE / "REAL_DATA_COMPARISONS.json")
    eligible = [r for r in comparisons if r["tradeoff_eligible"]]
    mixed = sum(r["observed_mixed_sign"] for r in comparisons)
    detections = []
    for r in eligible:
        result = source.infer_axis_statistics(r["mean_differences"], r["standard_errors"], r["units"]-1)
        p = float(result["tradeoff_iut_p"])
        assert isclose(p, r["iut_p"], abs_tol=1e-12)
        assert np.allclose(result["holm_adjusted_directional_p"], r["holm_adjusted_directional_p"], atol=1e-12)
        if p < .05:
            detections.append((r["dataset"], r["candidate"], r["reference"], p))
    assert (len(comparisons), len(eligible), mixed, len(detections)) == (34, 19, 13, 1)
    direct = [[float(measures["STAligner", s][key]) - float(measures["PCA", s][key]) for key in
               ("biological_separation", "conditional_batch_mixing", "spatial_lattice_overlap")] for s in slides]
    fitted = source.infer_axes(direct)
    hit, = detections
    assert hit[:3] == ("GSE307403", "STAligner", "PCA")
    assert isclose(hit[3], float(fitted["tradeoff_iut_p"]), abs_tol=1e-12)
    return {"comparisons": 34, "eligible": 19, "observed_mixed_sign": 13, "per_comparison_detection": hit,
            "independent_unit": "slide (GSE307403)", "numpy": np.__version__, "scipy": scipy.__version__}


def check_replay():
    saved = read(REPLAY / "ACCEPTANCE.json")
    scores = read(REPLAY / "SCORING.json")
    variants = read(REPLAY / "ABLATION.json")["variants"]
    counts = {r["variant"]: r for r in rows(REPLAY / "VARIANT_COUNTS.csv")}
    assert (saved["tool_version"], saved["cases"], saved["changed_case_count"]) == ("1.3.8", 880, 0)
    assert len({r["parent_id"] for r in scores["rows"]}) == 160
    assert len({r["case_id"] for r in scores["rows"]}) == 880 and len(variants) == 12
    for variant in variants:
        entries = variant["case_rows"]
        assert len({r["case_id"] for r in entries}) == 880
        counter = Counter(("clean" if r["truth"] == "clean" else "fault", r["decision"]) for r in entries)
        for group in ("clean", "fault"):
            for action in ("ACCEPT", "ABSTAIN", "REJECT"):
                assert counter[group, action] == int(counts[variant["variant_id"]][group + "_" + action])
        assert sum(counter.values()) == 880
    for name in ("full_framework", "drop_M9"):
        assert int(counts[name]["fault_REJECT"]) == 720 and int(counts[name]["clean_ABSTAIN"]) == 160
    return {"cases": 880, "parent_clusters": 160, "settings": 12, "STIA": "1.3.8",
            "full_and_M9": {"fault_reject": 720, "clean_abstain": 160}}


if __name__ == "__main__":
    slides, measures = check_spatial()
    print(json.dumps({"status": "PASS", "spatial": {"slides": len(slides), "eligible_donors": 48, "seeds": 3,
          "STAligner_10NN_mean": mean(float(measures["STAligner", s]["spatial_lattice_overlap"]) for s in slides)},
          "inference": check_inference(slides, measures), "replay": check_replay()}, indent=2))
