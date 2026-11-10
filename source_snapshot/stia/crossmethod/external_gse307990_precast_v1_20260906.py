# -*- coding: utf-8 -*-
"""GSE307990 PRECAST adapter (external confirmatory matrix, last method).

15k sampled (500/section x 30), 3000-gene INSPIRE panel, 3 seeds x 4 orders,
native R core (_x3_stageC_precast_core_v16_20260826.R) via input.npz contract.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import anndata as ad

ROOT = Path(__file__).resolve().parents[2]
H5AD = ROOT / "results/framework_evaluation/external_ai_20260903_g307990/gse307990_merged_20260903.h5ad"
MANIFEST = ROOT / "results/framework_evaluation/gse307990_confirmatory_20260903_r1/manifests/sampling_seed_20260903_15k.csv"
PANEL = ROOT / "results/framework_evaluation/gse307990_confirmatory_20260903_r1/cells/INSPIRE_panel_20260903.json"
ORDERS = ROOT / "Lunacy/runs/external-ai-three-day-20260903/evidence/D3/GSE307990_ORDER_DEFINITIONS_20260903_r1.json"
OUT_ROOT = ROOT / "results/framework_evaluation/gse307990_confirmatory_20260903_r1/cells_precast_15k"
R_EXE = Path(r"LOCAL_RUNTIME_ROOT\R\R-4.5.3\bin\Rscript.exe")
R_CORE = ROOT / "scripts/_x3_stageC_precast_core_v16_20260826.R"
SEEDS = [20260819, 20260820, 20260821]
ORDER_IDS = ["canonical", "reverse", "batch_grouped", "condition_grouped"]


def assemble_npz(order: str) -> dict:
    adata = ad.read_h5ad(str(H5AD))
    man = pd.read_csv(MANIFEST)
    orders = json.load(open(ORDERS, encoding="utf-8"))
    donor_list = orders[order]["donors"]
    genes = json.load(open(PANEL, encoding="utf-8"))["genes"]
    man["_key"] = man["barcode"] + "-" + man["gsm"]
    h5_idx = {n: i for i, n in enumerate(adata.obs_names)}
    idx = np.array([h5_idx[k] for k in man["_key"].tolist()], dtype=int)
    donor_vals = man["donor"].astype(str).values
    gpos = {g: i for i, g in enumerate(adata.var_names)}
    gcols = np.array([gpos[g] for g in genes])
    # donor blocks in `order` permutation
    blocks = []
    for d in donor_list:
        blocks.append(np.where(donor_vals == d)[0])
    rows = np.concatenate(blocks)
    X = adata.X[idx[rows]][:, gcols].toarray().astype(np.float64)
    spatial = np.column_stack([man["pxl_col"].values[rows].astype(np.float64),
                               man["pxl_row"].values[rows].astype(np.float64)])
    donor_code = np.concatenate([np.full(len(b), i, dtype=int) for i, b in enumerate(blocks)])
    block_sizes = np.array([len(b) for b in blocks], dtype=int)
    donor_levels = np.array(donor_list)
    return {"X": X, "spatial": spatial, "donor_code": donor_code, "donor_n": len(donor_list),
            "input_row_index": np.arange(len(rows), dtype=int), "donor_levels": donor_levels,
            "block_sizes_in": block_sizes}


def run_r_cell(order: str, seed: int) -> bool:
    out_dir = OUT_ROOT / f"PRECAST_{order}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = out_dir / "input.npz"
    outp = out_dir / "output.npz"
    if outp.exists() and (out_dir / "receipt.json").exists():
        return True
    data = assemble_npz(order)
    np.savez_compressed(inp, **data)
    env = dict(os.environ)
    env["R_LIBS_USER"] = r"LOCAL_RUNTIME_ROOT/x3_rlib45"
    env["RETICULATE_PYTHON"] = r"LOCAL_RUNTIME_ROOT/envs/INSPIRE_env/python.exe"
    t0 = time.time()
    proc = subprocess.run([str(R_EXE), "--vanilla", str(R_CORE), str(inp), str(outp), str(int(seed))],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=60 * 60 * 8, env=env)
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and outp.exists()
    (out_dir / "stderr.log").write_text(proc.stderr[-6000:], encoding="utf-8")
    if ok:
        z = np.load(outp, allow_pickle=False)
        latent = np.asarray(z["latent"], dtype=np.float32)
        row_idx = np.asarray(z["output_row_index"])
        donor_ids = np.array([data["donor_levels"][c] for c in data["donor_code"]])
        np.savez_compressed(out_dir / "latent.npz", latent=latent, donor_ids=donor_ids.astype("U"))
        rec = {"order": order, "seed": seed, "mode": "fit", "elapsed_s": round(elapsed, 1),
               "shape": list(latent.shape), "row_map_ok": bool(np.array_equal(row_idx, np.arange(len(rows := data["donor_code"])))),
               "n_donors": data["donor_n"], "out": str(out_dir / "latent.npz")}
        (out_dir / "receipt.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[{order} s{seed}] OK {elapsed/60:.0f}min shape={latent.shape}", flush=True)
    else:
        print(f"[{order} s{seed}] FAIL rc={proc.returncode} {elapsed/60:.0f}min", flush=True)
    return ok


def main() -> int:
    orders = sys.argv[1].split(",") if len(sys.argv) > 1 else ORDER_IDS
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS
    for s in seeds:
        for o in orders:
            if not run_r_cell(o, s):
                print(f"permanent fail {o} s{s}; stopping", flush=True)
                return 1
    print("PRECAST 15K ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
