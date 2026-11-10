# -*- coding: utf-8 -*-
"""GSE307990 STAligner adapter (external confirmatory matrix).

Sampled 15k (500/section x 30, user-approved shared subset) x 3 seeds x 4 orders.
Native STAligner train_STAligner, per-section KNN graphs (obs['batch']=section),
latent = obsm['STAligner'] (hidden 30-dim), donor-keyed drift -> OSI.
Modes: probe (short epochs, resource record) / fit (full epochs).
"""
from __future__ import annotations

import argparse
import json
import os
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
ORDERS = ROOT / "Lunacy/runs/external-ai-three-day-20260903/evidence/D3/GSE307990_ORDER_DEFINITIONS_20260903_r1.json"
OUT_ROOT = ROOT / "results/framework_evaluation/gse307990_confirmatory_20260903_r1/cells_staligner_15k"
SEEDS = [20260819, 20260820, 20260821]
ORDER_IDS = ["canonical", "reverse", "batch_grouped", "condition_grouped"]
N_EPOCHS_FULL = 1000
N_EPOCHS_PROBE = 60
KNN = 10


def assemble(arm: str):
    adata = ad.read_h5ad(str(H5AD))
    man = pd.read_csv(MANIFEST)
    orders = json.load(open(ORDERS, encoding="utf-8"))
    donor_list = orders[arm]["donors"]
    man["_key"] = man["barcode"] + "-" + man["gsm"]
    h5_idx = {n: i for i, n in enumerate(adata.obs_names)}
    idx = np.array([h5_idx[k] for k in man["_key"].tolist()], dtype=int)
    sub = adata[idx].copy()
    sec_map = dict(zip(man["_key"], man["section"]))
    sub.obs["section"] = [sec_map[k] for k in sub.obs_names]
    donor_vals = sub.obs["donor"].astype(str).values
    perm = np.concatenate([np.where(donor_vals == d)[0] for d in donor_list])
    sub = sub[perm].copy()
    coords = np.column_stack([sub.obs["pxl_col"].astype(np.float64), sub.obs["pxl_row"].astype(np.float64)])
    sub.obsm["spatial"] = coords
    sub.obs["batch_name"] = sub.obs["section"].astype(str).values  # STAligner library key
    return sub


def build_native_graph(sub, k: int = KNN):
    import scipy.sparse as sp
    from STAligner import Cal_Spatial_Net
    batches = np.asarray(sub.obs["batch_name"].astype(str).values)
    n_total = int(sub.n_obs)
    blocks = []
    for batch in list(dict.fromkeys(batches.tolist())):
        idx = np.flatnonzero(batches == batch)
        if idx.size < 2:
            raise RuntimeError(f"library {batch} has fewer than two spots")
        sub_i = sub[idx].copy()
        local_k = min(int(k), int(idx.size - 1))
        Cal_Spatial_Net(sub_i, k_cutoff=local_k, max_neigh=local_k + 1, model="KNN", verbose=False)
        adj = sub_i.uns.get("adj")
        if adj is None:
            raise RuntimeError(f"library {batch} graph returned no sparse adj")
        rows, cols = adj.nonzero()
        blocks.append(sp.coo_matrix((np.ones(rows.size, dtype=np.float32), (idx[rows], idx[cols])),
                                    shape=(n_total, n_total)))
    adjacency = sum(blocks, sp.coo_matrix((n_total, n_total))).tocsr()
    sub.uns["edgeList"] = adjacency.nonzero()
    return sub


def staligner_fit(sub, seed: int, epochs: int, device: str):
    import torch
    import torch.backends.cudnn as cudnn
    from STAligner import train_STAligner
    build_native_graph(sub)  # imports STAligner package (module-level cudnn.benchmark=True)
    # Deterministic fix AFTER all STAligner imports: module sets cudnn.benchmark=True
    # at import which defeats determinism (autotuning -> run-to-run noise ~0.026).
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    train_STAligner(sub, hidden_dims=[512, 30], n_epochs=int(epochs), lr=0.001,
                    random_seed=int(seed), knn_neigh=KNN, device=device, verbose=False)
    latent = np.asarray(sub.obsm["STAligner"], dtype=np.float32)
    return latent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=ORDER_IDS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["probe", "fit"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    epochs = N_EPOCHS_PROBE if args.mode == "probe" else N_EPOCHS_FULL
    t0 = time.time()
    sub = assemble(args.order)
    print(f"assembled {sub.shape} order={args.order}", flush=True)
    latent = staligner_fit(sub, args.seed, epochs, args.device)
    donor = sub.obs["donor"].astype(str).values
    section = sub.obs["section"].astype(str).values
    out_dir = Path(args.out_root) / f"STAligner_{args.order}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "latent.npz", latent=latent, donor_ids=donor.astype("U"),
                        section_ids=section.astype("U"))
    elapsed = time.time() - t0
    rec = {"order": args.order, "seed": args.seed, "mode": args.mode, "epochs": epochs,
           "device": args.device, "elapsed_s": round(elapsed, 1), "shape": list(latent.shape),
           "n_sections": len(np.unique(section)), "out": str(out_dir / "latent.npz")}
    (out_dir / "receipt.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
