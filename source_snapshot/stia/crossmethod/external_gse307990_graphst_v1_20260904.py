# -*- coding: utf-8 -*-
"""GSE307990 GraphST adapter (external confirmatory matrix, per Codex decision json).

Sampled run only: 2000 spots/section x 30 sections = 60k, seeds 20260819/20/21,
4 frozen orders, per-section 3-NN blockdiag graph, GraphST native recon_pca20
embedding, donor-keyed drift -> OSI. Modes: probe (1 order x 1 seed, small
epochs, resource record) and fit (full 600 epochs as frozen by caller).
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
MANIFEST = ROOT / "results/framework_evaluation/gse307990_confirmatory_20260903_r1/manifests/sampling_seed_20260903.csv"
ORDERS = ROOT / "Lunacy/runs/external-ai-three-day-20260903/evidence/D3/GSE307990_ORDER_DEFINITIONS_20260903_r1.json"
OUT_ROOT = ROOT / "results/framework_evaluation/gse307990_confirmatory_20260903_r1/cells_graphst"
SEEDS = [20260819, 20260820, 20260821]
ORDER_IDS = ["canonical", "reverse", "batch_grouped", "condition_grouped"]
N_EPOCHS_FULL = 600
N_EPOCHS_PROBE = 60
N_NEIGHBORS = 3


def assemble(arm: str, manifest_path: Path = MANIFEST):
    adata = ad.read_h5ad(str(H5AD))
    man = pd.read_csv(manifest_path)
    orders = json.load(open(ORDERS, encoding="utf-8"))
    donor_list = orders[arm]["donors"]
    man["_key"] = man["barcode"] + "-" + man["gsm"]
    h5_idx = {n: i for i, n in enumerate(adata.obs_names)}
    idx = np.array([h5_idx[k] for k in man["_key"].tolist()], dtype=int)
    sub = adata[idx].copy()
    # section column from manifest (h5ad obs has capture, not section)
    sec_map = dict(zip(man["_key"], man["section"]))
    sub.obs["section"] = [sec_map[k] for k in sub.obs_names]
    # per-arm row permutation: donor blocks in arm order
    donor_vals = sub.obs["donor"].astype(str).values
    perm = np.concatenate([np.where(donor_vals == d)[0] for d in donor_list])
    sub = sub[perm].copy()
    # spatial into obsm
    coords = np.column_stack([sub.obs["pxl_col"].astype(np.float64), sub.obs["pxl_row"].astype(np.float64)])
    sub.obsm["spatial"] = coords
    # deterministic top-3000 variance gene filter (memory safety; GraphST.preprocess reselects 3000 HVG consistently)
    X = sub.X
    if hasattr(X, "toarray"):
        xm = X.multiply(1.0)
        var = np.asarray((xm.multiply(xm)).mean(axis=0) - np.asarray(xm.mean(axis=0)) ** 2).ravel()
    else:
        var = X.var(axis=0)
    top = np.argsort(-var)[:3000]
    sub = sub[:, top].copy()
    return sub


def blockdiag_adj(adata, n_neighbors=3):
    """per-section 3-NN directed interaction + symmetrized adj, dense (GraphST requirement)."""
    from sklearn.neighbors import NearestNeighbors
    sections = adata.obs["section"].astype(str).values
    n = adata.shape[0]
    rows, cols = [], []
    for sec in np.unique(sections):
        m = np.where(sections == sec)[0]
        if len(m) < 2:
            continue
        xy = adata.obsm["spatial"][m]
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(m) - 1)).fit(xy)
        _, ind = nn.kneighbors(xy)
        for i, nb in zip(m, ind):
            for j in nb:
                rows.append(i)
                cols.append(m[j])
    interaction = np.zeros((n, n), dtype=np.float32)
    interaction[rows, cols] = 1.0
    adj = ((interaction + interaction.T) > 0).astype(np.float32)
    return interaction, adj


def graphst_fit(sub, model_seed: int, epochs: int, device: str):
    from GraphST import GraphST
    GraphST.preprocess(sub)  # normalize + log1p + scale
    interaction, adj = blockdiag_adj(sub)
    sub.obsm["graph_neigh"] = interaction
    sub.obsm["adj"] = adj
    model = GraphST.GraphST(sub, device=device, epochs=epochs, random_seed=int(model_seed),
                            dim_input=len(sub.var_names), dim_output=64)
    ret = model.train()
    emb = ret.obsm["emb"]  # (n, dim_input) reconstruction per GraphST convention
    emb = np.asarray(emb, dtype=np.float32)
    ce = emb - emb.mean(axis=0)
    U, S, _ = np.linalg.svd(ce, full_matrices=False)
    out_emb = (U[:, :20] * S[:20]).astype(np.float32)
    return out_emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=ORDER_IDS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["probe", "fit"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    epochs = N_EPOCHS_PROBE if args.mode == "probe" else N_EPOCHS_FULL
    t0 = time.time()
    sub = assemble(args.order, Path(args.manifest))
    print(f"assembled {sub.shape} order={args.order}", flush=True)
    emb = graphst_fit(sub, args.seed, epochs, args.device)
    donor = sub.obs["donor"].astype(str).values
    section = sub.obs["section"].astype(str).values
    out_dir = Path(args.out_root) / f"GraphST_{args.order}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "latent.npz", latent=emb, donor_ids=donor.astype("U"),
                        section_ids=section.astype("U"))
    elapsed = time.time() - t0
    rec = {"order": args.order, "seed": args.seed, "mode": args.mode, "epochs": epochs,
           "device": args.device, "elapsed_s": round(elapsed, 1), "shape": list(emb.shape),
           "n_sections": len(np.unique(section)), "out": str(out_dir / "latent.npz")}
    (out_dir / "receipt.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
