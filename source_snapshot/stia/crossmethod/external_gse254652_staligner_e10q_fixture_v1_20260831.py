"""Bounded native STAligner CPU fixture for E10Q.

This is not an external-data check.  It creates two synthetic libraries with
24 spots, invokes the real native graph constructor and ``train_STAligner``
for the hard-coded 500-step pretraining plus 500-step alignment boundary,
and records finite CPU embeddings/log markers.  Production E10C remains
1,000 epochs and is never changed by this fixture.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stia.crossmethod import external_gse254652_staligner_e10c_cpu_v1_20260831 as e10c


SEED = 20260819
FIXTURE_EPOCHS = 1000


def run_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    counts = rng.poisson(2.0, size=(24, 4)).astype(np.float32)
    counts[counts.sum(axis=1) == 0, 0] = 1.0
    obs = {"batch_name": ["fixture_library_1"] * 12 + ["fixture_library_2"] * 12}
    adata = AnnData(X=sp.csr_matrix(counts), obs=obs)
    adata.obsm["spatial"] = np.vstack([np.column_stack([np.arange(12), np.zeros(12)]), np.column_stack([np.arange(12), np.ones(12) * 100])]).astype(np.float32)
    adata.X = e10c.e10r.sparse_log1p_cpm(adata.X)
    e10c.e10r.build_native_graph(adata, k=e10c.e10r.KNN_NEIGH)
    stdout = io.StringIO(); stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        latent = e10c._fit_native_cpu(adata, SEED, n_epochs=FIXTURE_EPOCHS)
    logs = stdout.getvalue() + stderr.getvalue()
    if "Pretrain with STAGATE" not in logs or "Train with STAligner" not in logs:
        raise RuntimeError("native fixture did not expose both pretraining and alignment log markers")
    if latent.shape != (24, e10c.e10r.LATENT_DIM) or not np.isfinite(latent).all():
        raise RuntimeError("native fixture latent shape/finite check failed")
    return {"schema_version": "stia.external.gse254652.e10q.cpu_fixture.v1", "status": "PASS", "model_fit": "FIXTURE_ONLY", "backend": "native STAligner", "device": "cpu", "torch_num_threads": 4, "seed": SEED, "fixture_epochs": FIXTURE_EPOCHS, "production_epochs": e10c.e10r.N_EPOCHS, "pretraining_alignment": {"pretraining": 500, "alignment": 500, "markers_observed": ["Pretrain with STAGATE", "Train with STAligner"]}, "input": {"shape": [24, 4], "libraries": 2, "sparse_format": "CSR", "graph_edges": int(len(adata.uns["edgeList"][0]))}, "latent": {"shape": list(latent.shape), "dtype": str(latent.dtype), "finite": True, "mean": float(latent.mean()), "std": float(latent.std())}, "logs": {"captured": True, "bytes": len(logs.encode("utf-8"))}, "claim_boundary": {"external_input": "NOT_USED", "external_validation": "NOT_PERFORMED", "cuda_identity": "NOT_CLAIMED"}}


if __name__ == "__main__":
    print(json.dumps(run_fixture(), sort_keys=True))
