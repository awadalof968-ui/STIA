"""Small identity-only fixtures; no model, GPU, or project data is used."""

from __future__ import annotations

import numpy as np


class _Obs:
    def __init__(self, columns):
        self._columns = {key: np.asarray(value) for key, value in columns.items()}

    def __getitem__(self, key):
        return self._columns[key]


class SmokeAdata:
    def __init__(self, *, blocks_per_donor: int = 3, n_genes: int = 6):
        donors = [f"AD{i}" for i in range(5)] + [f"CT{i}" for i in range(5)]
        batches = [f"d{i}" for i in range(10)]
        donor_values = []
        batch_values = []
        group_values = []
        names = []
        for donor, batch in zip(donors, batches):
            for j in range(blocks_per_donor):
                donor_values.append(donor)
                batch_values.append(batch)
                group_values.append("AD" if donor.startswith("AD") else "CT")
                names.append(f"{donor}_{j}")
        n = len(names)
        self.obs = _Obs({
            "donor": donor_values,
            "batch": batch_values,
            "proc_batch": [str(i % 2) for i in range(n)],
            "group": group_values,
            "Layer": ["Layer 1" if i % 2 else "Layer 2" for i in range(n)],
        })
        self.obs_names = np.asarray(names, dtype=object)
        self.var_names = np.asarray([f"g{i}" for i in range(n_genes)], dtype=object)
        self.X = np.arange(n * n_genes, dtype=np.float32).reshape(n, n_genes) + 1
        self.obsm = {"spatial": np.column_stack((np.arange(n), np.arange(n)[::-1])).astype(float)}
