import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import RNA_ENCODINGS, find_h5mu, FALLBACK_PROTEINS


@dataclass
class Dataset:
    rna_encodings: np.ndarray
    cell_types: np.ndarray | None
    protein_names: list[str]
    h5mu_path: Path | None
    n_cells: int
    obs: pd.DataFrame | None = None

    @property
    def has_h5mu(self) -> bool:
        return self.cell_types is not None


_CD_HEAD = re.compile(r"^(CD\d+[a-zA-Z]?)-.+$")


def _clean_protein_name(name: str) -> str:
    name = name.split("|")[0]
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    m = _CD_HEAD.match(name)
    if m:
        name = m.group(1)
    return name.strip()


def load_dataset() -> Dataset:
    if not RNA_ENCODINGS.exists():
        raise FileNotFoundError(
            f"Missing RNA encodings at {RNA_ENCODINGS}. "
            "See README for where to put the precomputed embeddings file."
        )

    with h5py.File(RNA_ENCODINGS, "r") as f:
        rna_encodings = f["embeddings"][:].astype(np.float32, copy=False)
    n_cells = rna_encodings.shape[0]

    h5mu_path = find_h5mu()
    cell_types = None
    obs = None
    protein_names = list(FALLBACK_PROTEINS)

    if h5mu_path is not None:
        try:
            import muon as mu
            mdata = mu.read(str(h5mu_path), backed="r")
            rna = mdata["rna"]
            protein = mdata["protein"]

            if rna.shape[0] != n_cells:
                print(f"[data] warning: h5mu has {rna.shape[0]} cells, "
                      f"cache has {n_cells} — order may not align")

            obs = rna.obs.copy()
            if "CellType" in obs.columns:
                obs["CellType"] = obs["CellType"].astype(str)
                cell_types = obs["CellType"].values
            raw_names = list(protein.var_names)
            protein_names = sorted({_clean_protein_name(p) for p in raw_names})
        except Exception as e:
            print(f"[data] warning: failed to read {h5mu_path}: {e}")

    return Dataset(
        rna_encodings=rna_encodings,
        cell_types=cell_types,
        protein_names=protein_names,
        h5mu_path=h5mu_path,
        n_cells=n_cells,
        obs=obs,
    )
