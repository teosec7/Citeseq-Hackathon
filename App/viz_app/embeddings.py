import pickle

import numpy as np
import torch

from .config import (
    BIOBERT_NAME,
    RNA_EMB_CACHE,
    UMAP_COORDS_CACHE,
    UMAP_REDUCER_CACHE,
)
from .model import CLIPCITE


@torch.no_grad()
def compute_all_rna_embeddings(
    model: CLIPCITE,
    rna_encodings: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    if RNA_EMB_CACHE.exists():
        arr = np.load(RNA_EMB_CACHE)
        if arr.shape[0] == rna_encodings.shape[0]:
            return arr

    n = rna_encodings.shape[0]
    out = np.empty((n, 256), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(np.asarray(rna_encodings[start:end]), dtype=torch.float32, device=device)
        emb = model.get_rna_embedding(batch).cpu().numpy()
        out[start:end] = emb
    np.save(RNA_EMB_CACHE, out)
    return out


class QueryEncoder:
    """Lazy-loads BioBERT once, encodes a text query into the shared 256-d space."""

    def __init__(self, model: CLIPCITE, device: torch.device):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BIOBERT_NAME)
        self.biobert = AutoModel.from_pretrained(BIOBERT_NAME).to(device)
        self.biobert.eval()
        self.model = model
        self.device = device

    @torch.no_grad()
    def encode(self, query_text: str) -> np.ndarray:
        tokens = self.tokenizer(
            query_text, return_tensors="pt", truncation=True, max_length=128
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        bio_emb = self.biobert(**tokens).last_hidden_state[:, 0, :]
        q_emb = self.model.get_protein_embedding(bio_emb).cpu().numpy().squeeze()
        return q_emb  # (256,), L2-normalized


def fit_or_load_umap(all_rna_embs: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.3):
    """Return (reducer, cell_coords). Cache to disk so relaunch is instant."""
    import umap

    if UMAP_COORDS_CACHE.exists() and UMAP_REDUCER_CACHE.exists():
        coords = np.load(UMAP_COORDS_CACHE)
        if coords.shape[0] == all_rna_embs.shape[0]:
            with open(UMAP_REDUCER_CACHE, "rb") as f:
                reducer = pickle.load(f)
            return reducer, coords

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, verbose=True
    )
    coords = reducer.fit_transform(all_rna_embs)
    np.save(UMAP_COORDS_CACHE, coords)
    with open(UMAP_REDUCER_CACHE, "wb") as f:
        pickle.dump(reducer, f)
    return reducer, coords


def project_query_to_umap(reducer, query_emb: np.ndarray) -> np.ndarray:
    q = query_emb.reshape(1, -1)
    return reducer.transform(q)[0]


def similarity(query_emb: np.ndarray, all_rna_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity (embeddings are already L2-normalized) -> (n_cells,)."""
    return all_rna_embs @ query_emb
