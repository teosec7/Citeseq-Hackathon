"""RNA projection + text query encoding.

RNA inputs are the precomputed merged embeddings (e.g. scGPT + PCA + scVI +
C2S, 1626-d) stored in `data/rna_embeddings.h5`. We project them through the
trained RNA head, L2-normalise, and cache the result as a float32 .npy.

Text queries use the `nomic-ai/nomic-embed-text-v1` encoder with the required
`search_query: ` prefix; the [CLS] hidden state is passed through the trained
query head and L2-normalised.
"""
from __future__ import annotations

import pickle

import numpy as np
import torch

from .config import (
    D_EMB,
    RNA_EMB_CACHE,
    TEXT_ENCODER_NAME,
    TEXT_QUERY_PREFIX,
    UMAP_COORDS_CACHE,
    UMAP_REDUCER_CACHE,
)
from .model import CLIP


@torch.no_grad()
def compute_all_rna_embeddings(
    model: CLIP,
    rna_encodings: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    if RNA_EMB_CACHE.exists():
        arr = np.load(RNA_EMB_CACHE)
        if arr.shape[0] == rna_encodings.shape[0] and arr.shape[1] == D_EMB:
            return arr

    n = rna_encodings.shape[0]
    out = np.empty((n, D_EMB), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(
            np.asarray(rna_encodings[start:end]), dtype=torch.float32, device=device
        )
        out[start:end] = model.get_rna_embedding(batch).cpu().numpy()
    np.save(RNA_EMB_CACHE, out)
    return out


class QueryEncoder:
    """Nomic-embed-text-v1 + the trained query projection head."""

    def __init__(self, model: CLIP, device: torch.device):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
        self.text_model = AutoModel.from_pretrained(
            TEXT_ENCODER_NAME, trust_remote_code=True
        ).to(device)
        self.text_model.eval()
        self.model = model
        self.device = device

    @torch.no_grad()
    def encode(self, query_text: str) -> np.ndarray:
        prompt = TEXT_QUERY_PREFIX + query_text
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        hidden = self.text_model(**tokens).last_hidden_state[:, 0, :]
        q_emb = self.model.get_protein_embedding(hidden).cpu().numpy().squeeze()
        return q_emb.astype(np.float32)


def fit_or_load_umap(all_rna_embs: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.3):
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
    return reducer.transform(query_emb.reshape(1, -1))[0]


def similarity(query_emb: np.ndarray, all_rna_embs: np.ndarray) -> np.ndarray:
    return all_rna_embs @ query_emb
