"""CLIP model — two MLP projection heads onto a shared embedding space.

One head projects RNA features, the other projects text-query features. Both
are L2-normalised so cosine similarity reduces to a dot product.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D_EMB, D_QUERY, D_RNA, MODEL_WEIGHTS


class CLIP(nn.Module):
    def __init__(self, rna_dim=D_RNA, queries_dim=D_QUERY, proj_dim=D_EMB,
                 init_tau=float(np.log(1.0)), init_b=0.0):
        super().__init__()
        self.rna_proj_layer = nn.Sequential(
            nn.Linear(rna_dim, rna_dim, bias=False),
            nn.ReLU(),
            nn.Linear(rna_dim, proj_dim, bias=False),
        )
        self.queries_proj_layer = nn.Sequential(
            nn.Linear(queries_dim, queries_dim, bias=False),
            nn.ReLU(),
            nn.Linear(queries_dim, proj_dim, bias=False),
        )
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def get_rna_embedding(self, rna_emb):
        return F.normalize(self.rna_proj_layer(rna_emb), p=2, dim=-1)

    def get_protein_embedding(self, queries_emb):
        return F.normalize(self.queries_proj_layer(queries_emb), p=2, dim=-1)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_model(device: torch.device | None = None) -> tuple[CLIP, torch.device]:
    device = device or pick_device()
    model = CLIP().to(device)
    state = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device
