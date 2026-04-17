import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D_RNA, D_PROTEIN, D_HIDDEN, D_EMB, MODEL_WEIGHTS


class RNAEncoder(nn.Module):
    def __init__(self, d_in, d_hidden, d_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_emb),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, d_in, d_hidden, d_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_emb),
        )

    def forward(self, x):
        return self.net(x)


class CLIPCITE(nn.Module):
    def __init__(self, d_rna=D_RNA, d_protein=D_PROTEIN, d_hidden=D_HIDDEN, d_emb=D_EMB):
        super().__init__()
        self.rna_projection = RNAEncoder(d_rna, d_hidden, d_emb)
        self.protein_projection = ProjectionHead(d_protein, d_hidden, d_emb)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_rna_embedding(self, rna_enc):
        return F.normalize(self.rna_projection(rna_enc), dim=-1)

    def get_protein_embedding(self, protein_enc):
        return F.normalize(self.protein_projection(protein_enc), dim=-1)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_model(device: torch.device | None = None) -> tuple[CLIPCITE, torch.device]:
    device = device or pick_device()
    model = CLIPCITE().to(device)
    state = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device
