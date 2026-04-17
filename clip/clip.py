import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIP(torch.nn.Module):
    def __init__(self,
                    rna_dim, 
                    queries_dim,
                    proj_dim=512,
                    init_tau=np.log(1.0),
                    init_b=0):
        super(CLIP, self).__init__()
        
        # Learnable projections layer
        self.rna_proj_layer = nn.Sequential(
            torch.nn.Linear(rna_dim, rna_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(rna_dim, proj_dim, bias=False)
        )
        
        self.queries_proj_layer = nn.Sequential(
            torch.nn.Linear(queries_dim, queries_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(queries_dim, proj_dim, bias=False)
        )
        
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(self, rna_emb, queries_emb):
        rna_proj = self.rna_proj_layer(rna_emb)
        queries_proj = self.queries_proj_layer(queries_emb)

        # Normalize for cosine similarity
        rna_proj = F.normalize(rna_proj, p=2, dim=-1)
        queries_proj = F.normalize(queries_proj, p=2, dim=-1)
        
        return rna_proj @ queries_proj.t() * self.t_prime.exp() + self.b

    def get_rna_projection(self, rna_emb):
        return self.rna_proj_layer(rna_emb)
    
    def get_queries_projection(self, queries_emb):
        return self.queries_proj_layer(queries_emb)