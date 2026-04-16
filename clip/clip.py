import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIP(torch.nn.Module):
    def __init__(self,
                 rna_encoder,
                 queries_encoder,
                 rna_dim,
                 queries_dim,
                 proj_dim=512,
                 init_tau=np.log(1.0),
                 init_b=0):
        super(CLIP, self).__init__()

        self.rna_encoder = rna_encoder
        self.queries_encoder = queries_encoder

        # Learnable projection layer
        self.rna_proj = nn.Sequential(
            torch.nn.Linear(rna_dim, rna_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(rna_dim, proj_dim, bias=False)
        )
        
        self.queries_proj = nn.Sequential(
            torch.nn.Linear(queries_dim, queries_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(queries_dim, proj_dim, bias=False)
        )
        
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(self, rna, queries_tokens):
        rna_emb = self.rna_proj(self.rna_encoder(rna))
        queries_emb = self.queries_proj(self.queries_encoder(queries_tokens))

        # Normalize for cosine similarity
        rna_emb = F.normalize(rna_emb, p=2, dim=-1)
        queries_emb = F.normalize(queries_emb, p=2, dim=-1)
        
        return rna_emb @ queries_emb.t() * self.t_prime.exp() + self.b

    def get_query_embedding(self, query_tokens):
        return F.normalize(
            self.queries_proj(
                self.queries_encoder(query_tokens)
            ), p=2, dim=-1)

    def get_rna_embedding(self, rna):
        return F.normalize(
            self.rna_proj(
                self.rna_encoder(rna)
            ), p=2, dim=-1)
        

if __name__ == "__main__":
    # Example usage
    rna_encoder = nn.Linear(100, 256)  # Dummy RNA encoder
    queries_encoder = nn.Linear(768, 256)  # Dummy text encoder (e.g., BERT)
    
    model = CLIP(rna_encoder, queries_encoder, rna_dim=256, queries_dim=256)
    
    rna_input = torch.randn(32, 100)  # Batch of 32 RNA samples
    query_tokens = torch.randn(32, 768)  # Batch of 32 tokenized queries
    
    similarity_scores = model(rna_input, query_tokens)
    print(similarity_scores.shape)  # Should be (32, 32) for batch-wise similarity scores