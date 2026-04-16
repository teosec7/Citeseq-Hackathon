import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ProteinQueryEncoder(nn.Module):
    def __init__(self, model_name, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(ProteinQueryEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
        self.device = device
        self.model.to(device)

    def forward(self, queries):
        # tokenize the queries
        encoded = self.tokenizer(
            queries, 
            padding=True, 
            return_tensors='pt', 
        ).to(self.device)

        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = self.model(**encoded).last_hidden_state[:, 0, :]
        
        return embeds

if __name__ == "__main__":
    model_name = "ncbi/MedCPT-Query-Encoder"
    encoder = ProteinQueryEncoder(model_name, None)
    queries = ["What is the function of this protein?", "How does this protein interact with other molecules?"]
    embeddings = encoder(queries)
    print(embeddings.shape)