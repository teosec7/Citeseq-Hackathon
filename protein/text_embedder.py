import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

class TextProteinQueryEncoder(nn.Module):
    def __init__(self, model_name, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TextProteinQueryEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(device)

    def forward(self, sample):
        # Tokenize the queries
        _, queries = sample
        encoded = self.tokenizer(
            queries, 
            padding=True, 
            return_tensors='pt', 
        ).to(self.device)

        # Encode the queries : use the [CLS] last hidden states as the embedding
        embeds = self.model(**encoded).last_hidden_state[:, 0, :]
        
        return embeds