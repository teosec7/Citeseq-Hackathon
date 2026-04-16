import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

class ProteinQueryEncoder(nn.Module):
    def __init__(self, model_name, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(ProteinQueryEncoder, self).__init__()
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

if __name__ == "__main__":
    models = ["dmis-lab/biobert-v1.1", "ncbi/MedCPT-Query-Encoder", "PharMolix/BioMedGPT-LM-7B"]
    model_name = models[2] 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )

    encoder = ProteinQueryEncoder(model_name, bnb_config).to('cuda')
    queries = ([None, None], [
        "What is the function of protein X?", 
        "How does protein Y interact with other proteins?", 
        "Short query", 
        "A very long query that exceeds typical token limits and should be truncated appropriately to test the encoder's handling of long inputs. This query is intentionally verbose to ensure that it triggers the truncation mechanism in the tokenizer, allowing us to verify that the encoder can process long queries without errors."])
    embeddings = encoder(queries)
    print(embeddings.shape)