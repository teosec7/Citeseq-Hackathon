import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from clip import CLIP

# todo
class DummyDataset(Dataset):
    def __init__(self):
        pass
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

def clip_loss(logits):
    """
    Computes the symmetric contrastive cross-entropy loss.
    """
    # Logits shape: [batch_size, batch_size]
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    
    # Loss for RNA to queries
    loss_rna = F.cross_entropy(logits, labels)
    # Loss for queries to RNA
    loss_query = F.cross_entropy(logits.t(), labels)
    
    return (loss_rna + loss_query) / 2.0

def train_clip(model, dataloader, optimizer, device, epochs=10):
    """
    Training loop for the CLIP model.
    """
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (rna, queries) in enumerate(dataloader):
            rna = rna.to(device)
            queries = queries.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: compute similarity scores
            logits = model(rna, queries)
            
            # Compute loss
            loss = clip_loss(logits)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"End of Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n")

if __name__ == "__main__":
    # Hyperparameters
    RNA_INPUT_DIM = 100
    QUERY_INPUT_DIM = 768
    RNA_EMB_DIM = 256
    QUERY_EMB_DIM = 256
    PROJECTION_DIM = 512
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dummy encoders
    # Don't forget to freeze the pretrained models !
    rna_encoder = nn.Sequential(
        nn.Linear(RNA_INPUT_DIM, RNA_EMB_DIM),
        nn.ReLU()
    )
    
    queries_encoder = nn.Sequential(
        nn.Linear(QUERY_INPUT_DIM, QUERY_EMB_DIM),
        nn.ReLU()
    )
    
    
    # Setup Dataloader
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    #----------------------------- Do not change the code below this line -----------------------------#

    # Instantiate the CLIP model
    model = CLIP(
        rna_encoder=rna_encoder,
        queries_encoder=queries_encoder,
        rna_dim=RNA_EMB_DIM,
        queries_dim=QUERY_EMB_DIM,
        proj_dim=PROJECTION_DIM
    )
    
    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Run training
    print("Starting training...")
    train_clip(model, dataloader, optimizer, device, epochs=EPOCHS)
    print("Training finished!")
