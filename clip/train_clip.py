import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from clip import CLIP
from clip_dataset import CLIPDataset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Embeddings.dataset_rna_embeddings import EmbeddingRNADatasetIO
from protein.protein_dataset import EmbeddingProteinQueryDatasetNoIO, EmbeddingProteinQueryDatasetIO

import random
from tqdm import tqdm

def create_train_val_datasets(rna_dataset, query_dataset, train_ratio=0.8):
    """
    Split les datasets en train/val selon les cell_ids uniques pour éviter le dataleakage.
    """
    all_cell_ids = set()
    for idx in tqdm(range(len(rna_dataset))):
        _, cell_id = rna_dataset[idx]
        c_id = cell_id.item() if isinstance(cell_id, torch.Tensor) else cell_id
        all_cell_ids.add(c_id)
        
    all_cell_ids = list(all_cell_ids)
    random.shuffle(all_cell_ids)
    
    split_idx = int(len(all_cell_ids) * train_ratio)
    train_ids = set(all_cell_ids[:split_idx])
    val_ids = set(all_cell_ids[split_idx:])

    print("Instantiate train dataset")
    train_dataset = CLIPDataset(rna_dataset, query_dataset, allowed_cell_ids=train_ids)
    print("Instantiate validation dataset")
    val_dataset = CLIPDataset(rna_dataset, query_dataset, allowed_cell_ids=val_ids)
    
    return train_dataset, val_dataset


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

def train_clip(model, train_loader, val_loader, optimizer, device, epochs=10):
    """
    Training loop for the CLIP model.
    """
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (rna, queries, ids) in enumerate(train_loader):
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
            
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n")

        # Validation todo


if __name__ == "__main__":
    PROJECTION_DIM = 512
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    EPOCHS = 5
    TRAIN_VAL_SPLIT = 0.8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    query_embeddings_path = "./protein/notebooks/cell_texts_augmented_K20_subsetOnly.csv_embeddings.h5"
    rna_embeddings_path = "./Embeddings/RNA_embeddings_final.h5"

    print("Loading datasets")
    rna_dataset = EmbeddingRNADatasetIO(rna_embeddings_path)
    query_dataset = EmbeddingProteinQueryDatasetIO(query_embeddings_path)
        
    #----------------------------- Do not change the code below this line -----------------------------#

    print("Performing split")
    train_dataset, val_dataset = create_train_val_datasets(rna_dataset, query_dataset, train_ratio=TRAIN_VAL_SPLIT)
    
    print("Creating loaders")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Instantiate the CLIP model
    print("Instantiate the model & optimizer")
    model = CLIP(
        proj_dim=PROJECTION_DIM
    )
    
    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Run training
    print("Starting training...")
    train_clip(model, train_loader, val_loader, optimizer, device, epochs=EPOCHS)
    print("Training finished!")

    model_path = "clip_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
