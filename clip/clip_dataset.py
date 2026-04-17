from torch.utils.data import Dataset, DataLoader
import torch
from clip import CLIP

class CLIPDataset(Dataset):
    def __init__(self, rna_dataset: Dataset, query_dataset: Dataset, allowed_cell_ids=None):
        self.rna_dataset = rna_dataset
        self.query_dataset = query_dataset

        # Construction d'un mapping cell_id -> index pour rna_dataset
        self.cell_id_to_rna_idx = {}
        if hasattr(rna_dataset, 'get_all_cell_ids'):
            rna_cell_ids = rna_dataset.get_all_cell_ids()
            for idx, cell_id in enumerate(rna_cell_ids):
                c_id = cell_id.item() if isinstance(cell_id, torch.Tensor) else cell_id
                self.cell_id_to_rna_idx[c_id] = idx
        else:
            for idx in range(len(rna_dataset)):
                _, cell_id = rna_dataset[idx]
                # Si le cell_id est un tensor, on prend sa valeur
                c_id = cell_id.item() if isinstance(cell_id, torch.Tensor) else cell_id
                self.cell_id_to_rna_idx[c_id] = idx
            
        # Filtrage basé sur les cell_ids pour éviter le data leakage
        self.valid_query_indices = []
        
        if hasattr(query_dataset, 'get_all_cell_ids'):
            query_cell_ids = query_dataset.get_all_cell_ids()
            for idx, cell_id in enumerate(query_cell_ids):
                c_id = cell_id.item() if isinstance(cell_id, torch.Tensor) else cell_id
                
                if c_id in self.cell_id_to_rna_idx:
                    if allowed_cell_ids is None or c_id in allowed_cell_ids:
                        self.valid_query_indices.append(idx)
        else:
            for idx in range(len(query_dataset)):
                _, cell_id = query_dataset[idx]
                c_id = cell_id.item() if isinstance(cell_id, torch.Tensor) else cell_id
                
                # On s'assure que le cell_id existe dans le RNA et qu'il est autorisé (tr ou val)
                if c_id in self.cell_id_to_rna_idx:
                    if allowed_cell_ids is None or c_id in allowed_cell_ids:
                        self.valid_query_indices.append(idx)

    def __len__(self):
        return len(self.valid_query_indices)

    def __getitem__(self, idx):
        # On récupère l'index réel dans query_dataset
        real_idx = self.valid_query_indices[idx]
        query_emb, cell_id = self.query_dataset[real_idx]
        
        c_id = cell_id.item() if isinstance(cell_id, torch.Tensor) else cell_id
        rna_idx = self.cell_id_to_rna_idx[c_id]
        rna_emb, _ = self.rna_dataset[rna_idx]
        
        return rna_emb, query_emb, cell_id