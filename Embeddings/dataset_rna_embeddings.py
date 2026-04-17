import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py

class EmbeddingRNADatasetIO(Dataset):
    """
    Dataset reading the original file efficiently from disk without RAM overload.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None 
        
        with h5py.File(self.file_path, "r") as f:
            self.length = f["embeddings"].shape[0]

    def get_all_cell_ids(self):
        with h5py.File(self.file_path, "r") as f:
            ids = f["cell_ids"][:]
            return [cid.decode("utf-8") if isinstance(cid, bytes) else cid for cid in ids]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open once per worker
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
            
        embedding = torch.from_numpy(self.file["embeddings"][idx])
        cell_id = self.file["cell_ids"][idx]
        
        if isinstance(cell_id, bytes):
            cell_id = cell_id.decode("utf-8")
                
        return embedding, cell_id