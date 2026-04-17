import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd


class TextProteinQueryDataset(Dataset):
    def __init__(self, csv_file, query_transform=None, in_memory=True):
        data = pd.read_csv(csv_file)
        self.cell_ids = list(data["cell_id"])
        self.queries = list(data["text"])
        self.len = len(self.queries)

        if in_memory:
            if query_transform is not None:
                self.queries = [query_transform(query) for query in self.queries]
        else:
            self.query_transform = query_transform
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # If the dataset is not in memory, apply the query transform on-the-fly
        if hasattr(self, 'query_transform'):
            query = self.query_transform(self.queries[idx])
        else:
            query = self.queries[idx]

        return query, self.cell_ids[idx]
    
    def save_to_csv(self, output_file):
        df = pd.DataFrame({
            "cell_id": self.cell_ids,
            "text": self.queries
        })
        df.to_csv(output_file, index=False)

class EmbeddingProteinQueryDatasetNoIO(Dataset):
    """
    Dataset that loads all pairs of query, cell_id into the RAM at initialization
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.cell_ids = []
        self.embeddings = []
        with h5py.File(self.file_path, "r") as f:
            self.length = f["embeddings"].shape[0]
            
            for idx in range(self.length):
                self.embeddings.append(torch.from_numpy(f["embeddings"][idx]))
                
                cell_id = f["cell_ids"][idx]
                if isinstance(cell_id, bytes):
                    cell_id = cell_id.decode("utf-8")
                self.cell_ids.append(f["cell_ids"][idx])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.embeddings[idx], self.cell_ids[idx]


class EmbeddingProteinQueryDatasetIO(Dataset):
    """
    Dataset reading the original file at each retrieval
    """
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            self.length = f["embeddings"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            embedding = torch.from_numpy(f["embeddings"][idx])
            cell_id = f["cell_ids"][idx]
            
            if isinstance(cell_id, bytes):
                cell_id = cell_id.decode("utf-8")
                
        return embedding, cell_id
