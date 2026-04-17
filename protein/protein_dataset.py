import torch
from torch.utils.data import Dataset
import pandas as pd

class ProteinQueryDataset(Dataset):
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