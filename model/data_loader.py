import torch
from torch.utils.data import Dataset


class KM3Loader(Dataset):
    def __init__(self, hits_file, label_file):
        self.labels = torch.load(label_file)
        self.hits = torch.load(hits_file)

    def __len__(self):
        return len(self.hits)

    def __getitem__(self, idx):
        return self.hits[idx], self.labels[idx]
