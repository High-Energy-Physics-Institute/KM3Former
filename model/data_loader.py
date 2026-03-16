import torch
from torch.utils.data import Dataset


class KM3Loader(Dataset):
    def __init__(
        self,
        hits_file,
        raw_hits_file,
        padding_mask_file,
        label_file,
        rec_label_file=None,
    ):
        self.labels = torch.load(label_file, map_location="cpu")
        self.hits = torch.load(hits_file, map_location="cpu")
        self.raw_hits = torch.load(raw_hits_file, map_location="cpu")
        self.padding_mask = torch.load(padding_mask_file, map_location="cpu")
        self.rec_labels = None
        if rec_label_file is not None:
            self.rec_labels = torch.load(rec_label_file, map_location="cpu")

    def __len__(self):
        return len(self.hits)

    def __getitem__(self, idx):
        sample = (
            self.hits[idx],
            self.raw_hits[idx],
            self.padding_mask[idx],
            self.labels[idx],
        )
        if self.rec_labels is None:
            return sample

        return sample + (self.rec_labels[idx],)
