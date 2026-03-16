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
        load_strategy="lazy",
    ):
        if load_strategy not in {"eager", "lazy"}:
            raise ValueError(
                "load_strategy must be either 'eager' or 'lazy'. "
                f"Received: {load_strategy!r}."
            )

        self.load_strategy = load_strategy
        self._tensor_specs = [
            ("hits", hits_file),
            ("raw_hits", raw_hits_file),
            ("padding_mask", padding_mask_file),
            ("labels", label_file),
        ]
        if rec_label_file is not None:
            self._tensor_specs.append(("rec_labels", rec_label_file))

        self._tensors = None
        if self.load_strategy == "eager":
            self._ensure_loaded()

    def _ensure_loaded(self):
        if self._tensors is not None:
            return self._tensors

        loaded_entries = [
            (name, path, torch.load(path, map_location="cpu"))
            for name, path in self._tensor_specs
        ]
        self._validate_loaded_entries(loaded_entries)

        loaded_map = {name: tensor for name, _, tensor in loaded_entries}
        self._tensors = (
            loaded_map["hits"],
            loaded_map["raw_hits"],
            loaded_map["padding_mask"],
            loaded_map["labels"],
            loaded_map.get("rec_labels"),
        )
        return self._tensors

    def _validate_loaded_entries(self, loaded_entries):
        reference_name, reference_path, reference_tensor = loaded_entries[0]
        expected_length = len(reference_tensor)
        mismatches = []

        for name, path, tensor in loaded_entries[1:]:
            if len(tensor) != expected_length:
                mismatches.append(
                    (
                        name,
                        path,
                        len(tensor),
                    )
                )

        if mismatches:
            mismatch_summary = ", ".join(
                f"{name} ({path}) has {length} events"
                for name, path, length in mismatches
            )
            raise ValueError(
                "KM3Loader backing tensors must share the same first dimension. "
                f"{reference_name} ({reference_path}) has {expected_length} events; "
                f"mismatches: {mismatch_summary}."
            )

    def __len__(self):
        hits, _, _, _, _ = self._ensure_loaded()
        return len(hits)

    def __getitem__(self, idx):
        hits, raw_hits, padding_mask, labels, rec_labels = self._ensure_loaded()
        sample = (
            hits[idx],
            raw_hits[idx],
            padding_mask[idx],
            labels[idx],
        )
        if rec_labels is None:
            return sample

        return sample + (rec_labels[idx],)
