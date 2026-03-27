import json

import torch
from torch.utils.data import Dataset


class KM3Loader(Dataset):
    def __init__(
        self,
        hits_file,
        raw_hits_file,
        padding_mask_file,
        label_file=None,
        target_file=None,
        rec_label_file=None,
        energy_label_file=None,
        metadata_file=None,
        load_strategy="lazy",
    ):
        if load_strategy not in {"eager", "lazy"}:
            raise ValueError(
                "load_strategy must be either 'eager' or 'lazy'. "
                f"Received: {load_strategy!r}."
            )
        if label_file is None and target_file is None:
            raise ValueError("KM3Loader requires either label_file or target_file.")

        self.load_strategy = load_strategy
        self.metadata = None
        if metadata_file is not None:
            with open(metadata_file, "r", encoding="ascii") as metadata_handle:
                self.metadata = json.load(metadata_handle)

        resolved_target_file = target_file or label_file
        self._tensor_specs = [
            ("hits", hits_file),
            ("raw_hits", raw_hits_file),
            ("padding_mask", padding_mask_file),
            ("targets", resolved_target_file),
        ]
        # Keep optional targets append-only so older datasets still load unchanged.
        if rec_label_file is not None:
            self._tensor_specs.append(("rec_labels", rec_label_file))
        if energy_label_file is not None:
            self._tensor_specs.append(("energy_labels", energy_label_file))

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
        # Fail early if a partially written preprocessing run produced misaligned tensors.
        self._validate_loaded_entries(loaded_entries)

        loaded_map = {name: tensor for name, _, tensor in loaded_entries}
        self._tensors = (
            loaded_map["hits"],
            loaded_map["raw_hits"],
            loaded_map["padding_mask"],
            loaded_map["targets"],
            loaded_map.get("rec_labels"),
            loaded_map.get("energy_labels"),
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
        hits, _, _, _, _, _ = self._ensure_loaded()
        return len(hits)

    @property
    def has_rec_labels(self):
        return any(name == "rec_labels" for name, _ in self._tensor_specs)

    @property
    def has_energy_labels(self):
        return any(name == "energy_labels" for name, _ in self._tensor_specs)

    def _should_expose_muons_alias(self, targets):
        if self.metadata is not None:
            return self.metadata.get("task") == "direction"
        return targets.ndim > 1 and targets.shape[-1] == 3

    def __getitem__(self, idx):
        hits, raw_hits, padding_mask, targets, rec_labels, energy_labels = (
            self._ensure_loaded()
        )
        sample = {
            "hits": hits[idx],
            "raw_hits": raw_hits[idx],
            "padding_mask": padding_mask[idx],
            "target": targets[idx],
        }
        if self._should_expose_muons_alias(targets):
            sample["muons"] = targets[idx]
        if rec_labels is not None:
            sample["rec_muons"] = rec_labels[idx]
        if energy_labels is not None:
            sample["energy"] = energy_labels[idx]

        return sample
