import json
import os

import joblib
import torch
from normalisation import apply_nested_tensor_list_scalers, fit_nested_tensor_list_scalers
from padding import pad_tensor
from root_loader import process_path

MAX_HITS = 512
DATA_PATH = "./data"
ROOT_PATH_PATTERN = "/root/mcv7.1.mupage_tuned.sirene.jterbr0000*"
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SPLIT_SEED = 42
FEATURE_NAMES = [
    "relative_t",
    "pos_x",
    "pos_y",
    "pos_z",
    "dir_x",
    "dir_y",
    "dir_z",
    "log_tot",
    "hit_rank",
    "radius_xy",
]


def build_split_indices(num_samples):
    if not torch.isclose(
        torch.tensor(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION),
        torch.tensor(1.0),
    ):
        raise ValueError("Dataset split fractions must sum to 1.0.")

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    shuffled_indices = torch.randperm(num_samples, generator=generator).tolist()

    train_end = int(num_samples * TRAIN_FRACTION)
    val_end = train_end + int(num_samples * VAL_FRACTION)

    return {
        "train": shuffled_indices[:train_end],
        "val": shuffled_indices[train_end:val_end],
        "test": shuffled_indices[val_end:],
    }


def take_items(items, indices):
    return [items[index] for index in indices]


def pad_hits_list(hits_list, max_hits):
    padded_hits = []
    padding_masks = []

    for hit_tensor in hits_list:
        padded_tensor, padding_mask = pad_tensor(hit_tensor, target_length=max_hits)
        padded_hits.append(padded_tensor)
        padding_masks.append(padding_mask.cpu())

    return torch.stack(padded_hits), torch.stack(padding_masks)


def save_split(data_path, split_name, muons, rec_muons, muon_energies, raw_hits, hit_scalers):
    normalized_hits = apply_nested_tensor_list_scalers(raw_hits, hit_scalers)

    final_hits, final_masks = pad_hits_list(normalized_hits, max_hits=MAX_HITS)
    final_raw_hits, _ = pad_hits_list(raw_hits, max_hits=MAX_HITS)

    torch.save(torch.stack(muons), f"{data_path}/{split_name}_muons.pt")
    torch.save(torch.stack(rec_muons), f"{data_path}/{split_name}_muons_rec.pt")
    torch.save(torch.tensor(muon_energies, dtype=torch.float32), f"{data_path}/{split_name}_muons_e.pt")
    torch.save(final_hits, f"{data_path}/{split_name}_hits.pt")
    torch.save(final_raw_hits, f"{data_path}/{split_name}_hits_raw.pt")
    torch.save(final_masks, f"{data_path}/{split_name}_padding_mask.pt")


if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(f"{DATA_PATH}/scalers", exist_ok=True)

    muons, hits, rec_muons, muon_energies = process_path(
        f"{DATA_PATH}{ROOT_PATH_PATTERN}",
        max_hits=MAX_HITS,
    )

    if not hits:
        raise RuntimeError("No events were loaded from the ROOT files.")

    split_indices = build_split_indices(len(hits))
    train_hits = take_items(hits, split_indices["train"])
    hits_scalers = fit_nested_tensor_list_scalers(train_hits)

    for split_name, indices in split_indices.items():
        save_split(
            data_path=DATA_PATH,
            split_name=split_name,
            muons=take_items(muons, indices),
            rec_muons=take_items(rec_muons, indices),
            muon_energies=take_items(muon_energies, indices),
            raw_hits=take_items(hits, indices),
            hit_scalers=hits_scalers,
        )

    metadata = {
        "max_hits": MAX_HITS,
        "input_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "splits": {split_name: len(indices) for split_name, indices in split_indices.items()},
        "split_seed": SPLIT_SEED,
    }
    with open(f"{DATA_PATH}/metadata.json", "w", encoding="ascii") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    for scaler_index, scaler in enumerate(hits_scalers):
        joblib.dump(scaler, f"{DATA_PATH}/scalers/hits_scaler{scaler_index}.joblib")
