import argparse
import copy
import json
import os
from glob import glob

import h5py
import numpy as np
import torch
from normalisation import (
    DEFAULT_AFFINE_FEATURE_INDICES,
    DEFAULT_POSITION_SCALE,
    apply_deterministic_hit_transforms,
    apply_feature_stats,
    compute_feature_stats,
    save_feature_stats,
)
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
POSITION_SCALE = DEFAULT_POSITION_SCALE
AFFINE_FEATURE_NAMES = [FEATURE_NAMES[index] for index in DEFAULT_AFFINE_FEATURE_INDICES]
DEFAULT_HDF5_COLUMN_MAP = {
    "time": 0,
    "x": 1,
    "y": 2,
    "z": 3,
    "dir_x": 4,
    "dir_y": 5,
    "dir_z": 6,
    "tot": 7,
}
DEFAULT_PREPROCESS_CONFIG = {
    "paths": {
        "data_path": DATA_PATH,
        "root_path_pattern": f"{DATA_PATH}{ROOT_PATH_PATTERN}",
        "hdf5_path": f"{DATA_PATH}/muon_data_7224_7247.h5",
    },
    "source": {
        "format": "root",
    },
    "task": {
        "name": "direction",
    },
    "data": {
        "max_hits": MAX_HITS,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "split_seed": SPLIT_SEED,
    },
    "hdf5": {
        "hits_dataset": "hits",
        "label_dataset": "mc_muons",
        "label_col": 0,
        "sort_by_time": True,
        "count_class_values": None,
        "columns": DEFAULT_HDF5_COLUMN_MAP,
    },
}
REQUIRED_HDF5_COLUMNS = tuple(DEFAULT_HDF5_COLUMN_MAP)


def deep_merge_dicts(base, overrides):
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_json_config(config_path, default_config):
    config = copy.deepcopy(default_config)
    if config_path is None:
        return config

    with open(config_path, "r", encoding="ascii") as config_file:
        overrides = json.load(config_file)

    return deep_merge_dicts(config, overrides)


def resolve_preprocess_config(config_path=None, overrides=None):
    resolved = load_json_config(
        config_path=config_path,
        default_config=DEFAULT_PREPROCESS_CONFIG,
    )
    if overrides:
        resolved = deep_merge_dicts(resolved, overrides)
    return resolved


def build_split_indices(
    num_samples,
    train_fraction=TRAIN_FRACTION,
    val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION,
    split_seed=SPLIT_SEED,
):
    if not torch.isclose(
        torch.tensor(train_fraction + val_fraction + test_fraction),
        torch.tensor(1.0),
    ):
        raise ValueError("Dataset split fractions must sum to 1.0.")

    generator = torch.Generator().manual_seed(split_seed)
    shuffled_indices = torch.randperm(num_samples, generator=generator).tolist()

    train_end = int(num_samples * train_fraction)
    val_end = train_end + int(num_samples * val_fraction)

    return {
        "train": shuffled_indices[:train_end],
        "val": shuffled_indices[train_end:val_end],
        "test": shuffled_indices[val_end:],
    }


def take_items(items, indices):
    return [items[index] for index in indices]


def pad_hits_list(hits_list, max_hits, feature_dim=len(FEATURE_NAMES)):
    if not hits_list:
        return (
            torch.zeros((0, max_hits, feature_dim), dtype=torch.float32),
            torch.zeros((0, max_hits), dtype=torch.bool),
        )

    padded_hits = []
    padding_masks = []

    for hit_tensor in hits_list:
        padded_tensor, padding_mask = pad_tensor(hit_tensor, target_length=max_hits)
        padded_hits.append(padded_tensor)
        padding_masks.append(padding_mask.cpu())

    return torch.stack(padded_hits), torch.stack(padding_masks)


def _normalize_label_value(value):
    normalized = np.asarray(value).item()
    if isinstance(normalized, np.generic):
        normalized = normalized.item()
    if isinstance(normalized, float) and float(normalized).is_integer():
        return int(normalized)
    if isinstance(normalized, (np.integer, int)):
        return int(normalized)
    return normalized


def _resolve_input_paths(path_pattern):
    matched_paths = sorted(glob(path_pattern))
    if matched_paths:
        return matched_paths
    if os.path.exists(path_pattern):
        return [path_pattern]
    raise FileNotFoundError(f"No files matched input path or glob: {path_pattern}")


def _validate_hdf5_column_map(column_map):
    missing = [name for name in REQUIRED_HDF5_COLUMNS if name not in column_map]
    if missing:
        raise ValueError(
            "HDF5 column map is missing required semantic fields: "
            + ", ".join(missing)
        )


def _validate_hdf5_feature_indices(column_map, feature_count):
    invalid = [
        f"{name}={index}"
        for name, index in column_map.items()
        if index < 0 or index >= feature_count
    ]
    if invalid:
        raise ValueError(
            "HDF5 column map references indices outside the hits feature dimension: "
            + ", ".join(invalid)
        )


def _extract_label_values(label_dataset, label_col):
    if label_dataset.ndim == 1:
        if label_col not in (None, 0):
            raise ValueError(
                "label_col must be 0 or null when the HDF5 label dataset is 1-dimensional."
            )
        return np.asarray(label_dataset[:])
    return np.asarray(label_dataset[:, label_col])


def build_canonical_hdf5_hits(hit_rows, column_map, max_hits, sort_by_time=True):
    _validate_hdf5_column_map(column_map)

    valid_rows = hit_rows[~np.all(hit_rows == 0.0, axis=1)]
    if valid_rows.shape[0] == 0:
        return None

    if sort_by_time:
        order = np.argsort(valid_rows[:, column_map["time"]], kind="mergesort")
        valid_rows = valid_rows[order]

    if max_hits is not None:
        valid_rows = valid_rows[:max_hits]
    if valid_rows.shape[0] == 0:
        return None

    x_values = valid_rows[:, column_map["x"]].astype(np.float32, copy=False)
    y_values = valid_rows[:, column_map["y"]].astype(np.float32, copy=False)
    num_hits = valid_rows.shape[0]
    if num_hits == 1:
        hit_rank = np.zeros(1, dtype=np.float32)
    else:
        hit_rank = np.linspace(0.0, 1.0, num=num_hits, dtype=np.float32)
    radius_xy = np.sqrt(np.square(x_values) + np.square(y_values)).astype(np.float32)

    canonical = np.stack(
        [
            valid_rows[:, column_map["time"]].astype(np.float32, copy=False),
            x_values,
            y_values,
            valid_rows[:, column_map["z"]].astype(np.float32, copy=False),
            valid_rows[:, column_map["dir_x"]].astype(np.float32, copy=False),
            valid_rows[:, column_map["dir_y"]].astype(np.float32, copy=False),
            valid_rows[:, column_map["dir_z"]].astype(np.float32, copy=False),
            valid_rows[:, column_map["tot"]].astype(np.float32, copy=False),
            hit_rank,
            radius_xy,
        ],
        axis=-1,
    )
    return torch.from_numpy(canonical)


def _build_class_mapping(raw_targets, configured_class_values=None):
    if configured_class_values is None:
        class_values = sorted({_normalize_label_value(value) for value in raw_targets})
    else:
        class_values = [
            _normalize_label_value(value) for value in configured_class_values
        ]

    class_to_index = {value: index for index, value in enumerate(class_values)}
    missing = sorted(
        {
            _normalize_label_value(value)
            for value in raw_targets
            if _normalize_label_value(value) not in class_to_index
        }
    )
    if missing:
        raise ValueError(
            "Observed muon-count labels are missing from the configured class mapping: "
            + ", ".join(map(str, missing))
        )

    mapped_targets = [class_to_index[_normalize_label_value(value)] for value in raw_targets]
    return class_values, class_to_index, mapped_targets


def load_hdf5_muon_count_data(file_path_pattern, hdf5_config, max_hits):
    column_map = {
        name: int(index) for name, index in hdf5_config["columns"].items()
    }
    _validate_hdf5_column_map(column_map)

    hits_dataset_name = hdf5_config["hits_dataset"]
    label_dataset_name = hdf5_config["label_dataset"]
    label_col = hdf5_config.get("label_col", 0)
    sort_by_time = bool(hdf5_config.get("sort_by_time", True))
    configured_class_values = hdf5_config.get("count_class_values")

    hits = []
    raw_targets = []
    source_paths = _resolve_input_paths(file_path_pattern)

    for path in source_paths:
        with h5py.File(path, "r") as h5_file:
            hits_dataset = h5_file[hits_dataset_name]
            label_values = _extract_label_values(h5_file[label_dataset_name], label_col)
            if hits_dataset.shape[0] != len(label_values):
                raise ValueError(
                    f"HDF5 hits dataset and label dataset are misaligned in {path}."
                )

            _validate_hdf5_feature_indices(column_map, hits_dataset.shape[-1])

            for event_index in range(hits_dataset.shape[0]):
                canonical_hits = build_canonical_hdf5_hits(
                    hit_rows=hits_dataset[event_index],
                    column_map=column_map,
                    max_hits=max_hits,
                    sort_by_time=sort_by_time,
                )
                if canonical_hits is None:
                    continue

                hits.append(canonical_hits)
                raw_targets.append(_normalize_label_value(label_values[event_index]))

    class_values, class_to_index, mapped_targets = _build_class_mapping(
        raw_targets=raw_targets,
        configured_class_values=configured_class_values,
    )
    return {
        "hits": hits,
        "targets": mapped_targets,
        "rec_targets": None,
        "energies": None,
        "target_name": "muon_count",
        "target_kind": "multiclass",
        "target_dim": len(class_values),
        "class_values": class_values,
        "class_to_index": class_to_index,
        "source_details": {
            "hdf5_paths": source_paths,
            "hits_dataset": hits_dataset_name,
            "label_source": {
                "dataset": label_dataset_name,
                "column": label_col,
            },
            "hit_column_map": column_map,
            "sort_by_time": sort_by_time,
        },
    }


def load_source_data(resolved_config):
    source_format = resolved_config["source"]["format"]
    task_name = resolved_config["task"]["name"]
    max_hits = resolved_config["data"]["max_hits"]
    paths_config = resolved_config["paths"]

    if source_format == "root":
        if task_name != "direction":
            raise ValueError("The ROOT preprocessing path only supports the direction task.")

        targets, hits, rec_targets, energies = process_path(
            paths_config["root_path_pattern"],
            max_hits=max_hits,
        )
        return {
            "hits": hits,
            "targets": targets,
            "rec_targets": rec_targets,
            "energies": energies,
            "target_name": "muon_direction",
            "target_kind": "vector_regression",
            "target_dim": 3,
            "class_values": None,
            "class_to_index": None,
            "source_details": {
                "root_path_pattern": paths_config["root_path_pattern"],
            },
        }

    if source_format == "hdf5":
        if task_name != "muon_count":
            raise ValueError("The HDF5 preprocessing path currently supports only the muon_count task.")

        return load_hdf5_muon_count_data(
            file_path_pattern=paths_config["hdf5_path"],
            hdf5_config=resolved_config["hdf5"],
            max_hits=max_hits,
        )

    raise ValueError(f"Unsupported source format: {source_format!r}")


def _stack_primary_targets(targets, target_kind, target_dim):
    if target_kind == "vector_regression":
        if not targets:
            return torch.zeros((0, target_dim), dtype=torch.float32)
        return torch.stack([target.to(dtype=torch.float32) for target in targets])

    if target_kind == "multiclass":
        if not targets:
            return torch.zeros((0,), dtype=torch.long)
        return torch.tensor(targets, dtype=torch.long)

    raise ValueError(f"Unsupported target_kind: {target_kind!r}")


def _stack_optional_vectors(vectors, vector_dim):
    if vectors is None:
        return None
    if not vectors:
        return torch.zeros((0, vector_dim), dtype=torch.float32)
    return torch.stack([vector.to(dtype=torch.float32) for vector in vectors])


def _stack_optional_scalars(values):
    if values is None:
        return None
    if not values:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


def build_metadata(
    split_indices,
    *,
    max_hits=MAX_HITS,
    source_format="root",
    task="direction",
    target_name="muon_direction",
    target_kind="vector_regression",
    target_dim=3,
    class_values=None,
    class_to_index=None,
    source_details=None,
    split_seed=SPLIT_SEED,
):
    # Keep the legacy filename for compatibility, but describe the tensor by its actual role.
    saved_tensors = {
        "hits.pt": {
            "semantic_role": "model_input",
            "description": "deterministic transforms followed by train-fit affine normalization",
        },
        "hits_raw.pt": {
            "semantic_role": "pairwise_bias_input",
            "description": "legacy filename; contains deterministic geometry/time transforms for pairwise bias, not untouched ROOT hits",
        },
        "padding_mask.pt": {
            "semantic_role": "padding_mask",
            "description": "boolean mask where True marks padded hit rows",
        },
        "targets.pt": {
            "semantic_role": "primary_target",
            "description": "task-aware primary training target shared by all preprocessing sources",
        },
    }
    if task == "direction":
        saved_tensors["muons.pt"] = {
            "semantic_role": "truth_direction",
            "description": "normalized Monte Carlo muon direction",
        }
        saved_tensors["muons_rec.pt"] = {
            "semantic_role": "reconstructed_direction",
            "description": "normalized reconstructed track direction chosen by the ranking heuristic",
        }
        saved_tensors["muons_e.pt"] = {
            "semantic_role": "auxiliary_energy_label",
            "description": "Monte Carlo muon energy reserved for future multi-task training",
        }

    metadata = {
        "max_hits": max_hits,
        "input_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "splits": {
            split_name: len(indices) for split_name, indices in split_indices.items()
        },
        "split_seed": split_seed,
        "source_format": source_format,
        "task": task,
        "target_name": target_name,
        "target_kind": target_kind,
        "target_dim": target_dim,
        "saved_tensors": saved_tensors,
        "normalization": {
            "pairwise_bias_input": "deterministic physics transforms only",
            "model_input": "deterministic physics transforms + global affine stats",
            "position_scale": POSITION_SCALE,
            "affine_feature_indices": list(DEFAULT_AFFINE_FEATURE_INDICES),
            "affine_feature_names": AFFINE_FEATURE_NAMES,
            "unchanged_feature_names": ["dir_x", "dir_y", "dir_z"],
            "time_reference": "per-event minimum hit time",
            "tot_transform": "log1p(clamp(tot, min=0))",
        },
    }
    if class_values is not None:
        metadata["class_values"] = list(class_values)
        metadata["class_to_index"] = dict(class_to_index or {})
    if source_details is not None:
        metadata["source_details"] = source_details
    return metadata


def save_split(
    data_path,
    split_name,
    *,
    targets,
    raw_hits,
    hit_stats,
    target_kind,
    target_dim,
    task,
    rec_targets=None,
    energies=None,
    max_hits=MAX_HITS,
):
    # Persist both views: one for the model input and one for pairwise geometry features.
    transformed_hits = apply_deterministic_hit_transforms(
        raw_hits,
        position_scale=POSITION_SCALE,
    )
    normalized_hits = apply_feature_stats(transformed_hits, hit_stats)

    final_hits, final_masks = pad_hits_list(
        normalized_hits,
        max_hits=max_hits,
        feature_dim=len(FEATURE_NAMES),
    )
    # This tensor keeps deterministic geometry/time features for pairwise attention bias.
    final_pairwise_hits, _ = pad_hits_list(
        transformed_hits,
        max_hits=max_hits,
        feature_dim=len(FEATURE_NAMES),
    )

    target_tensor = _stack_primary_targets(
        targets=targets,
        target_kind=target_kind,
        target_dim=target_dim,
    )
    torch.save(target_tensor, f"{data_path}/{split_name}_targets.pt")
    torch.save(final_hits, f"{data_path}/{split_name}_hits.pt")
    torch.save(final_pairwise_hits, f"{data_path}/{split_name}_hits_raw.pt")
    torch.save(final_masks, f"{data_path}/{split_name}_padding_mask.pt")

    if task == "direction":
        torch.save(target_tensor, f"{data_path}/{split_name}_muons.pt")
        torch.save(
            _stack_optional_vectors(rec_targets, vector_dim=target_dim),
            f"{data_path}/{split_name}_muons_rec.pt",
        )
        torch.save(
            _stack_optional_scalars(energies),
            f"{data_path}/{split_name}_muons_e.pt",
        )


def preprocess_dataset(resolved_config):
    data_path = resolved_config["paths"]["data_path"]
    os.makedirs(data_path, exist_ok=True)

    source_data = load_source_data(resolved_config=resolved_config)
    if not source_data["hits"]:
        raise RuntimeError("No events were loaded from the configured input source.")

    split_indices = build_split_indices(
        len(source_data["hits"]),
        train_fraction=resolved_config["data"]["train_fraction"],
        val_fraction=resolved_config["data"]["val_fraction"],
        test_fraction=resolved_config["data"]["test_fraction"],
        split_seed=resolved_config["data"]["split_seed"],
    )
    train_hits = apply_deterministic_hit_transforms(
        take_items(source_data["hits"], split_indices["train"]),
        position_scale=POSITION_SCALE,
    )
    hit_stats = compute_feature_stats(train_hits)

    for split_name, indices in split_indices.items():
        save_split(
            data_path=data_path,
            split_name=split_name,
            targets=take_items(source_data["targets"], indices),
            raw_hits=take_items(source_data["hits"], indices),
            rec_targets=(
                take_items(source_data["rec_targets"], indices)
                if source_data["rec_targets"] is not None
                else None
            ),
            energies=(
                take_items(source_data["energies"], indices)
                if source_data["energies"] is not None
                else None
            ),
            hit_stats=hit_stats,
            target_kind=source_data["target_kind"],
            target_dim=source_data["target_dim"],
            task=resolved_config["task"]["name"],
            max_hits=resolved_config["data"]["max_hits"],
        )

    metadata = build_metadata(
        split_indices=split_indices,
        max_hits=resolved_config["data"]["max_hits"],
        source_format=resolved_config["source"]["format"],
        task=resolved_config["task"]["name"],
        target_name=source_data["target_name"],
        target_kind=source_data["target_kind"],
        target_dim=source_data["target_dim"],
        class_values=source_data["class_values"],
        class_to_index=source_data["class_to_index"],
        source_details=source_data["source_details"],
        split_seed=resolved_config["data"]["split_seed"],
    )
    with open(f"{data_path}/metadata.json", "w", encoding="ascii") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    # Save one compact tensor stats file instead of feature-by-feature sklearn scalers.
    save_feature_stats(hit_stats, f"{data_path}/hits_stats.pt")
    return metadata


def _parse_count_class_values(raw_value):
    if raw_value is None:
        return None
    values = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError:
            values.append(float(stripped))
    return values


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess ROOT direction data or HDF5 muon-count data into training tensors.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument(
        "--source-format",
        choices=("root", "hdf5"),
        default=None,
        help="Override the configured input source format.",
    )
    parser.add_argument(
        "--task",
        choices=("direction", "muon_count"),
        default=None,
        help="Override the configured task.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the output directory for preprocessed tensors.",
    )
    parser.add_argument(
        "--root-path-pattern",
        default=None,
        help="Override the ROOT file glob used by the legacy direction pipeline.",
    )
    parser.add_argument(
        "--h5-path",
        default=None,
        help="Override the HDF5 file path or glob.",
    )
    parser.add_argument(
        "--h5-hits-dataset",
        default=None,
        help="Override the HDF5 hits dataset name.",
    )
    parser.add_argument(
        "--h5-label-dataset",
        default=None,
        help="Override the HDF5 label dataset name.",
    )
    parser.add_argument(
        "--h5-label-col",
        type=int,
        default=None,
        help="Override the HDF5 label column index.",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=None,
        help="Override the maximum number of hits kept per event.",
    )
    parser.add_argument(
        "--count-class-values",
        default=None,
        help="Optional comma-separated muon-count class values to preserve in metadata and map to class indices.",
    )
    return parser


def build_cli_overrides(args):
    overrides = {}
    if args.source_format is not None:
        overrides.setdefault("source", {})["format"] = args.source_format
    if args.task is not None:
        overrides.setdefault("task", {})["name"] = args.task
    if args.data_path is not None:
        overrides.setdefault("paths", {})["data_path"] = args.data_path
    if args.root_path_pattern is not None:
        overrides.setdefault("paths", {})["root_path_pattern"] = args.root_path_pattern
    if args.h5_path is not None:
        overrides.setdefault("paths", {})["hdf5_path"] = args.h5_path
    if args.h5_hits_dataset is not None:
        overrides.setdefault("hdf5", {})["hits_dataset"] = args.h5_hits_dataset
    if args.h5_label_dataset is not None:
        overrides.setdefault("hdf5", {})["label_dataset"] = args.h5_label_dataset
    if args.h5_label_col is not None:
        overrides.setdefault("hdf5", {})["label_col"] = args.h5_label_col
    if args.max_hits is not None:
        overrides.setdefault("data", {})["max_hits"] = args.max_hits

    parsed_class_values = _parse_count_class_values(args.count_class_values)
    if parsed_class_values is not None:
        overrides.setdefault("hdf5", {})["count_class_values"] = parsed_class_values

    return overrides


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolved_config = resolve_preprocess_config(
        config_path=args.config,
        overrides=build_cli_overrides(args),
    )
    preprocess_dataset(resolved_config=resolved_config)


if __name__ == "__main__":
    main()
