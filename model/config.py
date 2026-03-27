import copy
import json
from pathlib import Path


DEFAULT_TRAIN_CONFIG = {
    "paths": {
        "data_path": "./data",
        "model_path": "./model",
    },
    "model": {
        "model_dim": 256,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "pairwise_neighbors": 32,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 8e-4,
        "epochs": 10,
    },
    "loader": {
        "train_max_workers": 4,
        "eval_max_workers": 2,
    },
}


def deep_merge_dicts(base, overrides):
    # Merge nested config sections without mutating either caller-owned dict.
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

    # Keep the config format dependency-free so the repo can stay JSON-only.
    with open(config_path, "r", encoding="ascii") as config_file:
        overrides = json.load(config_file)

    return deep_merge_dicts(config, overrides)


def resolve_train_config(config_path=None, overrides=None):
    resolved = load_json_config(
        config_path=config_path,
        default_config=DEFAULT_TRAIN_CONFIG,
    )
    if overrides:
        resolved = deep_merge_dicts(resolved, overrides)

    return resolved


def get_model_init_kwargs(metadata, resolved_config):
    # Model hyperparameters come from config, while tensor shape comes from metadata.
    model_config = resolved_config["model"]
    return {
        "input_dim": metadata["input_dim"],
        "target_dim": metadata.get("target_dim", 3),
        "target_kind": metadata.get("target_kind", "vector_regression"),
        "model_dim": model_config["model_dim"],
        "num_heads": model_config["num_heads"],
        "num_encoder_layers": model_config["num_encoder_layers"],
        "dim_feedforward": model_config["dim_feedforward"],
        "dropout": model_config["dropout"],
        "max_hits": metadata["max_hits"],
        "pairwise_neighbors": model_config["pairwise_neighbors"],
    }


def save_json_file(payload, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
