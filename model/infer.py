import json
from pathlib import Path

import click
import torch
from config import get_model_init_kwargs, resolve_train_config
from km3former import KM3Former
from torch.utils.data import DataLoader, TensorDataset


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="ascii") as metadata_file:
        return json.load(metadata_file)


def resolve_input_paths(
    split=None,
    data_path="./data",
    hits_file=None,
    raw_hits_file=None,
    padding_mask_file=None,
):
    if split is not None and any(
        path is not None for path in (hits_file, raw_hits_file, padding_mask_file)
    ):
        raise click.UsageError(
            "Use either --split or explicit tensor paths, not both."
        )

    if split is not None:
        return {
            "hits_file": f"{data_path}/{split}_hits.pt",
            "raw_hits_file": f"{data_path}/{split}_hits_raw.pt",
            "padding_mask_file": f"{data_path}/{split}_padding_mask.pt",
        }

    if None in (hits_file, raw_hits_file, padding_mask_file):
        raise click.UsageError(
            "Explicit inference requires --hits-file, --raw-hits-file, and --padding-mask-file."
        )

    return {
        "hits_file": hits_file,
        "raw_hits_file": raw_hits_file,
        "padding_mask_file": padding_mask_file,
    }


def load_inference_tensors(input_paths):
    hits = torch.load(input_paths["hits_file"], map_location="cpu")
    raw_hits = torch.load(input_paths["raw_hits_file"], map_location="cpu")
    padding_mask = torch.load(input_paths["padding_mask_file"], map_location="cpu")

    lengths = {len(hits), len(raw_hits), len(padding_mask)}
    if len(lengths) != 1:
        raise ValueError("Inference tensors must share the same first dimension.")

    return hits, raw_hits, padding_mask


def load_runtime_artifacts(data_path, metadata_path=None, stats_path=None):
    metadata_path = metadata_path or f"{data_path}/metadata.json"
    stats_path = stats_path or f"{data_path}/hits_stats.pt"

    metadata = load_metadata(metadata_path=metadata_path)
    # Loading feature stats here keeps inference tied to the saved preprocessing contract.
    feature_stats = torch.load(stats_path, map_location="cpu")
    if feature_stats["mean"].numel() != metadata["input_dim"]:
        raise ValueError(
            "Feature stats and metadata disagree on the input feature dimension."
        )

    return metadata, feature_stats, metadata_path, stats_path


def build_model_from_checkpoint(metadata, checkpoint, device):
    resolved_config = checkpoint.get("resolved_config") or resolve_train_config()
    model = KM3Former(
        **get_model_init_kwargs(
            metadata=metadata,
            resolved_config=resolved_config,
        )
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, resolved_config


def build_inference_loader(hits, raw_hits, padding_mask, batch_size):
    # Inference only needs the three tensors consumed by the forward path.
    dataset = TensorDataset(hits, raw_hits, padding_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def predict_batches(model, data_loader, device):
    predictions = []
    quality_scores = []

    for hits, raw_hits, padding_mask in data_loader:
        hits = hits.to(device)
        raw_hits = raw_hits.to(device)
        padding_mask = padding_mask.to(device)

        batch_predictions, batch_quality = model(
            hits,
            raw_hits=raw_hits,
            padding_mask=padding_mask,
            return_quality=True,
        )
        predictions.append(batch_predictions.cpu())
        quality_scores.append(batch_quality.cpu())

    return {
        "predictions": torch.cat(predictions, dim=0),
        "quality": torch.cat(quality_scores, dim=0),
    }


def save_inference_output(output_path, payload):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith(".json"):
        serializable_payload = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in payload.items()
        }
        with open(output_path, "w", encoding="ascii") as output_file:
            json.dump(serializable_payload, output_file, indent=2, sort_keys=True)
        return

    torch.save(payload, output_path)


def run_inference(
    checkpoint_path,
    output_path,
    split=None,
    data_path="./data",
    hits_file=None,
    raw_hits_file=None,
    padding_mask_file=None,
    metadata_path=None,
    stats_path=None,
    batch_size=256,
    device=None,
):
    input_paths = resolve_input_paths(
        split=split,
        data_path=data_path,
        hits_file=hits_file,
        raw_hits_file=raw_hits_file,
        padding_mask_file=padding_mask_file,
    )
    metadata, feature_stats, resolved_metadata_path, resolved_stats_path = (
        load_runtime_artifacts(
            data_path=data_path,
            metadata_path=metadata_path,
            stats_path=stats_path,
        )
    )
    hits, raw_hits, padding_mask = load_inference_tensors(input_paths=input_paths)

    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model, resolved_config = build_model_from_checkpoint(
        metadata=metadata,
        checkpoint=checkpoint,
        device=resolved_device,
    )
    data_loader = build_inference_loader(
        hits=hits,
        raw_hits=raw_hits,
        padding_mask=padding_mask,
        batch_size=batch_size,
    )
    predictions = predict_batches(
        model=model,
        data_loader=data_loader,
        device=resolved_device,
    )

    payload = {
        "checkpoint_path": checkpoint_path,
        "data_path": data_path,
        "metadata_path": resolved_metadata_path,
        "stats_path": resolved_stats_path,
        "input_paths": input_paths,
        "resolved_config": resolved_config,
        "feature_mean_shape": list(feature_stats["mean"].shape),
        "feature_std_shape": list(feature_stats["std"].shape),
        "predictions": predictions["predictions"],
        "quality": predictions["quality"],
    }
    save_inference_output(output_path=output_path, payload=payload)
    return payload


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="Checkpoint to load for inference.",
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, path_type=str),
    required=True,
    help="Where to save predictions. Use .pt for tensors or .json for a JSON payload.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default=None,
    help="Read tensors from a named preprocessed split under --data-path.",
)
@click.option(
    "--hits-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Explicit normalized hit tensor path.",
)
@click.option(
    "--raw-hits-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Explicit pairwise-bias tensor path.",
)
@click.option(
    "--padding-mask-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Explicit padding mask tensor path.",
)
@click.option(
    "--data-path",
    type=click.Path(file_okay=False, path_type=str),
    default="./data",
    show_default=True,
    help="Base directory for preprocessed tensors and metadata.",
)
@click.option(
    "--metadata-path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional explicit metadata path.",
)
@click.option(
    "--stats-path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional explicit feature-statistics path.",
)
@click.option(
    "--batch-size",
    type=int,
    default=256,
    show_default=True,
    help="Inference batch size.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Optional device override, for example cpu or cuda.",
)
def main(
    checkpoint_path,
    output_path,
    split,
    hits_file,
    raw_hits_file,
    padding_mask_file,
    data_path,
    metadata_path,
    stats_path,
    batch_size,
    device,
):
    run_inference(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        split=split,
        data_path=data_path,
        hits_file=hits_file,
        raw_hits_file=raw_hits_file,
        padding_mask_file=padding_mask_file,
        metadata_path=metadata_path,
        stats_path=stats_path,
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
