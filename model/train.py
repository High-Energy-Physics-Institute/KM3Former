import json
import os
import sys

import click
import torch
from config import get_model_init_kwargs, resolve_train_config, save_json_file
from data_loader import KM3Loader
from eval import (
    compute_task_loss,
    compute_task_metric,
    evaluate_model,
    move_batch_dict_to_device,
    reconstruction_quality_target,
    resolve_task_settings,
)
from km3former import KM3Former
from scheduler import create_optimizer_and_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_RESOLVED_CONFIG_PATH = "resolved_train_config.json"


def build_dataset(split_name, data_path, load_strategy="lazy"):
    target_file = f"{data_path}/{split_name}_targets.pt"
    legacy_label_file = f"{data_path}/{split_name}_muons.pt"
    rec_label_file = f"{data_path}/{split_name}_muons_rec.pt"
    energy_label_file = f"{data_path}/{split_name}_muons_e.pt"
    metadata_file = f"{data_path}/metadata.json"

    return KM3Loader(
        hits_file=f"{data_path}/{split_name}_hits.pt",
        raw_hits_file=f"{data_path}/{split_name}_hits_raw.pt",
        padding_mask_file=f"{data_path}/{split_name}_padding_mask.pt",
        target_file=target_file if os.path.exists(target_file) else None,
        label_file=legacy_label_file if os.path.exists(legacy_label_file) else None,
        rec_label_file=rec_label_file if os.path.exists(rec_label_file) else None,
        energy_label_file=(
            energy_label_file if os.path.exists(energy_label_file) else None
        ),
        metadata_file=metadata_file if os.path.exists(metadata_file) else None,
        load_strategy=load_strategy,
    )


def resolve_quality_supervision(*datasets, metadata=None):
    task_settings = resolve_task_settings(metadata or {})
    if not task_settings["supports_quality"]:
        return False

    availability = [dataset.has_rec_labels for dataset in datasets]
    if any(availability) and not all(availability):
        raise ValueError(
            "Reconstructed labels must be present for all train/val/test splits "
            "or omitted for all of them."
        )

    return all(availability)


def get_default_num_workers(max_workers=4):
    if sys.platform == "darwin":
        return 0

    return min(max_workers, os.cpu_count() or 1)


def build_data_loader(dataset, batch_size, shuffle, max_workers=4):
    num_workers = get_default_num_workers(max_workers=max_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


def load_metadata(data_path):
    with open(f"{data_path}/metadata.json", "r", encoding="ascii") as metadata_file:
        return json.load(metadata_file)


def build_model(metadata, resolved_config, device):
    # Metadata owns tensor shape; config owns architectural hyperparameters.
    return KM3Former(
        **get_model_init_kwargs(
            metadata=metadata,
            resolved_config=resolved_config,
        )
    ).to(device)


def get_cli_overrides(data_path=None, model_path=None):
    overrides = {}
    if data_path is not None:
        overrides.setdefault("paths", {})["data_path"] = data_path
    if model_path is not None:
        overrides.setdefault("paths", {})["model_path"] = model_path
    return overrides


def train_model(resolved_config):
    from torch.utils.tensorboard import SummaryWriter

    data_path = resolved_config["paths"]["data_path"]
    model_path = resolved_config["paths"]["model_path"]
    batch_size = resolved_config["training"]["batch_size"]
    learning_rate = resolved_config["training"]["learning_rate"]
    epochs = resolved_config["training"]["epochs"]
    train_max_workers = resolved_config["loader"]["train_max_workers"]
    eval_max_workers = resolved_config["loader"]["eval_max_workers"]

    metadata = load_metadata(data_path=data_path)
    task_settings = resolve_task_settings(metadata)

    os.makedirs(model_path, exist_ok=True)
    # Save the final merged config so each checkpoint directory is self-describing.
    save_json_file(
        resolved_config,
        f"{model_path}/{DEFAULT_RESOLVED_CONFIG_PATH}",
    )

    train_dataset = build_dataset("train", data_path=data_path)
    val_dataset = build_dataset("val", data_path=data_path)
    test_dataset = build_dataset("test", data_path=data_path)
    quality_supervised = resolve_quality_supervision(
        train_dataset,
        val_dataset,
        test_dataset,
        metadata=metadata,
    )

    train_loader = build_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        max_workers=train_max_workers,
    )
    val_loader = build_data_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        max_workers=eval_max_workers,
    )
    test_loader = build_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        max_workers=eval_max_workers,
    )

    train_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.05 * train_steps))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        metadata=metadata,
        resolved_config=resolved_config,
        device=device,
    )

    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model=model,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
    )

    writer = SummaryWriter(log_dir=f"{model_path}/tensorboard")
    best_val_loss = float("inf")
    model.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = 0.0
        running_metric = 0.0

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc="Training", leave=False)
        ):
            batch = move_batch_dict_to_device(batch, device=device)
            hits = batch["hits"]
            raw_hits = batch["raw_hits"]
            padding_mask = batch["padding_mask"]
            target = batch["target"]

            optimizer.zero_grad()
            model_output = model(
                hits,
                raw_hits=raw_hits,
                padding_mask=padding_mask,
                return_quality=quality_supervised,
            )
            if quality_supervised:
                prediction, quality = model_output
            else:
                prediction = model_output
                quality = None
            quality_target = None
            if quality_supervised:
                quality_target = reconstruction_quality_target(batch["rec_muons"], target)
            loss = compute_task_loss(
                prediction,
                target,
                task_settings=task_settings,
                quality_prediction=quality,
                quality_target=quality_target,
            )
            metric_value = compute_task_metric(
                prediction,
                target,
                task_settings=task_settings,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            running_metric += metric_value.item()

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Train_Batch", loss.item(), global_step)
            writer.add_scalar(
                f"{task_settings['metric_label']}/Train_Batch",
                metric_value.item(),
                global_step,
            )

        average_train_loss = running_loss / max(len(train_loader), 1)
        average_train_metric = running_metric / max(len(train_loader), 1)
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            task_settings=task_settings,
            quality_supervised=quality_supervised,
        )
        current_lr = lr_scheduler.get_last_lr()[0]

        writer.add_scalar("Loss/Train_Epoch", average_train_loss, epoch)
        writer.add_scalar(
            f"{task_settings['metric_label']}/Train_Epoch",
            average_train_metric,
            epoch,
        )
        writer.add_scalar("Loss/Val_Epoch", val_metrics["loss"], epoch)
        writer.add_scalar(
            f"{task_settings['metric_label']}/Val_Epoch",
            val_metrics[task_settings["metric_name"]],
            epoch,
        )
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "val_loss": val_metrics["loss"],
            "resolved_config": resolved_config,
            "quality_supervised": quality_supervised,
            "task": task_settings["task"],
            "target_kind": task_settings["target_kind"],
        }
        torch.save(checkpoint, f"{model_path}/model_epoch_{epoch + 1}.pth")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, f"{model_path}/best_model.pth")

    best_checkpoint = torch.load(f"{model_path}/best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        task_settings=task_settings,
        quality_supervised=quality_supervised,
    )
    hparam_metrics = {
        "hparam/test_loss": test_metrics["loss"],
        f"hparam/test_{task_settings['metric_name']}": test_metrics[
            task_settings["metric_name"]
        ],
    }
    writer.add_hparams(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "pairwise_neighbors": resolved_config["model"]["pairwise_neighbors"],
            "model_dim": resolved_config["model"]["model_dim"],
        },
        hparam_metrics,
    )
    writer.close()
    return test_metrics


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional JSON config file. Defaults stay identical when omitted.",
)
@click.option(
    "--data-path",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Override the configured preprocessing data directory.",
)
@click.option(
    "--model-path",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Override the configured output directory for checkpoints and logs.",
)
def main(config_path, data_path, model_path):
    resolved_config = resolve_train_config(
        config_path=config_path,
        overrides=get_cli_overrides(
            data_path=data_path,
            model_path=model_path,
        ),
    )
    train_model(resolved_config=resolved_config)


if __name__ == "__main__":
    main()
