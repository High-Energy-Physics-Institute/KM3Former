import json
import os
import sys

import torch
from data_loader import KM3Loader
from eval import (
    angular_error_degrees,
    combined_loss,
    evaluate_model,
    move_batch_to_device,
    reconstruction_quality_target,
)
from km3former import KM3Former
from scheduler import create_optimizer_and_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_PATH = "./data"
MODEL_PATH = "./model"

model_dim = 256
num_heads = 8
num_encoder_layers = 6
dim_feedforward = 512
dropout = 0.1
pairwise_neighbors = 32
batch_size = 256

learning_rate = 8e-4
epochs = 10


def build_dataset(split_name):
    return KM3Loader(
        hits_file=f"{DATA_PATH}/{split_name}_hits.pt",
        raw_hits_file=f"{DATA_PATH}/{split_name}_hits_raw.pt",
        padding_mask_file=f"{DATA_PATH}/{split_name}_padding_mask.pt",
        label_file=f"{DATA_PATH}/{split_name}_muons.pt",
        rec_label_file=f"{DATA_PATH}/{split_name}_muons_rec.pt",
    )


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


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    with open(f"{DATA_PATH}/metadata.json", "r", encoding="ascii") as metadata_file:
        metadata = json.load(metadata_file)

    os.makedirs(MODEL_PATH, exist_ok=True)

    train_dataset = build_dataset("train")
    val_dataset = build_dataset("val")
    test_dataset = build_dataset("test")

    train_loader = build_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        max_workers=4,
    )
    val_loader = build_data_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        max_workers=2,
    )
    test_loader = build_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        max_workers=2,
    )

    train_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.05 * train_steps))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KM3Former(
        input_dim=metadata["input_dim"],
        model_dim=model_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_hits=metadata["max_hits"],
        pairwise_neighbors=pairwise_neighbors,
    ).to(device)

    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model=model,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
    )

    writer = SummaryWriter(log_dir=f"{MODEL_PATH}/tensorboard")
    best_val_loss = float("inf")
    model.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = 0.0
        running_angle = 0.0

        for batch_idx, (hits, raw_hits, padding_mask, muons, rec_muons) in enumerate(
            tqdm(train_loader, desc="Training", leave=False)
        ):
            hits, raw_hits, padding_mask, muons, rec_muons = move_batch_to_device(
                hits,
                raw_hits,
                padding_mask,
                muons,
                rec_muons,
                device=device,
            )

            optimizer.zero_grad()
            prediction, quality = model(
                hits,
                raw_hits=raw_hits,
                padding_mask=padding_mask,
                return_quality=True,
            )
            quality_target = reconstruction_quality_target(rec_muons, muons)
            loss = combined_loss(
                prediction,
                muons,
                quality_prediction=quality,
                quality_target=quality_target,
            )
            angle = angular_error_degrees(prediction, muons)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            running_angle += angle.item()

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Train_Batch", loss.item(), global_step)
            writer.add_scalar("Angle/Train_Batch", angle.item(), global_step)

        average_train_loss = running_loss / max(len(train_loader), 1)
        average_train_angle = running_angle / max(len(train_loader), 1)
        val_metrics = evaluate_model(model, val_loader, device=device)
        current_lr = lr_scheduler.get_last_lr()[0]

        writer.add_scalar("Loss/Train_Epoch", average_train_loss, epoch)
        writer.add_scalar("Angle/Train_Epoch", average_train_angle, epoch)
        writer.add_scalar("Loss/Val_Epoch", val_metrics["loss"], epoch)
        writer.add_scalar("Angle/Val_Epoch", val_metrics["angle_deg"], epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "val_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, f"{MODEL_PATH}/model_epoch_{epoch + 1}.pth")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, f"{MODEL_PATH}/best_model.pth")

    best_checkpoint = torch.load(f"{MODEL_PATH}/best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device=device)
    writer.add_hparams(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "pairwise_neighbors": pairwise_neighbors,
            "model_dim": model_dim,
        },
        {
            "hparam/test_loss": test_metrics["loss"],
            "hparam/test_angle_deg": test_metrics["angle_deg"],
        },
    )
    writer.close()
