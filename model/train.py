import pandas as pd
import torch
import torch.nn as nn
from data_loader import KM3Loader
from km3former import KM3Former
from scheduler import create_optimizer_and_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DATA_PATH = "./data"
MODEL_PATH = "./model"

input_dim = 8
model_dim = 256
num_heads = 8
num_encoder_layers = 6
dim_feedforward = 512
dropout = 0.1
batch_size = 256

learning_rate = 8e-4
epochs = 10


dataset = KM3Loader(f"{DATA_PATH}/hits.pt", f"{DATA_PATH}/muons.pt")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_steps = len(data_loader) * epochs
warmup_steps = int(0.05 * train_steps)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KM3Former(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
).to(device)


criterion = nn.MSELoss()

optimizer, lr_scheduler = create_optimizer_and_scheduler(
    model, train_steps, warmup_steps
)

writer = SummaryWriter(log_dir=f"{MODEL_PATH}/tensorboard")

model.train()

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    for batch_idx, (hits, muons) in enumerate(tqdm(data_loader)):
        hits, muons = hits.to(device), muons.to(device)

        optimizer.zero_grad()
        output = model(hits)
        loss = criterion(output, muons)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        running_loss += loss.item()

        writer.add_scalar(
            "Loss/Train_Batch", loss.item(), epoch * len(data_loader) + batch_idx
        )

    average_loss = running_loss / len(data_loader)
    current_lr = lr_scheduler.get_last_lr()[0]

    writer.add_scalar("Loss/Train_Epoch", average_loss, epoch)
    writer.add_scalar("Learning_Rate", current_lr, epoch)

    torch.save(
        model.state_dict(),
        f"{MODEL_PATH}/model_epoch_{epoch+1}.pth",
    )
