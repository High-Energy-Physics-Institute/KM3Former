import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_optimizer_and_scheduler(
    model, train_steps, warmup_steps, learning_rate=2e-4
):
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create schedulers
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=train_steps - warmup_steps)

    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler
