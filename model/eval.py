import torch
import torch.nn.functional as F


def cosine_direction_loss(prediction, target):
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    cosine_similarity = torch.sum(prediction * target, dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine_similarity).mean()


def angular_error_degrees(prediction, target):
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    cosine_similarity = torch.sum(prediction * target, dim=-1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine_similarity))
    return angles.mean()


def reconstruction_quality_target(rec_track, target):
    rec_track = F.normalize(rec_track, dim=-1)
    target = F.normalize(target, dim=-1)
    return 0.5 * (torch.sum(rec_track * target, dim=-1).clamp(-1.0, 1.0) + 1.0)


def combined_loss(prediction, target, quality_prediction=None, quality_target=None):
    loss = cosine_direction_loss(prediction, target)
    if quality_prediction is not None and quality_target is not None:
        loss = loss + 0.1 * F.mse_loss(quality_prediction, quality_target)
    return loss


def move_batch_to_device(*tensors, device):
    non_blocking = device.type == "cuda"
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in tensors)


def move_batch_dict_to_device(batch, device):
    return {
        key: tensor.to(device, non_blocking=device.type == "cuda")
        for key, tensor in batch.items()
    }


@torch.no_grad()
def evaluate_model(model, data_loader, device, quality_supervised):
    model.eval()
    total_loss = 0.0
    total_angle = 0.0
    total_examples = 0

    for batch in data_loader:
        batch = move_batch_dict_to_device(batch, device=device)
        hits = batch["hits"]
        raw_hits = batch["raw_hits"]
        padding_mask = batch["padding_mask"]
        muons = batch["muons"]

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
            quality_target = reconstruction_quality_target(batch["rec_muons"], muons)
        batch_size = hits.size(0)

        total_loss += combined_loss(
            prediction,
            muons,
            quality_prediction=quality,
            quality_target=quality_target,
        ).item() * batch_size
        total_angle += angular_error_degrees(prediction, muons).item() * batch_size
        total_examples += batch_size

    model.train()
    return {
        "loss": total_loss / max(total_examples, 1),
        "angle_deg": total_angle / max(total_examples, 1),
    }
