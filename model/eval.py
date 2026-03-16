import torch
import torch.nn.functional as F


def unpack_supervised_batch(batch):
    # The current training loop is direction-supervised, with optional future energy labels.
    if len(batch) == 5:
        hits, raw_hits, padding_mask, muons, rec_muons = batch
        return hits, raw_hits, padding_mask, muons, rec_muons, None

    if len(batch) == 6:
        hits, raw_hits, padding_mask, muons, rec_muons, energy_labels = batch
        return hits, raw_hits, padding_mask, muons, rec_muons, energy_labels

    raise ValueError(
        "Expected supervised batches with 5 or 6 tensors, "
        f"received {len(batch)} entries."
    )


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


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_angle = 0.0
    total_examples = 0

    for batch in data_loader:
        hits, raw_hits, padding_mask, muons, rec_muons, energy_labels = (
            unpack_supervised_batch(batch)
        )
        tensors_to_move = [hits, raw_hits, padding_mask, muons, rec_muons]
        if energy_labels is not None:
            tensors_to_move.append(energy_labels)

        moved_tensors = move_batch_to_device(*tensors_to_move, device=device)
        # Ignore optional energy labels for now; they are only carried through for future heads.
        hits, raw_hits, padding_mask, muons, rec_muons = moved_tensors[:5]

        prediction, quality = model(
            hits,
            raw_hits=raw_hits,
            padding_mask=padding_mask,
            return_quality=True,
        )
        quality_target = reconstruction_quality_target(rec_muons, muons)
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
