import torch
import torch.nn.functional as F


def resolve_task_settings(metadata):
    task_name = metadata.get("task", "direction")
    target_kind = metadata.get("target_kind", "vector_regression")
    target_name = metadata.get("target_name", "muon_direction")
    target_dim = metadata.get("target_dim", 3)

    if target_kind == "vector_regression":
        return {
            "task": task_name,
            "target_kind": target_kind,
            "target_name": target_name,
            "target_dim": target_dim,
            "supports_quality": task_name == "direction",
            "metric_name": "angle_deg",
            "metric_label": "Angle",
            "prediction_label": "predictions",
        }

    if target_kind == "multiclass":
        return {
            "task": task_name,
            "target_kind": target_kind,
            "target_name": target_name,
            "target_dim": target_dim,
            "supports_quality": False,
            "metric_name": "accuracy",
            "metric_label": "Accuracy",
            "prediction_label": "logits",
        }

    raise ValueError(f"Unsupported target kind: {target_kind!r}")


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


def multiclass_accuracy(logits, target):
    predicted_classes = torch.argmax(logits, dim=-1)
    return (predicted_classes == target.long()).to(dtype=torch.float32).mean()


def reconstruction_quality_target(rec_track, target):
    rec_track = F.normalize(rec_track, dim=-1)
    target = F.normalize(target, dim=-1)
    return 0.5 * (torch.sum(rec_track * target, dim=-1).clamp(-1.0, 1.0) + 1.0)


def combined_loss(prediction, target, quality_prediction=None, quality_target=None):
    loss = cosine_direction_loss(prediction, target)
    if quality_prediction is not None and quality_target is not None:
        loss = loss + 0.1 * F.mse_loss(quality_prediction, quality_target)
    return loss


def classification_loss(logits, target):
    return F.cross_entropy(logits, target.long())


def compute_task_loss(
    prediction,
    target,
    task_settings,
    quality_prediction=None,
    quality_target=None,
):
    if task_settings["target_kind"] == "vector_regression":
        return combined_loss(
            prediction,
            target,
            quality_prediction=quality_prediction,
            quality_target=quality_target,
        )
    if task_settings["target_kind"] == "multiclass":
        return classification_loss(prediction, target)
    raise ValueError(f"Unsupported target kind: {task_settings['target_kind']!r}")


def compute_task_metric(prediction, target, task_settings):
    if task_settings["target_kind"] == "vector_regression":
        return angular_error_degrees(prediction, target)
    if task_settings["target_kind"] == "multiclass":
        return multiclass_accuracy(prediction, target)
    raise ValueError(f"Unsupported target kind: {task_settings['target_kind']!r}")


def move_batch_to_device(*tensors, device):
    non_blocking = device.type == "cuda"
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in tensors)


def move_batch_dict_to_device(batch, device):
    return {
        key: tensor.to(device, non_blocking=device.type == "cuda")
        for key, tensor in batch.items()
    }


@torch.no_grad()
def evaluate_model(model, data_loader, device, task_settings, quality_supervised):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_examples = 0

    for batch in data_loader:
        batch = move_batch_dict_to_device(batch, device=device)
        hits = batch["hits"]
        raw_hits = batch["raw_hits"]
        padding_mask = batch["padding_mask"]
        target = batch["target"]

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
        batch_size = hits.size(0)

        total_loss += compute_task_loss(
            prediction,
            target,
            task_settings=task_settings,
            quality_prediction=quality,
            quality_target=quality_target,
        ).item() * batch_size
        total_metric += (
            compute_task_metric(prediction, target, task_settings=task_settings).item()
            * batch_size
        )
        total_examples += batch_size

    model.train()
    return {
        "loss": total_loss / max(total_examples, 1),
        task_settings["metric_name"]: total_metric / max(total_examples, 1),
    }
