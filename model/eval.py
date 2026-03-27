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


def multiclass_outputs_from_logits(logits):
    probabilities = torch.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(logits, dim=-1)
    return {
        "probabilities": probabilities,
        "predicted_classes": predicted_classes,
    }


def confusion_matrix_from_predictions(predicted_classes, target, num_classes):
    flattened = target.long() * num_classes + predicted_classes.long()
    return torch.bincount(
        flattened,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def _label_strings(class_values):
    return [str(label) for label in class_values]


def build_multiclass_summary(predictions, targets, class_values=None):
    probabilities = torch.softmax(predictions, dim=-1)
    predicted_classes = torch.argmax(predictions, dim=-1)
    num_classes = predictions.size(-1)
    resolved_class_values = list(
        class_values if class_values is not None else range(num_classes)
    )

    confusion = confusion_matrix_from_predictions(
        predicted_classes=predicted_classes,
        target=targets,
        num_classes=num_classes,
    )
    confusion_float = confusion.to(dtype=torch.float32)
    true_positives = confusion_float.diag()
    support = confusion_float.sum(dim=1)
    predicted_support = confusion_float.sum(dim=0)
    total_examples = confusion_float.sum().clamp(min=1.0)
    precision = true_positives / predicted_support.clamp(min=1.0)
    recall = true_positives / support.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-6)
    true_negatives = total_examples - support - predicted_support + true_positives
    per_class_accuracy = (true_positives + true_negatives) / total_examples
    prediction_histogram = torch.bincount(
        predicted_classes,
        minlength=num_classes,
    )
    label_keys = _label_strings(resolved_class_values)

    return {
        "num_examples": int(targets.numel()),
        "accuracy": float((predicted_classes == targets.long()).to(dtype=torch.float32).mean()),
        "macro_f1": float(f1.mean()),
        "mean_confidence": float(probabilities.max(dim=-1).values.mean()),
        "class_values": resolved_class_values,
        "support": {
            label_keys[index]: int(support[index].item())
            for index in range(num_classes)
        },
        "prediction_histogram": {
            label_keys[index]: int(prediction_histogram[index].item())
            for index in range(num_classes)
        },
        "per_class_accuracy": {
            label_keys[index]: float(per_class_accuracy[index].item())
            for index in range(num_classes)
        },
        "per_class_recall": {
            label_keys[index]: float(recall[index].item())
            for index in range(num_classes)
        },
        "per_class_precision": {
            label_keys[index]: float(precision[index].item())
            for index in range(num_classes)
        },
        "per_class_f1": {
            label_keys[index]: float(f1[index].item())
            for index in range(num_classes)
        },
        "confusion_matrix": confusion.tolist(),
    }


def build_regression_summary(predictions, targets, quality_scores=None):
    summary = {
        "num_examples": int(targets.size(0)),
        "angle_deg": float(angular_error_degrees(predictions, targets).item()),
    }
    if quality_scores is not None:
        summary["mean_quality"] = float(quality_scores.mean().item())
    return summary


def build_evaluation_summary(outputs, task_settings, class_values=None):
    if task_settings["target_kind"] == "multiclass":
        return build_multiclass_summary(
            predictions=outputs["predictions"],
            targets=outputs["targets"],
            class_values=class_values,
        )

    return build_regression_summary(
        predictions=outputs["predictions"],
        targets=outputs["targets"],
        quality_scores=outputs.get("quality"),
    )


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    task_settings,
    quality_supervised,
    class_values=None,
    collect_outputs=False,
):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_examples = 0
    collected_predictions = [] if collect_outputs else None
    collected_targets = [] if collect_outputs else None
    collected_quality = [] if collect_outputs and quality_supervised else None

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

        if collect_outputs:
            collected_predictions.append(prediction.detach().cpu())
            collected_targets.append(target.detach().cpu())
            if quality_supervised:
                collected_quality.append(quality.detach().cpu())

    model.train()
    results = {
        "loss": total_loss / max(total_examples, 1),
        task_settings["metric_name"]: total_metric / max(total_examples, 1),
    }

    if not collect_outputs:
        return results

    outputs = {
        "predictions": torch.cat(collected_predictions, dim=0),
        "targets": torch.cat(collected_targets, dim=0),
    }
    if quality_supervised:
        outputs["quality"] = torch.cat(collected_quality, dim=0)
    if task_settings["target_kind"] == "multiclass":
        outputs.update(multiclass_outputs_from_logits(outputs["predictions"]))

    results["outputs"] = outputs
    results["summary"] = build_evaluation_summary(
        outputs=outputs,
        task_settings=task_settings,
        class_values=class_values,
    )
    return results
