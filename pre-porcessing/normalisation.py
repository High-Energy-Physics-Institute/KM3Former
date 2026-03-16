import torch

TIME_INDEX = 0
POSITION_SLICE = slice(1, 4)
TOT_INDEX = 7
HIT_RANK_INDEX = 8
RADIUS_INDEX = 9

DEFAULT_POSITION_SCALE = 500.0
DEFAULT_AFFINE_FEATURE_INDICES = (
    TIME_INDEX,
    1,
    2,
    3,
    TOT_INDEX,
    HIT_RANK_INDEX,
    RADIUS_INDEX,
)
EPSILON = 1e-6


def apply_deterministic_hit_transforms(hits_list, position_scale=DEFAULT_POSITION_SCALE):
    transformed_hits = []

    for hit_tensor in hits_list:
        transformed = hit_tensor.clone().to(dtype=torch.float32)
        if transformed.numel() == 0:
            transformed_hits.append(transformed)
            continue

        transformed[:, TIME_INDEX] = transformed[:, TIME_INDEX] - transformed[
            :, TIME_INDEX
        ].min()
        transformed[:, POSITION_SLICE] = (
            transformed[:, POSITION_SLICE] / position_scale
        )
        transformed[:, TOT_INDEX] = torch.log1p(
            torch.clamp(transformed[:, TOT_INDEX], min=0.0)
        )
        transformed[:, RADIUS_INDEX] = transformed[:, RADIUS_INDEX] / position_scale
        transformed_hits.append(transformed)

    return transformed_hits


def compute_feature_stats(
    hits_list,
    feature_indices=DEFAULT_AFFINE_FEATURE_INDICES,
):
    if not hits_list:
        raise ValueError("Cannot compute feature stats from an empty hits list.")

    all_data = torch.cat(hits_list, dim=0).to(dtype=torch.float32)
    n_features = all_data.shape[1]

    mean = torch.zeros(n_features, dtype=torch.float32)
    std = torch.ones(n_features, dtype=torch.float32)

    index_tensor = torch.tensor(feature_indices, dtype=torch.long)
    selected = all_data[:, index_tensor]
    mean[index_tensor] = selected.mean(dim=0)
    std[index_tensor] = selected.std(dim=0, unbiased=False).clamp(min=EPSILON)

    return {"mean": mean, "std": std}


def apply_feature_stats(hits_list, stats):
    mean = stats["mean"].to(dtype=torch.float32)
    std = stats["std"].to(dtype=torch.float32)

    normalized_hits = []
    for hit_tensor in hits_list:
        hits = hit_tensor.to(dtype=torch.float32)
        normalized_hits.append((hits - mean) / std)

    return normalized_hits


def denormalize_hits(hits_list, stats):
    mean = stats["mean"].to(dtype=torch.float32)
    std = stats["std"].to(dtype=torch.float32)

    denormalized_hits = []
    for hit_tensor in hits_list:
        hits = hit_tensor.to(dtype=torch.float32)
        denormalized_hits.append(hits * std + mean)

    return denormalized_hits


def save_feature_stats(stats, path):
    torch.save(
        {
            "mean": stats["mean"].detach().cpu(),
            "std": stats["std"].detach().cpu(),
        },
        path,
    )


def load_feature_stats(path):
    stats = torch.load(path, map_location="cpu")
    return {
        "mean": stats["mean"].to(dtype=torch.float32),
        "std": stats["std"].to(dtype=torch.float32),
    }
