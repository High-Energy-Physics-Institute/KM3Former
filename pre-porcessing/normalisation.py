import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def fit_tensor_list_scalers(tensor_list):
    data = np.stack([tensor.detach().cpu().numpy() for tensor in tensor_list], axis=0)
    scalers = []

    for feature_idx in range(data.shape[1]):
        scaler = StandardScaler()
        scaler.fit(data[:, feature_idx].reshape(-1, 1))
        scalers.append(scaler)

    return scalers


def apply_tensor_list_scalers(tensor_list, scalers):
    data = np.stack([tensor.detach().cpu().numpy() for tensor in tensor_list], axis=0)
    normalized_data = np.zeros_like(data, dtype=np.float32)

    for feature_idx, scaler in enumerate(scalers):
        normalized_data[:, feature_idx] = scaler.transform(
            data[:, feature_idx].reshape(-1, 1)
        ).reshape(-1)

    return [torch.tensor(row, dtype=torch.float32) for row in normalized_data]


def normalize_tensor_list(tensor_list):
    scalers = fit_tensor_list_scalers(tensor_list)
    normalized_list = apply_tensor_list_scalers(tensor_list, scalers)
    return normalized_list, scalers


def denormalize_tensor_list(normalized_list, scalers):
    normalized_data = np.stack(
        [tensor.detach().cpu().numpy() for tensor in normalized_list], axis=0
    )

    denormalized_data = np.zeros_like(normalized_data, dtype=np.float32)
    for feature_idx, scaler in enumerate(scalers):
        denormalized_data[:, feature_idx] = scaler.inverse_transform(
            normalized_data[:, feature_idx].reshape(-1, 1)
        ).reshape(-1)

    return [torch.tensor(row, dtype=torch.float32) for row in denormalized_data]


def fit_nested_tensor_list_scalers(hits_list):
    n_features = hits_list[0].shape[1]
    scalers = [StandardScaler() for _ in range(n_features)]
    all_data = torch.cat(hits_list, dim=0).detach().cpu().numpy()

    for feature_idx, scaler in enumerate(scalers):
        scaler.fit(all_data[:, feature_idx].reshape(-1, 1))

    return scalers


def apply_nested_tensor_list_scalers(hits_list, scalers):
    n_features = hits_list[0].shape[1]
    all_data = torch.cat(hits_list, dim=0).detach().cpu().numpy()
    normalized_data = np.zeros_like(all_data, dtype=np.float32)

    for feature_idx, scaler in enumerate(scalers):
        normalized_data[:, feature_idx] = scaler.transform(
            all_data[:, feature_idx].reshape(-1, 1)
        ).reshape(-1)

    normalized_hits = []
    start = 0
    for hit_tensor in hits_list:
        end = start + hit_tensor.shape[0]
        normalized_hits.append(
            torch.tensor(normalized_data[start:end], dtype=torch.float32)
        )
        start = end

    return normalized_hits


def normalize_nested_tensor_list(hits_list, scalers=None):
    if scalers is None:
        scalers = fit_nested_tensor_list_scalers(hits_list)

    normalized_hits = apply_nested_tensor_list_scalers(hits_list, scalers)
    return normalized_hits, scalers


def denormalize_nested_tensor_list(normalized_hits, scalers):
    n_features = normalized_hits[0].shape[1]
    all_data = torch.cat(normalized_hits, dim=0).detach().cpu().numpy()
    denormalized_data = np.zeros_like(all_data, dtype=np.float32)

    for feature_idx, scaler in enumerate(scalers):
        denormalized_data[:, feature_idx] = scaler.inverse_transform(
            all_data[:, feature_idx].reshape(-1, 1)
        ).reshape(-1)

    denormalized_hits = []
    start = 0
    for hit_tensor in normalized_hits:
        end = start + hit_tensor.shape[0]
        denormalized_hits.append(
            torch.tensor(denormalized_data[start:end], dtype=torch.float32)
        )
        start = end

    return denormalized_hits
