import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def normalize_tensor_list(tensor_list):
    # Convert list of tensors to a numpy array
    data = np.array([t.numpy() for t in tensor_list])

    # Initialize a list to store scalers
    scalers = []

    # Initialize an array to store normalized data
    normalized_data = np.zeros_like(data)

    # Normalize each component separately
    for i in range(data.shape[1]):
        scaler = StandardScaler()
        normalized_data[:, i] = scaler.fit_transform(
            data[:, i].reshape(-1, 1)
        ).flatten()
        scalers.append(scaler)

    # Convert back to a list of PyTorch tensors
    normalized_list = [
        torch.tensor(row, dtype=torch.float32) for row in normalized_data
    ]

    return normalized_list, scalers


def denormalize_tensor_list(normalized_list, scalers):
    # Convert list of tensors to a numpy array
    normalized_data = np.array([t.numpy() for t in normalized_list])

    # Initialize an array to store denormalized data
    denormalized_data = np.zeros_like(normalized_data)

    # Denormalize each component separately
    for i in range(normalized_data.shape[1]):
        denormalized_data[:, i] = (
            scalers[i].inverse_transform(normalized_data[:, i].reshape(-1, 1)).flatten()
        )

    # Convert back to a list of PyTorch tensors
    denormalized_list = [
        torch.tensor(row, dtype=torch.float32) for row in denormalized_data
    ]

    return denormalized_list


def normalize_nested_tensor_list(hits_list, scalers=None):
    # Get the dimensions
    n_samples = len(hits_list)
    n_hits, n_features = hits_list[0].shape

    # Initialize or reuse scalers
    if scalers is None:
        scalers = [StandardScaler() for _ in range(n_features)]
        fit_scalers = True
    else:
        fit_scalers = False

    # Prepare data for scaling
    flat_data = [hit.view(-1, n_features) for hit in hits_list]
    all_data = torch.cat(flat_data, dim=0).numpy()

    # Fit scalers (if necessary) and transform data
    normalized_data = np.zeros_like(all_data)
    for i in range(n_features):
        if fit_scalers:
            normalized_data[:, i] = (
                scalers[i].fit_transform(all_data[:, i].reshape(-1, 1)).flatten()
            )
        else:
            normalized_data[:, i] = (
                scalers[i].transform(all_data[:, i].reshape(-1, 1)).flatten()
            )

    # Reshape normalized data back to original structure
    normalized_hits = []
    start = 0
    for hit in hits_list:
        end = start + hit.shape[0]
        normalized_hit = torch.tensor(normalized_data[start:end], dtype=torch.float32)
        normalized_hits.append(normalized_hit)
        start = end

    return normalized_hits, scalers
