import torch


def pad_tensor(tensor, target_length):
    current_length = tensor.shape[0]
    if current_length >= target_length:
        return tensor[:target_length]
    padding = torch.zeros(
        target_length - current_length, 8, dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=0)
