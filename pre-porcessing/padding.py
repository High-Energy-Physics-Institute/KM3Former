import torch


def pad_tensor(tensor, target_length):
    current_length = tensor.shape[0]
    if current_length >= target_length:
        return tensor[:target_length], torch.zeros(target_length, dtype=torch.bool)

    padding = torch.zeros(
        target_length - current_length,
        tensor.shape[1],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded_tensor = torch.cat([tensor, padding], dim=0)
    padding_mask = torch.zeros(target_length, dtype=torch.bool, device=tensor.device)
    padding_mask[current_length:] = True
    return padded_tensor, padding_mask
