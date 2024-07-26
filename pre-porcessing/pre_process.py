import joblib
import torch
from normalisation import normalize_nested_tensor_list, normalize_tensor_list
from padding import pad_tensor
from root_loader import process_path

MAX_HITS = 512
DATA_PATH = "./data"
root_path_pattern = "/root/mcv7.1.mupage_tuned.sirene.jterbr0000*"


if __name__ == "__main__":
    muons, hits, rec_muons, muon_energies = process_path(
        f"{DATA_PATH}{root_path_pattern}", MAX_HITS
    )
    normalized_muons, muon_scalers = normalize_tensor_list(muons)
    normalized_hits, hits_scalers = normalize_nested_tensor_list(hits)
    padded_hits = [
        pad_tensor(tensor, target_length=MAX_HITS) for tensor in normalized_hits
    ]

    final_muons = torch.stack(normalized_muons)
    final_hits = torch.stack(padded_hits)
    final_rec_muons = torch.stack(rec_muons)
    final_energies = torch.tensor(muon_energies)

    torch.save(final_muons, f"{DATA_PATH}/muons.pt")
    torch.save(final_rec_muons, f"{DATA_PATH}/muons-rec.pt")
    torch.save(final_energies, f"{DATA_PATH}/muons-e.pt")
    torch.save(padded_hits, f"{DATA_PATH}/hits.pt")

    for i, scaler in enumerate(muon_scalers):
        joblib.dump(scaler, f"{DATA_PATH}/scalers/muon_scaler{i}.joblib")

    for i, scaler in enumerate(hits_scalers):
        joblib.dump(scaler, f"{DATA_PATH}/scalers/hits_scaler{i}.joblib")
