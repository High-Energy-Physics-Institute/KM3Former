from glob import glob

import awkward as ak
import km3io as k1
import torch
from tqdm import tqdm


def process_event(event, max_hits=512):
    if (
        event.n_mc_tracks != 2
        or event.mc_tracks.pdgid[1] != 13
        or event.mc_tracks.t[1] != 0
        or event.mc_tracks.E[1] > 3000
        or event.n_tracks == 0
        or event.n_hits > max_hits
    ):
        return None

    mc_track = torch.tensor(
        [
            event.mc_tracks.dir_x[1],
            event.mc_tracks.dir_y[1],
            event.mc_tracks.dir_z[1],
        ],
        dtype=torch.float32,
    )

    track = torch.tensor(
        [
            event.tracks.dir_x[1],
            event.tracks.dir_y[1],
            event.tracks.dir_z[1],
        ],
        dtype=torch.float32,
    )

    hits_tensor = torch.stack(
        [
            torch.tensor(ak.to_numpy(event.hits.t), dtype=torch.float32),
            torch.tensor(ak.to_numpy(event.hits.pos_x), dtype=torch.float32),
            torch.tensor(ak.to_numpy(event.hits.pos_y), dtype=torch.float32),
            torch.tensor(ak.to_numpy(event.hits.pos_z), dtype=torch.float32),
            torch.tensor(ak.to_numpy(event.hits.dir_x), dtype=torch.float32),
            torch.tensor(ak.to_numpy(event.hits.dir_y), dtype=torch.float32),
            torch.tensor(ak.to_numpy(event.hits.dir_z), dtype=torch.float32),
            torch.tensor(ak.to_numpy(event.hits.tot), dtype=torch.float32),
        ],
        dim=0,
    )
    hits_t = hits_tensor.transpose(0, 1)

    return mc_track, hits_t, track, event.mc_tracks.E[1]


def process_file(file_path, max_hits=512):
    file = k1.OfflineReader(file_path)
    muons_list, hits_list, reconstructed_muons_list, muon_e_list = [], [], [], []
    for event in tqdm(file):
        data = process_event(event)
        if data is not None:
            mc_track, hits_tensor, track, muon_e = data
            muons_list.append(mc_track)
            hits_list.append(hits_tensor)
            reconstructed_muons_list.append(track)
            muon_e_list.append(muon_e)
    return muons_list, hits_list, reconstructed_muons_list, muon_e_list


def process_path(MC_path_pattern, max_hits=512):
    muons, hits, rec_muons, muon_energies = [], [], [], []
    for file_path in tqdm(glob(MC_path_pattern)):
        muons_, hits_, rec_muons_, muon_energies_ = process_file(
            file_path=file_path, max_hits=max_hits
        )
        muons.extend(muons_)
        hits.extend(hits_)
        rec_muons.extend(rec_muons_)
        muon_energies.extend(muon_energies_)
        print(len(muons))
    return muons, hits, rec_muons, muon_energies
