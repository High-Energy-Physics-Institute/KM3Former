from glob import glob

import awkward as ak
import km3io as k1
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _to_numpy(field):
    return ak.to_numpy(field)


def _select_mc_track_index(event, max_energy=3000):
    if event.n_mc_tracks == 0:
        return None

    pdgid = _to_numpy(event.mc_tracks.pdgid)
    times = _to_numpy(event.mc_tracks.t)
    energies = _to_numpy(event.mc_tracks.E)

    candidate_indices = np.where(
        (np.abs(pdgid) == 13) & (times == 0) & (energies <= max_energy)
    )[0]
    if candidate_indices.size == 0:
        return None

    return int(candidate_indices[0])


def _select_reconstructed_track_index(event):
    if event.n_tracks == 0:
        return None

    ranked_fields = [
        ("lik", True),
        ("likelihood", True),
        ("quality", True),
        ("fitinf", True),
        ("chi2", False),
    ]

    for field_name, descending in ranked_fields:
        if hasattr(event.tracks, field_name):
            values = np.asarray(_to_numpy(getattr(event.tracks, field_name)))
            if values.size == 0:
                continue
            values = np.nan_to_num(
                values,
                nan=-np.inf if descending else np.inf,
            )
            if descending:
                return int(np.argmax(values))
            return int(np.argmin(values))

    return 0


def _build_hits_tensor(event, max_hits):
    if event.n_hits == 0:
        return None

    hit_time = torch.tensor(_to_numpy(event.hits.t), dtype=torch.float32)
    sort_indices = torch.argsort(hit_time)
    if max_hits is not None:
        sort_indices = sort_indices[:max_hits]

    hit_time = hit_time[sort_indices]
    hit_time = hit_time - hit_time[0]

    pos_x = torch.tensor(_to_numpy(event.hits.pos_x), dtype=torch.float32)[sort_indices]
    pos_y = torch.tensor(_to_numpy(event.hits.pos_y), dtype=torch.float32)[sort_indices]
    pos_z = torch.tensor(_to_numpy(event.hits.pos_z), dtype=torch.float32)[sort_indices]
    dir_x = torch.tensor(_to_numpy(event.hits.dir_x), dtype=torch.float32)[sort_indices]
    dir_y = torch.tensor(_to_numpy(event.hits.dir_y), dtype=torch.float32)[sort_indices]
    dir_z = torch.tensor(_to_numpy(event.hits.dir_z), dtype=torch.float32)[sort_indices]
    log_tot = torch.log1p(
        torch.clamp(
            torch.tensor(_to_numpy(event.hits.tot), dtype=torch.float32)[sort_indices],
            min=0.0,
        )
    )

    num_hits = hit_time.shape[0]
    if num_hits == 1:
        hit_rank = torch.zeros(1, dtype=torch.float32)
    else:
        hit_rank = torch.linspace(0.0, 1.0, steps=num_hits, dtype=torch.float32)
    radius_xy = torch.sqrt(pos_x.square() + pos_y.square())

    return torch.stack(
        [
            hit_time,
            pos_x,
            pos_y,
            pos_z,
            dir_x,
            dir_y,
            dir_z,
            log_tot,
            hit_rank,
            radius_xy,
        ],
        dim=-1,
    )


def process_event(event, max_hits=512):
    mc_track_index = _select_mc_track_index(event)
    reco_track_index = _select_reconstructed_track_index(event)
    hits_tensor = _build_hits_tensor(event, max_hits=max_hits)

    if mc_track_index is None or reco_track_index is None or hits_tensor is None:
        return None

    mc_track = torch.tensor(
        [
            event.mc_tracks.dir_x[mc_track_index],
            event.mc_tracks.dir_y[mc_track_index],
            event.mc_tracks.dir_z[mc_track_index],
        ],
        dtype=torch.float32,
    )
    mc_track = F.normalize(mc_track, dim=0)

    track = torch.tensor(
        [
            event.tracks.dir_x[reco_track_index],
            event.tracks.dir_y[reco_track_index],
            event.tracks.dir_z[reco_track_index],
        ],
        dtype=torch.float32,
    )
    track = F.normalize(track, dim=0)

    muon_energy = float(event.mc_tracks.E[mc_track_index])
    return mc_track, hits_tensor, track, muon_energy


def process_file(file_path, max_hits=512):
    file = k1.OfflineReader(file_path)
    muons_list, hits_list, reconstructed_muons_list, muon_e_list = [], [], [], []
    for event in tqdm(file):
        data = process_event(event, max_hits=max_hits)
        if data is None:
            continue

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
            file_path=file_path,
            max_hits=max_hits,
        )
        muons.extend(muons_)
        hits.extend(hits_)
        rec_muons.extend(rec_muons_)
        muon_energies.extend(muon_energies_)
        print(len(muons))

    return muons, hits, rec_muons, muon_energies
