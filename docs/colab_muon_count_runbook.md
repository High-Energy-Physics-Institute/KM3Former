# Colab Muon-Count Runbook

This runbook is for the fast ablation wave on Colab GPU using ephemeral `/content`.

## 1. Start The Runtime

1. Open a new Colab notebook.
2. Set runtime type to `GPU`.
3. Work in `/content`.
4. Download every finished run before the session ends.

## 2. Upload The Repo And Dataset

At the start of the session, upload:

- `KM3Former.zip`
- `muon_data_7224_7247.h5`

Notebook cell:

```python
from google.colab import files
uploaded = files.upload()
```

## 3. Unpack And Install Dependencies

```bash
!apt-get -qq update
!apt-get -qq install -y unzip
!unzip -q KM3Former.zip -d /content
```

If the zip expands to `/content/KM3Former`:

```bash
%cd /content/KM3Former
```

Install `uv` and sync the environment:

```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
```

```python
import os
os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
```

```bash
!uv sync
```

Move the uploaded HDF5 file into the repo data directory:

```bash
!mkdir -p /content/KM3Former/data
!mv /content/muon_data_7224_7247.h5 /content/KM3Former/data/
```

Sanity checks:

```bash
!nvidia-smi
!ls -lh data/muon_data_7224_7247.h5
```

## 4. Preprocess Once Per Session

```bash
!uv run python pre-porcessing/pre_process.py \
  --source-format hdf5 \
  --task muon_count \
  --data-path ./data \
  --h5-path ./data/muon_data_7224_7247.h5 \
  --h5-hits-dataset hits \
  --h5-label-dataset mc_muons \
  --h5-label-col 0 \
  --count-class-values 0,1,2
```

Validate the outputs:

```bash
!ls -lh data/metadata.json data/hits_stats.pt data/train_targets.pt data/val_targets.pt data/test_targets.pt
```

## 5. Run One Experiment At A Time

Training pattern:

```bash
!uv run python model/train.py \
  --config configs/experiments/EXP_NAME.json \
  --model-path ./runs/EXP_NAME
```

Inference pattern:

```bash
!uv run python model/infer.py \
  --checkpoint-path ./runs/EXP_NAME/best_model.pth \
  --output-path ./runs/EXP_NAME/test_predictions.pt \
  --split test \
  --data-path ./data
```

Collect preprocessing artifacts into the run directory:

```bash
!cp data/metadata.json ./runs/EXP_NAME/
!cp data/hits_stats.pt ./runs/EXP_NAME/
```

Add a short notes file:

```bash
!printf '%s\n' 'One-line summary of what changed in this experiment.' > ./runs/EXP_NAME/notes.txt
```

## 6. Archive Each Finished Run

Because `/content` is ephemeral, zip every run immediately:

```bash
!cd runs && zip -r EXP_NAME.zip EXP_NAME
```

Download the archive:

```python
from google.colab import files
files.download(f"/content/KM3Former/runs/EXP_NAME.zip")
```

If `results.md` changed, download that too:

```python
files.download("/content/KM3Former/results.md")
```

## 7. End-Of-Session Safeguard

Before closing the notebook, make sure you downloaded:

- every `runs/EXP_NAME.zip`
- the latest `results.md`
- any notebook notes you want to keep

## Experiment Ladder

Run experiments in this order while keeping the same preprocessing and split:

1. `exp_002_control_colab_current`
2. `exp_003_capacity_match`
3. `exp_004_pairbias_timecompress`
4. `exp_005_pairbias_dedup`
5. `exp_006_no_posenc`
6. `exp_007_pooling_upgrade`

Promotion rule:

- prefer better validation macro-F1 and class-`1` recall
- use test accuracy as a secondary signal, not the only one
