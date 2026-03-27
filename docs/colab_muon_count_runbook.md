# Colab Muon-Count Runbook

This runbook follows the checked-in `KM3Former.ipynb` notebook flow for Colab GPU runs in ephemeral `/content`.

## 1. Start The Runtime

1. Open a new Colab notebook.
2. Set runtime type to `GPU`.
3. Work in `/content`.
4. Download every finished run before the session ends.

## 2. Clone The Repo

The notebook clones the `muon-count` branch directly into `/content/KM3Former`.

```python
import os
if not os.path.exists('/content/KM3Former'):
    !git clone -b muon-count https://github.com/High-Energy-Physics-Institute/KM3Former.git KM3Former
%cd /content/KM3Former
```

## 3. Install The Notebook Dependencies

The notebook uses `uv pip install --system` instead of `uv sync`.

```python
!uv pip install km3io awkward tqdm pandas joblib --system

import sys
import os
repo_path = '/content/KM3Former'
if repo_path not in sys.path:
    sys.path.append(repo_path)
```

If the cloned branch does not already contain `data/muon_data_7224_7247.h5`, upload it manually and move it into `data/`:

```python
from google.colab import files
uploaded = files.upload()
```

```bash
!mkdir -p /content/KM3Former/data
!mv /content/muon_data_7224_7247.h5 /content/KM3Former/data/
```

Sanity checks:

```python
import torch
!nvidia-smi
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

```bash
!ls -lh data/muon_data_7224_7247.h5
```

## 4. Preprocess Once Per Session

```bash
!python pre-porcessing/pre_process.py \
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

Before training, the notebook enables expandable CUDA segments:

```python
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Notebook baseline training command:

```bash
!python model/train.py --config configs/train.muon_count.colab.json
```

General experiment pattern:

```bash
!python model/train.py \
  --config configs/experiments/EXP_NAME.json \
  --model-path ./runs/EXP_NAME
```

Inference pattern:

```bash
!python model/infer.py \
  --checkpoint-path ./runs/EXP_NAME/best_model.pth \
  --output-path ./runs/EXP_NAME/test_predictions.pt \
  --split test \
  --data-path ./data
```

The notebook baseline inference command is:

```bash
!python model/infer.py \
  --checkpoint-path ./runs/muon_count_colab/best_model.pth \
  --split test \
  --data-path ./data \
  --output-path ./runs/muon_count_colab/test_predictions.pt
```

Quick notebook-style checks after inference:

```python
import torch
p = torch.load('runs/EXP_NAME/test_predictions.pt', map_location='cpu')
print({
    k: (tuple(v.shape) if hasattr(v, 'shape') else v)
    for k, v in p.items()
    if k in ['task', 'target_kind', 'predictions', 'probabilities', 'predicted_classes', 'predicted_labels']
})
```

```python
y = torch.load('data/test_targets.pt', map_location='cpu')
pred = p['predicted_classes']
print({'test_accuracy': float((pred == y).float().mean())})
```

If you want the richer saved evaluation summary from the updated code, inspect:

```python
q = torch.load('runs/EXP_NAME/test_predictions.pt', map_location='cpu')
print(q.get('summary') or q.get('metrics'))
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

The notebook archives the default Colab run like this:

```bash
!zip -r /content/KM3Former/runs/muon_count_colab.zip /content/KM3Former/runs/muon_count_colab
```

For named ablation runs, zip every run immediately:

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
