# KM3Former

KM3Former is a transformer-based pipeline for KM3NeT-style event reconstruction. The repository currently supports two end-to-end workflows:

- ROOT -> `direction` vector regression
- HDF5 -> `muon_count` count classification with multiclass or ordinal heads

The codebase covers preprocessing, saved tensor datasets, training, inference, and basic regression tests.

## Current Status

The repository is no longer just a ROOT direction-regression prototype. The current code supports:

- preprocessing from ROOT files for `direction`
- preprocessing from HDF5 files for `muon_count`
- task-aware metadata and targets
- training with the same model code for regression or classification
- inference from saved preprocessed tensors

The `pre-porcessing/` directory name is intentionally spelled that way because that is the actual directory name in this repository.

## Repository Layout

```text
.
├── configs/
│   ├── train.default.json
│   └── train.muon_count.smoke.json
├── model/
│   ├── config.py
│   ├── data_loader.py
│   ├── eval.py
│   ├── infer.py
│   ├── km3former.py
│   ├── scheduler.py
│   └── train.py
├── pre-porcessing/
│   ├── normalisation.py
│   ├── padding.py
│   ├── pre_process.py
│   └── root_loader.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_train_and_infer.py
├── program.md
├── pyproject.toml
└── README.md
```

## Installation

The project targets Python `>=3.10,<3.14` and uses `uv` for environment management.

```bash
uv sync
```

Core dependencies include:

- `torch`
- `km3io`
- `awkward`
- `h5py`
- `click`
- `tensorboard`
- `scikit-learn`
- `tqdm`

Run tests with:

```bash
uv run pytest -q
```

## Supported Data Flows

### 1. ROOT -> `direction`

The legacy preprocessing path reads ROOT files and builds direction-regression targets.

- source format: `root`
- task: `direction`
- target kind: `vector_regression`
- target dimension: `3`

This path produces normalized truth direction vectors and can optionally supervise the auxiliary quality head from reconstructed-track agreement.

### 2. HDF5 -> `muon_count`

The newer preprocessing path reads HDF5 detector events and builds ordered muon-count targets.

- source format: `hdf5`
- task: `muon_count`
- target kind: `multiclass`
- target dimension: number of classes, not scalar target width

Important details:

- each event target is still a single scalar class index in `*_targets.pt`
- `target_dim` is the number of output classes the classifier predicts
- the training target stays `multiclass`, while the model head can be plain multiclass logits or `ordinal_coral`

For the checked-in HDF5 dataset used in local experiments, the current classes are:

- `0`
- `1`
- `2`

## Preprocessing

The preprocessing entry point is [pre_process.py](/Users/djaz89/projects/KM3Former/pre-porcessing/pre_process.py).

It:

1. loads ROOT or HDF5 source data
2. builds canonical hit tensors
3. splits events into train, validation, and test
4. computes train-only normalization statistics
5. pads events to a fixed number of hits
6. writes tensors plus `metadata.json` and `hits_stats.pt`

### Preprocessing CLI

```bash
uv run python pre-porcessing/pre_process.py --help
```

Useful options:

- `--source-format root|hdf5`
- `--task direction|muon_count`
- `--data-path`
- `--root-path-pattern`
- `--h5-path`
- `--h5-hits-dataset`
- `--h5-label-dataset`
- `--h5-label-col`
- `--max-hits`
- `--count-class-values`

### Saved Tensors

For every split, preprocessing writes:

- `*_hits.pt`
- `*_hits_raw.pt`
- `*_padding_mask.pt`
- `*_targets.pt`

For `direction`, preprocessing also writes:

- `*_muons.pt`
- `*_muons_rec.pt`
- `*_muons_e.pt`

Shared outputs:

- `data/metadata.json`
- `data/hits_stats.pt`

Tensor meanings:

- `*_hits.pt`: normalized model inputs
- `*_hits_raw.pt`: deterministic geometry/time view used for pairwise attention bias
- `*_padding_mask.pt`: `True` marks padded rows
- `*_targets.pt`: task-aware primary targets used by training

## Data Loader Contract

[data_loader.py](/Users/djaz89/projects/KM3Former/model/data_loader.py) defines `KM3Loader`.

Each sample is returned as a dictionary, not a tuple. Common keys are:

- `hits`
- `raw_hits`
- `padding_mask`
- `target`

Optional keys appear when present in the backing tensors:

- `muons`
- `rec_muons`
- `energy`

The loader supports `lazy` and `eager` loading and validates that all saved tensors share the same event count.

## Model Summary

[km3former.py](/Users/djaz89/projects/KM3Former/model/km3former.py) defines the current model.

High-level characteristics:

- hit tokens are embedded with a small MLP
- positional encoding is configurable; the maintained `muon_count` configs now use `none`
- attention uses sparse local neighborhoods instead of full dense self-attention
- local neighborhoods are built once per forward pass and reused across encoder layers
- time neighborhoods can be index-window or `delta_t_radius`
- pairwise geometry/time features contribute learned attention bias through `v1` or `v2_physics` feature sets
- pooling can be `attention`, `attention_mean_concat`, or `attention_mean_sum_count_concat`
- the final head is task-aware through `target_kind`

Outputs:

- `direction`: normalized 3D direction vector, with optional quality head
- `muon_count` with `count_head = "multiclass"`: class logits of shape `[batch, num_classes]`
- `muon_count` with `count_head = "ordinal_coral"`: threshold logits of shape `[batch, num_classes - 1]`

The quality head is only active for `direction`. For `muon_count`, training and inference always expose class probabilities and predicted classes; ordinal runs additionally expose `threshold_logits` and `confidence`.

## Training

The training entry point is [train.py](/Users/djaz89/projects/KM3Former/model/train.py).

### Training CLI

```bash
uv run python model/train.py --help
```

Options:

- `--config`
- `--data-path`
- `--model-path`

Default training config lives in [train.default.json](/Users/djaz89/projects/KM3Former/configs/train.default.json):

```json
{
  "paths": {
    "data_path": "./data",
    "model_path": "./model"
  },
  "model": {
    "model_dim": 256,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "pairwise_neighbors": 32,
    "time_neighbors": null,
    "spatial_neighbors": null,
    "position_encoding": "sinusoidal",
    "time_neighborhood_mode": "index_window",
    "time_radius": null,
    "pairwise_time_transform": "raw",
    "pairwise_distance_transform": "raw",
    "pairwise_feature_version": "v1",
    "exclude_self_from_spatial_knn": false,
    "deduplicate_neighbors": false,
    "pooling": "attention",
    "count_head": "multiclass",
    "propagation_speed": 0.225
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.0008,
    "epochs": 10
  },
  "loader": {
    "train_max_workers": 4,
    "eval_max_workers": 2
  }
}
```

Platform note:

- the code selects `cuda` when available, otherwise `cpu`
- there is no automatic `mps` selection
- on Apple Silicon, the current path therefore runs on `cpu`
- on macOS, the loader falls back to `num_workers=0`

Training outputs:

- `model_epoch_N.pth`
- `best_model.pth`
- `resolved_train_config.json`
- `metrics.json`
- `val_predictions.pt`
- `test_predictions.pt`
- `tensorboard/`

## Inference

The inference entry point is [infer.py](/Users/djaz89/projects/KM3Former/model/infer.py).

### Inference CLI

```bash
uv run python model/infer.py --help
```

Inference runs from already-preprocessed tensors, either by split name or by explicit tensor paths.

Example:

```bash
uv run python model/infer.py \
  --checkpoint-path ./runs/muon_count_smoke/best_model.pth \
  --split test \
  --data-path ./data \
  --output-path ./runs/muon_count_smoke/test_predictions.pt \
  --device cpu
```

For `muon_count`, the output payload includes:

- `predictions`
- `probabilities`
- `predicted_classes`
- `predicted_labels`
- `confidence`
- `threshold_logits` for ordinal runs
- `targets` when available
- `metrics` when targets are available
- `task`
- `target_kind`

## Local `muon_count` Experiment

The local experiment plan is documented in [program.md](/Users/djaz89/projects/KM3Former/program.md).

The HDF5 file currently used for local `muon_count` experiments is:

- `data/muon_data_7224_7247.h5`

Observed dataset contract:

- `hits`: `(21555, 512, 8)`
- `mc_muons`: `(21555, 7)`
- `rec_muons`: `(21555, 6)`
- `energies`: `(21555, 2)`

The `muon_count` preprocessing path currently uses:

- hit dataset: `hits`
- label dataset: `mc_muons`
- label column: `0`
- class values: `0,1,2`

### Smoke Config

A lightweight smoke config lives at [train.muon_count.smoke.json](/Users/djaz89/projects/KM3Former/configs/train.muon_count.smoke.json).

It uses:

- `model_dim = 128`
- `num_heads = 4`
- `num_encoder_layers = 4`
- `dim_feedforward = 256`
- `pairwise_neighbors = 16`
- `position_encoding = none`
- `exclude_self_from_spatial_knn = true`
- `deduplicate_neighbors = true`
- `pooling = attention_mean_sum_count_concat`
- `batch_size = 32`
- `epochs = 1`
- `model_path = ./runs/muon_count_smoke`

### Smoke Run Commands

Preprocess:

```bash
uv run python pre-porcessing/pre_process.py \
  --source-format hdf5 \
  --task muon_count \
  --data-path ./data \
  --h5-path ./data/muon_data_7224_7247.h5 \
  --h5-hits-dataset hits \
  --h5-label-dataset mc_muons \
  --h5-label-col 0 \
  --count-class-values 0,1,2
```

Train:

```bash
uv run python model/train.py --config configs/train.muon_count.smoke.json
```

Infer:

```bash
uv run python model/infer.py \
  --checkpoint-path ./runs/muon_count_smoke/best_model.pth \
  --split test \
  --data-path ./data \
  --output-path ./runs/muon_count_smoke/test_predictions.pt \
  --device cpu
```

### Current Smoke-Run Results

The latest smoke run completed successfully and produced:

- [best_model.pth](/Users/djaz89/projects/KM3Former/runs/muon_count_smoke/best_model.pth)
- [model_epoch_1.pth](/Users/djaz89/projects/KM3Former/runs/muon_count_smoke/model_epoch_1.pth)
- [resolved_train_config.json](/Users/djaz89/projects/KM3Former/runs/muon_count_smoke/resolved_train_config.json)
- [test_predictions.pt](/Users/djaz89/projects/KM3Former/runs/muon_count_smoke/test_predictions.pt)

Observed signals from that 1-epoch run:

- validation loss: about `0.986`
- test accuracy: about `50.2%`
- class `0` accuracy: about `62.3%`
- class `1` accuracy: about `22.5%`
- class `2` accuracy: about `64.8%`

Interpretation:

- the pipeline is working end to end
- the classifier is learning nontrivial structure after one epoch
- the middle class is currently the weakest and is often confused with the outer classes

## Tests

The current test suite lives in:

- [test_data_loader.py](/Users/djaz89/projects/KM3Former/tests/test_data_loader.py)
- [test_preprocessing.py](/Users/djaz89/projects/KM3Former/tests/test_preprocessing.py)
- [test_train_and_infer.py](/Users/djaz89/projects/KM3Former/tests/test_train_and_infer.py)

Coverage includes:

- preprocessing metadata and saved tensor semantics
- HDF5 `muon_count` preprocessing
- loader behavior and tensor alignment checks
- task-aware model construction
- training config resolution
- inference smoke tests
- macOS-safe worker defaults

## Important Notes

- Run commands from the repository root.
- Preprocessing must happen before training.
- Inference expects already-preprocessed tensors and does not read ROOT or HDF5 directly.
- `target_dim` means output width for the head, not the scalar width of each saved label.
- `*_hits_raw.pt` is a legacy filename and does not mean untouched raw detector values.

## Colab Ablations

The stable Colab baseline config now lives at [train.muon_count.colab.json](/Users/djaz89/projects/KM3Former/configs/train.muon_count.colab.json).

The fast ablation configs live under:

- [exp_002_control_colab_current.json](/Users/djaz89/projects/KM3Former/configs/experiments/exp_002_control_colab_current.json)
- [exp_003_capacity_match.json](/Users/djaz89/projects/KM3Former/configs/experiments/exp_003_capacity_match.json)
- [exp_004_pairbias_timecompress.json](/Users/djaz89/projects/KM3Former/configs/experiments/exp_004_pairbias_timecompress.json)
- [exp_005_pairbias_dedup.json](/Users/djaz89/projects/KM3Former/configs/experiments/exp_005_pairbias_dedup.json)
- [exp_006_no_posenc.json](/Users/djaz89/projects/KM3Former/configs/experiments/exp_006_no_posenc.json)
- [exp_007_pooling_upgrade.json](/Users/djaz89/projects/KM3Former/configs/experiments/exp_007_pooling_upgrade.json)
- [exp_008_capacity_push.json](/Users/djaz89/projects/KM3Former/configs/experiments/exp_008_capacity_push.json)

The detailed ephemeral Colab workflow is documented in [colab_muon_count_runbook.md](/Users/djaz89/projects/KM3Former/docs/colab_muon_count_runbook.md) and is aligned with [KM3Former.ipynb](/Users/djaz89/projects/KM3Former/KM3Former.ipynb).

## Next Useful Experiments

- compare `count_head = multiclass` against `count_head = ordinal_coral` on the same split
- benchmark `delta_t_radius` plus `pairwise_feature_version = v2_physics` against the current default sparse path
- inspect class-`1` confusion in more detail
- keep saving each experiment under its own run directory for clean comparisons
