# Local Mac M4 Muon Count Experiment

## Objective

Rewrite the experiment plan so it matches the code that is actually present in this repository.

This repository already supports two preprocessing paths:

- ROOT -> `direction` regression
- HDF5 -> `muon_count` multiclass classification

For this experiment we use the HDF5 path and the HDF5 file that already exists in the repo:

- `data/muon_data_7224_7247.h5`

The target task is:

- `muon_count`

## Repository Alignment

The old template should not refer to filenames that do not exist here. The real entry points in this repo are:

- preprocessing: `pre-porcessing/pre_process.py`
- training: `model/train.py`
- inference: `model/infer.py`
- training config baseline: `configs/train.default.json`

The preprocessing directory is intentionally named `pre-porcessing/` in this repository and must be referenced exactly like that.

## Local Machine Assumptions

Experiments are expected to run on a local Apple Silicon Mac M4.

Important repo-specific note:

- the current training and inference code auto-selects `cuda` when available, otherwise `cpu`
- there is no automatic `mps` device selection in the current code
- on a Mac M4 this means the existing code path runs on `cpu`
- the data loader already has a macOS-safe path and uses `num_workers=0` on Darwin

So this program is aligned with the current repo by running locally on the Mac M4 CPU path, without renaming files or assuming a different launcher.

## HDF5 Dataset Contract Used By This Repo

The checked-in data file is:

- `data/muon_data_7224_7247.h5`

The current file contains:

- dataset `hits` with shape `(21555, 512, 8)`
- dataset `mc_muons` with shape `(21555, 7)`
- dataset `rec_muons` with shape `(21555, 6)`
- dataset `energies` with shape `(21555, 2)`

For the `muon_count` experiment, the preprocessing code uses:

- HDF5 hits dataset: `hits`
- HDF5 label dataset: `mc_muons`
- label column: `0`

The label values in `mc_muons[:, 0]` are balanced and currently map to:

- `0`
- `1`
- `2`

Each class has `7185` examples.

## Experiment Flow

The experiment should be run in four stages:

1. install dependencies
2. preprocess the HDF5 file into split `.pt` tensors under `data/`
3. train KM3Former for the `muon_count` task on the generated tensors
4. run inference on the test split

In addition, outputs from every experiment must be collected and preserved for the next experiments so changes can be tracked instead of overwritten.

## Experiment Tracking Requirement

This was part of the original intent and must remain part of the program:

- every experiment must write to its own run directory
- no experiment should overwrite the outputs of a previous experiment
- each run must preserve the exact training configuration, model outputs, and preprocessing snapshot used for that run
- later experiments should be compared against earlier saved outputs

Recommended run naming:

- `runs/muon_count_mac_m4/exp_001_baseline`
- `runs/muon_count_mac_m4/exp_002_model_dim_256`
- `runs/muon_count_mac_m4/exp_003_more_epochs`

Each experiment directory should collect at least:

- `best_model.pth`
- epoch checkpoints
- `resolved_train_config.json`
- TensorBoard logs
- test inference output such as `test_predictions.pt`
- a copy of `data/metadata.json`
- a copy of `data/hits_stats.pt`
- a short notes file describing what changed in that experiment

This makes the outputs reusable for the next experiments and keeps a clear history of model and configuration changes.

## Experiment Log Requirement

In addition to saving each run directory, every finished experiment should be logged to a root-level Markdown file:

- `results.md`

This log should be laconic. Each experiment record should be at most 2 to 3 sentences.

Each record should include the most important facts only:

- run name or run path
- short git commit hash
- how much data was used
- what was changed or kept as baseline
- best validation loss
- test accuracy when available
- confusion matrix when available
- status such as `keep`, `discard`, or `crash`

Recommended structure:

```markdown
## exp_001_baseline

Sentence 1: what data was used, which config or model setup was used, and whether this is a baseline or a modified run.
Sentence 2: the key result numbers such as best validation loss, test accuracy, and confusion matrix.
Sentence 3 if needed: one short interpretation or status note such as keep, discard, or crash.
```

Crash convention:

- if a run crashes before producing a usable checkpoint, say so explicitly
- if a metric is unavailable, say that it was not recorded instead of inventing a placeholder number

When an experiment is complete:

1. save the run directory under a unique `runs/.../exp_...` path
2. copy `metadata.json` and `hits_stats.pt` into that run directory
3. record the best validation loss from training
4. run inference and capture the confusion matrix when possible
5. append a short Markdown entry to `results.md`

This gives the project two layers of experiment tracking:

- full artifacts in each run directory
- a compact narrative log in `results.md` for quick comparison

## 1. Environment Setup

From the repository root:

```bash
uv sync
```

## 2. Preprocess The HDF5 File For `muon_count`

Run:

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

This command is aligned with the existing preprocessing code in `pre-porcessing/pre_process.py`.

It will generate the files the training pipeline expects:

- `data/train_hits.pt`
- `data/train_hits_raw.pt`
- `data/train_padding_mask.pt`
- `data/train_targets.pt`
- `data/val_hits.pt`
- `data/val_hits_raw.pt`
- `data/val_padding_mask.pt`
- `data/val_targets.pt`
- `data/test_hits.pt`
- `data/test_hits_raw.pt`
- `data/test_padding_mask.pt`
- `data/test_targets.pt`
- `data/metadata.json`
- `data/hits_stats.pt`

For the `muon_count` task, preprocessing does not write `*_muons.pt` files. Training should use `*_targets.pt`.

## 3. Training Configuration For A Local Mac M4 Run

The repo already includes `configs/train.default.json`, but its default `model_path` is `./model`, which would place checkpoints inside the source-code directory.

For a cleaner local experiment, use a dedicated config such as `configs/train.muon_count.mac_m4.json` with values like:

```json
{
  "paths": {
    "data_path": "./data",
    "model_path": "./runs/muon_count_mac_m4"
  },
  "model": {
    "model_dim": 128,
    "num_heads": 4,
    "num_encoder_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "pairwise_neighbors": 16
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0008,
    "epochs": 10
  },
  "loader": {
    "train_max_workers": 0,
    "eval_max_workers": 0
  }
}
```

Why this is suitable for the local Mac M4 setup:

- it keeps the experiment local and lightweight
- it avoids mixing checkpoints with source files
- it remains fully compatible with `model/train.py`
- the smaller model is more practical for CPU training on macOS

The config gives a Mac-friendly baseline, but each experiment should still use its own `model_path` so outputs are tracked separately.

## 4. Train The Muon Count Model

Run:

```bash
uv run python model/train.py \
  --config configs/train.muon_count.mac_m4.json \
  --model-path ./runs/muon_count_mac_m4/exp_001_baseline
```

This command will:

- read `data/metadata.json`
- detect `task = muon_count`
- build a multiclass model head using `target_dim` from metadata
- load `train`, `val`, and `test` tensors from `data/`
- train and save checkpoints

Expected outputs under `./runs/muon_count_mac_m4/exp_001_baseline/`:

- `best_model.pth`
- `model_epoch_1.pth`, `model_epoch_2.pth`, ...
- `resolved_train_config.json`
- `tensorboard/`

After training, collect the preprocessing snapshot into the same experiment directory:

```bash
mkdir -p ./runs/muon_count_mac_m4/exp_001_baseline
cp ./data/metadata.json ./runs/muon_count_mac_m4/exp_001_baseline/metadata.json
cp ./data/hits_stats.pt ./runs/muon_count_mac_m4/exp_001_baseline/hits_stats.pt
```

Also add a short notes file for change tracking, for example:

```text
experiment: exp_001_baseline
task: muon_count
input: data/muon_data_7224_7247.h5
changes: initial local Mac M4 baseline
```

## 5. Run Inference On The Test Split

Run:

```bash
uv run python model/infer.py \
  --checkpoint-path ./runs/muon_count_mac_m4/exp_001_baseline/best_model.pth \
  --split test \
  --data-path ./data \
  --output-path ./runs/muon_count_mac_m4/exp_001_baseline/test_predictions.pt \
  --device cpu
```

For this `muon_count` task, the inference payload should contain:

- `predictions`
- `probabilities`
- `predicted_classes`
- `predicted_labels`
- `task = muon_count`
- `target_kind = multiclass`

That inference file should stay inside the same experiment directory so it can be compared directly with later runs.

## 6. How To Use The Outputs In Later Experiments

For every new experiment:

1. keep preprocessing in `data/` unless the dataset itself changes
2. launch training with a new `--model-path`
3. save inference output inside that same new run directory
4. append the experiment summary to `results.md`
5. compare the new run against earlier runs using the saved config, checkpoint, TensorBoard logs, predictions, and Markdown log

Examples:

- baseline: `./runs/muon_count_mac_m4/exp_001_baseline`
- changed model size: `./runs/muon_count_mac_m4/exp_002_model_dim_256`
- changed training length: `./runs/muon_count_mac_m4/exp_003_epochs_20`

This is the part that keeps a track of changes across experiments: every run remains self-contained and reviewable.

## 7. Optional Validation Commands

Check that preprocessing produced the expected files:

```bash
find data -maxdepth 1 -type f | sort
```

Run the repository test suite:

```bash
uv run pytest
```

## Final Program Summary

This repository should be documented as an HDF5-based `muon_count` experiment on a local Mac M4 using:

- input file `data/muon_data_7224_7247.h5`
- preprocessing script `pre-porcessing/pre_process.py`
- training script `model/train.py`
- inference script `model/infer.py`

The program should no longer describe nonexistent filenames or a generic template pipeline. It should describe the real repo flow and preserve outputs between experiments:

1. preprocess `data/muon_data_7224_7247.h5` into split tensors in `data/`
2. train KM3Former for `muon_count`
3. save each experiment under its own run directory
4. collect checkpoint, config, metadata snapshot, and predictions for later comparison
5. append a short summary entry to `results.md`
6. infer on the test split and keep the predictions with the same experiment artifacts

## Colab Fast Ablation Wave

For the GPU ablation wave, use the stable Colab baseline config:

- `configs/train.muon_count.colab.json`

Experiment-specific overrides live in:

- `configs/experiments/exp_002_control_colab_current.json`
- `configs/experiments/exp_003_capacity_match.json`
- `configs/experiments/exp_004_pairbias_timecompress.json`
- `configs/experiments/exp_005_pairbias_dedup.json`
- `configs/experiments/exp_006_no_posenc.json`
- `configs/experiments/exp_007_pooling_upgrade.json`

The detailed Colab instructions for ephemeral `/content` runs are documented in:

- `docs/colab_muon_count_runbook.md`

New experiment outputs now include:

- `metrics.json`
- `val_predictions.pt`
- `test_predictions.pt`

For the ablation ladder, prefer validation macro-F1 and class-`1` recall as the main promotion signal, and use test accuracy as a secondary comparison metric.
