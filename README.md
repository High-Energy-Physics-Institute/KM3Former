# KM3Former

KM3Former is a transformer-based pipeline for muon direction reconstruction on KM3NeT-style detector data. The repository covers the full flow from ROOT event ingestion to padded tensor generation, model training, evaluation, and basic dataset-loading tests.

The project is now closer to a detector-aware space-time transformer than a plain sequence encoder. It still uses a relatively compact architecture, but it includes the most important pieces that were previously missing: padding masks, time-sorted hits, event-relative timing, geometry-aware pairwise attention bias, split-safe normalization, and direction-aware loss functions.

## What the repository does

At a high level, the codebase does five things:

1. Reads KM3NeT ROOT files and extracts truth tracks, reconstructed tracks, hit features, and event energy.
2. Converts each event into a variable-length hit tensor with deterministic physics-inspired features.
3. Splits the dataset into train, validation, and test subsets before fitting normalization statistics.
4. Pads every event to a fixed number of hits and saves the resulting tensors to disk.
5. Trains a transformer that uses normalized hits as token inputs and raw geometric hit features for pairwise attention bias.

## Current model summary

The current implementation is designed around these ideas:

- The sequence order is meaningful because hits are explicitly sorted by time.
- Padding is treated correctly through a boolean `src_key_padding_mask`.
- Event pooling is learned with masked attention pooling rather than a naive mean over padded rows.
- Direction regression uses normalized output vectors and a cosine-based loss instead of plain MSE alone.
- Attention is biased with pairwise geometry/time features so the network does not have to infer all detector structure from token embeddings alone.
- A small quality head is trained jointly using the agreement between the reconstructed track and Monte Carlo truth as a proxy target.

## Repository structure

```text
.
├── model/
│   ├── data_loader.py
│   ├── eval.py
│   ├── km3former.py
│   ├── scheduler.py
│   └── train.py
├── pre-porcessing/
│   ├── normalisation.py
│   ├── padding.py
│   ├── pre_process.py
│   └── root_loader.py
├── tests/
│   └── test_data_loader.py
├── pyproject.toml
└── README.md
```

`pre-porcessing/` is intentionally spelled that way because that is the directory name in the repository.

## Installation

The project is configured through [pyproject.toml](./pyproject.toml) and currently targets Python `>=3.10,<3.14`.

Install dependencies with `uv`:

```bash
uv sync
```

Core dependencies:

- `torch`
- `km3io`
- `awkward`
- `click`
- `tensorboard`
- `scikit-learn`
- `tqdm`

Developer dependency:

- `pytest`

## End-to-end pipeline

The main workflow is:

1. Run preprocessing to build `.pt` tensors from ROOT files.
2. Train the model on the saved train/validation/test splits.
3. Monitor metrics in TensorBoard and evaluate the best checkpoint on the test split.

Commands:

```bash
uv run python pre-porcessing/pre_process.py
uv run python model/train.py
uv run python model/train.py --config configs/train.default.json
uv run python model/infer.py --checkpoint-path ./model/best_model.pth --split test --output-path ./model/test_predictions.pt
uv run pytest
```

## Data preprocessing in detail

The preprocessing entry point is [pre_process.py](./pre-porcessing/pre_process.py). It orchestrates event loading, splitting, normalization, padding, and serialization.

### ROOT input discovery

The preprocessing script uses:

- `DATA_PATH = "./data"`
- `ROOT_PATH_PATTERN = "/root/mcv7.1.mupage_tuned.sirene.jterbr0000*"`

These are combined into:

```text
./data/root/mcv7.1.mupage_tuned.sirene.jterbr0000*
```

So the script expects the ROOT files to live under `./data/root/`.

### Event filtering and track selection

The event-level extraction logic lives in [root_loader.py](./pre-porcessing/root_loader.py).

For each event, the code:

- looks for a Monte Carlo muon track with `abs(pdgid) == 13`
- requires the selected MC track to have `t == 0`
- requires the selected MC track to have energy `<= 3000`
- rejects events without a valid MC muon candidate
- rejects events without a reconstructed track
- rejects events without any hits

Monte Carlo track selection:

- `_select_mc_track_index()` scans the MC track table.
- It returns the first track satisfying the muon, time-zero, and energy conditions.

Reconstructed track selection:

- `_select_reconstructed_track_index()` tries to choose the best available reconstructed track.
- It prefers the maximum of `lik`, `likelihood`, `quality`, or `fitinf` if those fields exist.
- If none of those exist but `chi2` exists, it chooses the minimum `chi2`.
- If no ranking field is available, it falls back to the first reconstructed track.

This is much safer than the old fixed `event.tracks.dir_x[1]` style indexing because it no longer assumes a specific reconstructed-track index is always valid.

### Hit ordering and hit tensor construction

For every accepted event, `_build_hits_tensor()` builds a hit matrix and sorts hits by time before any padding is applied.

That means the sequence dimension now has a defined meaning:

- earlier hits appear before later hits
- the transformer's positional encoding is applied over time-sorted hit order
- truncation to `MAX_HITS` keeps the earliest hits rather than arbitrary file order

Current hit fields produced by `_build_hits_tensor()` before deterministic transforms:

1. `hit_t`
2. `pos_x`
3. `pos_y`
4. `pos_z`
5. `dir_x`
6. `dir_y`
7. `dir_z`
8. `tot`
9. `hit_rank`
10. `radius_xy`

Feature meanings at the ROOT-loading stage:

- `hit_t`: raw hit time before per-event re-anchoring
- `pos_x`, `pos_y`, `pos_z`: detector coordinates
- `dir_x`, `dir_y`, `dir_z`: PMT or hit orientation direction
- `tot`: charge-like TOT feature before logarithmic compression
- `hit_rank`: normalized hit order index from 0 to 1 within the event
- `radius_xy`: radial distance in the detector transverse plane, computed as `sqrt(x^2 + y^2)`

After deterministic preprocessing:

- `*_hits.pt` stores the normalized model-input view, where `hit_t` becomes event-relative time and `tot` becomes `log_tot`
- `*_hits_raw.pt` keeps the legacy filename but now stores the deterministic pairwise-bias view, not untouched ROOT hits

The truth label saved as `*_muons.pt` is the normalized MC direction vector `[dir_x, dir_y, dir_z]`.

The auxiliary reconstructed label saved as `*_muons_rec.pt` is the normalized reconstructed direction vector chosen by the ranking heuristic above.

### Train/validation/test split

The split happens before fitted normalization statistics are computed.

Split configuration:

- train: `0.8`
- validation: `0.1`
- test: `0.1`
- random seed: `42`

The split indices are generated once by `torch.randperm(...)` and stored implicitly through the resulting saved files plus explicit metadata in `metadata.json`.

This is important because it prevents normalization leakage from validation or test data into the training preprocessing stage.

### Deterministic feature transforms

The deterministic feature transform logic lives in [normalisation.py](./pre-porcessing/normalisation.py).

Before affine normalization, the preprocessing applies physics-motivated fixed transforms:

- relative time is re-anchored per event
- spatial coordinates are divided by a fixed detector scale of `500.0`
- `log_tot` is compressed with `log1p(clamp(value, min=0))`
- `radius_xy` is divided by the same position scale
- direction vectors are left unchanged

The fixed position scale is:

```python
DEFAULT_POSITION_SCALE = 500.0
```

This gives the model a stable geometric scale even before any train-fit normalization is applied.

### Affine normalization

After the deterministic transforms, the code computes mean and standard deviation on the training split only.

The current affine-normalized feature indices are:

- `relative_t`
- `pos_x`
- `pos_y`
- `pos_z`
- `log_tot`
- `hit_rank`
- `radius_xy`

Direction components are intentionally not standardized:

- `dir_x`
- `dir_y`
- `dir_z`

This choice preserves directional semantics while still normalizing the scalar features that vary in scale.

The saved stats file is:

```text
./data/hits_stats.pt
```

Its content is:

```python
{
    "mean": mean_tensor,
    "std": std_tensor,
}
```

### Padding and masks

Variable-length events are padded to:

```python
MAX_HITS = 512
```

Padding behavior is implemented in [padding.py](./pre-porcessing/padding.py).

Important details:

- if an event has more than `MAX_HITS`, it is truncated earlier during hit sorting
- if an event has fewer than `MAX_HITS`, zero rows are appended
- the feature dimension is inferred dynamically from `tensor.shape[1]`
- a boolean padding mask is produced alongside the padded tensor
- `True` means the row is padding
- `False` means the row is a real hit

This mask is later passed directly into the transformer as `src_key_padding_mask`.

### Saved files

For each split, preprocessing writes:

- `train_hits.pt`, `val_hits.pt`, `test_hits.pt`
- `train_hits_raw.pt`, `val_hits_raw.pt`, `test_hits_raw.pt`
- `train_padding_mask.pt`, `val_padding_mask.pt`, `test_padding_mask.pt`
- `train_muons.pt`, `val_muons.pt`, `test_muons.pt`
- `train_muons_rec.pt`, `val_muons_rec.pt`, `test_muons_rec.pt`
- `train_muons_e.pt`, `val_muons_e.pt`, `test_muons_e.pt`

It also writes:

- `metadata.json`
- `hits_stats.pt`

Meaning of the saved tensors:

- `*_hits.pt`: normalized transformer inputs
- `*_hits_raw.pt`: legacy filename for the deterministically transformed pairwise-bias input view
- `*_padding_mask.pt`: boolean mask marking padded rows
- `*_muons.pt`: normalized truth direction vectors
- `*_muons_rec.pt`: normalized reconstructed direction vectors
- `*_muons_e.pt`: scalar MC energies

## Data loading

[data_loader.py](./model/data_loader.py) defines `KM3Loader`, the dataset class used by training and evaluation.

### Loader behavior

`KM3Loader` supports two loading strategies:

- `load_strategy="lazy"`: load the tensors the first time they are needed
- `load_strategy="eager"`: load all tensors immediately during dataset construction

The default is `lazy`.

### What each sample contains

When reconstructed labels are provided, each dataset sample is a 5-tuple:

1. normalized hits
2. raw hits
3. padding mask
4. truth muon direction
5. reconstructed muon direction

If reconstructed labels are omitted, the sample becomes a 4-tuple without the final item.

If energy labels are also provided, they are appended after the reconstructed direction so existing supervised consumers keep the same leading fields.

### Validation checks

The loader validates that all backing tensors have the same first dimension. If the event counts do not match, it raises a `ValueError` with a detailed mismatch summary.

## Model architecture in detail

The main model is defined in [km3former.py](./model/km3former.py).

### Inputs

The forward method accepts:

- `src`: normalized hit tokens with shape `[batch, hits, features]`
- `raw_hits`: raw geometry/time hit features used only for pairwise bias construction
- `padding_mask`: boolean mask with shape `[batch, hits]`
- `return_quality`: whether to also return the confidence head output

### Token embedding

Each hit is embedded with a small MLP:

```python
nn.Sequential(
    nn.Linear(input_dim, model_dim),
    nn.GELU(),
    nn.Linear(model_dim, model_dim),
    nn.LayerNorm(model_dim),
)
```

This is stronger than a single linear projection because it gives the network one nonlinear mixing stage before self-attention begins.

### Positional encoding

The model still uses sinusoidal positional encoding, but it is now implemented correctly for `batch_first=True`.

The shape convention is:

- input tokens: `[B, N, D]`
- positional encoding buffer: `[1, max_len, D]`

The encoding is applied as:

```python
x = x + self.pe[:, :x.size(1)]
```

Because hits are explicitly sorted by time, this positional encoding now corresponds to time order rather than arbitrary array order.

### Padding-aware transformer stack

Each transformer layer is a `torch.nn.TransformerEncoderLayer` configured with:

- `batch_first=True`
- `norm_first=True`
- GELU activation

Padding is handled in two ways:

- padded tokens are zeroed before the encoder
- `src_key_padding_mask=padding_mask` is passed to every encoder layer

This prevents fake padded rows from being attended to as if they were real hits.

### Pairwise geometry/time attention bias

The most detector-specific part of the architecture is `PairwiseAttentionBias`.

Instead of making attention depend only on token content, the model computes pairwise features between every pair of hits using `raw_hits`.

Current pairwise features:

1. `delta_t`
2. `abs(delta_t)`
3. `delta_x`
4. `delta_y`
5. `delta_z`
6. pairwise distance
7. source-hit orientation projected onto the connecting direction
8. target-hit orientation projected onto the reverse connecting direction

These pairwise features are mapped through an MLP to produce one bias value per attention head.

Conceptually, this means the attention score is no longer just:

```text
QK^T
```

It becomes content-based attention plus a learned physics-aware bias derived from geometry and timing.

### Sparse attention neighborhoods

The pairwise module also prunes the full `N x N` attention graph by keeping only local neighborhoods.

For each hit, the model keeps:

- the nearest `pairwise_neighbors` hits in time
- the nearest `pairwise_neighbors` hits in space
- the hit itself through the diagonal

The default is:

```python
pairwise_neighbors = 32
```

This gives the model a more detector-aware inductive bias and avoids treating every distant hit pair as equally important.

### Event pooling

After the encoder stack, the model uses `MaskedAttentionPooling` instead of a plain mean.

Why this matters:

- padded rows are excluded
- real hits can contribute with learned importance weights
- sparse, noisy, or highly structured events are summarized more flexibly than with `mean(dim=1)`

### Output heads

The model has two output heads:

- `fc_out`: predicts a 3D direction vector
- `quality_head`: predicts a scalar confidence/quality value in `[0, 1]`

The direction output is normalized with:

```python
F.normalize(self.fc_out(pooled), dim=-1)
```

This enforces the unit-vector structure expected for direction regression.

## Losses and evaluation

[eval.py](./model/eval.py) defines the main losses and evaluation metrics.

### Direction loss

The primary regression loss is cosine-based:

```python
loss = 1 - dot(normalized_prediction, normalized_target)
```

This is better aligned with directional reconstruction than plain coordinate MSE because it directly rewards angular agreement.

### Angular metric

For reporting, the evaluation code computes the mean angular error in degrees:

```python
angle_deg = rad2deg(arccos(clamped_cosine_similarity))
```

This is easier to interpret physically than raw loss values.

### Quality target

The code trains a quality head using a proxy target derived from the agreement between:

- the reconstructed track direction
- the true MC track direction

The target is:

```python
0.5 * (cosine_similarity(rec_track, truth_track) + 1.0)
```

So:

- `1.0` means excellent agreement
- `0.5` means orthogonal agreement
- `0.0` means opposite direction

### Combined loss

Training optimizes:

```python
combined_loss = cosine_direction_loss + 0.1 * quality_mse
```

The quality term is auxiliary. The main task is still direction reconstruction.

## Training loop in detail

[train.py](./model/train.py) drives model training.

### Default hyperparameters

Current defaults:

- `model_dim = 256`
- `num_heads = 8`
- `num_encoder_layers = 6`
- `dim_feedforward = 512`
- `dropout = 0.1`
- `pairwise_neighbors = 32`
- `batch_size = 256`
- `learning_rate = 8e-4`
- `epochs = 10`

### DataLoader defaults

The training script uses platform-aware DataLoader settings:

- on macOS, `num_workers = 0`
- on Linux, up to 4 workers for training
- for validation and test, up to 2 workers
- pinned memory is enabled when CUDA is available
- persistent workers and prefetching are used when multiprocessing workers are enabled

This behavior is implemented in:

- `get_default_num_workers()`
- `build_data_loader()`

### Optimizer and scheduler

[scheduler.py](./model/scheduler.py) builds:

- `AdamW` optimizer
- linear warmup schedule
- cosine annealing schedule
- `SequentialLR` to join them

Warmup uses 5% of the total training steps, with guard rails to avoid zero-step schedulers.

### Training behavior

Per batch, the script:

1. moves tensors to the selected device
2. runs the forward pass with `return_quality=True`
3. computes the proxy quality target
4. computes combined loss
5. computes the mean angular error for logging
6. backpropagates
7. clips gradients to `max_norm=1.0`
8. steps the optimizer
9. steps the scheduler

Per epoch, the script:

- logs training loss and angle
- evaluates on the validation split
- writes TensorBoard scalars
- saves `model_epoch_{N}.pth`
- updates `best_model.pth` when validation loss improves

At the end, it reloads `best_model.pth` and evaluates on the test split.

### Output locations

Training writes outputs under `./model/`:

- `model_epoch_1.pth`, `model_epoch_2.pth`, ...
- `best_model.pth`
- `resolved_train_config.json`
- `tensorboard/`

## Tests

The current automated tests live in:

- [test_data_loader.py](./tests/test_data_loader.py)
- [test_preprocessing.py](./tests/test_preprocessing.py)
- [test_train_and_infer.py](./tests/test_train_and_infer.py)

They cover:

- lazy and eager dataset loading returning identical samples
- optional energy-label loading without breaking older dataset shapes
- behavior when reconstructed labels are omitted
- validation of mismatched tensor lengths
- expected batch shapes from the DataLoader
- deterministic preprocessing transforms and padding-mask alignment
- metadata semantics for saved tensor roles
- config resolution and model construction smoke coverage
- checkpoint loading and inference smoke coverage
- safe macOS worker defaults
- Linux worker and pinned-memory defaults
- `move_batch_to_device()` behavior

Run the test suite with:

```bash
uv run pytest
```

## Important implementation notes

- The scripts assume they are run from the repository root because they use relative paths like `./data` and `./model`.
- Preprocessing must run before training.
- The code stores both normalized hits and a deterministic pairwise-bias view because the transformer uses them for different purposes.
- `*_hits_raw.pt` is a legacy filename and no longer means untouched ROOT hit values.
- The padding mask uses `True` for fake padded rows, matching PyTorch transformer conventions.
- The truth and reconstructed directions are normalized before being saved.
- Training now supports a JSON config file through a `click` CLI and saves the resolved run config next to checkpoints.
- Inference currently targets already-preprocessed tensors only; it does not run directly from ROOT files.

## Current limitations

The project is much stronger than the original baseline, but it is not a full hierarchical KM3NeT reconstruction system yet.

Important limitations that still remain:

- no categorical embeddings yet for DOM id, PMT id, line id, or detector-unit id
- no explicit DOM-level or line-level hierarchy
- no explicit causal residual term such as `delta_t - distance / v_light_in_water`
- no dedicated energy regression head yet
- no uncertainty calibration beyond the simple proxy quality head
- the experiment configuration and inference tooling are still minimal rather than a full experiment-management stack
- no ablation framework for comparing model variants

There is also one implementation detail worth being aware of: the ROOT-loading stage now keeps raw hit time and TOT, while deterministic transforms are applied centrally before tensors are saved. The legacy `*_hits_raw.pt` filename still refers to the transformed pairwise-bias view rather than untouched detector hits, so that distinction is a good one to keep explicit as preprocessing evolves.

## Suggested next improvements

If you want to keep pushing this toward a more KM3NeT-specific architecture, the next high-value upgrades would be:

1. add categorical detector embeddings such as DOM id, PMT id, and line id
2. add a more explicit causal physics feature in the pairwise bias
3. add energy or uncertainty heads for multi-task training
4. evaluate learned `[CLS]` pooling or a hybrid pooling strategy
5. extend the JSON config and inference utilities into a fuller experiment-management and ablation workflow
6. add end-to-end tests for preprocessing and a real training smoke test

## Acknowledgments

This work is developed in the context of KM3NeT-inspired muon reconstruction research with support from the High Energy Physics Institute, Tbilisi, Georgia, and the wider KM3NeT community.
