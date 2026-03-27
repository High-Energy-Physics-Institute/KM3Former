# Results

## exp_001_baseline_local_smoke

Commit `4cec1d0`; used `data/muon_data_7224_7247.h5` with 21,555 events split into 17,244 train / 2,155 val / 2,156 test, and kept the baseline smoke setup in `runs/muon_count_smoke` (`model_dim=128`, `num_heads=4`, `num_encoder_layers=4`, `dim_feedforward=256`, `pairwise_neighbors=16`, `batch_size=32`, `epochs=1`, CPU). Best validation loss was `0.986383` and test accuracy was `0.502319`; confusion matrix (true rows, predicted cols) was `[[439, 140, 126], [309, 157, 233], [138, 127, 487]]`, so this is a usable baseline to keep. Initial note: the model already learned nontrivial structure after one epoch, but class `1` was the main weakness and was frequently confused with classes `0` and `2`.
