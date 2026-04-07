import math
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = ROOT / "pre-porcessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from normalisation import apply_deterministic_hit_transforms
from pre_process import (
    DEFAULT_HDF5_COLUMN_MAP,
    build_canonical_hdf5_hits,
    build_metadata,
    pad_hits_list,
    preprocess_dataset,
    resolve_preprocess_config,
)


class PreprocessingContractTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deterministic_transforms_anchor_time_and_scale_geometry(self):
        hit_tensor = torch.tensor(
            [
                [10.0, 500.0, -250.0, 100.0, 0.1, 0.2, 0.3, 9.0, 0.0, math.sqrt(500.0**2 + 250.0**2)],
                [13.0, 1000.0, 0.0, -400.0, -0.1, 0.0, 1.0, -2.0, 1.0, 1000.0],
            ],
            dtype=torch.float32,
        )

        transformed = apply_deterministic_hit_transforms([hit_tensor])[0]

        expected = torch.tensor(
            [
                [0.0, 1.0, -0.5, 0.2, 0.1, 0.2, 0.3, math.log1p(9.0), 0.0, math.sqrt(1.25)],
                [3.0, 2.0, 0.0, -0.8, -0.1, 0.0, 1.0, 0.0, 1.0, 2.0],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(transformed, expected, atol=1e-6))

    def test_padding_masks_stay_aligned_across_tensor_views(self):
        first_view = [
            torch.ones(2, 10, dtype=torch.float32),
            torch.full((1, 10), 2.0, dtype=torch.float32),
        ]
        second_view = [
            torch.full((2, 10), 3.0, dtype=torch.float32),
            torch.full((1, 10), 4.0, dtype=torch.float32),
        ]

        padded_first, first_mask = pad_hits_list(first_view, max_hits=4)
        padded_second, second_mask = pad_hits_list(second_view, max_hits=4)

        self.assertTrue(torch.equal(first_mask, second_mask))
        self.assertTrue(torch.equal(first_mask[0], torch.tensor([False, False, True, True])))
        self.assertTrue(torch.equal(first_mask[1], torch.tensor([False, True, True, True])))
        self.assertTrue(torch.equal(padded_first[0, 2:], torch.zeros(2, 10)))
        self.assertTrue(torch.equal(padded_second[1, 1:], torch.zeros(3, 10)))

    def test_metadata_describes_pairwise_bias_tensor_semantics(self):
        metadata = build_metadata(
            split_indices={
                "train": [0, 1, 2],
                "val": [3],
                "test": [4],
            }
        )

        self.assertEqual(
            metadata["saved_tensors"]["hits_raw.pt"]["semantic_role"],
            "pairwise_bias_input",
        )
        self.assertIn(
            "legacy filename",
            metadata["saved_tensors"]["hits_raw.pt"]["description"],
        )
        self.assertEqual(
            metadata["normalization"]["pairwise_bias_input"],
            "deterministic physics transforms only",
        )

    def test_hdf5_hits_are_canonicalized_into_ten_features(self):
        hits = torch.tensor(
            [
                [5.0, 4.0, 3.0, 2.0, 0.1, 0.2, 0.3, 7.0],
                [2.0, 1.0, -2.0, 5.0, -0.1, 0.5, 0.0, 9.0],
                [0.0] * 8,
            ],
            dtype=torch.float32,
        ).numpy()

        canonical = build_canonical_hdf5_hits(
            hit_rows=hits,
            column_map=DEFAULT_HDF5_COLUMN_MAP,
            max_hits=8,
            sort_by_time=True,
        )

        expected = torch.tensor(
            [
                [2.0, 1.0, -2.0, 5.0, -0.1, 0.5, 0.0, 9.0, 0.0, math.sqrt(5.0)],
                [5.0, 4.0, 3.0, 2.0, 0.1, 0.2, 0.3, 7.0, 1.0, 5.0],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(canonical, expected, atol=1e-6))

    def test_preprocess_dataset_writes_hdf5_count_outputs_and_metadata(self):
        h5_path = self.workspace / "counts.h5"
        output_dir = self.workspace / "preprocessed"

        num_events = 10
        hits = torch.zeros(num_events, 4, 8, dtype=torch.float32)
        labels = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=torch.int64)
        for event_index in range(num_events):
            base_time = float(event_index * 10)
            hits[event_index, 0] = torch.tensor(
                [base_time + 5.0, 10.0, 20.0, 30.0, 0.0, 1.0, 0.0, 8.0]
            )
            hits[event_index, 1] = torch.tensor(
                [base_time + 1.0, 13.0, 24.0, 35.0, 1.0, 0.0, 0.0, 4.0]
            )

        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("hits", data=hits.numpy())
            label_matrix = torch.stack(
                [labels.to(dtype=torch.float32), torch.zeros(num_events)],
                dim=1,
            )
            h5_file.create_dataset("mc_muons", data=label_matrix.numpy())

        resolved_config = resolve_preprocess_config(
            overrides={
                "paths": {
                    "data_path": str(output_dir),
                    "hdf5_path": str(h5_path),
                },
                "source": {"format": "hdf5"},
                "task": {"name": "muon_count"},
                "data": {
                    "max_hits": 4,
                    "train_fraction": 0.8,
                    "val_fraction": 0.1,
                    "test_fraction": 0.1,
                    "split_seed": 7,
                },
                "hdf5": {
                    "count_class_values": [1, 2, 3],
                },
            }
        )

        metadata = preprocess_dataset(resolved_config=resolved_config)

        self.assertEqual(metadata["task"], "muon_count")
        self.assertEqual(metadata["source_format"], "hdf5")
        self.assertEqual(metadata["target_kind"], "multiclass")
        self.assertEqual(metadata["class_values"], [1, 2, 3])
        self.assertEqual(metadata["class_to_index"], {1: 0, 2: 1, 3: 2})
        self.assertEqual(metadata["source_details"]["label_source"]["dataset"], "mc_muons")
        self.assertEqual(
            metadata["source_details"]["hit_column_map"],
            DEFAULT_HDF5_COLUMN_MAP,
        )

        train_targets = torch.load(output_dir / "train_targets.pt")
        train_hits = torch.load(output_dir / "train_hits.pt")
        train_raw_hits = torch.load(output_dir / "train_hits_raw.pt")
        train_padding_mask = torch.load(output_dir / "train_padding_mask.pt")
        metadata_payload = (output_dir / "metadata.json").read_text(encoding="ascii")

        self.assertEqual(train_targets.dtype, torch.long)
        self.assertEqual(train_hits.shape[1:], torch.Size([4, 10]))
        self.assertEqual(train_raw_hits.shape[1:], torch.Size([4, 10]))
        self.assertEqual(train_padding_mask.shape[1:], torch.Size([4]))
        self.assertTrue((output_dir / "hits_stats.pt").exists())
        self.assertNotIn("train_muons.pt", {path.name for path in output_dir.iterdir()})
        self.assertIn('"task": "muon_count"', metadata_payload)


if __name__ == "__main__":
    unittest.main()
