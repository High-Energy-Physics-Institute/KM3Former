import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = ROOT / "pre-porcessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from normalisation import apply_deterministic_hit_transforms
from pre_process import build_metadata, pad_hits_list


class PreprocessingContractTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
