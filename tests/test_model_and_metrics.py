import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from eval import build_multiclass_summary
from km3former import KM3Former, PairwiseAttentionBias, SparseNeighborhoodSelfAttention


class ModelAblationTestCase(unittest.TestCase):
    def test_pairwise_bias_log_transforms_stay_finite(self):
        bias = PairwiseAttentionBias(
            num_heads=2,
            hidden_dim=4,
            time_transform="signed_log1p",
            distance_transform="log1p",
        )
        raw_hits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [100.0, 0.2, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [250.0, 0.4, 0.3, 0.1, 0.0, 0.0, 1.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        neighbor_indices = torch.tensor(
            [[[0, 1], [1, 2], [2, 1]]],
            dtype=torch.long,
        )

        output = bias(raw_hits, neighbor_indices)

        self.assertEqual(output.shape, torch.Size([1, 2, 3, 2]))
        self.assertTrue(torch.isfinite(output).all())

    def test_spatial_knn_can_exclude_self(self):
        attention = SparseNeighborhoodSelfAttention(
            model_dim=8,
            num_heads=2,
            dropout=0.0,
            pairwise_hidden_dim=8,
            time_neighbors=1,
            spatial_neighbors=2,
            exclude_self_from_spatial_knn=True,
        )
        raw_hits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )

        spatial_indices, spatial_mask = attention._build_spatial_knn_indices(raw_hits)

        for query_index in range(raw_hits.size(1)):
            valid_neighbors = spatial_indices[0, query_index][spatial_mask[0, query_index]]
            self.assertNotIn(query_index, valid_neighbors.tolist())

    def test_deduplicated_neighbors_keep_unique_valid_entries(self):
        attention = SparseNeighborhoodSelfAttention(
            model_dim=8,
            num_heads=2,
            dropout=0.0,
            pairwise_hidden_dim=8,
            time_neighbors=1,
            spatial_neighbors=2,
            deduplicate_neighbors=True,
        )
        raw_hits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )

        neighbor_indices, neighbor_mask = attention.build_neighborhood_indices(raw_hits)
        valid_neighbors = neighbor_indices[0, 1][neighbor_mask[0, 1]].tolist()

        self.assertEqual(len(valid_neighbors), len(set(valid_neighbors)))
        self.assertEqual(valid_neighbors.count(1), 1)

    def test_model_supports_attention_mean_concat_pooling(self):
        model = KM3Former(
            input_dim=10,
            model_dim=16,
            num_heads=4,
            num_encoder_layers=2,
            dim_feedforward=32,
            pairwise_neighbors=2,
            target_dim=3,
            target_kind="multiclass",
            position_encoding="none",
            pooling="attention_mean_concat",
            pairwise_time_transform="signed_log1p",
            pairwise_distance_transform="log1p",
            exclude_self_from_spatial_knn=True,
            deduplicate_neighbors=True,
        )
        hits = torch.randn(2, 4, 10)
        raw_hits = torch.randn(2, 4, 10)
        padding_mask = torch.tensor(
            [[False, False, False, True], [False, False, True, True]],
            dtype=torch.bool,
        )

        predictions = model(hits, raw_hits=raw_hits, padding_mask=padding_mask)

        self.assertEqual(predictions.shape, torch.Size([2, 3]))


class MetricsSummaryTestCase(unittest.TestCase):
    def test_multiclass_summary_includes_confusion_and_macro_f1(self):
        predictions = torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.5, 1.0, 3.0],
                [2.5, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        summary = build_multiclass_summary(
            predictions=predictions,
            targets=targets,
            class_values=[0, 1, 2],
        )

        self.assertIn("confusion_matrix", summary)
        self.assertIn("macro_f1", summary)
        self.assertIn("per_class_recall", summary)
        self.assertIn("per_class_accuracy", summary)
        self.assertEqual(summary["confusion_matrix"], [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(summary["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
