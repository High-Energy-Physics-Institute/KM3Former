import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from eval import build_multiclass_summary
from km3former import (
    AttentionMeanSumCountConcatPooling,
    KM3Former,
    PairwiseAttentionBias,
    SparseNeighborhoodSelfAttention,
)


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

    def test_attention_context_is_cached_across_encoder_layers(self):
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
            exclude_self_from_spatial_knn=True,
            deduplicate_neighbors=True,
        )
        first_attention = model.encoder_layers[0].self_attention
        second_attention = model.encoder_layers[1].self_attention
        first_attention.build_attention_context = Mock(
            wraps=first_attention.build_attention_context
        )
        first_attention.pairwise_bias.build_pair_features = Mock(
            wraps=first_attention.pairwise_bias.build_pair_features
        )
        second_attention.build_attention_context = Mock(
            side_effect=AssertionError("Second encoder layer should reuse cached context.")
        )
        second_attention.pairwise_bias.build_pair_features = Mock(
            side_effect=AssertionError("Pair features should be computed once per batch.")
        )

        _ = model(
            torch.randn(2, 4, 10),
            raw_hits=torch.randn(2, 4, 10),
            padding_mask=torch.tensor(
                [[False, False, False, True], [False, False, True, True]],
                dtype=torch.bool,
            ),
        )

        self.assertEqual(first_attention.build_attention_context.call_count, 1)
        self.assertEqual(first_attention.pairwise_bias.build_pair_features.call_count, 1)

    def test_delta_t_radius_neighborhood_uses_time_values(self):
        attention = SparseNeighborhoodSelfAttention(
            model_dim=8,
            num_heads=2,
            dropout=0.0,
            pairwise_hidden_dim=8,
            time_neighbors=1,
            spatial_neighbors=0,
            time_neighborhood_mode="delta_t_radius",
            time_radius=5.0,
        )
        raw_hits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1],
                    [100.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2],
                    [2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
                    [1000.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.4],
                ]
            ],
            dtype=torch.float32,
        )

        neighbor_indices, neighbor_mask = attention.build_neighborhood_indices(raw_hits)
        valid_neighbors = neighbor_indices[0, 0][neighbor_mask[0, 0]].tolist()

        self.assertEqual(valid_neighbors, [0, 2])

    def test_pairwise_bias_v2_physics_uses_position_scale_for_causal_residual(self):
        bias = PairwiseAttentionBias(
            num_heads=2,
            hidden_dim=4,
            feature_version="v2_physics",
            position_scale=500.0,
            propagation_speed=2.0,
        )
        raw_hits = torch.tensor(
            [
                [
                    [100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5],
                    [160.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        neighbor_indices = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)

        pair_features = bias.build_pair_features(raw_hits, neighbor_indices)
        output = bias(pair_features=pair_features)

        self.assertEqual(pair_features.shape, torch.Size([1, 2, 2, 13]))
        self.assertAlmostEqual(pair_features[0, 0, 1, 8].item(), -110.0, places=4)
        self.assertAlmostEqual(pair_features[0, 0, 1, 10].item(), 1.5, places=4)
        self.assertAlmostEqual(pair_features[0, 0, 1, 11].item(), 0.5, places=4)
        self.assertTrue(torch.isfinite(output).all())

    def test_count_aware_pooling_tracks_valid_hit_count(self):
        pooler = AttentionMeanSumCountConcatPooling(model_dim=4)
        memory = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )

        pooled_two_hits = pooler(
            memory,
            padding_mask=torch.tensor([[False, False, True]], dtype=torch.bool),
        )
        pooled_three_hits = pooler(
            memory,
            padding_mask=torch.tensor([[False, False, False]], dtype=torch.bool),
        )

        self.assertEqual(pooled_two_hits.shape[-1], 13)
        self.assertAlmostEqual(
            pooled_two_hits[0, -1].item(),
            torch.log1p(torch.tensor(2.0)).item(),
            places=5,
        )
        self.assertAlmostEqual(
            pooled_three_hits[0, -1].item(),
            torch.log1p(torch.tensor(3.0)).item(),
            places=5,
        )
        self.assertFalse(torch.allclose(pooled_two_hits, pooled_three_hits))

    def test_ordinal_coral_head_outputs_threshold_logits(self):
        model = KM3Former(
            input_dim=10,
            model_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            dim_feedforward=32,
            pairwise_neighbors=2,
            target_dim=3,
            target_kind="multiclass",
            position_encoding="none",
            count_head="ordinal_coral",
        )

        predictions = model(
            torch.randn(2, 4, 10),
            raw_hits=torch.randn(2, 4, 10),
            padding_mask=torch.tensor(
                [[False, False, False, True], [False, False, True, True]],
                dtype=torch.bool,
            ),
        )

        self.assertEqual(predictions.shape, torch.Size([2, 2]))


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
