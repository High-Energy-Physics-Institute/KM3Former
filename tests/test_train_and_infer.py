import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from config import DEFAULT_TRAIN_CONFIG, resolve_train_config
from infer import main as infer_cli, run_inference
from train import (
    DEFAULT_METRICS_PATH,
    build_model,
    main as train_cli,
    resolve_quality_supervision,
    train_model,
)


class TrainConfigAndInferenceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.data_dir = self.workspace / "data"
        self.model_dir = self.workspace / "model"
        self.data_dir.mkdir()
        self.model_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, name, payload):
        path = self.workspace / name
        with open(path, "w", encoding="ascii") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
        return path

    def _write_tensor(self, relative_path, tensor):
        path = self.workspace / relative_path
        torch.save(tensor, path)
        return path

    def _write_runtime_files(self, metadata_overrides=None):
        metadata = {
            "task": "direction",
            "target_kind": "vector_regression",
            "target_name": "muon_direction",
            "target_dim": 3,
            "input_dim": 10,
            "max_hits": 4,
            "feature_names": [f"feature_{index}" for index in range(10)],
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        self._write_json("data/metadata.json", metadata)
        self._write_tensor(
            "data/hits_stats.pt",
            {
                "mean": torch.zeros(10, dtype=torch.float32),
                "std": torch.ones(10, dtype=torch.float32),
            },
        )
        hits = torch.randn(2, 4, 10, dtype=torch.float32)
        raw_hits = torch.randn(2, 4, 10, dtype=torch.float32)
        padding_mask = torch.tensor(
            [[False, False, False, True], [False, False, True, True]],
            dtype=torch.bool,
        )
        hits_path = self._write_tensor("explicit_hits.pt", hits)
        raw_hits_path = self._write_tensor("explicit_raw_hits.pt", raw_hits)
        padding_mask_path = self._write_tensor("explicit_padding_mask.pt", padding_mask)
        return metadata, hits_path, raw_hits_path, padding_mask_path

    def _write_split_tensors(self, split_name, event_count, target_kind="multiclass"):
        hits = torch.randn(event_count, 4, 10, dtype=torch.float32)
        raw_hits = torch.randn(event_count, 4, 10, dtype=torch.float32)
        padding_mask = torch.tensor(
            [[False, False, False, True]] * event_count,
            dtype=torch.bool,
        )
        self._write_tensor(f"data/{split_name}_hits.pt", hits)
        self._write_tensor(f"data/{split_name}_hits_raw.pt", raw_hits)
        self._write_tensor(f"data/{split_name}_padding_mask.pt", padding_mask)
        if target_kind == "multiclass":
            targets = torch.arange(event_count, dtype=torch.long) % 3
        else:
            targets = torch.randn(event_count, 3, dtype=torch.float32)
        self._write_tensor(f"data/{split_name}_targets.pt", targets)

    def test_resolve_train_config_keeps_defaults_while_applying_overrides(self):
        config_path = self._write_json(
            "train_override.json",
            {
                "paths": {
                    "data_path": str(self.data_dir),
                    "model_path": str(self.model_dir),
                },
                "model": {
                    "model_dim": 64,
                    "num_heads": 4,
                    "num_encoder_layers": 2,
                    "dim_feedforward": 128,
                    "pairwise_neighbors": 8,
                },
                "training": {
                    "batch_size": 8,
                },
            },
        )

        resolved = resolve_train_config(config_path=str(config_path))
        model = build_model(
            metadata={"input_dim": 10, "max_hits": 4},
            resolved_config=resolved,
            device=torch.device("cpu"),
        )
        predictions, quality = model(
            torch.randn(2, 4, 10),
            raw_hits=torch.randn(2, 4, 10),
            padding_mask=torch.tensor(
                [[False, False, False, True], [False, False, True, True]],
                dtype=torch.bool,
            ),
            return_quality=True,
        )

        self.assertEqual(resolved["training"]["batch_size"], 8)
        self.assertEqual(
            resolved["training"]["learning_rate"],
            DEFAULT_TRAIN_CONFIG["training"]["learning_rate"],
        )
        self.assertEqual(
            resolved["model"]["position_encoding"],
            DEFAULT_TRAIN_CONFIG["model"]["position_encoding"],
        )
        self.assertEqual(predictions.shape, torch.Size([2, 3]))
        self.assertEqual(quality.shape, torch.Size([2]))

    def test_build_model_supports_multiclass_targets(self):
        resolved = resolve_train_config(
            overrides={
                "paths": {
                    "data_path": str(self.data_dir),
                    "model_path": str(self.model_dir),
                },
                "model": {
                    "model_dim": 64,
                    "num_heads": 4,
                    "num_encoder_layers": 2,
                    "dim_feedforward": 128,
                    "pairwise_neighbors": 8,
                },
            }
        )
        model = build_model(
            metadata={
                "task": "muon_count",
                "target_kind": "multiclass",
                "target_dim": 3,
                "input_dim": 10,
                "max_hits": 4,
            },
            resolved_config=resolved,
            device=torch.device("cpu"),
        )
        predictions = model(
            torch.randn(2, 4, 10),
            raw_hits=torch.randn(2, 4, 10),
            padding_mask=torch.tensor(
                [[False, False, False, True], [False, False, True, True]],
                dtype=torch.bool,
            ),
            return_quality=True,
        )

        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape, torch.Size([2, 3]))

    def test_click_train_cli_exposes_help(self):
        result = CliRunner().invoke(train_cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--config", result.output)

    def test_inference_smoke_runs_on_explicit_tensor_paths(self):
        metadata, hits_path, raw_hits_path, padding_mask_path = self._write_runtime_files()
        resolved_config = resolve_train_config(
            overrides={
                "paths": {
                    "data_path": str(self.data_dir),
                    "model_path": str(self.model_dir),
                },
                "model": {
                    "model_dim": 64,
                    "num_heads": 4,
                    "num_encoder_layers": 2,
                    "dim_feedforward": 128,
                    "pairwise_neighbors": 8,
                },
            }
        )
        model = build_model(
            metadata=metadata,
            resolved_config=resolved_config,
            device=torch.device("cpu"),
        )
        checkpoint_path = self.workspace / "model" / "checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "resolved_config": resolved_config,
                "quality_supervised": True,
            },
            checkpoint_path,
        )

        output_path = self.workspace / "predictions.pt"
        payload = run_inference(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            data_path=str(self.data_dir),
            hits_file=str(hits_path),
            raw_hits_file=str(raw_hits_path),
            padding_mask_file=str(padding_mask_path),
            batch_size=1,
            device="cpu",
        )

        self.assertTrue(output_path.exists())
        self.assertEqual(payload["predictions"].shape, torch.Size([2, 3]))
        self.assertEqual(payload["quality"].shape, torch.Size([2]))
        self.assertTrue(payload["supports_quality"])

    def test_click_inference_cli_smoke_runs(self):
        metadata, hits_path, raw_hits_path, padding_mask_path = self._write_runtime_files()
        resolved_config = resolve_train_config(
            overrides={
                "paths": {
                    "data_path": str(self.data_dir),
                    "model_path": str(self.model_dir),
                },
                "model": {
                    "model_dim": 64,
                    "num_heads": 4,
                    "num_encoder_layers": 2,
                    "dim_feedforward": 128,
                    "pairwise_neighbors": 8,
                },
            }
        )
        model = build_model(
            metadata=metadata,
            resolved_config=resolved_config,
            device=torch.device("cpu"),
        )
        checkpoint_path = self.workspace / "model" / "cli_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "resolved_config": resolved_config,
                "quality_supervised": True,
            },
            checkpoint_path,
        )
        output_path = self.workspace / "predictions.json"

        result = CliRunner().invoke(
            infer_cli,
            [
                "--checkpoint-path",
                str(checkpoint_path),
                "--output-path",
                str(output_path),
                "--data-path",
                str(self.data_dir),
                "--hits-file",
                str(hits_path),
                "--raw-hits-file",
                str(raw_hits_path),
                "--padding-mask-file",
                str(padding_mask_path),
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(output_path.exists())

    def test_inference_omits_quality_for_direction_only_checkpoint(self):
        metadata, hits_path, raw_hits_path, padding_mask_path = self._write_runtime_files()
        resolved_config = resolve_train_config(
            overrides={
                "paths": {
                    "data_path": str(self.data_dir),
                    "model_path": str(self.model_dir),
                },
                "model": {
                    "model_dim": 64,
                    "num_heads": 4,
                    "num_encoder_layers": 2,
                    "dim_feedforward": 128,
                    "pairwise_neighbors": 8,
                },
            }
        )
        model = build_model(
            metadata=metadata,
            resolved_config=resolved_config,
            device=torch.device("cpu"),
        )
        checkpoint_path = self.workspace / "model" / "direction_only_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "resolved_config": resolved_config,
                "quality_supervised": False,
            },
            checkpoint_path,
        )

        payload = run_inference(
            checkpoint_path=str(checkpoint_path),
            output_path=str(self.workspace / "direction_only_predictions.pt"),
            data_path=str(self.data_dir),
            hits_file=str(hits_path),
            raw_hits_file=str(raw_hits_path),
            padding_mask_file=str(padding_mask_path),
            batch_size=1,
            device="cpu",
        )

        self.assertEqual(payload["predictions"].shape, torch.Size([2, 3]))
        self.assertFalse(payload["supports_quality"])
        self.assertNotIn("quality", payload)

    def test_inference_exposes_multiclass_outputs_for_count_task(self):
        metadata, hits_path, raw_hits_path, padding_mask_path = self._write_runtime_files(
            metadata_overrides={
                "task": "muon_count",
                "target_kind": "multiclass",
                "target_name": "muon_count",
                "target_dim": 3,
                "class_values": [1, 2, 3],
            }
        )
        resolved_config = resolve_train_config(
            overrides={
                "paths": {
                    "data_path": str(self.data_dir),
                    "model_path": str(self.model_dir),
                },
                "model": {
                    "model_dim": 64,
                    "num_heads": 4,
                    "num_encoder_layers": 2,
                    "dim_feedforward": 128,
                    "pairwise_neighbors": 8,
                },
            }
        )
        model = build_model(
            metadata=metadata,
            resolved_config=resolved_config,
            device=torch.device("cpu"),
        )
        checkpoint_path = self.workspace / "model" / "count_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "resolved_config": resolved_config,
                "quality_supervised": False,
                "task": "muon_count",
                "target_kind": "multiclass",
            },
            checkpoint_path,
        )

        payload = run_inference(
            checkpoint_path=str(checkpoint_path),
            output_path=str(self.workspace / "count_predictions.pt"),
            data_path=str(self.data_dir),
            hits_file=str(hits_path),
            raw_hits_file=str(raw_hits_path),
            padding_mask_file=str(padding_mask_path),
            target_file=str(self._write_tensor("explicit_targets.pt", torch.tensor([0, 2]))),
            batch_size=1,
            device="cpu",
        )

        self.assertEqual(payload["task"], "muon_count")
        self.assertEqual(payload["target_kind"], "multiclass")
        self.assertEqual(payload["predictions"].shape, torch.Size([2, 3]))
        self.assertEqual(payload["probabilities"].shape, torch.Size([2, 3]))
        self.assertEqual(payload["predicted_classes"].shape, torch.Size([2]))
        self.assertEqual(payload["predicted_labels"].shape, torch.Size([2]))
        self.assertEqual(payload["targets"].shape, torch.Size([2]))
        self.assertIn("metrics", payload)
        self.assertIn("confusion_matrix", payload["metrics"])
        self.assertFalse(payload["supports_quality"])
        self.assertNotIn("quality", payload)

    def test_train_model_writes_metrics_and_prediction_artifacts(self):
        self._write_json(
            "data/metadata.json",
            {
                "task": "muon_count",
                "target_kind": "multiclass",
                "target_name": "muon_count",
                "target_dim": 3,
                "input_dim": 10,
                "max_hits": 4,
                "class_values": [0, 1, 2],
            },
        )
        self._write_split_tensors("train", event_count=3)
        self._write_split_tensors("val", event_count=2)
        self._write_split_tensors("test", event_count=2)
        resolved_config = resolve_train_config(
            overrides={
                "paths": {
                    "data_path": str(self.data_dir),
                    "model_path": str(self.model_dir),
                },
                "model": {
                    "model_dim": 16,
                    "num_heads": 4,
                    "num_encoder_layers": 1,
                    "dim_feedforward": 32,
                    "pairwise_neighbors": 2,
                    "position_encoding": "none",
                    "pairwise_time_transform": "signed_log1p",
                    "pairwise_distance_transform": "log1p",
                    "exclude_self_from_spatial_knn": True,
                    "deduplicate_neighbors": True,
                    "pooling": "attention_mean_concat",
                },
                "training": {
                    "batch_size": 1,
                    "epochs": 1,
                },
                "loader": {
                    "train_max_workers": 0,
                    "eval_max_workers": 0,
                },
            }
        )

        metrics = train_model(resolved_config=resolved_config)

        self.assertIn("accuracy", metrics)
        self.assertTrue((self.model_dir / DEFAULT_METRICS_PATH).exists())
        self.assertTrue((self.model_dir / "val_predictions.pt").exists())
        self.assertTrue((self.model_dir / "test_predictions.pt").exists())
        metrics_payload = json.loads(
            (self.model_dir / DEFAULT_METRICS_PATH).read_text(encoding="ascii")
        )
        self.assertIn("val", metrics_payload)
        self.assertIn("test", metrics_payload)
        self.assertIn("macro_f1", metrics_payload["test"])
        self.assertIn("confusion_matrix", metrics_payload["test"])

    def test_resolve_quality_supervision_rejects_mixed_split_capabilities(self):
        class DatasetStub:
            def __init__(self, has_rec_labels):
                self.has_rec_labels = has_rec_labels

        with self.assertRaisesRegex(
            ValueError,
            "Reconstructed labels must be present",
        ):
            resolve_quality_supervision(
                DatasetStub(True),
                DatasetStub(False),
                DatasetStub(True),
            )

    def test_resolve_quality_supervision_disables_quality_for_multiclass_tasks(self):
        class DatasetStub:
            def __init__(self, has_rec_labels):
                self.has_rec_labels = has_rec_labels

        self.assertFalse(
            resolve_quality_supervision(
                DatasetStub(False),
                DatasetStub(False),
                DatasetStub(False),
                metadata={"task": "muon_count", "target_kind": "multiclass"},
            )
        )


if __name__ == "__main__":
    unittest.main()
