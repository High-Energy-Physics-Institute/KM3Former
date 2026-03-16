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
from train import build_model, main as train_cli


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

    def _write_runtime_files(self):
        metadata = {
            "input_dim": 10,
            "max_hits": 4,
            "feature_names": [f"feature_{index}" for index in range(10)],
        }
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
        self.assertEqual(predictions.shape, torch.Size([2, 3]))
        self.assertEqual(quality.shape, torch.Size([2]))

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


if __name__ == "__main__":
    unittest.main()
