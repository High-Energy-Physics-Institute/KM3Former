import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import KM3Loader
from eval import move_batch_to_device
from train import build_data_loader


class KM3LoaderTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_tensor(self, name, tensor):
        path = self.data_dir / f"{name}.pt"
        torch.save(tensor, path)
        return path

    def _write_dataset_files(self, include_rec_labels=True, label_length=2):
        hits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        raw_hits = hits + 100.0
        padding_mask = torch.tensor(
            [[False, False, True], [False, True, True]],
            dtype=torch.bool,
        )
        labels = torch.arange(label_length * 3, dtype=torch.float32).reshape(
            label_length,
            3,
        )

        paths = {
            "hits_file": self._write_tensor("hits", hits),
            "raw_hits_file": self._write_tensor("raw_hits", raw_hits),
            "padding_mask_file": self._write_tensor("padding_mask", padding_mask),
            "label_file": self._write_tensor("labels", labels),
        }
        if include_rec_labels:
            rec_labels = labels + 10.0
            paths["rec_label_file"] = self._write_tensor("rec_labels", rec_labels)

        return paths

    def test_lazy_and_eager_loading_return_identical_samples(self):
        dataset_kwargs = self._write_dataset_files()

        eager_dataset = KM3Loader(load_strategy="eager", **dataset_kwargs)
        lazy_dataset = KM3Loader(load_strategy="lazy", **dataset_kwargs)

        self.assertEqual(len(eager_dataset), len(lazy_dataset))

        eager_sample = eager_dataset[1]
        lazy_sample = lazy_dataset[1]
        self.assertEqual(len(eager_sample), len(lazy_sample))

        for eager_tensor, lazy_tensor in zip(eager_sample, lazy_sample):
            self.assertTrue(torch.equal(eager_tensor, lazy_tensor))

    def test_loader_without_rec_labels_returns_four_item_samples(self):
        dataset_kwargs = self._write_dataset_files(include_rec_labels=False)
        dataset = KM3Loader(load_strategy="lazy", **dataset_kwargs)

        sample = dataset[0]
        self.assertEqual(len(sample), 4)
        self.assertEqual(sample[0].shape, torch.Size([3, 4]))
        self.assertEqual(sample[2].dtype, torch.bool)

    def test_mismatched_tensor_lengths_raise_value_error(self):
        dataset_kwargs = self._write_dataset_files(label_length=1)

        with self.assertRaisesRegex(
            ValueError,
            "backing tensors must share the same first dimension",
        ):
            KM3Loader(load_strategy="eager", **dataset_kwargs)

    def test_dataloader_batch_still_unpacks_with_expected_shapes(self):
        dataset_kwargs = self._write_dataset_files()
        dataset = KM3Loader(load_strategy="lazy", **dataset_kwargs)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(loader))
        self.assertEqual(len(batch), 5)
        self.assertEqual(batch[0].shape, torch.Size([2, 3, 4]))
        self.assertEqual(batch[1].shape, torch.Size([2, 3, 4]))
        self.assertEqual(batch[2].shape, torch.Size([2, 3]))
        self.assertEqual(batch[3].shape, torch.Size([2, 3]))
        self.assertEqual(batch[4].shape, torch.Size([2, 3]))
        self.assertEqual(batch[2].dtype, torch.bool)

        moved_batch = move_batch_to_device(*batch, device=torch.device("cpu"))
        for original, moved in zip(batch, moved_batch):
            self.assertTrue(torch.equal(original, moved))

    def test_build_data_loader_uses_mac_safe_defaults(self):
        dataset_kwargs = self._write_dataset_files()
        dataset = KM3Loader(load_strategy="lazy", **dataset_kwargs)

        with mock.patch("train.sys.platform", "darwin"), mock.patch(
            "train.torch.cuda.is_available",
            return_value=False,
        ):
            loader = build_data_loader(
                dataset,
                batch_size=2,
                shuffle=True,
                max_workers=4,
            )

        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.pin_memory)
        self.assertFalse(loader.persistent_workers)

    def test_build_data_loader_uses_linux_worker_defaults(self):
        dataset_kwargs = self._write_dataset_files()
        dataset = KM3Loader(load_strategy="lazy", **dataset_kwargs)

        with mock.patch("train.sys.platform", "linux"), mock.patch(
            "train.os.cpu_count",
            return_value=8,
        ), mock.patch(
            "train.torch.cuda.is_available",
            return_value=True,
        ):
            loader = build_data_loader(
                dataset,
                batch_size=2,
                shuffle=False,
                max_workers=4,
            )

        self.assertEqual(loader.num_workers, 4)
        self.assertTrue(loader.pin_memory)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.prefetch_factor, 2)


if __name__ == "__main__":
    unittest.main()
