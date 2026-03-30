import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join saved muon-count prediction tensors with raw HDF5 Monte Carlo "
            "energy and direction metadata for EDA."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Prediction files or run directories. Directories default to "
            "`test_predictions.pt`."
        ),
    )
    parser.add_argument(
        "--h5-path",
        default="data/muon_data_7224_7247.h5",
        help="Path to the raw HDF5 file used for preprocessing.",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/metadata.json",
        help="Path to preprocessing metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory for outputs. By default, files are written next "
            "to each prediction payload."
        ),
    )
    return parser.parse_args()


def load_metadata(path):
    with open(path, "r", encoding="ascii") as handle:
        return json.load(handle)


def build_split_indices(total_events, metadata):
    split_seed = int(metadata["split_seed"])
    split_sizes = metadata["splits"]
    train_size = int(split_sizes["train"])
    val_size = int(split_sizes["val"])
    test_size = int(split_sizes["test"])

    if train_size + val_size + test_size != total_events:
        raise ValueError(
            "Metadata split sizes do not match raw HDF5 event count: "
            f"{train_size + val_size + test_size} != {total_events}"
        )

    generator = torch.Generator().manual_seed(split_seed)
    shuffled = torch.randperm(total_events, generator=generator).tolist()

    train_end = train_size
    val_end = train_end + val_size
    test_end = val_end + test_size

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:test_end],
    }


def resolve_prediction_path(raw_input):
    input_path = Path(raw_input)
    if input_path.is_dir():
        candidate = input_path / "test_predictions.pt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Could not find test_predictions.pt inside directory: {input_path}"
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction payload does not exist: {input_path}")
    return input_path


def infer_split_name(prediction_path, payload):
    split_name = payload.get("split")
    if isinstance(split_name, str):
        return split_name

    stem = prediction_path.stem.lower()
    for candidate in ("train", "val", "test"):
        if candidate in stem:
            return candidate
    return "test"


def load_hdf5_arrays(path):
    with h5py.File(path, "r") as handle:
        return {
            "mc_muons": handle["mc_muons"][:],
            "rec_muons": handle["rec_muons"][:],
            "energies": handle["energies"][:],
        }


def raw_label_to_class_index(raw_labels, metadata):
    class_values = metadata.get("class_values")
    class_to_index = metadata.get("class_to_index")

    if not class_values or not class_to_index:
        return raw_labels.astype(np.int64)

    mapping = {float(key): int(value) for key, value in class_to_index.items()}
    return np.array([mapping[float(label)] for label in raw_labels], dtype=np.int64)


def off_vertical_degrees(dir_z_values):
    clipped = np.clip(-dir_z_values, -1.0, 1.0)
    return np.degrees(np.arccos(clipped))


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def build_event_rows(
    *,
    split_name,
    event_indices,
    payload,
    metadata,
    mc_muons,
    rec_muons,
    energies,
):
    split_mc = mc_muons[event_indices]
    split_rec = rec_muons[event_indices]
    split_energies = energies[event_indices]

    raw_labels = split_mc[:, 0]
    mapped_labels = raw_label_to_class_index(raw_labels, metadata)

    targets = payload.get("targets")
    if targets is None:
        raise ValueError("Prediction payload does not include targets; cannot run EDA.")
    targets = targets.cpu().numpy().astype(np.int64)

    predicted = payload["predicted_labels"].cpu().numpy().astype(np.int64)
    confidence = payload.get("confidence")
    if confidence is None:
        confidence_values = np.full(targets.shape, np.nan, dtype=np.float32)
    else:
        confidence_values = confidence.cpu().numpy().astype(np.float32)

    probabilities = payload.get("probabilities")
    probability_values = (
        probabilities.cpu().numpy().astype(np.float32) if probabilities is not None else None
    )

    if not np.array_equal(mapped_labels, targets):
        mismatch_count = int(np.count_nonzero(mapped_labels != targets))
        raise ValueError(
            "Reconstructed split labels do not match payload targets. "
            f"Mismatched rows: {mismatch_count}"
        )

    primary_dir = split_mc[:, 1:4]
    secondary_dir = split_mc[:, 4:7]
    secondary_direction_present = ~np.isclose(secondary_dir, 0.0).all(axis=1)

    rows = []
    for split_position, source_event_index in enumerate(event_indices):
        row = {
            "source_event_index": int(source_event_index),
            "split": split_name,
            "split_position": int(split_position),
            "raw_label_value": float(raw_labels[split_position]),
            "target_class": int(targets[split_position]),
            "predicted_class": int(predicted[split_position]),
            "correct": bool(targets[split_position] == predicted[split_position]),
            "confidence": float(confidence_values[split_position]),
            "primary_dir_x": float(primary_dir[split_position, 0]),
            "primary_dir_y": float(primary_dir[split_position, 1]),
            "primary_dir_z": float(primary_dir[split_position, 2]),
            "secondary_dir_x": float(secondary_dir[split_position, 0]),
            "secondary_dir_y": float(secondary_dir[split_position, 1]),
            "secondary_dir_z": float(secondary_dir[split_position, 2]),
            "secondary_direction_present": bool(
                secondary_direction_present[split_position]
            ),
            "primary_off_vertical_deg": float(
                off_vertical_degrees(primary_dir[split_position, 2:3])[0]
            ),
            "secondary_off_vertical_deg": float(
                off_vertical_degrees(secondary_dir[split_position, 2:3])[0]
            )
            if secondary_direction_present[split_position]
            else math.nan,
            "primary_energy": float(split_energies[split_position, 0]),
            "secondary_energy": float(split_energies[split_position, 1]),
            "total_energy_first_two": float(split_energies[split_position].sum()),
            "rec_muons_0": float(split_rec[split_position, 0]),
            "rec_muons_1": float(split_rec[split_position, 1]),
            "rec_muons_2": float(split_rec[split_position, 2]),
            "rec_muons_3": float(split_rec[split_position, 3]),
            "rec_muons_4": float(split_rec[split_position, 4]),
            "rec_muons_5": float(split_rec[split_position, 5]),
        }
        if probability_values is not None:
            for class_index in range(probability_values.shape[1]):
                row[f"prob_class_{class_index}"] = float(
                    probability_values[split_position, class_index]
                )
        rows.append(row)
    return rows


def summarize_rows(rows):
    total = len(rows)
    correct_mask = np.array([row["correct"] for row in rows], dtype=bool)
    targets = np.array([row["target_class"] for row in rows], dtype=np.int64)
    predicted = np.array([row["predicted_class"] for row in rows], dtype=np.int64)

    primary_energy = np.array([row["primary_energy"] for row in rows], dtype=np.float64)
    secondary_energy = np.array(
        [row["secondary_energy"] for row in rows], dtype=np.float64
    )
    total_energy = np.array(
        [row["total_energy_first_two"] for row in rows], dtype=np.float64
    )
    primary_off_vertical = np.array(
        [row["primary_off_vertical_deg"] for row in rows], dtype=np.float64
    )

    summary = {
        "num_examples": total,
        "num_correct": int(correct_mask.sum()),
        "num_failed": int((~correct_mask).sum()),
        "accuracy": float(correct_mask.mean()),
        "overall": {
            "successful": {
                "primary_energy": describe(primary_energy[correct_mask]),
                "secondary_energy": describe(secondary_energy[correct_mask]),
                "total_energy_first_two": describe(total_energy[correct_mask]),
                "primary_off_vertical_deg": describe(primary_off_vertical[correct_mask]),
            },
            "failed": {
                "primary_energy": describe(primary_energy[~correct_mask]),
                "secondary_energy": describe(secondary_energy[~correct_mask]),
                "total_energy_first_two": describe(total_energy[~correct_mask]),
                "primary_off_vertical_deg": describe(primary_off_vertical[~correct_mask]),
            },
        },
        "per_true_class": {},
        "confusion_pairs": {},
    }

    class_values = sorted(set(targets.tolist()))
    for class_index in class_values:
        class_mask = targets == class_index
        class_correct = correct_mask & class_mask
        class_failed = (~correct_mask) & class_mask
        summary["per_true_class"][str(class_index)] = {
            "support": int(class_mask.sum()),
            "accuracy": float(correct_mask[class_mask].mean()),
            "successful": {
                "primary_energy": describe(primary_energy[class_correct]),
                "secondary_energy": describe(secondary_energy[class_correct]),
                "total_energy_first_two": describe(total_energy[class_correct]),
                "primary_off_vertical_deg": describe(primary_off_vertical[class_correct]),
            },
            "failed": {
                "primary_energy": describe(primary_energy[class_failed]),
                "secondary_energy": describe(secondary_energy[class_failed]),
                "total_energy_first_two": describe(total_energy[class_failed]),
                "primary_off_vertical_deg": describe(primary_off_vertical[class_failed]),
            },
        }

    for true_class in class_values:
        for predicted_class in class_values:
            pair_mask = (targets == true_class) & (predicted == predicted_class)
            pair_key = f"{true_class}->{predicted_class}"
            summary["confusion_pairs"][pair_key] = {
                "count": int(pair_mask.sum()),
                "primary_energy": describe(primary_energy[pair_mask]),
                "secondary_energy": describe(secondary_energy[pair_mask]),
                "total_energy_first_two": describe(total_energy[pair_mask]),
                "primary_off_vertical_deg": describe(primary_off_vertical[pair_mask]),
            }
    return summary


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(run_name, split_name, summary):
    print(f"\n=== {run_name} [{split_name}] ===")
    print(
        "accuracy={accuracy:.4f} correct={num_correct} failed={num_failed}".format(
            **summary
        )
    )
    for bucket_name in ("successful", "failed"):
        bucket = summary["overall"][bucket_name]
        print(
            "  {bucket}: e1_mean={e1:.1f} e2_mean={e2:.1f} etot_mean={etot:.1f} "
            "off_vertical_mean={theta:.2f}".format(
                bucket=bucket_name,
                e1=bucket["primary_energy"].get("mean", math.nan),
                e2=bucket["secondary_energy"].get("mean", math.nan),
                etot=bucket["total_energy_first_two"].get("mean", math.nan),
                theta=bucket["primary_off_vertical_deg"].get("mean", math.nan),
            )
        )

    print("  per_true_class:")
    for class_key, class_summary in sorted(summary["per_true_class"].items()):
        print(
            "    class {class_key}: acc={acc:.4f} "
            "e1_success={e1s:.1f} e1_fail={e1f:.1f} "
            "etot_success={ets:.1f} etot_fail={etf:.1f}".format(
                class_key=class_key,
                acc=class_summary["accuracy"],
                e1s=class_summary["successful"]["primary_energy"].get("mean", math.nan),
                e1f=class_summary["failed"]["primary_energy"].get("mean", math.nan),
                ets=class_summary["successful"]["total_energy_first_two"].get(
                    "mean", math.nan
                ),
                etf=class_summary["failed"]["total_energy_first_two"].get(
                    "mean", math.nan
                ),
            )
        )


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata_path)
    hdf5_arrays = load_hdf5_arrays(args.h5_path)
    split_indices = build_split_indices(len(hdf5_arrays["mc_muons"]), metadata)

    for raw_input in args.inputs:
        prediction_path = resolve_prediction_path(raw_input)
        payload = torch.load(prediction_path, map_location="cpu")
        split_name = infer_split_name(prediction_path, payload)
        event_indices = split_indices[split_name]

        rows = build_event_rows(
            split_name=split_name,
            event_indices=event_indices,
            payload=payload,
            metadata=metadata,
            mc_muons=hdf5_arrays["mc_muons"],
            rec_muons=hdf5_arrays["rec_muons"],
            energies=hdf5_arrays["energies"],
        )
        summary = summarize_rows(rows)

        base_dir = Path(args.output_dir) if args.output_dir else prediction_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        base_name = prediction_path.stem
        csv_path = base_dir / f"{base_name}_joined.csv"
        summary_path = base_dir / f"{base_name}_eda_summary.json"

        write_csv(csv_path, rows)
        with open(summary_path, "w", encoding="ascii") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        print_summary(prediction_path.parent.name, split_name, summary)
        print(f"  wrote {csv_path}")
        print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
