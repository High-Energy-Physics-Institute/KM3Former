import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


CLASS_LABELS = ["1 muon", "2 muons", "3 muons"]
CLASS_INDICES = [0, 1, 2]
RUN_COLORS = {
    "exp_008_capacity_push": "#4477AA",
    "exp_009_capacity_push_retuned": "#CC6677",
}
CLASS_COLORS = {
    0: "#4477AA",
    1: "#228833",
    2: "#CC6677",
}
REPRESENTATIVE_CATEGORIES = {
    "downward": {
        "label": "True 2-muon predicted as 1 muon",
        "predicted_class": 0,
    },
    "middle": {
        "label": "True 2-muon predicted as 2 muons",
        "predicted_class": 1,
    },
    "upward": {
        "label": "True 2-muon predicted as 3 muons",
        "predicted_class": 2,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build detector-overlap diagnostics for the muon-count task, "
            "including detector summary tables, baseline comparisons, model "
            "ambiguity plots, and representative event visualizations."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run directories containing `test_predictions.pt`.",
    )
    parser.add_argument(
        "--h5-path",
        default="data/muon_data_7224_7247.h5",
        help="Path to the raw HDF5 file.",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/metadata.json",
        help="Path to preprocessing metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/detector_overlap_analysis",
        help="Directory where analysis artifacts are written.",
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


def off_vertical_degrees(dir_z_values):
    clipped = np.clip(-dir_z_values, -1.0, 1.0)
    return np.degrees(np.arccos(clipped))


def configure_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def _safe_nanmean(values, counts):
    return np.divide(
        values.sum(axis=1),
        np.maximum(counts, 1),
        out=np.zeros(values.shape[0], dtype=np.float64),
        where=np.maximum(counts, 1) > 0,
    )


def _safe_nanstd(values, mean_values, counts):
    centered = values - mean_values[:, None]
    squared = centered**2
    return np.sqrt(
        np.divide(
            squared.sum(axis=1),
            np.maximum(counts, 1),
            out=np.zeros(values.shape[0], dtype=np.float64),
            where=np.maximum(counts, 1) > 0,
        )
    )


def _masked_min(values, mask):
    masked = np.where(mask, values, np.inf)
    result = masked.min(axis=1)
    result[~mask.any(axis=1)] = 0.0
    return result


def _masked_max(values, mask):
    masked = np.where(mask, values, -np.inf)
    result = masked.max(axis=1)
    result[~mask.any(axis=1)] = 0.0
    return result


def build_detector_summary_dataframe(h5_path, metadata):
    with h5py.File(h5_path, "r") as handle:
        hits = handle["hits"][:]
        mc_muons = handle["mc_muons"][:]
        rec_muons = handle["rec_muons"][:]
        energies = handle["energies"][:]

    split_indices = build_split_indices(len(hits), metadata)
    split_by_index = np.empty(len(hits), dtype=object)
    for split_name, indices in split_indices.items():
        split_by_index[np.asarray(indices, dtype=np.int64)] = split_name

    tot = np.clip(hits[:, :, 7], 0.0, None)
    active_mask = tot > 0.0
    active_hits = active_mask.sum(axis=1).astype(np.int64)
    active_hit_fraction = active_hits / hits.shape[1]
    total_tot = tot.sum(axis=1)
    mean_tot_active = np.divide(
        total_tot,
        np.maximum(active_hits, 1),
        out=np.zeros_like(total_tot, dtype=np.float64),
        where=np.maximum(active_hits, 1) > 0,
    )
    max_tot_active = tot.max(axis=1)

    time = hits[:, :, 0]
    x = hits[:, :, 1]
    y = hits[:, :, 2]
    z = hits[:, :, 3]
    hit_dir_z = hits[:, :, 6]
    radius = np.sqrt(x**2 + y**2)

    time_span = _masked_max(time, active_mask) - _masked_min(time, active_mask)
    z_span = _masked_max(z, active_mask) - _masked_min(z, active_mask)
    z_mean = _safe_nanmean(np.where(active_mask, z, 0.0), active_hits)
    radius_mean = _safe_nanmean(np.where(active_mask, radius, 0.0), active_hits)
    radius_std = _safe_nanstd(
        np.where(active_mask, radius, 0.0),
        radius_mean,
        active_hits,
    )
    hit_dir_z_mean = _safe_nanmean(
        np.where(active_mask, hit_dir_z, 0.0),
        active_hits,
    )
    hit_dir_z_std = _safe_nanstd(
        np.where(active_mask, hit_dir_z, 0.0),
        hit_dir_z_mean,
        active_hits,
    )

    target_class = mc_muons[:, 0].astype(np.int64)
    primary_dir = mc_muons[:, 1:4]
    secondary_dir = mc_muons[:, 4:7]
    secondary_direction_present = ~np.isclose(secondary_dir, 0.0).all(axis=1)

    dataframe = pd.DataFrame(
        {
            "source_event_index": np.arange(len(hits), dtype=np.int64),
            "split": split_by_index,
            "target_class": target_class,
            "physical_muon_count": target_class + 1,
            "active_hits": active_hits,
            "active_hit_fraction": active_hit_fraction,
            "total_tot": total_tot,
            "mean_tot_active": mean_tot_active,
            "max_tot_active": max_tot_active,
            "time_span": time_span,
            "z_span": z_span,
            "z_mean": z_mean,
            "radius_mean": radius_mean,
            "radius_std": radius_std,
            "hit_dir_z_mean": hit_dir_z_mean,
            "hit_dir_z_std": hit_dir_z_std,
            "primary_energy_mc": energies[:, 0],
            "secondary_energy_mc": energies[:, 1],
            "total_energy_mc": energies.sum(axis=1),
            "primary_dir_x_mc": primary_dir[:, 0],
            "primary_dir_y_mc": primary_dir[:, 1],
            "primary_dir_z_mc": primary_dir[:, 2],
            "secondary_dir_x_mc": secondary_dir[:, 0],
            "secondary_dir_y_mc": secondary_dir[:, 1],
            "secondary_dir_z_mc": secondary_dir[:, 2],
            "secondary_direction_present_mc": secondary_direction_present,
            "primary_off_vertical_deg_mc": off_vertical_degrees(primary_dir[:, 2]),
            "secondary_off_vertical_deg_mc": np.where(
                secondary_direction_present,
                off_vertical_degrees(secondary_dir[:, 2]),
                np.nan,
            ),
            "rec_dir_x_0": rec_muons[:, 0],
            "rec_dir_y_0": rec_muons[:, 1],
            "rec_dir_z_0": rec_muons[:, 2],
            "rec_dir_x_1": rec_muons[:, 3],
            "rec_dir_y_1": rec_muons[:, 4],
            "rec_dir_z_1": rec_muons[:, 5],
        }
    )
    return dataframe, hits, split_indices


def validate_run_payload(payload):
    computed = confusion_matrix(
        payload["targets"].cpu().numpy().astype(np.int64),
        payload["predicted_labels"].cpu().numpy().astype(np.int64),
        labels=CLASS_INDICES,
    ).tolist()
    stored = payload.get("metrics", {}).get("confusion_matrix")
    if stored is not None and computed != stored:
        raise ValueError(
            "Stored confusion matrix does not match payload predictions: "
            f"{computed} != {stored}"
        )


def join_run_predictions(run_dir, detector_summary, split_indices):
    run_path = Path(run_dir)
    payload_path = run_path / "test_predictions.pt"
    payload = torch.load(payload_path, map_location="cpu")
    validate_run_payload(payload)

    event_indices = np.asarray(split_indices["test"], dtype=np.int64)
    detector_test = (
        detector_summary.loc[
            detector_summary["source_event_index"].isin(event_indices)
        ]
        .set_index("source_event_index")
        .loc[event_indices]
        .reset_index()
    )

    targets = payload["targets"].cpu().numpy().astype(np.int64)
    predicted = payload["predicted_labels"].cpu().numpy().astype(np.int64)
    probabilities = payload["probabilities"].cpu().numpy().astype(np.float64)
    confidence = payload["confidence"].cpu().numpy().astype(np.float64)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    if not np.array_equal(detector_test["target_class"].to_numpy(), targets):
        raise ValueError(
            f"Joined detector summary targets do not match payload targets for {run_path}."
        )

    joined = detector_test.copy()
    joined["predicted_class"] = predicted
    joined["correct"] = predicted == targets
    joined["confidence"] = confidence
    joined["entropy"] = entropy
    joined["margin"] = margin
    joined["run_name"] = run_path.name
    for class_index in CLASS_INDICES:
        joined[f"prob_class_{class_index}"] = probabilities[:, class_index]
    return joined, payload


def compute_pairwise_auc(values, labels, lower_class, upper_class):
    mask = (labels == lower_class) | (labels == upper_class)
    subset_values = np.asarray(values[mask], dtype=np.float64)
    subset_labels = labels[mask]
    if subset_values.size == 0 or np.allclose(subset_values, subset_values[0]):
        return 0.5
    binary_targets = (subset_labels == upper_class).astype(np.int64)
    return float(roc_auc_score(binary_targets, subset_values))


def compute_cohens_d(a_values, b_values):
    a_values = np.asarray(a_values, dtype=np.float64)
    b_values = np.asarray(b_values, dtype=np.float64)
    if len(a_values) < 2 or len(b_values) < 2:
        return 0.0
    pooled_variance = (
        ((len(a_values) - 1) * a_values.var(ddof=1))
        + ((len(b_values) - 1) * b_values.var(ddof=1))
    ) / (len(a_values) + len(b_values) - 2)
    if pooled_variance <= 0.0:
        return 0.0
    return float((b_values.mean() - a_values.mean()) / math.sqrt(pooled_variance))


def compute_interval_overlap(a_values, b_values, lower_percentile=10, upper_percentile=90):
    a_low, a_high = np.percentile(a_values, [lower_percentile, upper_percentile])
    b_low, b_high = np.percentile(b_values, [lower_percentile, upper_percentile])
    overlap = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    union = max(a_high, b_high) - min(a_low, b_low)
    if union <= 0.0:
        return 0.0
    return float(overlap / union)


def compute_detector_overlap_summary(detector_summary, detector_feature_columns):
    test_df = detector_summary.loc[detector_summary["split"] == "test"].copy()
    labels = test_df["target_class"].to_numpy().astype(np.int64)
    rows = []
    for feature_name in detector_feature_columns:
        values = test_df[feature_name].to_numpy(dtype=np.float64)
        class_values = {
            class_index: values[labels == class_index] for class_index in CLASS_INDICES
        }
        medians = {
            class_index: float(np.median(class_values[class_index]))
            for class_index in CLASS_INDICES
        }
        ordered_increasing = medians[0] < medians[1] < medians[2]
        ordered_decreasing = medians[0] > medians[1] > medians[2]
        sandwich_ordered = ordered_increasing or ordered_decreasing

        auc_01 = compute_pairwise_auc(values, labels, 0, 1)
        auc_12 = compute_pairwise_auc(values, labels, 1, 2)
        auc_02 = compute_pairwise_auc(values, labels, 0, 2)

        row = {
            "feature": feature_name,
            "median_class_0": medians[0],
            "median_class_1": medians[1],
            "median_class_2": medians[2],
            "ordered_sandwich": sandwich_ordered,
            "auc_0_vs_1": auc_01,
            "auc_1_vs_2": auc_12,
            "auc_0_vs_2": auc_02,
            "auc_strength_0_vs_1": max(auc_01, 1.0 - auc_01),
            "auc_strength_1_vs_2": max(auc_12, 1.0 - auc_12),
            "auc_strength_0_vs_2": max(auc_02, 1.0 - auc_02),
            "cohens_d_0_vs_1": compute_cohens_d(class_values[0], class_values[1]),
            "cohens_d_1_vs_2": compute_cohens_d(class_values[1], class_values[2]),
            "cohens_d_0_vs_2": compute_cohens_d(class_values[0], class_values[2]),
            "interval_overlap_0_vs_1": compute_interval_overlap(
                class_values[0], class_values[1]
            ),
            "interval_overlap_1_vs_2": compute_interval_overlap(
                class_values[1], class_values[2]
            ),
            "interval_overlap_0_vs_2": compute_interval_overlap(
                class_values[0], class_values[2]
            ),
        }
        row["adjacent_auc_strength_mean"] = (
            row["auc_strength_0_vs_1"] + row["auc_strength_1_vs_2"]
        ) / 2.0
        rows.append(row)

    overlap_df = pd.DataFrame(rows).sort_values(
        by=["ordered_sandwich", "adjacent_auc_strength_mean", "auc_strength_0_vs_2"],
        ascending=[False, False, False],
    )
    return overlap_df.reset_index(drop=True)


def train_detector_baselines(detector_summary, detector_feature_columns):
    split_masks = {
        split_name: detector_summary["split"] == split_name
        for split_name in ("train", "val", "test")
    }
    x_train = detector_summary.loc[split_masks["train"], detector_feature_columns].to_numpy(
        dtype=np.float64
    )
    y_train = detector_summary.loc[split_masks["train"], "target_class"].to_numpy(
        dtype=np.int64
    )

    x_by_split = {
        split_name: detector_summary.loc[split_masks[split_name], detector_feature_columns].to_numpy(
            dtype=np.float64
        )
        for split_name in ("val", "test")
    }
    y_by_split = {
        split_name: detector_summary.loc[split_masks[split_name], "target_class"].to_numpy(
            dtype=np.int64
        )
        for split_name in ("val", "test")
    }

    detector_models = {
        "logistic_detector": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
        ),
        "random_forest_detector": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
    }

    mc_augmented_columns = detector_feature_columns + [
        "primary_energy_mc",
        "secondary_energy_mc",
        "total_energy_mc",
        "primary_off_vertical_deg_mc",
    ]
    x_train_augmented = detector_summary.loc[
        split_masks["train"], mc_augmented_columns
    ].to_numpy(dtype=np.float64)
    x_augmented_by_split = {
        split_name: detector_summary.loc[
            split_masks[split_name], mc_augmented_columns
        ].to_numpy(dtype=np.float64)
        for split_name in ("val", "test")
    }
    diagnostic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
    )

    results = {}
    for model_name, model in detector_models.items():
        model.fit(x_train, y_train)
        results[model_name] = {
            "feature_type": "detector_only",
            "feature_columns": detector_feature_columns,
            "split_metrics": {},
        }
        for split_name in ("val", "test"):
            predictions = model.predict(x_by_split[split_name])
            probabilities = model.predict_proba(x_by_split[split_name])
            results[model_name]["split_metrics"][split_name] = summarize_classifier_outputs(
                y_by_split[split_name],
                predictions,
                probabilities,
            )

    diagnostic_model.fit(x_train_augmented, y_train)
    results["logistic_detector_plus_mc_diagnostic"] = {
        "feature_type": "detector_plus_mc_truth_diagnostic_only",
        "feature_columns": mc_augmented_columns,
        "split_metrics": {},
    }
    for split_name in ("val", "test"):
        predictions = diagnostic_model.predict(x_augmented_by_split[split_name])
        probabilities = diagnostic_model.predict_proba(x_augmented_by_split[split_name])
        results["logistic_detector_plus_mc_diagnostic"]["split_metrics"][
            split_name
        ] = summarize_classifier_outputs(
            y_by_split[split_name],
            predictions,
            probabilities,
        )

    test_event_indices = detector_summary.loc[
        split_masks["test"], "source_event_index"
    ].to_numpy(dtype=np.int64)
    baseline_predictions = pd.DataFrame(
        {
            "source_event_index": test_event_indices,
            "target_class": y_by_split["test"],
        }
    )
    for model_name, model in detector_models.items():
        predictions = model.predict(x_by_split["test"])
        probabilities = model.predict_proba(x_by_split["test"])
        baseline_predictions[f"{model_name}_predicted_class"] = predictions
        baseline_predictions[f"{model_name}_confidence"] = probabilities.max(axis=1)

    return results, baseline_predictions


def summarize_classifier_outputs(targets, predictions, probabilities):
    confusion = confusion_matrix(targets, predictions, labels=CLASS_INDICES)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        np.maximum(row_sums, 1),
        out=np.zeros_like(confusion, dtype=np.float64),
        where=np.maximum(row_sums, 1) > 0,
    )
    per_class_recall = {}
    for class_index in CLASS_INDICES:
        support = int((targets == class_index).sum())
        recall = float((predictions[targets == class_index] == class_index).mean())
        per_class_recall[str(class_index)] = {
            "support": support,
            "recall": recall,
        }
    return {
        "accuracy": float((predictions == targets).mean()),
        "confusion_matrix": confusion.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "per_class_recall": per_class_recall,
    }


def build_reliability_curve(y_true_binary, y_probabilities, num_bins=10):
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(y_probabilities, bins[1:-1], right=True)
    curve_rows = []
    for bin_index in range(num_bins):
        mask = bin_ids == bin_index
        if not mask.any():
            curve_rows.append(
                {
                    "bin_index": bin_index,
                    "bin_left": float(bins[bin_index]),
                    "bin_right": float(bins[bin_index + 1]),
                    "bin_center": float((bins[bin_index] + bins[bin_index + 1]) / 2.0),
                    "count": 0,
                    "mean_predicted_probability": float(
                        (bins[bin_index] + bins[bin_index + 1]) / 2.0
                    ),
                    "fraction_positive": math.nan,
                }
            )
            continue
        curve_rows.append(
            {
                "bin_index": bin_index,
                "bin_left": float(bins[bin_index]),
                "bin_right": float(bins[bin_index + 1]),
                "bin_center": float((bins[bin_index] + bins[bin_index + 1]) / 2.0),
                "count": int(mask.sum()),
                "mean_predicted_probability": float(y_probabilities[mask].mean()),
                "fraction_positive": float(y_true_binary[mask].mean()),
            }
        )
    return pd.DataFrame(curve_rows)


def summarize_model_ambiguity(run_joined_dataframe):
    grouped_summary = {}
    for true_class in CLASS_INDICES:
        mask = run_joined_dataframe["target_class"] == true_class
        subset = run_joined_dataframe.loc[mask]
        grouped_summary[str(true_class)] = {
            "support": int(len(subset)),
            "accuracy": float(subset["correct"].mean()),
            "median_confidence": float(subset["confidence"].median()),
            "median_entropy": float(subset["entropy"].median()),
            "median_margin": float(subset["margin"].median()),
        }

    class_one = run_joined_dataframe.loc[run_joined_dataframe["target_class"] == 1]
    class_one_by_prediction = {}
    for predicted_class in CLASS_INDICES:
        subset = class_one.loc[class_one["predicted_class"] == predicted_class]
        class_one_by_prediction[str(predicted_class)] = {
            "count": int(len(subset)),
            "median_confidence": float(subset["confidence"].median()),
            "median_entropy": float(subset["entropy"].median()),
            "median_margin": float(subset["margin"].median()),
        }

    class_one_binary = (run_joined_dataframe["target_class"] == 1).to_numpy(dtype=np.int64)
    class_one_probability = run_joined_dataframe["prob_class_1"].to_numpy(dtype=np.float64)
    reliability = build_reliability_curve(class_one_binary, class_one_probability)

    return {
        "per_true_class": grouped_summary,
        "class_1_by_prediction": class_one_by_prediction,
        "class_1_reliability": reliability.to_dict(orient="records"),
    }


def render_confusion_matrix(axis, matrix, title, annotate_percentages=True):
    matrix = np.asarray(matrix, dtype=np.float64)
    image = axis.imshow(
        matrix,
        cmap=LinearSegmentedColormap.from_list(
            "km3former_heat", ["#F7FBFF", "#6BAED6", "#08306B"]
        ),
        vmin=0.0,
        vmax=1.0 if annotate_percentages else matrix.max(),
        aspect="equal",
    )
    axis.set_xticks(range(3), CLASS_LABELS, rotation=20)
    axis.set_yticks(range(3), CLASS_LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row_index in range(3):
        for col_index in range(3):
            value = matrix[row_index, col_index]
            label = f"{value:.2f}" if annotate_percentages else str(int(round(value)))
            axis.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=9,
            )
    return image


def render_detector_feature_cdfs(detector_summary, overlap_df, output_path):
    test_df = detector_summary.loc[detector_summary["split"] == "test"].copy()
    top_features = overlap_df.loc[overlap_df["ordered_sandwich"]].head(6)["feature"].tolist()
    if len(top_features) < 6:
        additional = overlap_df.loc[
            ~overlap_df["feature"].isin(top_features)
        ].head(6 - len(top_features))["feature"].tolist()
        top_features.extend(additional)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    axes = axes.ravel()
    for axis, feature_name in zip(axes, top_features):
        feature_row = overlap_df.loc[overlap_df["feature"] == feature_name].iloc[0]
        for class_index in CLASS_INDICES:
            values = np.sort(
                test_df.loc[test_df["target_class"] == class_index, feature_name].to_numpy(
                    dtype=np.float64
                )
            )
            ecdf = np.arange(1, len(values) + 1) / len(values)
            axis.plot(
                values,
                ecdf,
                color=CLASS_COLORS[class_index],
                linewidth=1.8,
                label=CLASS_LABELS[class_index],
            )
        axis.set_title(
            f"{feature_name}\nAUC(1vs2)={feature_row['auc_strength_0_vs_1']:.3f}, "
            f"AUC(2vs3)={feature_row['auc_strength_1_vs_2']:.3f}"
        )
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Detector-summary feature overlap on the test split", y=1.02)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_path}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def render_pairwise_auc_heatmap(overlap_df, output_path):
    feature_subset = overlap_df.head(10).copy()
    auc_matrix = feature_subset[
        ["auc_strength_0_vs_1", "auc_strength_1_vs_2", "auc_strength_0_vs_2"]
    ].to_numpy(dtype=np.float64)

    fig, axis = plt.subplots(figsize=(8.8, 7.2), constrained_layout=True)
    image = axis.imshow(auc_matrix, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")
    axis.set_xticks([0, 1, 2], ["1 vs 2", "2 vs 3", "1 vs 3"])
    axis.set_yticks(range(len(feature_subset)), feature_subset["feature"].tolist())
    axis.set_xlabel("Class pair")
    axis.set_title("Pairwise AUC by detector feature")
    for row_index in range(auc_matrix.shape[0]):
        for col_index in range(auc_matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                f"{auc_matrix[row_index, col_index]:.3f}",
                ha="center",
                va="center",
                color="white" if auc_matrix[row_index, col_index] > 0.74 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="AUC strength")
    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_path}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def render_confusion_comparison_figure(baseline_results, run_joined_dataframes, output_path):
    figure_specs = [
        (
            "Detector logistic",
            baseline_results["logistic_detector"]["split_metrics"]["test"],
        ),
        (
            "Detector random forest",
            baseline_results["random_forest_detector"]["split_metrics"]["test"],
        ),
        (
            "KM3Former exp_008",
            summarize_classifier_outputs(
                run_joined_dataframes["exp_008_capacity_push"]["target_class"].to_numpy(
                    dtype=np.int64
                ),
                run_joined_dataframes["exp_008_capacity_push"]["predicted_class"].to_numpy(
                    dtype=np.int64
                ),
                run_joined_dataframes["exp_008_capacity_push"][
                    [f"prob_class_{index}" for index in CLASS_INDICES]
                ].to_numpy(dtype=np.float64),
            ),
        ),
        (
            "KM3Former exp_009",
            summarize_classifier_outputs(
                run_joined_dataframes["exp_009_capacity_push_retuned"][
                    "target_class"
                ].to_numpy(dtype=np.int64),
                run_joined_dataframes["exp_009_capacity_push_retuned"][
                    "predicted_class"
                ].to_numpy(dtype=np.int64),
                run_joined_dataframes["exp_009_capacity_push_retuned"][
                    [f"prob_class_{index}" for index in CLASS_INDICES]
                ].to_numpy(dtype=np.float64),
            ),
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.2), constrained_layout=True)
    axes = axes.ravel()
    images = []
    for axis, (title, metrics) in zip(axes, figure_specs):
        class_one_recall = metrics["per_class_recall"]["1"]["recall"]
        image = render_confusion_matrix(
            axis,
            metrics["normalized_confusion_matrix"],
            f"{title}\nacc={metrics['accuracy']:.3f}, recall(2 muons)={class_one_recall:.3f}",
        )
        images.append(image)
    fig.colorbar(images[0], ax=axes.tolist(), fraction=0.024, pad=0.03, label="Row-normalized fraction")
    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_path}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def render_model_confidence_by_true_class(run_joined_dataframes, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for axis, (run_name, dataframe) in zip(axes, run_joined_dataframes.items()):
        for true_class in CLASS_INDICES:
            values = np.sort(
                dataframe.loc[dataframe["target_class"] == true_class, "confidence"].to_numpy(
                    dtype=np.float64
                )
            )
            ecdf = np.arange(1, len(values) + 1) / len(values)
            axis.plot(
                values,
                ecdf,
                color=CLASS_COLORS[true_class],
                linewidth=1.8,
                label=CLASS_LABELS[true_class],
            )
        axis.set_title(run_name)
        axis.set_xlabel("Prediction confidence")
        axis.set_ylabel("Empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Confidence by true class", y=1.02)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_path}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def render_class_one_reliability_and_margin(run_joined_dataframes, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    run_items = list(run_joined_dataframes.items())
    for column_index, (run_name, dataframe) in enumerate(run_items):
        reliability = build_reliability_curve(
            (dataframe["target_class"] == 1).to_numpy(dtype=np.int64),
            dataframe["prob_class_1"].to_numpy(dtype=np.float64),
        )
        axis = axes[0, column_index]
        valid = reliability["count"] > 0
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.0)
        axis.plot(
            reliability.loc[valid, "mean_predicted_probability"],
            reliability.loc[valid, "fraction_positive"],
            color=RUN_COLORS.get(run_name, "#333333"),
            linewidth=2.0,
            marker="o",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Predicted P(2 muons)")
        axis.set_ylabel("Observed fraction of 2-muon events")
        axis.set_title(f"{run_name}: class-1 reliability")

        axis = axes[1, column_index]
        class_one = dataframe.loc[dataframe["target_class"] == 1]
        for predicted_class in CLASS_INDICES:
            values = class_one.loc[
                class_one["predicted_class"] == predicted_class, "margin"
            ].to_numpy(dtype=np.float64)
            axis.hist(
                values,
                bins=np.linspace(0.0, 1.0, 21),
                density=True,
                histtype="step",
                linewidth=1.8,
                color=CLASS_COLORS[predicted_class],
                label=f"Pred {CLASS_LABELS[predicted_class]} (n={len(values)})",
            )
        axis.set_xlabel("Prediction margin")
        axis.set_ylabel("Density")
        axis.set_title(f"{run_name}: true 2-muon margins")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Class-1 reliability and ambiguity", y=1.02)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_path}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def select_representative_events(detector_test, run_joined_dataframes):
    combined = detector_test.copy()
    exp008 = run_joined_dataframes["exp_008_capacity_push"][
        ["source_event_index", "predicted_class", "confidence", "correct"]
    ].rename(
        columns={
            "predicted_class": "exp008_predicted_class",
            "confidence": "exp008_confidence",
            "correct": "exp008_correct",
        }
    )
    exp009 = run_joined_dataframes["exp_009_capacity_push_retuned"][
        ["source_event_index", "predicted_class", "confidence", "correct"]
    ].rename(
        columns={
            "predicted_class": "exp009_predicted_class",
            "confidence": "exp009_confidence",
            "correct": "exp009_correct",
        }
    )
    combined = combined.merge(exp008, on="source_event_index", how="inner")
    combined = combined.merge(exp009, on="source_event_index", how="inner")
    combined = combined.loc[combined["target_class"] == 1].copy()
    combined["confidence_bias"] = (combined["exp009_confidence"] - 0.5).abs()

    selected_rows = []
    for category_name, category_spec in REPRESENTATIVE_CATEGORIES.items():
        predicted_class = category_spec["predicted_class"]
        candidate_subset = combined.loc[
            (combined["exp008_predicted_class"] == predicted_class)
            & (combined["exp009_predicted_class"] == predicted_class)
        ].copy()
        if candidate_subset.empty:
            candidate_subset = combined.loc[
                combined["exp009_predicted_class"] == predicted_class
            ].copy()
        if category_name == "downward":
            chosen = candidate_subset.sort_values(
                by=["active_hits", "total_tot", "confidence_bias"],
                ascending=[True, True, True],
            ).head(1)
        elif category_name == "upward":
            chosen = candidate_subset.sort_values(
                by=["active_hits", "total_tot"],
                ascending=[False, False],
            ).head(1)
        else:
            median_activity = candidate_subset["active_hits"].median()
            chosen = candidate_subset.assign(
                median_distance=(candidate_subset["active_hits"] - median_activity).abs()
            ).sort_values(
                by=["median_distance", "total_tot"],
                ascending=[True, True],
            ).head(1)
        if chosen.empty:
            continue
        chosen = chosen.copy()
        chosen["category"] = category_name
        chosen["category_label"] = category_spec["label"]
        selected_rows.append(chosen)

    if not selected_rows:
        raise RuntimeError("Could not select representative events.")

    selected = pd.concat(selected_rows, ignore_index=True)
    return selected[
        [
            "category",
            "category_label",
            "source_event_index",
            "target_class",
            "active_hits",
            "total_tot",
            "time_span",
            "z_span",
            "total_energy_mc",
            "primary_energy_mc",
            "secondary_energy_mc",
            "primary_off_vertical_deg_mc",
            "exp008_predicted_class",
            "exp008_confidence",
            "exp009_predicted_class",
            "exp009_confidence",
        ]
    ].sort_values(
        by="category",
        key=lambda series: series.map(
            {"downward": 0, "middle": 1, "upward": 2}
        ),
    )
def render_representative_event_panel(hits, representative_events, output_path):
    category_order = ["downward", "middle", "upward"]
    representative_events = representative_events.set_index("category").loc[category_order].reset_index()

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.0), constrained_layout=True)
    scatter_reference = None
    for row_index, event_row in enumerate(representative_events.itertuples(index=False)):
        event_hits = hits[int(event_row.source_event_index)]
        tot = np.clip(event_hits[:, 7], 0.0, None)
        active_mask = tot > 0.0
        active_hits = event_hits[active_mask]
        active_tot = tot[active_mask]
        if active_hits.size == 0:
            continue

        point_sizes = 18.0 + 14.0 * np.log1p(active_tot)
        point_colors = np.log1p(active_tot)

        time_axis = axes[row_index, 0]
        scatter_reference = time_axis.scatter(
            active_hits[:, 0],
            active_hits[:, 3],
            s=point_sizes,
            c=point_colors,
            cmap="viridis",
            alpha=0.75,
            edgecolors="none",
        )
        time_axis.set_xlabel("Hit time")
        time_axis.set_ylabel("z")
        time_axis.set_title(
            f"{event_row.category_label}\n"
            f"event={event_row.source_event_index}, active_hits={event_row.active_hits}, "
            f"total_tot={event_row.total_tot:.1f}"
        )

        geometry_axis = axes[row_index, 1]
        geometry_axis.scatter(
            active_hits[:, 1],
            active_hits[:, 2],
            s=point_sizes,
            c=point_colors,
            cmap="viridis",
            alpha=0.75,
            edgecolors="none",
        )
        geometry_axis.set_xlabel("x")
        geometry_axis.set_ylabel("y")
        geometry_axis.set_title(
            f"exp008 -> {CLASS_LABELS[event_row.exp008_predicted_class]}, "
            f"exp009 -> {CLASS_LABELS[event_row.exp009_predicted_class]}\n"
            f"E1={event_row.primary_energy_mc:.1f}, E2={event_row.secondary_energy_mc:.1f}, "
            f"theta={event_row.primary_off_vertical_deg_mc:.1f} deg"
        )
        geometry_axis.set_aspect("equal")

    if scatter_reference is not None:
        fig.colorbar(
            scatter_reference,
            ax=axes.ravel().tolist(),
            fraction=0.024,
            pad=0.02,
            label="log1p(TOT)",
        )
    fig.suptitle("Representative true 2-muon events across the detector-activity continuum", y=1.01)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_path}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, payload):
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_ready)


def dataframe_to_markdown_table(dataframe, floatfmt=".3f"):
    columns = list(dataframe.columns)

    def _format_value(value):
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (bool, np.bool_)):
            return "True" if value else "False"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(_format_value(row[column]) for column in columns) + " |"
        for _, row in dataframe.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def build_report_markdown(
    output_dir,
    detector_summary,
    overlap_df,
    baseline_results,
    model_ambiguity,
    representative_events,
):
    top_features = overlap_df.head(5)[
        [
            "feature",
            "ordered_sandwich",
            "auc_strength_0_vs_1",
            "auc_strength_1_vs_2",
            "auc_strength_0_vs_2",
        ]
    ]
    top_features_table = dataframe_to_markdown_table(top_features, floatfmt=".3f")

    logistic_test = baseline_results["logistic_detector"]["split_metrics"]["test"]
    forest_test = baseline_results["random_forest_detector"]["split_metrics"]["test"]
    diagnostic_test = baseline_results["logistic_detector_plus_mc_diagnostic"][
        "split_metrics"
    ]["test"]

    exp008 = model_ambiguity["exp_008_capacity_push"]
    exp009 = model_ambiguity["exp_009_capacity_push_retuned"]

    representative_rows = dataframe_to_markdown_table(
        representative_events,
        floatfmt=".3f",
    )

    report_path = output_dir / "README.md"
    report = f"""# Detector Overlap Analysis

## Summary

This analysis treats class `0/1/2` as physical `1/2/3` muons and focuses on why the true two-muon class collapses toward one- and three-muon predictions.

Main conclusion:

- the bottleneck is **mostly an information limit in detector response**, not a clean direction-based mistake
- true two-muon events occupy an overlap region in detector-summary space
- KM3Former improves over cheap detector-only baselines, but not by a large margin
- if Monte Carlo truth energies are added to the same cheap baseline, performance rises sharply, which implies that generator-level multiplicity information is only partially visible to the detector

## Artifacts

- [detector summary table]({(output_dir / "detector_summary_all_splits.csv").resolve()})
- [detector feature overlap]({(output_dir / "detector_feature_overlap_test.csv").resolve()})
- [baseline metrics]({(output_dir / "baseline_metrics.json").resolve()})
- [model ambiguity summary]({(output_dir / "model_ambiguity_summary.json").resolve()})
- [representative events]({(output_dir / "representative_events.csv").resolve()})

Plots:

- [detector feature CDFs]({(output_dir / "plots" / "detector_feature_cdfs.png").resolve()})
- [pairwise AUC heatmap]({(output_dir / "plots" / "pairwise_auc_heatmap.png").resolve()})
- [baseline vs KM3Former confusion matrices]({(output_dir / "plots" / "confusion_comparison.png").resolve()})
- [confidence by true class]({(output_dir / "plots" / "confidence_by_true_class.png").resolve()})
- [class-1 reliability and margin]({(output_dir / "plots" / "class1_reliability_and_margin.png").resolve()})
- [representative event panel]({(output_dir / "plots" / "representative_events_panel.png").resolve()})

## Detector Overlap Clues

The strongest detector-summary features are the ones that track event activity and light yield, not direction. Top detector features by pairwise AUC strength on the test split:

{top_features_table}

Interpretation:

- if a feature is `ordered_sandwich = True`, the class medians are monotonic across `1 -> 2 -> 3` muons
- strong `1 vs 2` and `2 vs 3` AUC values mean the feature changes in the expected direction even before using a transformer
- the top-ranked features are primarily activity proxies such as hit count and TOT-derived summaries

## Baseline Comparison

Detector-only baselines on the test split:

- logistic regression: accuracy `{"{:.3f}".format(logistic_test["accuracy"])}`, recall(2 muons) `{"{:.3f}".format(logistic_test["per_class_recall"]["1"]["recall"])}`
- random forest: accuracy `{"{:.3f}".format(forest_test["accuracy"])}`, recall(2 muons) `{"{:.3f}".format(forest_test["per_class_recall"]["1"]["recall"])}`

Diagnostic upper-bound style check, not a fair detector-only baseline:

- logistic regression with detector features plus MC truth energies: accuracy `{"{:.3f}".format(diagnostic_test["accuracy"])}`, recall(2 muons) `{"{:.3f}".format(diagnostic_test["per_class_recall"]["1"]["recall"])}`

This gap is important:

- detector-only logistic regression is well below the transformer, so KM3Former is learning something useful
- but the detector-only baseline is still close enough to show that the overlap is already present in coarse hit-space
- the MC-augmented diagnostic rise indicates the missing signal is not purely a model-capacity issue

## Model Ambiguity

`exp_008_capacity_push`:

- true two-muon accuracy `{"{:.3f}".format(exp008["per_true_class"]["1"]["accuracy"])}`
- median confidence for true two-muon events `{"{:.3f}".format(exp008["per_true_class"]["1"]["median_confidence"])}`
- median confidence when true two-muon is predicted as one muon `{"{:.3f}".format(exp008["class_1_by_prediction"]["0"]["median_confidence"])}`
- median confidence when true two-muon is predicted correctly `{"{:.3f}".format(exp008["class_1_by_prediction"]["1"]["median_confidence"])}`
- median confidence when true two-muon is predicted as three muons `{"{:.3f}".format(exp008["class_1_by_prediction"]["2"]["median_confidence"])}`

`exp_009_capacity_push_retuned`:

- true two-muon accuracy `{"{:.3f}".format(exp009["per_true_class"]["1"]["accuracy"])}`
- median confidence for true two-muon events `{"{:.3f}".format(exp009["per_true_class"]["1"]["median_confidence"])}`
- median confidence when true two-muon is predicted as one muon `{"{:.3f}".format(exp009["class_1_by_prediction"]["0"]["median_confidence"])}`
- median confidence when true two-muon is predicted correctly `{"{:.3f}".format(exp009["class_1_by_prediction"]["1"]["median_confidence"])}`
- median confidence when true two-muon is predicted as three muons `{"{:.3f}".format(exp009["class_1_by_prediction"]["2"]["median_confidence"])}`

This is a strong sign that the middle class is not a compact cluster in the learned representation:

- the model is least confident when it predicts the two-muon class correctly
- it is often more confident when it pushes those same events to the neighboring classes

## Representative Events

Representative true two-muon events were chosen from the test split to cover:

- a detector-poor event pushed to one muon
- a mid-activity event predicted correctly
- a detector-rich event pushed to three muons

{representative_rows}

The representative event panel should be read as:

- left column: hit time vs `z`
- right column: hit geometry in the `x-y` plane
- point size and color scale with `TOT`

These examples are intended as qualitative checks of whether the detector response itself looks lower-count, middle, or upper-count.

## Verdict

Best-supported explanation at this stage:

- **primary bottleneck:** information limit in detector response
- **secondary factor:** some remaining model underfitting is still plausible because KM3Former beats the cheap detector-only baselines
- **not supported as the main issue:** a purely direction-driven failure mode

## Reproduction

```bash
uv run --with matplotlib python scripts/detector_overlap_analysis.py \\
  runs/exp_008_capacity_push \\
  runs/exp_009_capacity_push_retuned
```
"""
    with open(report_path, "w", encoding="ascii") as handle:
        handle.write(report)
    return report_path


def main():
    args = parse_args()
    configure_plot_style()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    detector_summary, hits, split_indices = build_detector_summary_dataframe(
        args.h5_path,
        metadata,
    )
    detector_summary_path = output_dir / "detector_summary_all_splits.csv"
    detector_summary.to_csv(detector_summary_path, index=False)

    detector_feature_columns = [
        "active_hits",
        "active_hit_fraction",
        "total_tot",
        "mean_tot_active",
        "max_tot_active",
        "time_span",
        "z_span",
        "z_mean",
        "radius_mean",
        "radius_std",
        "hit_dir_z_mean",
        "hit_dir_z_std",
    ]

    overlap_df = compute_detector_overlap_summary(
        detector_summary,
        detector_feature_columns,
    )
    overlap_path = output_dir / "detector_feature_overlap_test.csv"
    overlap_df.to_csv(overlap_path, index=False)

    baseline_results, baseline_predictions = train_detector_baselines(
        detector_summary,
        detector_feature_columns,
    )
    baseline_predictions_path = output_dir / "baseline_test_predictions.csv"
    baseline_predictions.to_csv(baseline_predictions_path, index=False)
    baseline_metrics_path = output_dir / "baseline_metrics.json"
    write_json(baseline_metrics_path, baseline_results)

    run_joined_dataframes = {}
    model_ambiguity = {}
    for run_dir in args.run_dirs:
        joined, payload = join_run_predictions(run_dir, detector_summary, split_indices)
        run_name = Path(run_dir).name
        run_joined_dataframes[run_name] = joined
        joined_path = output_dir / f"{run_name}_test_joined.csv"
        joined.to_csv(joined_path, index=False)
        model_ambiguity[run_name] = summarize_model_ambiguity(joined)

    model_ambiguity_path = output_dir / "model_ambiguity_summary.json"
    write_json(model_ambiguity_path, model_ambiguity)

    render_detector_feature_cdfs(
        detector_summary,
        overlap_df,
        plots_dir / "detector_feature_cdfs",
    )
    render_pairwise_auc_heatmap(
        overlap_df,
        plots_dir / "pairwise_auc_heatmap",
    )
    render_confusion_comparison_figure(
        baseline_results,
        run_joined_dataframes,
        plots_dir / "confusion_comparison",
    )
    render_model_confidence_by_true_class(
        run_joined_dataframes,
        plots_dir / "confidence_by_true_class",
    )
    render_class_one_reliability_and_margin(
        run_joined_dataframes,
        plots_dir / "class1_reliability_and_margin",
    )

    detector_test = detector_summary.loc[detector_summary["split"] == "test"].copy()
    representative_events = select_representative_events(
        detector_test,
        run_joined_dataframes,
    )
    representative_events_path = output_dir / "representative_events.csv"
    representative_events.to_csv(representative_events_path, index=False)
    render_representative_event_panel(
        hits,
        representative_events,
        plots_dir / "representative_events_panel",
    )

    report_path = build_report_markdown(
        output_dir,
        detector_summary,
        overlap_df,
        baseline_results,
        model_ambiguity,
        representative_events,
    )

    print(f"Wrote detector summary: {detector_summary_path}")
    print(f"Wrote overlap summary:  {overlap_path}")
    print(f"Wrote baselines:        {baseline_metrics_path}")
    print(f"Wrote ambiguity:        {model_ambiguity_path}")
    print(f"Wrote representative events: {representative_events_path}")
    print(f"Wrote report:           {report_path}")
    print(f"Wrote plots under:      {plots_dir}")


if __name__ == "__main__":
    main()
