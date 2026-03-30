import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PREDICTION_COLORS = {
    0: "#4477AA",
    1: "#228833",
    2: "#CC6677",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render publication-style plots for muon-count failure analysis, "
            "with special focus on the difficult middle class."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Run directories or joined CSV files produced by "
            "scripts/analyze_muon_count_predictions.py. Directories default to "
            "`test_predictions_joined.csv`."
        ),
    )
    parser.add_argument(
        "--h5-path",
        default="data/muon_data_7224_7247.h5",
        help="Path to the raw HDF5 file with `hits`, `mc_muons`, and `energies`.",
    )
    parser.add_argument(
        "--focus-class",
        type=int,
        default=1,
        help="True class to focus on. Default is the middle class `1`.",
    )
    parser.add_argument(
        "--focus-label",
        default="True class 1",
        help="Human-readable label used in plot titles.",
    )
    parser.add_argument(
        "--output-subdir",
        default="eda_plots",
        help="Subdirectory created next to each joined CSV for plot outputs.",
    )
    return parser.parse_args()


def resolve_joined_csv(raw_input):
    path = Path(raw_input)
    if path.is_dir():
        candidate = path / "test_predictions_joined.csv"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not find test_predictions_joined.csv inside: {path}"
            )
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return path


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


def load_joined_dataframe(path):
    dataframe = pd.read_csv(path)
    bool_map = {"True": True, "False": False, True: True, False: False}
    dataframe["correct"] = dataframe["correct"].map(bool_map).astype(bool)
    dataframe["secondary_direction_present"] = (
        dataframe["secondary_direction_present"].map(bool_map).astype(bool)
    )

    numeric_columns = [
        column
        for column in dataframe.columns
        if column
        not in {"split", "correct", "secondary_direction_present"}
    ]
    dataframe[numeric_columns] = dataframe[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    return dataframe


def compute_hit_activity(hits):
    tot = hits[:, :, 7]
    active_mask = tot > 0.0
    active_hits = active_mask.sum(axis=1).astype(np.int64)
    total_tot = np.clip(tot, 0.0, None).sum(axis=1)
    return active_hits, total_tot


def enrich_with_hit_activity(dataframe, h5_path):
    source_indices = dataframe["source_event_index"].astype(int).to_numpy()
    sort_order = np.argsort(source_indices)
    inverse_order = np.argsort(sort_order)
    sorted_indices = source_indices[sort_order]
    with h5py.File(h5_path, "r") as handle:
        hits = handle["hits"][sorted_indices][inverse_order]

    active_hits, total_tot = compute_hit_activity(hits)
    enriched = dataframe.copy()
    enriched["active_hits"] = active_hits
    enriched["total_tot"] = total_tot
    enriched["log10_total_energy"] = np.log10(
        np.clip(enriched["total_energy_first_two"].to_numpy(), 1e-6, None)
    )
    enriched["log10_primary_energy"] = np.log10(
        np.clip(enriched["primary_energy"].to_numpy(), 1e-6, None)
    )
    enriched["log10_secondary_energy"] = np.log10(
        np.clip(enriched["secondary_energy"].to_numpy(), 1e-6, None)
    )
    enriched["log10_total_tot"] = np.log10(
        np.clip(enriched["total_tot"].to_numpy(), 1e-6, None)
    )
    enriched["secondary_energy_fraction"] = (
        enriched["secondary_energy"] / enriched["total_energy_first_two"]
    )
    return enriched


def format_prediction_label(class_index):
    return f"Predicted class {class_index}"


def add_density_histogram(axis, series_by_prediction, bins, xlabel):
    for predicted_class, series in series_by_prediction.items():
        if len(series) == 0:
            continue
        axis.hist(
            series,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=PREDICTION_COLORS[predicted_class],
            label=f"{format_prediction_label(predicted_class)} (n={len(series)})",
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")


def render_confusion_heatmap(axis, dataframe):
    confusion = pd.crosstab(
        dataframe["target_class"],
        dataframe["predicted_class"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    image = axis.imshow(
        confusion.to_numpy(),
        cmap=LinearSegmentedColormap.from_list(
            "km3former_heat", ["#F7FBFF", "#6BAED6", "#08306B"]
        ),
        aspect="equal",
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks([0, 1, 2], labels=["0", "1", "2"])
    axis.set_yticks([0, 1, 2], labels=["0", "1", "2"])

    for true_index in range(3):
        for predicted_index in range(3):
            value = int(confusion.iloc[true_index, predicted_index])
            axis.text(
                predicted_index,
                true_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > confusion.to_numpy().max() * 0.45 else "black",
                fontsize=10,
            )
    return image


def render_overview_figure(dataframe, focus_dataframe, output_prefix, focus_label):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    image = render_confusion_heatmap(axes[0, 0], dataframe)
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 0].set_title("Confusion matrix")

    grouped = {
        predicted_class: focus_dataframe.loc[
            focus_dataframe["predicted_class"] == predicted_class
        ]
        for predicted_class in [0, 1, 2]
    }

    add_density_histogram(
        axes[0, 1],
        {
            class_index: group["log10_total_energy"].to_numpy()
            for class_index, group in grouped.items()
        },
        bins=np.linspace(
            focus_dataframe["log10_total_energy"].min(),
            focus_dataframe["log10_total_energy"].max(),
            28,
        ),
        xlabel="log10(total MC energy of first two muons)",
    )
    axes[0, 1].set_title(f"{focus_label}: total energy")

    add_density_histogram(
        axes[0, 2],
        {
            class_index: group["active_hits"].to_numpy()
            for class_index, group in grouped.items()
        },
        bins=np.arange(
            max(1, int(focus_dataframe["active_hits"].min()) - 1),
            int(focus_dataframe["active_hits"].max()) + 3,
            4,
        ),
        xlabel="Active hits (TOT > 0)",
    )
    axes[0, 2].set_title(f"{focus_label}: detector activity")

    add_density_histogram(
        axes[1, 0],
        {
            class_index: group["log10_total_tot"].to_numpy()
            for class_index, group in grouped.items()
        },
        bins=np.linspace(
            focus_dataframe["log10_total_tot"].min(),
            focus_dataframe["log10_total_tot"].max(),
            28,
        ),
        xlabel="log10(sum TOT)",
    )
    axes[1, 0].set_title(f"{focus_label}: total TOT")

    add_density_histogram(
        axes[1, 1],
        {
            class_index: group["primary_off_vertical_deg"].to_numpy()
            for class_index, group in grouped.items()
        },
        bins=np.linspace(0.0, max(70.0, focus_dataframe["primary_off_vertical_deg"].max()), 28),
        xlabel="Primary off-vertical angle [deg]",
    )
    axes[1, 1].set_title(f"{focus_label}: direction")

    add_density_histogram(
        axes[1, 2],
        {
            class_index: group["secondary_energy_fraction"].to_numpy()
            for class_index, group in grouped.items()
        },
        bins=np.linspace(0.0, 1.0, 24),
        xlabel="Secondary energy / total energy",
    )
    axes[1, 2].set_title(f"{focus_label}: energy sharing")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        f"Muon-count failure overview for {focus_label}",
        y=1.02,
        fontsize=14,
    )

    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_prefix}_overview.{suffix}", bbox_inches="tight")
    plt.close(fig)


def render_scatter_figure(focus_dataframe, output_prefix, focus_label):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)

    for predicted_class in [0, 1, 2]:
        group = focus_dataframe.loc[focus_dataframe["predicted_class"] == predicted_class]
        axes[0].scatter(
            group["primary_energy"],
            group["secondary_energy"],
            s=18,
            alpha=0.5,
            color=PREDICTION_COLORS[predicted_class],
            label=f"{format_prediction_label(predicted_class)} (n={len(group)})",
            edgecolors="none",
        )
        axes[1].scatter(
            group["total_energy_first_two"],
            group["active_hits"],
            s=18,
            alpha=0.5,
            color=PREDICTION_COLORS[predicted_class],
            label=f"{format_prediction_label(predicted_class)} (n={len(group)})",
            edgecolors="none",
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Primary MC energy")
    axes[0].set_ylabel("Secondary MC energy")
    axes[0].set_title(f"{focus_label}: energy plane")

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Total MC energy of first two muons")
    axes[1].set_ylabel("Active hits (TOT > 0)")
    axes[1].set_title(f"{focus_label}: energy vs detector activity")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    for suffix in ("png", "pdf"):
        fig.savefig(f"{output_prefix}_scatter.{suffix}", bbox_inches="tight")
    plt.close(fig)


def build_summary(dataframe, focus_dataframe, focus_class, focus_label):
    overall_accuracy = float((dataframe["target_class"] == dataframe["predicted_class"]).mean())
    focus_groups = {}
    for predicted_class in [0, 1, 2]:
        group = focus_dataframe.loc[focus_dataframe["predicted_class"] == predicted_class]
        focus_groups[str(predicted_class)] = {
            "count": int(len(group)),
            "fraction_within_focus_class": float(len(group) / len(focus_dataframe)),
            "median_total_energy_first_two": float(group["total_energy_first_two"].median()),
            "median_active_hits": float(group["active_hits"].median()),
            "median_total_tot": float(group["total_tot"].median()),
            "median_primary_off_vertical_deg": float(
                group["primary_off_vertical_deg"].median()
            ),
            "median_secondary_energy_fraction": float(
                group["secondary_energy_fraction"].median()
            ),
        }
    return {
        "overall_accuracy": overall_accuracy,
        "focus_class": int(focus_class),
        "focus_label": focus_label,
        "focus_support": int(len(focus_dataframe)),
        "per_predicted_class": focus_groups,
    }


def write_summary(summary, output_prefix):
    summary_path = Path(f"{output_prefix}_summary.json")
    with open(summary_path, "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary_path


def main():
    args = parse_args()
    configure_plot_style()

    for raw_input in args.inputs:
        joined_csv_path = resolve_joined_csv(raw_input)
        dataframe = load_joined_dataframe(joined_csv_path)
        dataframe = enrich_with_hit_activity(dataframe, args.h5_path)
        focus_dataframe = dataframe.loc[dataframe["target_class"] == args.focus_class].copy()

        if focus_dataframe.empty:
            raise ValueError(
                f"No rows with target_class={args.focus_class} found in {joined_csv_path}"
            )

        output_dir = joined_csv_path.parent / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = output_dir / f"{joined_csv_path.stem}_focus_class_{args.focus_class}"

        render_overview_figure(
            dataframe=dataframe,
            focus_dataframe=focus_dataframe,
            output_prefix=output_prefix,
            focus_label=args.focus_label,
        )
        render_scatter_figure(
            focus_dataframe=focus_dataframe,
            output_prefix=output_prefix,
            focus_label=args.focus_label,
        )
        summary = build_summary(
            dataframe=dataframe,
            focus_dataframe=focus_dataframe,
            focus_class=args.focus_class,
            focus_label=args.focus_label,
        )
        summary_path = write_summary(summary, output_prefix)

        print(f"\nGenerated plots for {joined_csv_path.parent.name}")
        print(f"  overview: {output_prefix}_overview.png")
        print(f"  scatter:  {output_prefix}_scatter.png")
        print(f"  summary:  {summary_path}")


if __name__ == "__main__":
    main()
