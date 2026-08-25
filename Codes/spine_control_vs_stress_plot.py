"""
plot_spine_control_vs_stress.py — Generate the Control-vs-Stress spine
type distribution comparison plot by brain region, in English, using
the established Golgi color scheme.

Reads (for BOTH the Control and the Stress dataset):
    - summary_per_neuron.csv (neuron, region, n_thin, n_mushroom, n_stubby, n_filopodia, ...)

Produces:
    plot_type_distribution_by_region_control_vs_stress.png
    Single-panel grouped bar chart: x-axis = brain region, bars colored
    by spine type (thin/mushroom/stubby/filopodia), Control shown as
    solid bars and Stress as hatched bars of the same color, so both
    groups are directly compared within one plot.

Usage:
    python3 plot_spine_control_vs_stress.py \
        --control-per-neuron path/to/control/summary_per_neuron.csv \
        --stress-per-neuron path/to/stress/summary_per_neuron.csv \
        --outdir .
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGION_ORDER = ["Cortex", "Hippocampus", "Striatum", "Amygdala"]
GROUP_COLORS = {"Control": "#1f4e9c", "Stress": "#2EC4B6"}
TYPE_COLORS = {
    "thin": "#5DA9E9",
    "mushroom": "#6A4C93",
    "stubby": "#2EC4B6",
    "filopodia": "#1f4e9c",
}
TYPE_ORDER = ["thin", "mushroom", "stubby", "filopodia"]


def plot_type_distribution(control_per_neuron_df, stress_per_neuron_df, outpath):
    """Single-panel grouped bar chart: x-axis = brain region, bars colored
    by spine type, with Control shown as solid bars and Stress as
    hatched bars of the same color -- so both groups are directly
    compared within one plot instead of two separate side-by-side
    panels.

    Bars show the MEAN PER-NEURON DENSITY (spines / 10 um dendrite) for
    each type, rather than raw summed counts -- this normalizes for
    unequal numbers of neurons and unequal total dendrite length
    sampled between groups, while remaining purely descriptive (no
    formal statistical test, given the nested spines-within-neuron
    data structure)."""
    type_cols = {"thin": "n_thin", "mushroom": "n_mushroom",
                 "stubby": "n_stubby", "filopodia": "n_filopodia"}

    control_per_neuron_df = control_per_neuron_df.copy()
    stress_per_neuron_df = stress_per_neuron_df.copy()
    control_per_neuron_df["group"] = "Control"
    def normalize_segment_col(d, source_name):
        if "segment_length_um" not in d.columns and "segment_um" in d.columns:
            print(f"  Note: {source_name} uses column name 'segment_um' -- renaming to 'segment_length_um'.")
            d = d.rename(columns={"segment_um": "segment_length_um"})
        if "segment_length_um" not in d.columns:
            raise ValueError(f"{source_name} has neither 'segment_length_um' nor 'segment_um' column.")
        return d

    control_per_neuron_df = normalize_segment_col(control_per_neuron_df, "control CSV")
    stress_per_neuron_df = normalize_segment_col(stress_per_neuron_df, "stress CSV")

    stress_per_neuron_df["group"] = "Stress"
    df = pd.concat([control_per_neuron_df, stress_per_neuron_df], ignore_index=True)

    # per-neuron density for each spine type
    for spine_type, count_col in type_cols.items():
        df[f"density_{spine_type}"] = (df[count_col] / df["segment_length_um"]) * 10

    regions_present = [r for r in REGION_ORDER if r in df["region"].unique()]
    mean_density = (df.groupby(["region", "group"])[[f"density_{t}" for t in TYPE_ORDER]]
                     .mean().reindex(regions_present, level="region"))

    n_types = len(TYPE_ORDER)
    n_groups = 2  # Control, Stress
    # each region gets a cluster of n_types * n_groups bars
    cluster_width = 0.8
    bar_width = cluster_width / (n_types * n_groups)
    x = np.arange(len(regions_present))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for t_idx, spine_type in enumerate(TYPE_ORDER):
        for g_idx, group in enumerate(["Control", "Stress"]):
            # position within the cluster: type blocks, with Control/Stress as adjacent sub-bars
            slot = t_idx * n_groups + g_idx
            offset = (slot - (n_types * n_groups - 1) / 2) * bar_width

            vals = []
            for region in regions_present:
                try:
                    vals.append(mean_density.loc[(region, group), f"density_{spine_type}"])
                except KeyError:
                    vals.append(0)

            hatch = "" if group == "Control" else "///"
            ax.bar(x + offset, vals, width=bar_width * 0.95, color=TYPE_COLORS[spine_type],
                   edgecolor="black", linewidth=0.5, hatch=hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(regions_present)
    ax.set_xlabel("Brain Region")
    ax.set_ylabel("Mean Density (spines / 10 µm)")
    ax.set_title("Spine Type Density by Brain Region — Control vs. Stress")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # two separate legends: one for type (color), one for group (hatch)
    type_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=TYPE_COLORS[t], edgecolor="black")
                     for t in TYPE_ORDER]
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=""),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="///"),
    ]
    legend1 = ax.legend(type_handles, TYPE_ORDER, title="Type", loc="upper left",
                         bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
    ax.add_artist(legend1)
    legend2 = ax.legend(group_handles, ["Control", "Stress"], title="Group", loc="upper left",
                         bbox_to_anchor=(1.0, 0.6), borderaxespad=0.5)

    fig.subplots_adjust(right=0.75)
    plt.savefig(outpath, dpi=200, bbox_inches="tight", bbox_extra_artists=(legend1, legend2),
                pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Control vs Stress spine type distribution plot")
    parser.add_argument("--control-per-neuron", required=True)
    parser.add_argument("--stress-per-neuron", required=True)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    control_per_neuron_df = pd.read_csv(args.control_per_neuron)
    stress_per_neuron_df = pd.read_csv(args.stress_per_neuron)

    plot_type_distribution(
        control_per_neuron_df, stress_per_neuron_df,
        os.path.join(args.outdir, "plot_type_distribution_by_region_control_vs_stress.png"))


if __name__ == "__main__":
    main()
