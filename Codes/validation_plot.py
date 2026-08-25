"""
plot_figure6_validation.py

Recreates Figure 6 exactly: dot plots of dendritic morphology metrics
(Total Dendrite Length, Branch Points, Primary Dendrites, Max Branch
Order) for a SINGLE experimental condition (e.g. EP), colored by BRAIN
REGION (Caudoputamen, Nucleus Accumbens, Cortex) rather than by
condition. This is the visual/quantitative validation figure -- one
condition, all its regions side by side.

Usage:
    python3 plot_figure6_validation.py Metrics_validation.csv

Region is auto-detected from the SWC filename. Flagged neurons
(flag_incomplete == True) are kept in the plot but rendered with an
open/hollow marker to visually distinguish them from confidently
complete reconstructions.

Developed with the assistance of Claude (Anthropic, claude.ai, accessed August 2026).
"""

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REGION_PATTERNS = [
    (r"caudoputamen", "Caudoputamen"),
    (r"nucleus[_ ]?accumbens", "Nucleus Accumbens"),
    (r"nca\d", "Nucleus Accumbens"),
    (r"\bnca\b", "Nucleus Accumbens"),
    (r"hypothalamus", "Hypothalamus"),
    (r"hippocampus", "Hippocampus"),
    (r"amygdala|\bbla\b", "Amygdala"),
    (r"thalamus", "Thalamus"),
    (r"ipacl", "IPACL"),
    (r"cortex", "Cortex"),
]

# Region colors, consistent with the scheme used throughout the thesis
# (dark blue / purple / turquoise), matching Figure 6's original coloring.
REGION_COLORS = {
    "Caudoputamen":      "#2056A8",  # dark blue
    "Nucleus Accumbens": "#6A4C93",  # purple
    "Cortex":            "#2EC4B6",  # turquoise
}
REGION_ORDER = ["Caudoputamen", "Nucleus Accumbens", "Cortex"]

METRICS = [
    ("TDL_um",        "Total Dendrite Length (µm)"),
    ("branch_points", "Branch Points (n)"),
    ("primaries",     "Primary Dendrites (n)"),
    ("max_order",     "Max Branch Order (n)"),
]

# Font sizes (increased per Simon's feedback: "Font too small")
FONT_AXIS_LABEL = 14
FONT_TICK_LABEL = 13
FONT_SUPTITLE = 17
FONT_LEGEND = 13


def detect_region(filename):
    name = filename.lower()
    for pattern, label in REGION_PATTERNS:
        if re.search(pattern, name):
            return label
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the single-condition metrics CSV")
    parser.add_argument("--condition-label", default=None,
                         help="Condition name shown in the title (e.g. 'EP'). "
                              "If omitted, no condition name is shown in the title.")
    parser.add_argument("--output", default="figure6_validation.png",
                         help="Output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df["region"] = df["file"].apply(detect_region)
    df["flag_incomplete"] = df["flag_incomplete"].astype(str).str.lower().eq("true")

    unknown = df[df["region"] == "Unknown"]["file"].tolist()
    if unknown:
        print(f"⚠️  Could not detect region for {len(unknown)} file(s); excluded from plot:")
        for u in unknown:
            print(f"    - {u}")
        df = df[df["region"] != "Unknown"]

    regions_present = [r for r in REGION_ORDER if r in df["region"].unique()]
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.6 * n_metrics, 5.2))

    x_positions = {r: i for i, r in enumerate(regions_present)}

    for ax, (col, ylabel) in zip(axes, METRICS):
        for region in regions_present:
            color = REGION_COLORS[region]
            sub = df[df["region"] == region]
            x0 = x_positions[region]

            complete = sub[~sub["flag_incomplete"]]
            flagged = sub[sub["flag_incomplete"]]

            if len(complete):
                ax.scatter(
                    x0 + np.random.uniform(-0.08, 0.08, size=len(complete)),
                    complete[col], color=color, s=32, alpha=0.85,
                    edgecolors="none", zorder=3,
                )
            if len(flagged):
                ax.scatter(
                    x0 + np.random.uniform(-0.08, 0.08, size=len(flagged)),
                    flagged[col], facecolors="none", edgecolors=color,
                    s=44, linewidths=1.4, zorder=3,
                )

            mean = sub[col].mean()
            sd = sub[col].std()
            ax.errorbar(
                x0, mean, yerr=sd, fmt="o", color=color,
                markersize=10, markeredgecolor="black", markeredgewidth=0.6,
                elinewidth=1.4, capsize=4, zorder=4,
            )

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(list(x_positions.keys()), fontsize=FONT_TICK_LABEL,
                            rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
        ax.set_xlim(-0.6, len(regions_present) - 0.4)
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=REGION_COLORS[r],
                    markersize=9, label=r)
        for r in regions_present
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                    markeredgecolor="gray", markersize=9,
                    label="flagged (possibly incomplete tracing)")
    )
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.1), frameon=False, fontsize=FONT_LEGEND)

    title = "Validation — Neuronal Morphology across Brain Regions"
    if args.condition_label:
        title += f"\n({args.condition_label})"
    fig.suptitle(title, y=1.2, fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"✅ Figure saved to: {args.output}")

    n_flagged = df["flag_incomplete"].sum()
    print(f"\n{len(df)} neuron(s) total across {len(regions_present)} region(s), "
          f"{n_flagged} flagged as potentially incomplete.")
    for region in regions_present:
        n = (df["region"] == region).sum()
        print(f"  {region}: n={n}")


if __name__ == "__main__":
    main()
