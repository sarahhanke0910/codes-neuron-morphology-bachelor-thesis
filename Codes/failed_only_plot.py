"""
plot_failed_only.py

Produces BOTH validation figures from a single script run:

  1. Panel B — reconstruction_success_overview.png
     Stacked bar chart of reconstruction outcomes (used / failed /
     unmarked) per condition and brain region, with success rate
     (used/total, %) annotated above each bar. Ported from the
     original reconstruction_success_overview.R.

  2. Panel C — failed_only_plot.png
     Morphological metrics (Total Dendrite Length, Branch Points,
     Primary Dendrites, Max Branch Order) computed from the excluded
     ("failed") reconstructions only, grouped by condition and brain
     region (Cortex vs. second region within each condition).

Both panels read from the same All_reconstructions_overview.xlsx for
condition/region/outcome classification; Panel C additionally needs
the separately computed morphology metrics CSV
(failed_reconstructions_metrics.csv).

Usage:
    python3 plot_failed_only.py failed_reconstructions_metrics.csv \
        --overview-excel All_reconstructions_overview.xlsx \
        --output-b reconstruction_success_overview.png \
        --output-c failed_only_plot.png

Developed with the assistance of Claude (Anthropic, claude.ai, accessed August 2026).
"""

import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

CONDITION_COLOR_ORDER = ["SOC", "EP", "CON", "Q"]
CONDITION_COLORS = {
    "SOC": "#2EC4B6",
    "EP":  "#5DA9E9",
    "CON": "#2056A8",
    "Q":   "#6A4C93",
}

# Status colors for Panel B, matching reconstruction_success_overview.R exactly
STATUS_COLORS = {
    "used":     "#2AB7A9",
    "failed":   "#7B3FA0",
    "unmarked": "#CCCCCC",
}

METRICS = [
    ("TDL_um",        "Total Dendrite Length (µm)"),
    ("branch_points", "Branch Points (n)"),
    ("primaries",     "Primary Dendrites (n)"),
    ("max_order",     "Max Branch Order (n)"),
]

# ---- Font sizes (matching the size that worked well when printed, same as
# plot_morphology_data_analysis.py / plot_figure6_validation.py) ----
FONT_AXIS_LABEL = 14
FONT_TICK_LABEL = 13
FONT_CONDITION_LABEL = 13
FONT_SUPTITLE = 17
FONT_LEGEND = 13
FONT_BAR_LABEL = 12


def detect_region(filename):
    name = filename.lower()
    for pattern, label in REGION_PATTERNS:
        if re.search(pattern, name):
            return label
    return "Unknown"


def get_color(condition, idx_fallback):
    if condition in CONDITION_COLORS:
        return CONDITION_COLORS[condition]
    cmap = plt.get_cmap("tab10")
    return cmap(idx_fallback % 10)


def load_overview(overview_excel):
    """Load and classify All_reconstructions_overview.xlsx (used/failed/unmarked
    per condition and region), matching reconstruction_success_overview.R."""
    df = pd.read_excel(overview_excel)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "all reconstructions":     "file",
        "Condition":               "condition",
        "Brain Region":            "region",
        "used for analysis":       "used_col",
        "not used for analysis":   "notused_col",
    })
    df["condition"] = df["condition"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    df["status"] = np.where(
        df["used_col"].notna(), "used",
        np.where(df["notused_col"].notna(), "failed", "unmarked")
    )
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_COLOR_ORDER, ordered=True)
    return df


def make_success_rate_plot(overview_df, output_path):
    """Panel B: stacked bar chart of used/failed/unmarked reconstructions
    per condition-region group, with success rate (%) annotated above."""
    df = overview_df.copy()
    df["group_label"] = df["condition"].astype(str) + " – " + df["region"]

    group_order = (
        df[["condition", "region", "group_label"]]
        .drop_duplicates()
        .sort_values(["condition", "region"])["group_label"]
        .tolist()
    )
    df["group_label"] = pd.Categorical(df["group_label"], categories=group_order, ordered=True)

    counts = df.groupby(["group_label", "status"], observed=True).size().unstack(fill_value=0)
    for status in ["used", "failed", "unmarked"]:
        if status not in counts.columns:
            counts[status] = 0
    counts = counts.reindex(group_order)

    n_unmarked = int(counts["unmarked"].sum())
    if n_unmarked > 0:
        print(f"WARNING: {n_unmarked} reconstruction(s) are classified as 'unmarked' "
              f"(neither used nor failed) and will NOT be shown in the plot, per request "
              f"to remove 'unmarked' from the legend.")

    totals = counts["used"] + counts["failed"]
    success_rate = (counts["used"] / totals * 100).round(1)

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(group_order)), 6.5))
    x = np.arange(len(group_order))
    bottom = np.zeros(len(group_order))

    for status in ["used", "failed"]:
        vals = counts[status].to_numpy()
        bars = ax.bar(x, vals, bottom=bottom, width=0.65, color=STATUS_COLORS[status], label=status)
        for xi, v, b in zip(x, vals, bottom):
            if v > 0:
                ax.text(xi, b + v / 2, str(int(v)), ha="center", va="center",
                        color="white", fontweight="bold", fontsize=FONT_BAR_LABEL)
        bottom += vals

    for xi, total, rate, used_n in zip(x, totals, success_rate, counts["used"]):
        ax.text(xi, total + max(totals) * 0.02, f"{int(used_n)}/{int(total)} ({rate}%)",
                ha="center", va="bottom", fontsize=FONT_BAR_LABEL - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=40, ha="right", fontsize=FONT_TICK_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
    ax.set_ylabel("Number of reconstructions", fontsize=FONT_AXIS_LABEL)
    ax.set_ylim(0, max(totals) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Reconstruction outcome", title_fontsize=FONT_LEGEND,
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              frameon=False, fontsize=FONT_LEGEND)
    ax.set_title("Neuron Reconstruction Outcomes by Condition and Brain Region",
                  fontsize=FONT_SUPTITLE - 3, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")


def make_failed_metrics_plot(input_csv, overview_df, output_path):
    """Panel C: morphological metrics of failed reconstructions, grouped
    by condition and region (Cortex vs. second region per condition)."""
    df = pd.read_csv(input_csv)
    df["region"] = df["file"].apply(detect_region)
    df["flag_incomplete"] = df["flag_incomplete"].astype(str).str.lower().eq("true")

    overview = overview_df[["file", "condition"]].copy()
    overview["file_key"] = overview["file"].astype(str).str.replace(r"\.swc$", "", regex=True).str.strip()

    df["file_key"] = df["file"].astype(str).str.replace(r"\.swc$", "", regex=True).str.strip()
    df = df.merge(overview[["file_key", "condition"]], on="file_key", how="left")
    df["condition"] = df["condition"].astype(str).str.strip()

    unmatched = df[df["condition"].isna() | (df["condition"] == "nan")]
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} file(s) could not be matched to a condition:")
        for f in unmatched["file"]:
            print(f"  - {f}")
        df = df[~(df["condition"].isna() | (df["condition"] == "nan"))]

    conditions_present = [c for c in CONDITION_COLOR_ORDER if c in df["condition"].unique()]
    conditions_present += [c for c in df["condition"].unique() if c not in CONDITION_COLOR_ORDER]

    group_order = []
    group_to_condition = {}
    for cond in conditions_present:
        sub_cond = df[df["condition"] == cond]
        regions_here = sorted(sub_cond["region"].unique())
        ordered_regions = ([r for r in regions_here if r == "Cortex"] +
                            [r for r in regions_here if r != "Cortex"])
        for region in ordered_regions:
            label = f"{cond}\n{region}"
            group_order.append(label)
            group_to_condition[label] = cond

    df["group_label"] = df["condition"] + "\n" + df["region"]

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(2.4 * len(group_order) + 2, 8.2))

    x_positions = {g: i for i, g in enumerate(group_order)}

    for ax, (col, ylabel) in zip(axes, METRICS):
        for group in group_order:
            condition = group_to_condition[group]
            color = get_color(condition, conditions_present.index(condition))
            sub = df[df["group_label"] == group]
            if sub.empty:
                continue
            x0 = x_positions[group]

            mean = sub[col].mean()
            sd = sub[col].std()

            ax.bar(x0, mean, width=0.65, color=color, alpha=0.35,
                   edgecolor=color, linewidth=1.2, zorder=2)
            ax.errorbar(x0, mean, yerr=sd, fmt="none", ecolor="dimgrey",
                        elinewidth=1.3, capsize=3.5, zorder=3)

            complete = sub[~sub["flag_incomplete"]]
            flagged = sub[sub["flag_incomplete"]]

            if len(complete):
                ax.scatter(
                    x0 + np.random.uniform(-0.12, 0.12, size=len(complete)),
                    complete[col], color=color, s=30, alpha=0.9,
                    edgecolors="black", linewidths=0.4, zorder=4,
                )
            if len(flagged):
                ax.scatter(
                    x0 + np.random.uniform(-0.12, 0.12, size=len(flagged)),
                    flagged[col], facecolors="none", edgecolors=color,
                    s=42, linewidths=1.3, zorder=4,
                )

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([g.split("\n")[1] for g in group_order],
                            fontsize=FONT_TICK_LABEL, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
        ax.set_xlim(-0.6, len(group_order) - 0.4)
        ax.margins(y=0.1)
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for cond in conditions_present:
            cond_x = [x_positions[g] for g in group_order if group_to_condition[g] == cond]
            if not cond_x:
                continue
            x_center = sum(cond_x) / len(cond_x)
            ax.text(
                x_center, -0.32, cond, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FONT_CONDITION_LABEL, fontweight="bold",
                color=get_color(cond, conditions_present.index(cond)), clip_on=False,
            )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=get_color(c, i),
                    markersize=9, label=c)
        for i, c in enumerate(conditions_present)
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                    markeredgecolor="gray", markersize=9,
                    label="flagged (possibly incomplete tracing)")
    )

    fig.subplots_adjust(top=0.78, bottom=0.32, wspace=0.35)

    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98),
               frameon=False, fontsize=FONT_LEGEND)
    fig.suptitle("Neuronal Morphology — Failed Reconstructions\n(Cortex vs. Second Region per Condition)",
                 y=1.05, fontsize=FONT_SUPTITLE)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"\nn per condition/region:")
    print(df.groupby(["condition", "region"]).size())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Path to failed_reconstructions_metrics.csv (for Panel C)")
    parser.add_argument("--overview-excel", required=True,
                         help="Path to All_reconstructions_overview.xlsx (used for both panels)")
    parser.add_argument("--output-b", default="reconstruction_success_overview.png",
                         help="Output PNG path for Panel B (success rate overview)")
    parser.add_argument("--output-c", default="failed_only_plot.png",
                         help="Output PNG path for Panel C (failed reconstruction metrics)")
    args = parser.parse_args()

    overview_df = load_overview(args.overview_excel)

    print("Generating Panel B (reconstruction success overview)...")
    make_success_rate_plot(overview_df, args.output_b)

    print("\nGenerating Panel C (failed reconstruction metrics)...")
    make_failed_metrics_plot(args.input_csv, overview_df, args.output_c)

    print("\nDone. Both panels saved.")


if __name__ == "__main__":
    main()

