#!/usr/bin/env python3
"""
sholl_curves_control_vs_stress.py
===========================================================
Per-region Sholl profile plots: Control vs. Stress overlaid in one panel
per brain region (Cortex, Hippocampus, Striatum, Amygdala), mean +/- SEM,
using the exact same mean/SEM computation as sholl_analysis_golgi.py.

Color palette matches golgi_sholl_stats_2.R:
    Control = "#1f4e9c"
    Stress  = "#2EC4B6"

Font sizes match the enlarged RABIES/Golgi print scale (title 22, axis
labels 20, tick labels 17, legend 15) so the figure stays legible when
printed.

Input: the two sholl_curves.csv files produced by sholl_analysis_golgi.py
    (one from the Control run, one from the Stress run) — each with
    columns: cell_id, region, radius_um, intersections

Usage:
    python3 sholl_curves_control_vs_stress.py \\
        --control "/Volumes/.../Control/Ctrl_metrics/sholl_results/sholl_curves.csv" \\
        --stress  "/Volumes/.../Experimental/Exp_metrics/sholl_results/sholl_curves.csv" \\
        --output  "/Volumes/.../Golgi/sholl_control_vs_stress"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================================
#  KONFIGURATION
# ================================================================

# Region colors (identical to REGION_COLORS in sholl_analysis_golgi.py)
REGION_COLORS = {
    'Cortex':      '#2EC4B6',
    'Hippocampus': '#5DA9E9',
    'Striatum':    '#1f4e9c',
    'Amygdala':    '#6A4C93',
}

# Control vs. Stress are now encoded via line style + SEM-ribbon style
# (not color, since color now encodes brain region):
#   Stress:  solid line,  solid SEM ribbon
#   Control: dashed line, hatched SEM ribbon (same color, diagonal hatch)
GROUP_STYLE = {
    'Control': dict(linestyle='--', hatch='///',  fill_alpha=0.12, line_alpha=0.95),
    'Stress':  dict(linestyle='-',  hatch=None,   fill_alpha=0.22, line_alpha=1.0),
}

REGION_ORDER = ['Cortex', 'Hippocampus', 'Striatum', 'Amygdala']

# Font sizes matched to the enlarged RABIES/Golgi print scale
FONT_AXIS_LABEL   = 20
FONT_TICK_LABEL   = 17
FONT_TITLE        = 22
FONT_SUPTITLE     = 24
FONT_LEGEND       = 15
FONT_LEGEND_TITLE = 16


# ================================================================
#  MEAN +/- SEM (identical logic to sholl_analysis_golgi.py)
# ================================================================

def _mean_sem(g):
    """
    Compute mean +/- SEM per radius using NaN-fill for cells that have
    ended. Only cells that actually have intersections at a given radius
    contribute to the mean -- cells that have ended are treated as NaN
    (not 0 and not forward-filled), preventing artificial plateau
    artifacts in the mean curve.
    """
    all_radii = sorted(g['radius_um'].unique())
    pivot = g.pivot_table(index='radius_um', columns='cell_id',
                          values='intersections', aggfunc='mean')
    pivot = pivot.reindex(all_radii)
    ms = np.array([np.nanmean(pivot.loc[r].values) for r in all_radii])
    ss = np.array([np.nanstd(pivot.loc[r].dropna().values) /
                   np.sqrt(pivot.loc[r].notna().sum())
                   if pivot.loc[r].notna().sum() > 1 else 0
                   for r in all_radii])
    ns = np.array([pivot.loc[r].notna().sum() for r in all_radii])
    return np.array(all_radii), ms, ss, ns


def _plot_mean_sem_line(ax, r, m, s, n, color, label, style):
    """
    Plot mean line + SEM ribbon for one group within one region.
    - color: region color (shared between Control and Stress)
    - style: dict from GROUP_STYLE (linestyle, hatch, fill_alpha, line_alpha)
    """
    ax.plot(r, m, color=color, lw=2.6, label=label,
            linestyle=style['linestyle'], alpha=style['line_alpha'])

    multi = n > 1
    if np.any(multi):
        if style['hatch'] is None:
            # Stress: solid shaded ribbon
            ax.fill_between(r[multi], (m - s)[multi], (m + s)[multi],
                            color=color, alpha=style['fill_alpha'], linewidth=0)
        else:
            # Control: hatched ribbon (same color, diagonal pattern),
            # so it stays visually distinct from the solid Stress ribbon
            # even though both share the region color.
            ax.fill_between(r[multi], (m - s)[multi], (m + s)[multi],
                            facecolor=color, alpha=style['fill_alpha'],
                            hatch=style['hatch'], edgecolor=color, linewidth=0.0)


def _region_order(regions_present):
    return [r for r in REGION_ORDER if r in regions_present] + \
           sorted([r for r in regions_present if r not in REGION_ORDER])


# ================================================================
#  MAIN PLOTTING
# ================================================================

def plot_control_vs_stress_by_region(df, out):
    """
    One panel per region, Control and Stress curves overlaid.
    Saves BOTH a combined multi-panel figure AND one separate PNG/PDF
    per region.
    """
    regions = _region_order(df['region'].unique())

    # -- combined multi-panel version --
    fig, axes = plt.subplots(1, len(regions), figsize=(6.5 * len(regions), 6.5), sharey=False)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        sub_r = df[df['region'] == region]
        color = REGION_COLORS.get(region, '#888888')
        for group in ['Control', 'Stress']:
            sub = sub_r[sub_r['group'] == group]
            if sub.empty:
                continue
            r, m, s, n = _mean_sem(sub)
            _plot_mean_sem_line(ax, r, m, s, n, color, label=group, style=GROUP_STYLE[group])

        ax.set_title(region, color=color, fontweight='bold', fontsize=FONT_TITLE, pad=12)
        ax.set_xlabel('Distance from soma (µm)', fontsize=FONT_AXIS_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK_LABEL)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, title='Group', title_fontsize=FONT_LEGEND_TITLE)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=0.08)
        if ax == axes[0]:
            ax.set_ylabel('Intersections (n)', fontsize=FONT_AXIS_LABEL)

    fig.suptitle('Sholl Analysis — Control vs. Stress by Region\nMean ± SEM',
                 fontsize=FONT_SUPTITLE, y=1.06)
    plt.tight_layout()
    for ext in ['.pdf', '.png']:
        plt.savefig(out / f'sholl_curves_control_vs_stress_combined{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> sholl_curves_control_vs_stress_combined.pdf/.png")

    # -- separate per-region version --
    for region in regions:
        sub_r = df[df['region'] == region]
        color = REGION_COLORS.get(region, '#888888')

        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        for group in ['Control', 'Stress']:
            sub = sub_r[sub_r['group'] == group]
            if sub.empty:
                continue
            r, m, s, n = _mean_sem(sub)
            _plot_mean_sem_line(ax, r, m, s, n, color, label=group, style=GROUP_STYLE[group])

        ax.set_title(f'Sholl Analysis — {region}\nMean ± SEM', color=color, fontweight='bold',
                     fontsize=FONT_TITLE, pad=12)
        ax.set_xlabel('Distance from soma (µm)', fontsize=FONT_AXIS_LABEL)
        ax.set_ylabel('Intersections (n)', fontsize=FONT_AXIS_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK_LABEL)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, title='Group', title_fontsize=FONT_LEGEND_TITLE)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=0.08)
        plt.tight_layout()
        safe_name = region.replace(' ', '_')
        for ext in ['.pdf', '.png']:
            plt.savefig(out / f'sholl_curves_control_vs_stress_{safe_name}{ext}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> sholl_curves_control_vs_stress_{safe_name}.pdf/.png")


# ================================================================
#  MAIN
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Per-region Sholl profile plots: Control vs. Stress overlaid (mean +/- SEM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiel:
  python3 sholl_curves_control_vs_stress.py \\
      --control "/Volumes/PortableSSD/BACHELOR_THESIS/Golgi/Control/Ctrl_metrics/sholl_results/sholl_curves.csv" \\
      --stress  "/Volumes/PortableSSD/BACHELOR_THESIS/Golgi/Experimental/Exp_metrics/sholl_results/sholl_curves.csv" \\
      --output  "/Volumes/PortableSSD/BACHELOR_THESIS/Golgi/sholl_control_vs_stress"
        """
    )
    parser.add_argument('--control', required=True, help='Pfad zur sholl_curves.csv aus dem Control-Lauf')
    parser.add_argument('--stress',  required=True, help='Pfad zur sholl_curves.csv aus dem Stress-Lauf')
    parser.add_argument('--output',  required=True, help='Ausgabe-Ordner')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrl_df = pd.read_csv(args.control)
    ctrl_df['group'] = 'Control'

    stress_df = pd.read_csv(args.stress)
    stress_df['group'] = 'Stress'

    df = pd.concat([ctrl_df, stress_df], ignore_index=True)

    missing_cols = {'cell_id', 'region', 'radius_um', 'intersections'} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Fehlende Spalten in den sholl_curves.csv-Dateien: {missing_cols}")

    print(f"Geladen: {len(ctrl_df)} Control-Zeilen, {len(stress_df)} Stress-Zeilen")
    print(df.groupby(['region', 'group'])['cell_id'].nunique().to_string())
    print()

    plot_control_vs_stress_by_region(df, out_dir)
    print(f"\nFertig. Abbildungen gespeichert in: {out_dir}")
