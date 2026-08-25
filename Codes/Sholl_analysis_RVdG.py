#!/usr/bin/env python3
"""
Sholl Analysis Pipeline for Vaa3D APP2 SWC Reconstructions (RABIES)
===========================================================
Autoren: Sarah (Genopuzzle) + Claude
Datum:   2026-08 (merged version: individual cells + region curves +
         bar-chart parameter comparison, all with enlarged fonts)

Ordnerstruktur erwartet:
    Reconstruction/
        SOC/   <- SWC-Dateien
        EP/
        CON/
        Q/
    Confocal/ oder anderswo: die originalen .tif Dateien
    (Script sucht automatisch nach der passenden .tif)

Verwendung:
    python3 sholl_analysis.py \\
        --input  /Volumes/.../Reconstruction \\
        --output /Volumes/.../Reconstruction/sholl_results \\
        --tifdir /Volumes/.../Confocal          # Ordner mit .tif Dateien

    Falls kein --tifdir: Voxelgroesse aus --voxel-xy angeben (Fallback).

Erzeugte Plots:
    1. sholl_individual_cells.png/.pdf   -- Panel A style (per condition,
       individual cell traces + region means)
    2. sholl_curves_by_region.png/.pdf   -- mean +/- SEM curves per
       condition, colored/split by region
    3. sholl_params_bar_chart.png/.pdf   -- Panel B style (bar chart,
       mean +/- SD, individual points, per condition x region)

Benoete Pakete (einmalig):
    pip install numpy pandas matplotlib scipy tifffile
"""

import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# ================================================================
#  KONFIGURATION
# ================================================================

CONDITION_COLORS = {
    'SOC': '#00C4B4',   # kräftiges Türkis
    'EP':  '#6BB8FF',   # helles Himmelblau
    'CON': '#1A3A8C',   # tiefes Marineblau
    'Q':   '#9B4DCA',   # kräftiges Lila
}
CONDITION_COLORS_DARK = {
    'SOC': '#007A70',
    'EP':  '#2A6FBF',
    'CON': '#0A1F4A',
    'Q':   '#5A1F8A',
}
CONDITION_ORDER = ['SOC', 'EP', 'CON', 'Q']

SHOLL_STEP = 5.0

# Fallback-Voxelgroesse falls keine .tif gefunden wird
DEFAULT_VOXEL_XY = 0.3    # µm/px — typisch fuer 20x Konfokalobjektiv
DEFAULT_VOXEL_Z  = 1.0    # µm/px

VALID_CONDITIONS = {'SOC', 'EP', 'CON', 'Q'}

REGION_KEYWORDS = [
    ('Hypothalamus',      'Hypothalamus'),
    ('NucleusAccumbens',  'Nucleus Accumbens'),
    ('Nucleus_Accumbens', 'Nucleus Accumbens'),
    ('NcA',               'Nucleus Accumbens'),
    ('Thalamus',          'Thalamus'),
    ('Cortex',            'Cortex'),
]

PARAMS = ['N_max', 'r_critical', 'AUC', 'enclosing_r', 'sholl_k']
PARAM_LABELS = {
    'N_max':       'N_max',
    'r_critical':  'r_critical (µm)',
    'AUC':         'AUC',
    'enclosing_r': 'Enclosing radius (µm)',
    'sholl_k':     'Sholl k',
}

# ---- Font sizes (increased across the board per Simon's feedback: "Font too small";
# further substantially increased on request, especially the legend text ----
FONT_AXIS_LABEL   = 24
FONT_TICK_LABEL   = 20
FONT_TITLE        = 26
FONT_SUPTITLE     = 28
FONT_LEGEND       = 20
FONT_LEGEND_TITLE = 21

# ================================================================
#  VOXELGROESSE AUS TIF-METADATEN LESEN
# ================================================================

def read_voxel_size_from_tif(tif_path):
    try:
        import tifffile
    except ImportError:
        return None, None
    try:
        with tifffile.TiffFile(str(tif_path), mode='rb') as tif:
            if tif.imagej_metadata:
                meta = tif.imagej_metadata
                unit = meta.get('unit', 'um')
                factor = 1.0
                if unit in ('nm', 'nanometer'):
                    factor = 0.001
                elif unit in ('mm', 'millimeter'):
                    factor = 1000.0
                spacing = meta.get('spacing', None)
                voxel_z = float(spacing) * factor if spacing else None
                page = tif.pages[0]
                xres = page.tags.get('XResolution')
                if xres:
                    num, den = xres.value
                    res_unit = page.tags.get('ResolutionUnit')
                    res_unit_val = res_unit.value if res_unit else 2
                    if den > 0:
                        px_per_unit = num / den
                        if res_unit_val == 2:
                            voxel_xy = 25400.0 / px_per_unit
                        elif res_unit_val == 3:
                            voxel_xy = 10000.0 / px_per_unit
                        else:
                            voxel_xy = 1.0 / px_per_unit
                        voxel_xy *= factor
                    else:
                        voxel_xy = None
                else:
                    voxel_xy = None
                if voxel_xy and voxel_xy < 0.05:
                    voxel_xy *= 1000.0
                if voxel_xy and voxel_z:
                    return round(voxel_xy, 4), round(voxel_z, 4)
                if voxel_xy:
                    return round(voxel_xy, 4), DEFAULT_VOXEL_Z

            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                px = root.find('.//ome:Pixels', ns)
                if px is not None:
                    vx = float(px.get('PhysicalSizeX', 0))
                    vz = float(px.get('PhysicalSizeZ', 0))
                    if vx > 0:
                        return round(vx, 6), round(vz, 6) if vz > 0 else DEFAULT_VOXEL_Z

            page = tif.pages[0]
            img_desc = page.tags.get('ImageDescription')
            if img_desc:
                desc = str(img_desc.value)
                m = re.search(r'VoxelSizeX["\s:=]+([0-9.eE+-]+)', desc)
                if m:
                    voxel_xy = float(m.group(1))
                    m2 = re.search(r'VoxelSizeZ["\s:=]+([0-9.eE+-]+)', desc)
                    voxel_z = float(m2.group(1)) if m2 else DEFAULT_VOXEL_Z
                    return round(voxel_xy, 6), round(voxel_z, 6)
    except Exception:
        pass
    return None, None


def load_resolution_table(table_path):
    import pandas as pd
    path = Path(table_path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df[df["swc_file"].notna() & (df["swc_file"].astype(str).str.strip() != "")]
    lookup = {}
    for _, row in df.iterrows():
        key = str(row["swc_file"]).strip()
        if key.lower().endswith(".swc"):
            key = key[:-4]
        xy = float(row["xy_resolution_um"]) if pd.notna(row.get("xy_resolution_um")) else None
        z = float(row["z_resolution_um"]) if "z_resolution_um" in row and pd.notna(row.get("z_resolution_um")) else None
        if xy:
            lookup[key] = (xy, z)
    return lookup


def find_tif_for_swc(swc_path, tif_dir=None):
    swc_stem = swc_path.stem
    m = re.search(r'([\w.\-]+\.tif)', swc_stem, re.IGNORECASE)
    tif_name = m.group(1) if m else None
    search_dirs = []
    if tif_dir:
        search_dirs.append(Path(tif_dir))
    search_dirs.extend([swc_path.parent, swc_path.parent.parent, swc_path.parent.parent.parent])
    if tif_name:
        for d in search_dirs:
            matches = list(d.rglob(tif_name))
            if matches:
                return matches[0]
    return None


def get_voxel_size(swc_path, tif_dir=None, res_table=None):
    if res_table:
        stem = Path(swc_path).stem
        if stem in res_table:
            xy, z = res_table[stem]
            return xy, (z if z else DEFAULT_VOXEL_Z), "TABLE"
    tif_path = find_tif_for_swc(swc_path, tif_dir)
    if tif_path and tif_path.exists():
        vxy, vz = read_voxel_size_from_tif(tif_path)
        if vxy:
            return vxy, vz or DEFAULT_VOXEL_Z, str(tif_path.name)
    return DEFAULT_VOXEL_XY, DEFAULT_VOXEL_Z, 'FALLBACK'


# ================================================================
#  SWC PARSING & SHOLL ANALYSIS
# ================================================================

def parse_swc(filepath):
    nodes = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    nodes.append({
                        'index':  int(parts[0]), 'type':   int(parts[1]),
                        'x':      float(parts[2]), 'y':      float(parts[3]),
                        'z':      float(parts[4]), 'radius': float(parts[5]),
                        'parent': int(parts[6]),
                    })
                except ValueError:
                    continue
    if not nodes:
        raise ValueError(f"Keine validen Nodes in {filepath}")
    return pd.DataFrame(nodes)


def get_soma_center(df, voxel_xy, voxel_z):
    soma = df[df['type'] == 1]
    if len(soma) > 0:
        center = soma[['x', 'y', 'z']].mean().values
    else:
        root = df[df['parent'] == -1]
        center = root[['x', 'y', 'z']].iloc[0].values if len(root) > 0 else df[['x', 'y', 'z']].iloc[0].values
    return np.array([center[0] * voxel_xy, center[1] * voxel_xy, center[2] * voxel_z])


def compute_sholl_intersections(df, soma_center, voxel_xy, voxel_z, step=5.0):
    df = df.copy()
    df['x_um'] = df['x'] * voxel_xy
    df['y_um'] = df['y'] * voxel_xy
    df['z_um'] = df['z'] * voxel_z
    coords = df[['x_um', 'y_um', 'z_um']].values
    df['dist'] = np.sqrt(np.sum((coords - soma_center) ** 2, axis=1))
    max_radius = df['dist'].max() * 1.05
    radii = np.arange(step, max_radius + step, step)
    intersections = np.zeros(len(radii), dtype=float)
    dist_lookup = dict(zip(df['index'], df['dist']))
    for _, node in df[df['parent'] != -1].iterrows():
        d_child  = dist_lookup.get(node['index'])
        d_parent = dist_lookup.get(node['parent'])
        if d_child is None or d_parent is None:
            continue
        d_min, d_max = min(d_child, d_parent), max(d_child, d_parent)
        intersections[(radii > d_min) & (radii <= d_max)] += 1
    return radii, intersections


def compute_sholl_params(radii, intersections):
    valid = intersections > 0
    if not np.any(valid):
        return {k: np.nan for k in ['N_max', 'r_critical', 'AUC', 'enclosing_r', 'mean_N', 'sholl_k', 'sholl_r2']}
    params = {
        'N_max':       float(np.max(intersections)),
        'r_critical':  float(radii[np.argmax(intersections)]),
        'AUC':         float(np.trapezoid(intersections, radii)),
        'mean_N':      float(np.mean(intersections[valid])),
        'enclosing_r': float(radii[np.where(valid)[0][-1]]),
    }
    valid_r, valid_n = radii[valid], intersections[valid]
    if len(valid_r) >= 4:
        slope, _, r_val, _, _ = stats.linregress(valid_r, np.log(valid_n))
        params['sholl_k']  = float(slope)
        params['sholl_r2'] = float(r_val ** 2)
    else:
        params['sholl_k'] = params['sholl_r2'] = np.nan
    return params


def load_flagged_set(reconstruction_dir):
    import pandas as pd
    flagged = set()
    recon_dir = Path(reconstruction_dir)
    for csv_path in recon_dir.rglob("Metrics_complete.csv"):
        try:
            df = pd.read_csv(csv_path)
            if "flag_incomplete" in df.columns and "file" in df.columns:
                bad = df[df["flag_incomplete"].astype(str).str.lower() == "true"]["file"]
                for f in bad:
                    stem = str(f).replace(".swc", "").replace(".SWC", "")
                    flagged.add(stem)
        except Exception:
            pass
    return flagged


def parse_meta_from_path(filepath, flagged_set=None):
    path = Path(filepath)
    fname = path.stem
    condition = None
    for parent in path.parents:
        if parent.name.upper() in VALID_CONDITIONS:
            condition = parent.name.upper()
            break
    if condition is None:
        raise ValueError(f"Kein gueltiger Condition-Ordner im Pfad: {filepath}")
    region = None
    for keyword, region_name in REGION_KEYWORDS:
        if keyword.lower() in fname.lower():
            region = region_name
            break
    if region is None:
        print(f"    WARNUNG: Region nicht erkannt in '{fname[:60]}' -> 'Unknown'")
        region = 'Unknown'
    if flagged_set is not None:
        flagged = fname in flagged_set
    else:
        flagged = any(kw in fname.lower() for kw in ['flagged', 'incomplete', 'partial'])
    return condition, region, flagged


# ================================================================
#  HAUPTPIPELINE
# ================================================================

def run_sholl_pipeline(input_dir, output_dir, tif_dir=None, res_table=None):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flagged_set = load_flagged_set(input_dir)
    if flagged_set:
        print(f"📋 Flagged neurons loaded from Metrics_complete.csv: {len(flagged_set)} file(s)")
    else:
        print("ℹ️  No Metrics_complete.csv found — flagged status will be inferred from filename keywords.")

    swc_files = sorted(f for f in input_dir.glob('**/*.swc') if not f.name.startswith('._'))
    if not swc_files:
        swc_files = sorted(f for f in input_dir.glob('**/*.SWC') if not f.name.startswith('._'))
    if not swc_files:
        print(f"Keine .swc Dateien in {input_dir} gefunden!")
        return

    print(f"\n{len(swc_files)} SWC-Dateien gefunden (macOS ._-Dateien ignoriert).\n")

    print("Voxelgroesse-Check (erste 3 Dateien):")
    for swc in swc_files[:3]:
        vxy, vz, src = get_voxel_size(swc, tif_dir, res_table)
        print(f"  {swc.name[:60]}")
        print(f"    XY={vxy} µm/px, Z={vz} µm/px  [Quelle: {src}]")
    print()

    records_curves, records_params, errors = [], [], []
    voxel_log = []

    for swc_path in swc_files:
        fname = swc_path.name
        print(f"  {fname[:70]}...")
        try:
            condition, region, flagged = parse_meta_from_path(swc_path, flagged_set)
        except ValueError as e:
            print(f"    WARNUNG: {e}")
            errors.append(fname)
            continue

        voxel_xy, voxel_z, voxel_src = get_voxel_size(swc_path, tif_dir, res_table)
        voxel_log.append({'cell_id': swc_path.stem, 'voxel_xy_um': voxel_xy, 'voxel_z_um': voxel_z, 'source': voxel_src})

        try:
            df   = parse_swc(swc_path)
            soma = get_soma_center(df, voxel_xy, voxel_z)
            radii, intersections = compute_sholl_intersections(df, soma, voxel_xy, voxel_z, step=SHOLL_STEP)
            params = compute_sholl_params(radii, intersections)
        except Exception as e:
            print(f"    FEHLER: {e}")
            errors.append(fname)
            continue

        print(f"    -> {condition} | {region} | N_max={params['N_max']:.0f} | "
              f"r_crit={params['r_critical']:.0f} µm | vxy={voxel_xy} µm/px [{voxel_src}]")

        cell_id = swc_path.stem
        for r, n in zip(radii, intersections):
            records_curves.append({'cell_id': cell_id, 'condition': condition, 'region': region,
                                    'flagged': flagged, 'radius_um': r, 'intersections': n})
        records_params.append({'cell_id': cell_id, 'condition': condition, 'region': region,
                                'flagged': flagged, 'voxel_xy_um': voxel_xy, 'voxel_z_um': voxel_z, **params})

    if not records_curves:
        print("\nKeine Daten verarbeitet.")
        return

    df_curves = pd.DataFrame(records_curves)
    df_params = pd.DataFrame(records_params)
    df_voxels = pd.DataFrame(voxel_log)

    df_curves.to_csv(output_dir / 'sholl_curves.csv', index=False)
    df_params.to_csv(output_dir / 'sholl_params.csv', index=False)
    df_voxels.to_csv(output_dir / 'voxel_sizes_used.csv', index=False)

    print(f"\nGespeichert in: {output_dir}")
    print(f"  sholl_curves.csv, sholl_params.csv, voxel_sizes_used.csv")
    print("\nZellen pro Bedingung/Region:")
    print(df_params.groupby(['condition', 'region']).size().to_string())

    n_fallback = (df_voxels['source'] == 'FALLBACK').sum()
    if n_fallback > 0:
        print(f"\n  WARNUNG: Fuer {n_fallback} Dateien wurde Fallback-Voxelgroesse ({DEFAULT_VOXEL_XY} µm/px) verwendet.")
        print(f"  Tipp: --resolution-table angeben (empfohlen) oder --tifdir, oder DEFAULT_VOXEL_XY im Script anpassen.")

    if errors:
        print(f"\nFehler bei {len(errors)} Dateien.")

    print("\nErzeuge Plots...")
    plot_sholl_curves_by_region(df_curves, output_dir)
    plot_individual_cells(df_curves, output_dir)
    plot_sholl_params_bar_chart(df_params, output_dir)
    print(f"\nFertig!")


# ================================================================
#  PLOTS
# ================================================================

def _get_color(c):  return CONDITION_COLORS.get(c, '#888888')
def _darken(h, f=0.55):
    h = h.lstrip('#')
    return '#{:02x}{:02x}{:02x}'.format(*[int(max(0, int(h[i:i+2],16)/255*f)*255) for i in (0,2,4)])
def _cond_order(cs):
    o = list(CONDITION_COLORS.keys())
    return sorted(cs, key=lambda c: o.index(c) if c in o else 99)

def _mean_sem(g):
    all_radii = sorted(g['radius_um'].unique())
    pivot = g.pivot_table(index='radius_um', columns='cell_id', values='intersections', aggfunc='mean')
    pivot = pivot.reindex(all_radii)
    ms = np.array([np.nanmean(pivot.loc[r].values) for r in all_radii])
    ss = np.array([np.nanstd(pivot.loc[r].dropna().values) / np.sqrt(pivot.loc[r].notna().sum())
                   if pivot.loc[r].notna().sum() > 1 else 0 for r in all_radii])
    ns = np.array([pivot.loc[r].notna().sum() for r in all_radii])
    return np.array(all_radii), ms, ss, ns

def _plot_mean_sem_line(ax, r, m, s, n, color, label):
    ax.plot(r, m, color=color, lw=2.2, label=label)
    multi = n > 1
    if np.any(multi):
        ax.fill_between(r[multi], (m - s)[multi], (m + s)[multi], color=color, alpha=0.18)


def plot_sholl_curves_by_region(df, out):
    """Mean +/- SEM Sholl curves, one panel per condition, split by region.
    Saves BOTH a combined multi-panel figure AND one separate PNG/PDF per
    condition."""
    conditions = _cond_order(df['condition'].unique())

    # -- combined multi-panel version --
    fig, axes = plt.subplots(1, len(conditions), figsize=(8.0*len(conditions), 8.5), sharey=False)
    if len(conditions)==1: axes=[axes]
    for ax, cond in zip(axes, conditions):
        sub_c = df[df['condition']==cond]; color = _get_color(cond)
        regions = sorted(sub_c['region'].unique())
        colors = [color, _darken(color)] if len(regions)>1 else [color]
        for reg, rc in zip(regions, colors):
            sub = sub_c[sub_c['region']==reg]; r, m, s, n = _mean_sem(sub)
            _plot_mean_sem_line(ax, r, m, s, n, rc, label=reg)
        ax.set_title(cond, color=color, fontweight='bold', fontsize=FONT_TITLE, pad=12)
        ax.set_xlabel('Distance from soma (µm)', fontsize=FONT_AXIS_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK_LABEL)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, loc='upper right')
        ax.spines[['top','right']].set_visible(False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.4)
        if ax==axes[0]: ax.set_ylabel('Intersections (n)', fontsize=FONT_AXIS_LABEL)
    fig.suptitle('Sholl Analysis — Regions\nMean ± SEM', fontsize=FONT_SUPTITLE, y=1.08)
    fig.subplots_adjust(wspace=0.3, top=0.8)
    plt.tight_layout()
    for ext in ['.pdf','.png']: plt.savefig(out/f'sholl_curves_by_region{ext}', dpi=300, bbox_inches='tight')
    plt.close(); print("  -> sholl_curves_by_region.pdf/.png (combined)")

    # -- separate per-condition version --
    for cond in conditions:
        sub_c = df[df['condition']==cond]; color = _get_color(cond)
        regions = sorted(sub_c['region'].unique())
        colors = [color, _darken(color)] if len(regions)>1 else [color]

        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        for reg, rc in zip(regions, colors):
            sub = sub_c[sub_c['region']==reg]; r, m, s, n = _mean_sem(sub)
            _plot_mean_sem_line(ax, r, m, s, n, rc, label=reg)
        ax.set_title(f'Sholl Analysis — {cond}\nMean ± SEM', color=color, fontweight='bold',
                     fontsize=FONT_TITLE, pad=14)
        ax.set_xlabel('Distance from soma (µm)', fontsize=FONT_AXIS_LABEL)
        ax.set_ylabel('Intersections (n)', fontsize=FONT_AXIS_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK_LABEL)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, title='Region', title_fontsize=FONT_LEGEND_TITLE,
                  loc='upper right')
        ax.spines[['top','right']].set_visible(False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.4)
        plt.tight_layout()
        for ext in ['.pdf', '.png']:
            plt.savefig(out / f'sholl_curves_by_region_{cond}{ext}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> sholl_curves_by_region_{cond}.pdf/.png")


def plot_individual_cells(df, out):
    """Panel A style: individual cell traces per condition, split by region, with region means."""
    conds = _cond_order(df['condition'].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(8.0*len(conds), 8.5), sharey=False)
    if len(conds)==1: axes=[axes]

    for ax, cond in zip(axes, conds):
        sub_c = df[df['condition']==cond]
        regions = sorted(sub_c['region'].unique())
        color_main = _get_color(cond)
        region_colors = {regions[0]: color_main}
        if len(regions) > 1:
            region_colors[regions[1]] = _darken(color_main)
        mean_lw = {regions[0]: 3.0}
        if len(regions) > 1:
            mean_lw[regions[1]] = 2.0

        for region in regions:
            rc = region_colors[region]
            sub_r = sub_c[sub_c['region']==region]
            for cell in sub_r['cell_id'].unique():
                sub = sub_r[sub_r['cell_id']==cell]
                ax.plot(sub['radius_um'], sub['intersections'], color=rc, lw=1.1, alpha=0.5, linestyle='-')
            r_reg, m_reg, _, __ = _mean_sem(sub_r)
            ax.plot(r_reg, m_reg, color=rc, lw=mean_lw[region])

        ax.set_title(cond, color=color_main, fontweight='bold', fontsize=FONT_TITLE, pad=12)
        ax.set_xlabel('Distance from soma (µm)', fontsize=FONT_AXIS_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK_LABEL)
        if ax==axes[0]: ax.set_ylabel('Intersections (n)', fontsize=FONT_AXIS_LABEL)
        ax.spines[['top','right']].set_visible(False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.4)

        handles = [plt.Line2D([],[],color=region_colors[reg],lw=mean_lw[reg], label=f'{reg} (mean)')
                   for reg in regions]
        ax.legend(handles=handles, frameon=False, fontsize=FONT_LEGEND, loc='upper right')

    fig.suptitle('Sholl Analysis — Individual Cells', fontsize=FONT_SUPTITLE, y=1.06)
    fig.subplots_adjust(wspace=0.3, top=0.82)
    plt.tight_layout()
    for ext in ['.pdf','.png']:
        plt.savefig(out/f'sholl_individual_cells{ext}', dpi=300, bbox_inches='tight')
    plt.close(); print("  -> sholl_individual_cells.pdf/.png")


def plot_sholl_params_bar_chart(df, out):
    """Panel B style: grouped bar chart, conditions on x-axis, one bar per
    region within each condition, mean +/- SD with individual data points."""
    df = df.copy()
    df['condition'] = pd.Categorical(df['condition'], categories=CONDITION_ORDER, ordered=True)
    conditions = [c for c in CONDITION_ORDER if c in df['condition'].values]
    n_params = len(PARAMS)

    fig, axes = plt.subplots(1, n_params, figsize=(7.5 * n_params, 9.0))
    np.random.seed(42)

    for ax, param in zip(axes, PARAMS):
        x_ticks, x_labels = [], []
        legend_handles = {}
        bar_width = 0.35
        group_gap = 0.9

        for ci, cond in enumerate(conditions):
            sub_c = df[df['condition'] == cond]
            regions = sorted(sub_c['region'].unique())
            n_regions = len(regions)
            offsets = np.linspace(-(n_regions-1)*bar_width/2, (n_regions-1)*bar_width/2, n_regions)
            group_center = ci * group_gap

            for ri, (region, offset) in enumerate(zip(regions, offsets)):
                sub = sub_c[sub_c['region'] == region][param].dropna()
                if sub.empty:
                    continue
                mean, sd = sub.mean(), sub.std()
                color = CONDITION_COLORS[cond] if ri == 0 else CONDITION_COLORS_DARK[cond]

                ax.bar(group_center + offset, mean, width=bar_width * 0.85,
                       color=color, alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
                ax.errorbar(group_center + offset, mean, yerr=sd, fmt='none', color='#333333',
                            capsize=5, capthick=1.4, elinewidth=1.4, zorder=4)

                jitter = np.random.uniform(-0.06, 0.06, size=len(sub))
                ax.scatter(group_center + offset + jitter, sub.values, color='white', s=28,
                           alpha=0.9, edgecolors=color, linewidths=1.0, zorder=5)

                legend_key = f"{cond}_{region}"
                if legend_key not in legend_handles:
                    legend_handles[legend_key] = plt.Rectangle((0,0), 1, 1, fc=color, alpha=0.85,
                                                                label=f"{cond} — {region}")
            x_ticks.append(group_center)
            x_labels.append(cond)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK_LABEL)
        ax.tick_params(axis='y', labelsize=FONT_TICK_LABEL)
        ax.set_ylabel(PARAM_LABELS[param], fontsize=FONT_AXIS_LABEL)
        ax.set_title(PARAM_LABELS[param].split(' (')[0], fontsize=FONT_TITLE, fontweight='bold', pad=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(-0.5, (len(conditions)-1) * group_gap + 0.5)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 2.0)

        if param == 'sholl_k':
            ax.axhline(0, color='#AAAAAA', lw=0.8, ls='--', zorder=1)

        if legend_handles:
            ax.legend(handles=list(legend_handles.values()), frameon=False,
                      fontsize=FONT_LEGEND - 2, loc='upper right')

    fig.suptitle('Sholl Parameters — Mean ± SD\nby condition and brain region',
                 fontsize=FONT_SUPTITLE, y=1.06)
    fig.subplots_adjust(wspace=0.4, top=0.8)
    plt.tight_layout()
    for ext in ['.pdf', '.png']:
        plt.savefig(out / f'sholl_params_bar_chart{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> sholl_params_bar_chart.pdf/.png")


# ================================================================
#  MAIN
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sholl-Analyse fuer Vaa3D APP2 SWC (liest Voxelgroesse automatisch aus TIF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python3 sholl_analysis.py \\
      --input  /Volumes/SSD/BACHELOR_THESIS/Confocal/RABIES/Reconstruction \\
      --output /Volumes/SSD/BACHELOR_THESIS/Confocal/RABIES/Reconstruction/sholl_results \\
      --tifdir /Volumes/SSD/BACHELOR_THESIS/Confocal/RABIES

  python3 sholl_analysis.py \\
      --input  /Volumes/SSD/.../Reconstruction \\
      --output /Volumes/SSD/.../sholl_results \\
      --voxel-xy 0.284 --voxel-z 1.0
        """
    )
    parser.add_argument('--input',    required=True, help='Ordner mit SOC/EP/CON/Q Unterordnern')
    parser.add_argument('--output',   required=True, help='Ausgabe-Ordner')
    parser.add_argument('--tifdir',   default=None,  help='Ordner mit .tif Originaldateien (fuer automatische Voxelgroesse)')
    parser.add_argument('--resolution-table', default=None,
                         help='Excel/CSV-Tabelle mit Spalten swc_file, xy_resolution_um, z_resolution_um')
    parser.add_argument('--step',     type=float, default=SHOLL_STEP, help=f'Radius-Step µm (Standard: {SHOLL_STEP})')
    parser.add_argument('--voxel-xy', type=float, default=DEFAULT_VOXEL_XY, help=f'Fallback XY µm/px (Standard: {DEFAULT_VOXEL_XY})')
    parser.add_argument('--voxel-z',  type=float, default=DEFAULT_VOXEL_Z,  help=f'Fallback Z µm/px (Standard: {DEFAULT_VOXEL_Z})')

    args = parser.parse_args()
    SHOLL_STEP       = args.step
    DEFAULT_VOXEL_XY = args.voxel_xy
    DEFAULT_VOXEL_Z  = args.voxel_z

    res_table = None
    if args.resolution_table:
        res_table = load_resolution_table(args.resolution_table)
        print(f"📋 Resolution table loaded: {len(res_table)} entries from {args.resolution_table}")

    run_sholl_pipeline(args.input, args.output, args.tifdir, res_table)
