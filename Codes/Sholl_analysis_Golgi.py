#!/usr/bin/env python3
"""
Sholl Analysis Pipeline for Golgi-Stained Control Brains (Vaa3D APP2 SWC)
===========================================================
Adapted from the RABIES version for Golgi control-brain data (no
experimental conditions — SOC/EP/CON/Q folders — just brain regions).
Autoren: Sarah (Genopuzzle) + Claude
Datum:   2026-07

Ordnerstruktur erwartet:
    Beliebiger Ordner mit .swc-Dateien (auch in Unterordnern, z.B. pro
    Tier). Region wird direkt aus dem Dateinamen erkannt (Cortex,
    Hippocampus, Striatum, Amygdala) — kein Condition-Unterordner noetig.

Verwendung:
    python3 sholl_analysis_golgi.py \\
        --input  "/Volumes/.../Golgi/Control/Ctrl_metrics" \\
        --output "/Volumes/.../Golgi/Control/Ctrl_metrics/sholl_results" \\
        --xy 0.1625 --z 0.2508

    Empfohlen: --resolution-table statt --xy/--z, falls die Aufloesung
    pro Datei variiert (zuverlaessiger als ein fixer Wert fuer alle Files).

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

REGION_COLORS = {
    'Cortex':      '#2EC4B6',
    'Hippocampus': '#5DA9E9',
    'Striatum':    '#1f4e9c',
    'Amygdala':    '#6A4C93',
}

SHOLL_STEP = 5.0

# Fallback-Voxelgroesse falls keine .tif gefunden wird
# (wird ueberschrieben wenn .tif-Metadaten lesbar sind)
DEFAULT_VOXEL_XY = 0.1625  # µm/px — typisch fuer Golgi-Datensatz dieser Arbeit
DEFAULT_VOXEL_Z  = 0.2508  # µm/px

# Region-Keywords im Dateinamen
REGION_KEYWORDS = [
    ('Hippocampus', 'Hippocampus'),
    ('Striatum',    'Striatum'),
    ('Amygdala',    'Amygdala'),
    ('Cortex',      'Cortex'),
]

# ================================================================
#  VOXELGROESSE AUS TIF-METADATEN LESEN
# ================================================================

def read_voxel_size_from_tif(tif_path):
    """
    Voxelgroesse aus TIFF-Metadaten lesen.
    Unterstuetzt: ImageJ-TIFF, OME-TIFF, Leica/Zeiss/Nikon (via tifffile).
    Gibt (voxel_xy, voxel_z) in µm zurueck.
    """
    try:
        import tifffile
    except ImportError:
        return None, None

    try:
        with tifffile.TiffFile(str(tif_path), mode='rb') as tif:

            # 1. ImageJ-Metadaten (haeufigste Variante bei Konfokaldaten)
            if tif.imagej_metadata:
                meta = tif.imagej_metadata
                # Einheit pruefen
                unit = meta.get('unit', 'um')
                factor = 1.0
                if unit in ('nm', 'nanometer'):
                    factor = 0.001
                elif unit in ('mm', 'millimeter'):
                    factor = 1000.0

                spacing = meta.get('spacing', None)  # Z-Abstand
                voxel_z = float(spacing) * factor if spacing else None

                # XY aus TIFF-Resolution-Tags
                page = tif.pages[0]
                xres = page.tags.get('XResolution')
                if xres:
                    num, den = xres.value
                    res_unit = page.tags.get('ResolutionUnit')
                    res_unit_val = res_unit.value if res_unit else 2  # 2=inch, 3=cm
                    if den > 0:
                        px_per_unit = num / den
                        if res_unit_val == 2:    # inch -> µm
                            voxel_xy = 25400.0 / px_per_unit
                        elif res_unit_val == 3:  # cm -> µm
                            voxel_xy = 10000.0 / px_per_unit
                        else:
                            voxel_xy = 1.0 / px_per_unit
                        voxel_xy *= factor
                    else:
                        voxel_xy = None
                else:
                    voxel_xy = None

                # Leica/Zeiss speichern XResolution oft als px/mm statt px/inch
                # → Ergebnis waere in mm/px statt µm/px → ×1000 korrigieren
                if voxel_xy and voxel_xy < 0.05:
                    voxel_xy *= 1000.0

                if voxel_xy and voxel_z:
                    return round(voxel_xy, 4), round(voxel_z, 4)
                if voxel_xy:
                    return round(voxel_xy, 4), DEFAULT_VOXEL_Z

            # 2. OME-TIFF (Zeiss, Leica neuere Versionen)
            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                px = root.find('.//ome:Pixels', ns)
                if px is not None:
                    vx = float(px.get('PhysicalSizeX', 0))
                    vz = float(px.get('PhysicalSizeZ', 0))
                    unit_x = px.get('PhysicalSizeXUnit', 'µm')
                    # Nur µm unterstuetzt (Standard bei Konfokaldaten)
                    if vx > 0:
                        return round(vx, 6), round(vz, 6) if vz > 0 else DEFAULT_VOXEL_Z

            # 3. Leica LIF-basierte TIFFs: Metadaten im ImageDescription-Tag
            page = tif.pages[0]
            img_desc = page.tags.get('ImageDescription')
            if img_desc:
                desc = str(img_desc.value)
                # Versuche µm/px aus beschreibendem Text zu lesen
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
    """
    Load a resolution-matching table (xlsx or csv) with columns:
    swc_file, xy_resolution_um, z_resolution_um (z optional).
    Returns a dict: {swc_filename_stem: (xy, z)}.
    """
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
    """
    Findet die passende .tif-Datei fuer eine SWC-Datei.

    Strategie:
    1. Tif-Dateiname ist im SWC-Namen eingebettet
       z.B. RABIES_CON_Q_Cortex_24.06_Cortex2.tif_x866_y428_z12_app2.swc
            -> suche nach 'Cortex2.tif'
    2. Falls nicht gefunden: suche nach aehnlichen .tif-Dateien im tif_dir
    """
    swc_stem = swc_path.stem  # z.B. RABIES_CON_Q_Cortex_24.06_Cortex2.tif_x866_y428_z12_app2

    # Tif-Dateiname aus SWC-Namen extrahieren (Pattern: xxxxx.tif_x...)
    m = re.search(r'([\w.\-]+\.tif)', swc_stem, re.IGNORECASE)
    tif_name = m.group(1) if m else None

    search_dirs = []
    if tif_dir:
        search_dirs.append(Path(tif_dir))
    # Auch im gleichen Ordner und Elternordnern suchen
    search_dirs.extend([swc_path.parent, swc_path.parent.parent,
                        swc_path.parent.parent.parent])

    if tif_name:
        for d in search_dirs:
            # Rekursiv suchen
            matches = list(d.rglob(tif_name))
            if matches:
                return matches[0]

    return None


def get_voxel_size(swc_path, tif_dir=None, res_table=None, xy_override=None, z_override=None):
    """
    Voxelgroesse fuer eine SWC-Datei ermitteln.
    Prioritaet: manueller Override (--voxel-xy/--voxel-z, wenn vom Nutzer
    explizit angegeben) > resolution table > matching .tif metadata >
    hartcodierter Fallback. Diese Reihenfolge entspricht swc_metrics.py:
    ein manuell angegebener Wert soll IMMER Vorrang haben, auch wenn eine
    (evtl. fehlerhafte) TIF-Metadaten-Aufloesung gefunden wird.
    """
    if xy_override is not None:
        return xy_override, (z_override if z_override is not None else DEFAULT_VOXEL_Z), "MANUAL"

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
    """SWC-Datei einlesen. Unterstuetzt Vaa3D APP2 (Typ 274) und Standard."""
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
                        'index':  int(parts[0]),
                        'type':   int(parts[1]),
                        'x':      float(parts[2]),
                        'y':      float(parts[3]),
                        'z':      float(parts[4]),
                        'radius': float(parts[5]),
                        'parent': int(parts[6]),
                    })
                except ValueError:
                    continue
    if not nodes:
        raise ValueError(f"Keine validen Nodes in {filepath}")
    return pd.DataFrame(nodes)


def get_soma_center(df, voxel_xy, voxel_z):
    """Root-Node (parent=-1) als Soma verwenden, Voxelgroesse anwenden."""
    soma = df[df['type'] == 1]
    if len(soma) > 0:
        center = soma[['x', 'y', 'z']].mean().values
    else:
        root = df[df['parent'] == -1]
        center = root[['x', 'y', 'z']].iloc[0].values if len(root) > 0 \
                 else df[['x', 'y', 'z']].iloc[0].values
    return np.array([center[0] * voxel_xy, center[1] * voxel_xy, center[2] * voxel_z])


def compute_sholl_intersections(df, soma_center, voxel_xy, voxel_z, step=5.0):
    """
    Sholl-Analyse via Segment-Crossing-Methode.
    Alle Nicht-Root-Nodes werden als Dendriten behandelt (Vaa3D APP2).
    """
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
    """Sholl-Parameter: N_max, r_critical, AUC, enclosing_r, mean_N, sholl_k, sholl_r2."""
    valid = intersections > 0
    if not np.any(valid):
        return {k: np.nan for k in
                ['N_max', 'r_critical', 'AUC', 'enclosing_r', 'mean_N', 'sholl_k', 'sholl_r2']}

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


def load_flagged_set(reconstruction_dir, metrics_csv=None):
    """
    Load the set of SWC filenames flagged as potentially incomplete
    (flag_incomplete == True), so the Sholl analysis uses the SAME
    flagged status as swc_metrics.py for the same underlying neurons.

    Priority:
      1. If metrics_csv is given, read flag_incomplete directly from
         that file (most reliable -- e.g. your merged
         controls_all_merged.csv / exp_all_merged.csv).
      2. Otherwise, search reconstruction_dir recursively for any CSV
         whose name contains "metrics" and "complete" (case-insensitive),
         covering naming variants like "Ctrl5_metrics_complete.csv" as
         well as the original "Metrics_complete.csv".

    Returns a set of SWC filename stems (no .swc extension) that are
    flagged.
    """
    import pandas as pd
    flagged = set()

    def _extract_flagged(df, source_name):
        if "flag_incomplete" not in df.columns or "file" not in df.columns:
            print(f"  WARNUNG: '{source_name}' hat keine 'flag_incomplete'/'file'-Spalten -- uebersprungen.")
            return
        bad = df[df["flag_incomplete"].astype(str).str.strip().str.lower() == "true"]["file"]
        for f in bad:
            stem = str(f).replace(".swc", "").replace(".SWC", "")
            flagged.add(stem)

    if metrics_csv:
        csv_path = Path(metrics_csv)
        if not csv_path.exists():
            print(f"  WARNUNG: --metrics-csv Datei nicht gefunden: {csv_path}")
        else:
            try:
                df = pd.read_csv(csv_path)
                _extract_flagged(df, csv_path.name)
                print(f"Flagged status loaded from --metrics-csv ({csv_path.name}): "
                      f"{len(flagged)} flagged neuron(s) found.")
            except Exception as e:
                print(f"  WARNUNG: Konnte --metrics-csv nicht lesen: {e}")
        return flagged

    recon_dir = Path(reconstruction_dir)
    for csv_path in recon_dir.rglob("*.csv"):
        name_lower = csv_path.name.lower()
        if "metrics" in name_lower and "complete" in name_lower:
            try:
                df = pd.read_csv(csv_path)
                _extract_flagged(df, csv_path.name)
            except Exception:
                pass
    return flagged


def parse_meta_from_path(filepath, flagged_set=None):
    """Region aus Dateinamen erkennen. Kein Condition-Ordner noetig (nur Controls)."""
    path = Path(filepath)
    fname = path.stem
    region = None
    for keyword, region_name in REGION_KEYWORDS:
        if keyword.lower() in fname.lower():
            region = region_name
            break
    if region is None:
        print(f"    WARNUNG: Region nicht erkannt in '{fname[:60]}' -> 'Unknown'")
        region = 'Unknown'
    # Animal-ID aus Dateinamen (z.B. 'Ctrl5.2', 'Ctrl9') fuer Referenz/spaetere Gruppierung
    m = re.match(r'([Cc]trl\d+(?:\.\d+)?)', fname)
    animal = m.group(1) if m else 'Unknown'
    if flagged_set is not None:
        flagged = fname in flagged_set
    else:
        flagged = any(kw in fname.lower() for kw in ['flagged', 'incomplete', 'partial'])
    return region, animal, flagged


# ================================================================
#  HAUPTPIPELINE
# ================================================================

def run_sholl_pipeline(input_dir, output_dir, tif_dir=None, res_table=None, xy_override=None, z_override=None, group_label='Control Brains', metrics_csv=None):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load flagged neurons from Metrics_complete.csv files (more reliable than filename keywords)
    flagged_set = load_flagged_set(input_dir, metrics_csv)
    if flagged_set:
        print(f"📋 Flagged neurons loaded from Metrics_complete.csv: {len(flagged_set)} file(s)")
    else:
        print("ℹ️  No Metrics_complete.csv found — flagged status will be inferred from filename keywords.")

    # macOS erstellt versteckte "._"-Dateien (Resource Forks) — diese ausfiltern
    swc_files = sorted(
        f for f in input_dir.glob('**/*.swc')
        if not f.name.startswith('._')
    )
    if not swc_files:
        swc_files = sorted(
            f for f in input_dir.glob('**/*.SWC')
            if not f.name.startswith('._')
        )
    if not swc_files:
        print(f"Keine .swc Dateien in {input_dir} gefunden!")
        return

    print(f"\n{len(swc_files)} SWC-Dateien gefunden (macOS ._-Dateien ignoriert).\n")

    # Kurze Voxelgroessen-Vorschau fuer erste 3 Dateien
    print("Voxelgroesse-Check (erste 3 Dateien):")
    for swc in swc_files[:3]:
        vxy, vz, src = get_voxel_size(swc, tif_dir, res_table, xy_override, z_override)
        print(f"  {swc.name[:60]}")
        print(f"    XY={vxy} µm/px, Z={vz} µm/px  [Quelle: {src}]")
    print()

    records_curves, records_params, errors = [], [], []
    voxel_log = []

    for swc_path in swc_files:
        fname = swc_path.name
        print(f"  {fname[:70]}...")

        try:
            region, animal, flagged = parse_meta_from_path(swc_path, flagged_set)
        except ValueError as e:
            print(f"    WARNUNG: {e}")
            errors.append(fname)
            continue

        voxel_xy, voxel_z, voxel_src = get_voxel_size(swc_path, tif_dir, res_table, xy_override, z_override)
        voxel_log.append({
            'cell_id': swc_path.stem,
            'voxel_xy_um': voxel_xy,
            'voxel_z_um':  voxel_z,
            'source':      voxel_src,
        })

        try:
            df   = parse_swc(swc_path)
            soma = get_soma_center(df, voxel_xy, voxel_z)
            radii, intersections = compute_sholl_intersections(
                df, soma, voxel_xy, voxel_z, step=SHOLL_STEP)
            params = compute_sholl_params(radii, intersections)
        except Exception as e:
            print(f"    FEHLER: {e}")
            errors.append(fname)
            continue

        print(f"    -> {region} | {animal} | "
              f"N_max={params['N_max']:.0f} | r_crit={params['r_critical']:.0f} µm | "
              f"vxy={voxel_xy} µm/px [{voxel_src}]")

        cell_id = swc_path.stem
        for r, n in zip(radii, intersections):
            records_curves.append({
                'cell_id': cell_id, 'region': region,
                'animal': animal, 'flagged': flagged,
                'radius_um': r, 'intersections': n,
            })
        records_params.append({
            'cell_id': cell_id, 'region': region,
            'animal': animal, 'flagged': flagged,
            'voxel_xy_um': voxel_xy, 'voxel_z_um': voxel_z,
            **params,
        })

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
    print("\nZellen pro Region:")
    print(df_params.groupby(['region']).size().to_string())

    # Warnung falls viele Fallbacks
    n_fallback = (df_voxels['source'] == 'FALLBACK').sum()
    if n_fallback > 0:
        print(f"\n  WARNUNG: Fuer {n_fallback} Dateien wurde Fallback-Voxelgroesse "
              f"({DEFAULT_VOXEL_XY} µm/px) verwendet.")
        print(f"  Tipp: --resolution-table angeben (empfohlen) oder --tifdir, "
              f"oder DEFAULT_VOXEL_XY im Script anpassen.")

    if errors:
        print(f"\nFehler bei {len(errors)} Dateien.")

    print("\nErzeuge Plots...")
    plot_sholl_curves_by_region(df_curves, output_dir, group_label)
    plot_sholl_curves_by_region_separate(df_curves, output_dir)
    plot_sholl_params(df_params, output_dir, group_label)
    plot_individual_cells(df_curves, output_dir, group_label)
    print(f"\nFertig!")


# ================================================================
#  PLOTS
# ================================================================

def _get_color(r):  return REGION_COLORS.get(r, '#888888')
def _region_order(rs):
    o = list(REGION_COLORS.keys())
    return sorted(rs, key=lambda r: o.index(r) if r in o else 99)
def _mean_sem(g):
    """
    Compute mean ± SEM per radius using NaN-fill for cells that have ended.
    Only cells that actually have intersections at a given radius contribute
    to the mean — cells that have ended are treated as NaN (not 0 and not
    forward-filled), preventing artificial plateau artifacts in the mean curve.
    """
    import pandas as pd
    all_radii = sorted(g['radius_um'].unique())
    pivot = g.pivot_table(index='radius_um', columns='cell_id',
                          values='intersections', aggfunc='mean')
    # Reindex to full radius grid — missing entries become NaN (not filled)
    pivot = pivot.reindex(all_radii)
    ms = np.array([np.nanmean(pivot.loc[r].values) for r in all_radii])
    ss = np.array([np.nanstd(pivot.loc[r].dropna().values) /
                   np.sqrt(pivot.loc[r].notna().sum())
                   if pivot.loc[r].notna().sum() > 1 else 0
                   for r in all_radii])
    ns = np.array([pivot.loc[r].notna().sum() for r in all_radii])
    return np.array(all_radii), ms, ss, ns

def _plot_mean_sem_line(ax, r, m, s, n, color, label):
    """
    Plot mean line + SEM ribbon.
    - Continuous solid line throughout (no gaps)
    - SEM ribbon only where n > 1 (statistically valid)
    - Where n == 1: line only, no ribbon (SEM undefined for single observation)
    """
    # Full continuous mean line
    ax.plot(r, m, color=color, lw=2.0, label=label)

    # SEM ribbon only where n > 1
    multi = n > 1
    if np.any(multi):
        ax.fill_between(r[multi], (m - s)[multi], (m + s)[multi],
                        color=color, alpha=0.18)

def plot_sholl_curves_by_region(df, out, group_label='Control Brains'):
    """Single plot: mean +/- SEM Sholl curve per region, all four regions
    overlaid for direct comparison (replaces the RABIES per-condition
    paired-region panels, since Golgi controls have no conditions)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for region in _region_order(df['region'].unique()):
        sub = df[df['region'] == region]
        color = _get_color(region)
        r, m, s, n = _mean_sem(sub)
        _plot_mean_sem_line(ax, r, m, s, n, color, label=region)
    ax.set_xlabel('Distance from soma (\u00b5m)', fontsize=12)
    ax.set_ylabel('Intersections (n)', fontsize=12)
    ax.set_title(f'Sholl Analysis \u2014 {group_label} by Region\nMean \u00b1 SEM', fontsize=13)
    ax.legend(title='Region', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    for ext in ['.pdf', '.png']:
        plt.savefig(out / f'sholl_curves_by_region{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print("  -> sholl_curves_by_region.pdf/.png")


def plot_sholl_curves_by_region_separate(df, out):
    """Save ONE separate PNG/PDF per region (mean +/- SEM), in addition to
    the combined multi-region overview plot."""
    for region in _region_order(df['region'].unique()):
        sub = df[df['region'] == region]
        color = _get_color(region)
        r, m, s, n = _mean_sem(sub)

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_mean_sem_line(ax, r, m, s, n, color, label=region)
        ax.set_xlabel('Distance from soma (µm)', fontsize=11)
        ax.set_ylabel('Intersections (n)', fontsize=11)
        ax.set_title(f'Sholl Analysis — {region}\nMean ± SEM', color=color, fontweight='bold', fontsize=13)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        safe_name = region.replace(' ', '_')
        for ext in ['.pdf', '.png']:
            plt.savefig(out / f'sholl_curves_{safe_name}{ext}', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> sholl_curves_{safe_name}.pdf/.png")


def plot_sholl_params(df, out, group_label='Control Brains'):
    params = [('N_max','N_max'),('r_critical','r_c (µm)'),
              ('AUC','AUC'),('enclosing_r','Enclosing r (µm)'),('sholl_k','Sholl k')]
    regions = _region_order(df['region'].unique())
    fig, axes = plt.subplots(1, len(params), figsize=(3.8*len(params), 5))
    np.random.seed(42)
    for ax, (param, label) in zip(axes, params):
        for i, region in enumerate(regions):
            color = _get_color(region)
            all_v = df[df['region']==region][param].dropna()
            if len(all_v)==0:
                continue

            mean = all_v.mean()
            sd = all_v.std()

            ax.bar(i, mean, width=0.6, color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.errorbar(i, mean, yerr=sd, fmt='none', color='#333333',
                        capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)

            # individual data points on top of the bar, flagged shown as
            # open/outlined markers instead of filled white ones
            comp = df[(df['region']==region)&(~df['flagged'])][param].dropna()
            flag = df[(df['region']==region)&(df['flagged'])][param].dropna()
            if len(comp) > 0:
                jit = np.random.uniform(-0.15, 0.15, len(comp))
                ax.scatter(i + jit, comp, color='white', s=22, alpha=0.9,
                           edgecolors=color, linewidths=0.9, zorder=5)
            if len(flag) > 0:
                jit = np.random.uniform(-0.15, 0.15, len(flag))
                ax.scatter(i + jit, flag, facecolors='none', edgecolors=color,
                           s=22, alpha=0.9, linewidths=1.4, zorder=5)

        if param == 'sholl_k':
            ax.axhline(0, color='#AAAAAA', lw=0.8, ls='--', zorder=1)

        ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, fontsize=9, rotation=15)
        ax.set_ylabel(label, fontsize=9); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Sholl Parameters — {group_label} by Region\nMean ± SD (open markers = flagged)', fontsize=12)
    plt.tight_layout()
    for ext in ['.pdf','.png']: plt.savefig(out/f'sholl_params_by_region{ext}', dpi=200, bbox_inches='tight')
    plt.close(); print("  -> sholl_params_by_region.pdf/.png")

def plot_individual_cells(df, out, group_label='Control Brains'):
    """
    Individual Sholl curves per region (one panel per region), showing
    every traced cell plus the region mean curve. Simpler than the RABIES
    version since Golgi controls have no experimental conditions to facet
    by — one panel per brain region is sufficient.
    """
    regions = _region_order(df['region'].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(4.0*len(regions), 4.5), sharey=False)
    if len(regions)==1: axes=[axes]

    for ax, region in zip(axes, regions):
        color = _get_color(region)
        sub_r = df[df['region']==region]
        for cell in sub_r['cell_id'].unique():
            sub = sub_r[sub_r['cell_id']==cell]
            ax.plot(sub['radius_um'], sub['intersections'],
                    color=color, lw=0.9, alpha=0.5, linestyle='-')

        r_reg, m_reg, _, __ = _mean_sem(sub_r)
        ax.plot(r_reg, m_reg, color=color, lw=2.5, label=f'{region} (mean)')

        ax.set_title(region, color=color, fontweight='bold', fontsize=12)
        ax.set_xlabel('Distance from soma (µm)', fontsize=9)
        if ax==axes[0]: ax.set_ylabel('Intersections (n)', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f'Sholl Analysis — Individual Cells ({group_label})', fontsize=12)
    plt.tight_layout()
    for ext in ['.pdf','.png']:
        plt.savefig(out/f'sholl_individual_cells{ext}', dpi=200, bbox_inches='tight')
    plt.close(); print("  -> sholl_individual_cells.pdf/.png")


# ================================================================
#  MAIN
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sholl-Analyse fuer Golgi-Kontrollhirne (Vaa3D APP2 SWC, region-basiert, keine Conditions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Mit manueller Voxelgroesse (wie bei den anderen Golgi-Skripten):
  python3 sholl_analysis_golgi.py \\
      --input  "/Volumes/PortableSSD/BACHELOR_THESIS/Golgi/Control/Ctrl_metrics" \\
      --output "/Volumes/PortableSSD/BACHELOR_THESIS/Golgi/Control/Ctrl_metrics/sholl_results" \\
      --voxel-xy 0.1625 --voxel-z 0.2508

  # Mit Resolution-Tabelle (empfohlen, falls Aufloesung pro Datei variiert):
  python3 sholl_analysis_golgi.py \\
      --input  "/Volumes/PortableSSD/.../Ctrl_metrics" \\
      --output "/Volumes/PortableSSD/.../Ctrl_metrics/sholl_results" \\
      --resolution-table resolution_table.xlsx
        """
    )
    parser.add_argument('--input',    required=True, help='Ordner mit .swc-Dateien (rekursiv durchsucht, keine Unterordner-Struktur noetig)')
    parser.add_argument('--output',   required=True, help='Ausgabe-Ordner')
    parser.add_argument('--tifdir',   default=None,  help='Ordner mit .tif Originaldateien (fuer automatische Voxelgroesse)')
    parser.add_argument('--resolution-table', default=None,
                         help='Excel/CSV-Tabelle mit Spalten swc_file, xy_resolution_um, '
                              'z_resolution_um (empfohlen, zuverlaessiger als --tifdir)')
    parser.add_argument('--metrics-csv', default=None,
                         help='Pfad zur swc_metrics.py-Ergebnis-CSV (z.B. controls_all_merged.csv), '
                              'aus der die flag_incomplete-Spalte gelesen wird, damit dieselben Neurone '
                              'in Sholl-Analyse und Morphologie-Metriken konsistent als geflaggt gelten. '
                              'Falls nicht angegeben, wird versucht, automatisch eine passende Datei '
                              '(Name enthaelt "metrics" und "complete") unter --input zu finden.')
    parser.add_argument('--step',     type=float, default=SHOLL_STEP, help=f'Radius-Step µm (Standard: {SHOLL_STEP})')
    parser.add_argument('--xy', type=float, default=None,
                         help='MANUELLER Override der XY-Aufloesung (µm/px) fuer ALLE Dateien -- hat '
                              'IMMER Vorrang vor Resolution-Tabelle und TIF-Metadaten (wie bei '
                              'swc_metrics.py). Nutze dies, wenn du dir sicher bist, welche Aufloesung '
                              'stimmt, und TIF-Metadaten unzuverlaessig sind.')
    parser.add_argument('--z', type=float, default=None,
                         help='MANUELLER Override der Z-Aufloesung (µm/px) fuer ALLE Dateien, siehe --xy.')
    parser.add_argument('--voxel-xy', type=float, default=DEFAULT_VOXEL_XY,
                         help=f'Letzter Fallback-Wert XY (µm/px), NUR falls weder --xy, Resolution-Tabelle '
                              f'noch TIF-Metadaten verfuegbar sind (Standard: {DEFAULT_VOXEL_XY})')
    parser.add_argument('--voxel-z',  type=float, default=DEFAULT_VOXEL_Z,
                         help=f'Letzter Fallback-Wert Z (µm/px), siehe --voxel-xy (Standard: {DEFAULT_VOXEL_Z})')
    parser.add_argument('--group-label', type=str, default='Control Brains',
                         help='Label used in plot titles (default: "Control Brains"). '
                              'Set to "Stress Brains" (or similar) when running on the '
                              'Experimental dataset instead of Controls.')

    args = parser.parse_args()
    SHOLL_STEP       = args.step
    DEFAULT_VOXEL_XY = args.voxel_xy
    DEFAULT_VOXEL_Z  = args.voxel_z

    res_table = None
    if args.resolution_table:
        res_table = load_resolution_table(args.resolution_table)
        print(f"📋 Resolution table loaded: {len(res_table)} entries from {args.resolution_table}")

    if args.xy is not None:
        print(f"⚠️  Manual override active: ALL files will use XY={args.xy} µm/px"
              f"{f', Z={args.z} µm/px' if args.z is not None else ''} "
              f"(ignoring TIF metadata and resolution table).")

    run_sholl_pipeline(args.input, args.output, args.tifdir, res_table, args.xy, args.z, args.group_label, args.metrics_csv)
