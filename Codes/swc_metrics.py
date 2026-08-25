"""
swc_metrics.py — Morphological metrics from Vaa3D APP2 .swc files
====================================================================
Reads the correct, per-file XY pixel resolution directly from the
matching TIF file's metadata (instead of a single fixed default),
since zoom factor was adjusted individually per neuron during
acquisition. Z resolution defaults to a fixed value (Z-step was
constant across acquisitions) but can be overridden.

Usage:
    python3 swc_metrics.py file.swc --tifdir /path/to/tifs
    python3 swc_metrics.py folder/ --tifdir /path/to/tifs --csv
    python3 swc_metrics.py folder/ --xy 0.13 --z 2.0 --csv   # manual override, all files

Developed with the assistance of Claude (Anthropic, claude.ai,
accessed June 2026).
"""

import numpy as np
import sys
import os
import re
import csv
import argparse
from pathlib import Path

# ── default Z resolution (µm/slice); constant Z-step across acquisitions ────
DEFAULT_Z = 2.0
# fallback XY only used if no matching TIF is found and no --xy override given
FALLBACK_XY = 0.078


def parse_swc(path):
    nodes, children = {}, {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            n, t, x, y, z, r, p = (int(parts[0]), int(parts[1]),
                float(parts[2]), float(parts[3]), float(parts[4]),
                float(parts[5]), int(parts[6]))
            nodes[n] = (x, y, z, r, p)
            children.setdefault(p, []).append(n)
    return nodes, children


# ── voxel size reading (adapted from sholl_analysis.py) ─────────────────────

def read_voxel_size_from_tif(tif_path):
    """Read XY/Z voxel size (µm) from TIFF metadata. Returns (xy, z) or (None, None)."""
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
                voxel_xy = None
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

                if voxel_xy and voxel_xy < 0.05:
                    voxel_xy *= 1000.0

                if voxel_xy and voxel_z:
                    return round(voxel_xy, 4), round(voxel_z, 4)
                if voxel_xy:
                    return round(voxel_xy, 4), None

            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                px = root.find('.//ome:Pixels', ns)
                if px is not None:
                    vx = float(px.get('PhysicalSizeX', 0))
                    vz = float(px.get('PhysicalSizeZ', 0))
                    if vx > 0:
                        return round(vx, 6), round(vz, 6) if vz > 0 else None

            page = tif.pages[0]
            img_desc = page.tags.get('ImageDescription')
            if img_desc:
                desc = str(img_desc.value)
                m = re.search(r'VoxelSizeX["\s:=]+([0-9.eE+-]+)', desc)
                if m:
                    voxel_xy = float(m.group(1))
                    m2 = re.search(r'VoxelSizeZ["\s:=]+([0-9.eE+-]+)', desc)
                    voxel_z = float(m2.group(1)) if m2 else None
                    return round(voxel_xy, 6), round(voxel_z, 6) if voxel_z else None

    except Exception:
        pass

    return None, None


def find_tif_for_swc(swc_path, tif_dir):
    """
    Find the matching TIF for an SWC file, embedded in the SWC filename
    (pattern: somename.tif_x123_y456...). Searches recursively under
    tif_dir, explicitly skipping macOS AppleDouble metadata files
    (filenames starting with '._').
    """
    swc_stem = swc_path.stem
    m = re.search(r'([\w.\-]+\.tif)', swc_stem, re.IGNORECASE)
    tif_name = m.group(1) if m else None
    if not tif_name or not tif_dir:
        return None

    tif_dir = Path(tif_dir)
    matches = [p for p in tif_dir.rglob(tif_name) if not p.name.startswith('._')]
    return matches[0] if matches else None


def load_resolution_table(table_path):
    """
    Load a resolution-matching table (xlsx or csv) with columns:
    swc_file, xy_resolution_um, z_resolution_um (z optional).
    Returns a dict: {swc_filename_stem: (xy, z)}.
    SWC filenames are matched by stem (without .swc extension) to be
    robust to minor naming differences.
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


def get_voxel_size(swc_path, tif_dir, xy_override, z_override, res_table):
    """
    Resolve (xy, z, source_label) for a given SWC file.
    Priority: explicit --xy/--z override > resolution table match
              > matching TIF metadata > fallback.
    """
    if xy_override is not None:
        return xy_override, (z_override if z_override is not None else DEFAULT_Z), "MANUAL"

    if res_table:
        stem = Path(swc_path).stem
        if stem in res_table:
            xy, z = res_table[stem]
            return xy, (z if z else (z_override if z_override is not None else DEFAULT_Z)), "TABLE"

    tif_path = find_tif_for_swc(swc_path, tif_dir) if tif_dir else None
    if tif_path and tif_path.exists():
        vxy, vz = read_voxel_size_from_tif(tif_path)
        if vxy:
            return vxy, (vz if vz else (z_override if z_override is not None else DEFAULT_Z)), tif_path.name

    return FALLBACK_XY, (z_override if z_override is not None else DEFAULT_Z), "FALLBACK"


# ── metric computation (unchanged logic, validated previously) ──────────────

def compute_metrics(nodes, children, xy_res, z_res):
    roots = [nid for nid, (x, y, z, r, p) in nodes.items() if p == -1]
    root = roots[0] if roots else None

    tdl = 0
    for nid, (x, y, z, r, p) in nodes.items():
        if p == -1 or p not in nodes:
            continue
        px, py, pz, _, _ = nodes[p]
        tdl += np.sqrt(((x - px) * xy_res) ** 2 + ((y - py) * xy_res) ** 2 + ((z - pz) * z_res) ** 2)

    primaries = len(children.get(root, [])) if root is not None else 0
    branch_points = sum(1 for nid, kids in children.items() if len(kids) > 1 and nid != -1)
    tips = sum(1 for nid in nodes if nid not in children)

    max_order = 0
    if root is not None:
        stack = [(root, 0)]
        while stack:
            nid, order = stack.pop()
            max_order = max(max_order, order)
            kids = children.get(nid, [])
            is_branch = len(kids) > 1
            next_order = order + 1 if is_branch else order
            for k in kids:
                stack.append((k, next_order))

    return {
        "nodes": len(nodes),
        "primaries": primaries,
        "branch_points": branch_points,
        "tips": tips,
        "max_order": max_order,
        "TDL_um": round(tdl, 2),
    }


def check_completeness(nodes, children, xy_res, z_res,
                        dist_percentile=97, radius_percentile=95,
                        flag_fraction_threshold=0.20):
    """
    Heuristic flag for potentially incomplete/truncated tracing.

    Compares each tip's step-distance and radius against the distribution
    of OTHER TIPS specifically (not all nodes in the tree). Comparing tip
    radius against all node radii is wrong, since proximal/soma-adjacent
    nodes are structurally thicker than tips - that comparison flags most
    tips spuriously.

    A neuron is only flagged as 'potentially incomplete' if the FRACTION
    of suspicious tips exceeds flag_fraction_threshold (default 20%), not
    if a single tip happens to fall in the outlier percentile - with many
    tips (e.g. 40+), having 1-2 statistical outliers by chance is expected
    and does not indicate a true tracing problem.
    """
    tips = [nid for nid in nodes if nid not in children]

    step_dists = []
    for nid, (x, y, z, r, p) in nodes.items():
        if p == -1 or p not in nodes:
            continue
        px, py, pz, _, _ = nodes[p]
        d = np.sqrt(((x - px) * xy_res) ** 2 + ((y - py) * xy_res) ** 2 + ((z - pz) * z_res) ** 2)
        step_dists.append(d)

    tip_radii = [nodes[t][3] for t in tips]

    if not step_dists or not tips:
        return {"flag_incomplete": False, "n_flagged_tips": 0, "n_tips_checked": len(tips)}

    step_thresh = np.percentile(step_dists, dist_percentile)
    radius_thresh = np.percentile(tip_radii, radius_percentile) if len(tip_radii) > 1 else None

    flagged = 0
    for t in tips:
        x, y, z, r, p = nodes[t]
        if p == -1 or p not in nodes:
            continue
        px, py, pz, _, _ = nodes[p]
        step = np.sqrt(((x - px) * xy_res) ** 2 + ((y - py) * xy_res) ** 2 + ((z - pz) * z_res) ** 2)
        thick_tip = radius_thresh is not None and r >= radius_thresh and r > 1.5
        if step >= step_thresh or thick_tip:
            flagged += 1

    fraction_flagged = flagged / len(tips) if tips else 0

    return {
        "flag_incomplete": fraction_flagged > flag_fraction_threshold,
        "n_flagged_tips": flagged,
        "n_tips_checked": len(tips),
    }


def process_file(path, tif_dir, xy_override, z_override, res_table):
    path = Path(path)
    nodes, children = parse_swc(path)
    xy_res, z_res, source = get_voxel_size(path, tif_dir, xy_override, z_override, res_table)
    m = compute_metrics(nodes, children, xy_res, z_res)
    qc = check_completeness(nodes, children, xy_res, z_res)
    m["flag_incomplete"] = qc["flag_incomplete"]
    m["n_flagged_tips"] = qc["n_flagged_tips"]
    m["n_tips_checked"] = qc["n_tips_checked"]
    m["xy_res_um"] = xy_res
    m["z_res_um"] = z_res
    m["resolution_source"] = source
    return m


def print_table(results):
    header = (f"{'File':<45} {'TDL (µm)':>10} {'Primaries':>10} {'BranchPts':>10} "
              f"{'Tips':>6} {'MaxOrder':>9} {'Nodes':>7} {'XY(µm)':>8} {'Source':<30}")
    print("\n" + header)
    print("-" * len(header))
    for name, m in results:
        print(f"{name:<45} {m['TDL_um']:>10.1f} {m['primaries']:>10} {m['branch_points']:>10} "
              f"{m['tips']:>6} {m['max_order']:>9} {m['nodes']:>7} {m['xy_res_um']:>8.4f} "
              f"{m['resolution_source']:<30}")


def save_csv(results, out_path):
    fieldnames = ["file", "TDL_um", "primaries", "branch_points", "tips", "max_order",
                  "nodes", "flag_incomplete", "n_flagged_tips", "n_tips_checked",
                  "xy_res_um", "z_res_um", "resolution_source"]
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in results:
            w.writerow({"file": name, **m})
    print(f"\n✅ CSV saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".swc file or folder containing .swc files")
    ap.add_argument("--tifdir", default=None,
                     help="Folder containing the original TIF files (searched recursively) "
                          "to auto-detect per-file XY/Z resolution from metadata")
    ap.add_argument("--resolution-table", default=None,
                     help="Excel (.xlsx) or CSV file with columns 'swc_file', "
                          "'xy_resolution_um', 'z_resolution_um' (z optional). "
                          "Takes priority over --tifdir, used to look up the correct "
                          "per-file resolution by SWC filename.")
    ap.add_argument("--xy", type=float, default=None,
                     help="Manually override XY resolution (µm/px) for ALL files, "
                          "skipping TIF lookup")
    ap.add_argument("--z", type=float, default=None,
                     help=f"Manually override Z resolution (µm/slice) for ALL files "
                          f"(default if not from TIF: {DEFAULT_Z})")
    ap.add_argument("--csv", action="store_true", help="also save results as CSV")
    ap.add_argument("--output", "-out", default=None,
                     help="CSV output filename (default: swc_metrics.csv)")
    a = ap.parse_args()

    res_table = None
    if a.resolution_table:
        res_table = load_resolution_table(a.resolution_table)
        print(f"📋 Loaded resolution table: {len(res_table)} SWC entries from {a.resolution_table}")

    if os.path.isdir(a.input):
        files = sorted([os.path.join(a.input, f) for f in os.listdir(a.input)
                         if f.endswith('.swc') and not f.startswith('._')])
        if not files:
            print("No .swc files found in folder.")
            sys.exit(1)
    else:
        files = [a.input]

    results = []
    n_fallback = 0
    for path in files:
        try:
            m = process_file(path, a.tifdir, a.xy, a.z, res_table)
            if m["resolution_source"] == "FALLBACK":
                n_fallback += 1
            results.append((os.path.basename(path), m))
            if len(files) == 1:
                print(f"\n=== {os.path.basename(path)} ===")
                for k, v in m.items():
                    print(f"  {k:<20} {v}")
        except Exception as e:
            print(f"⚠️  Error processing {path}: {e}")

    if len(files) > 1:
        print_table(results)

    if n_fallback > 0:
        print(f"\n⚠️  WARNING: {n_fallback}/{len(results)} file(s) used FALLBACK resolution "
              f"({FALLBACK_XY} µm/px) — no matching TIF found or --tifdir not provided. "
              f"These TDL values may be inaccurate. Check 'resolution_source' column.")

    if a.csv:
        filename = a.output if a.output else "swc_metrics.csv"
        if not filename.endswith(".csv"):
            filename += ".csv"
        csv_path = os.path.join(os.path.dirname(files[0]) if os.path.isdir(a.input) else ".", filename)
        save_csv(results, csv_path)
