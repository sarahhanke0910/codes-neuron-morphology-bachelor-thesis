"""
export_lif_resolutions.py

Reads all image series from one or more LIF files and exports their
names and XY pixel resolution (µm/px) to an Excel table. You can then
manually add an "swc_file" column, filling in the SWC filename for
each series you actually used for reconstruction, and leave the rest
blank/delete them.

That completed table is then used by swc_metrics.py (via --resolution-table)
to look up the correct, per-file resolution for TDL calculation.

Usage:
    python3 export_lif_resolutions.py file1.lif file2.lif ... -o resolutions.xlsx

Developed with the assistance of Claude (Anthropic, claude.ai,
accessed June 2026).
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from readlif.reader import LifFile


def extract_series_info(lif_path):
    lif = LifFile(str(lif_path))
    series_list = list(lif.get_iter_image())

    rows = []
    for i, series in enumerate(series_list):
        try:
            scale = series.scale
            px_size_x = 1.0 / scale[0] if scale[0] else None
            px_size_z = (1.0 / scale[2]) if len(scale) > 2 and scale[2] else None
        except Exception:
            px_size_x = px_size_z = None

        rows.append({
            "lif_file": lif_path.name,
            "series_index": i,
            "series_name": series.name,
            "xy_resolution_um": round(px_size_x, 5) if px_size_x else None,
            "z_resolution_um": round(px_size_z, 5) if px_size_z else None,
            "swc_file": "",  # to be filled in manually
        })
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="A folder containing .lif files, or one/more .lif file paths")
    ap.add_argument("-o", "--output", default="lif_resolutions.xlsx",
                     help="Output Excel filename (default: lif_resolutions.xlsx)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        lif_files = sorted(input_path.glob("*.lif"))
        if not lif_files:
            print(f"No .lif files found in folder: {input_path}")
            sys.exit(1)
    else:
        lif_files = [input_path]

    all_rows = []
    for path in lif_files:
        if not path.exists():
            print(f"⚠️  File not found, skipping: {path}")
            continue
        print(f"Reading: {path.name} ...")
        rows = extract_series_info(path)
        all_rows.extend(rows)
        print(f"  -> {len(rows)} series found")

    if not all_rows:
        print("No series found in any file.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_excel(args.output, index=False)
    print(f"\n✅ Saved {len(df)} series to: {args.output}")
    print("\nNext steps:")
    print("  1. Open the Excel file.")
    print("  2. In the 'swc_file' column, enter the exact SWC filename")
    print("     for each series you used for reconstruction (leave others blank).")
    print("  3. Save the file (keep .xlsx or save as .csv).")
    print("  4. Use it with: python3 swc_metrics.py . --resolution-table lif_resolutions.xlsx --csv")
