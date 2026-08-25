"""
batch_spine_classification.py

Batch spine classification & density calculation for multiple
PyReconstruct "Export all traces" CSVs (single-plane method, one CSV
per neuron/segment).

Expected columns per CSV: Name, Section, Index, Hidden, Closed, Length,
Area, Radius, Centroid-x, Centroid-y, ...

Classification criteria (Rubio-de Anda et al. 2026, JoVE):
    Filopodia: neck length > 2 um
    Thin:      neck length < 2 um (and not Stubby/Mushroom)
    Stubby:    neck length / head width <= 1 (includes spines with no neck)
    Mushroom:  head width > 0.6 um

Region is auto-detected from the filename (Cortex, Hippocampus,
Striatum, Amygdala), matching the naming convention used throughout
the rest of the Golgi pipeline.

Usage:
    python3 batch_spine_classification.py <csv_folder> [--max-pair-dist 3.0]

Example:
    python3 batch_spine_classification.py \
        "/Volumes/.../Golgi/Control/control_tif/PyReconstruct/PyR_csv"

    python3 batch_spine_classification.py \
        "/Volumes/.../Golgi/Experimental/exp_tif/PyReconstruct/PyR_csv"

Output (saved in csv_folder):
    - all_spines_classified.csv   (every individual spine, with type)
    - summary_per_neuron.csv      (per-neuron summary)
    - region_summary.csv          (per-region summary)

Requires:
    pip install pandas numpy --break-system-packages
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

REGION_KEYWORDS = [
    ("hippocampus", "Hippocampus"),
    ("striatum", "Striatum"),
    ("amygdala", "Amygdala"),
    ("cortex", "Cortex"),
]


def detect_region(filename):
    fname_lower = filename.lower()
    for keyword, region_name in REGION_KEYWORDS:
        if keyword in fname_lower:
            return region_name
    return None


def pair_head_neck(head_df, neck_df, max_dist):
    """Greedily pair each spine head with its nearest unused spine neck,
    within max_dist. Returns a DataFrame aligned with head_df's row
    order, with columns neck_length_um and neck_id (NaN if unmatched)."""
    if len(head_df) == 0:
        return pd.DataFrame({"neck_length_um": [], "neck_id": []})

    neck_available = neck_df.copy()
    neck_available["used"] = False

    results = []
    for _, h in head_df.iterrows():
        avail = neck_available[~neck_available["used"]]
        if len(avail) == 0:
            results.append({"neck_length_um": np.nan, "neck_id": np.nan})
            continue

        d = np.sqrt((avail["centroid_x"] - h["centroid_x"]) ** 2 +
                    (avail["centroid_y"] - h["centroid_y"]) ** 2)
        best_idx = d.idxmin()

        if d.loc[best_idx] <= max_dist:
            chosen_neck_id = avail.loc[best_idx, "neck_id"]
            neck_available.loc[neck_available["neck_id"] == chosen_neck_id, "used"] = True
            results.append({
                "neck_length_um": avail.loc[best_idx, "Length"],
                "neck_id": chosen_neck_id,
            })
        else:
            results.append({"neck_length_um": np.nan, "neck_id": np.nan})

    return pd.DataFrame(results).reset_index(drop=True)


def classify_spine(head_width, neck_length):
    if pd.isna(neck_length):
        return "stubby"
    ratio = neck_length / head_width
    if ratio <= 1:
        return "stubby"
    if neck_length > 2:
        return "filopodia"
    if head_width > 0.6:
        return "mushroom"
    return "thin"


def process_one_file(path, max_pair_dist_um):
    """Process a single neuron's trace CSV. Returns a DataFrame of
    classified spines, or None if the file couldn't be processed."""
    neuron_id = os.path.splitext(os.path.basename(path))[0]
    region = detect_region(os.path.basename(path))

    raw = pd.read_csv(path)
    raw = raw.rename(columns={"Centroid-x": "centroid_x", "Centroid-y": "centroid_y"})

    segment_df = raw[raw["Name"].astype(str).str.lower().str.match(r"^segment")]
    head_df = raw[raw["Name"] == "spine_head"].reset_index(drop=True)
    head_df["head_id"] = range(len(head_df))
    neck_df = raw[raw["Name"] == "spine_neck"].reset_index(drop=True)
    neck_df["neck_id"] = range(len(neck_df))

    if len(segment_df) == 0:
        warnings.warn(f"'{neuron_id}': no 'segment' trace found - file skipped.")
        return None
    if len(head_df) == 0:
        warnings.warn(f"'{neuron_id}': no spine_head traces found - file skipped.")
        return None

    segment_length_um = segment_df["Length"].sum()

    pairs = pair_head_neck(head_df, neck_df, max_pair_dist_um)

    spines = head_df[["head_id", "Length", "centroid_x", "centroid_y"]].rename(
        columns={"Length": "head_width_um"})
    spines = pd.concat([spines.reset_index(drop=True), pairs], axis=1)
    spines["type"] = spines.apply(
        lambda r: classify_spine(r["head_width_um"], r["neck_length_um"]), axis=1)
    spines["neuron"] = neuron_id
    spines["region"] = region
    spines["segment_length_um"] = segment_length_um

    matched_neck_ids = set(spines["neck_id"].dropna())
    all_neck_ids = set(neck_df["neck_id"])
    unmatched = all_neck_ids - matched_neck_ids
    if unmatched:
        warnings.warn(f"'{neuron_id}': {len(unmatched)} neck trace(s) unmatched "
                       f"(distance > {max_pair_dist_um:.1f} um). Row(s): {sorted(unmatched)}")

    return spines


def main():
    parser = argparse.ArgumentParser(description="Batch spine classification and density calculation")
    parser.add_argument("csv_folder", help="Folder containing PyReconstruct 'Export all traces' CSVs "
                                            "(one CSV per neuron)")
    parser.add_argument("--max-pair-dist", type=float, default=3.0,
                         help="Maximum distance (um) for pairing a spine head with a spine neck "
                              "(default: 3.0)")
    args = parser.parse_args()

    print(f"CSV folder: {args.csv_folder}")
    print(f"Max head-neck pairing distance: {args.max_pair_dist} um")

    files = sorted(glob.glob(os.path.join(args.csv_folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in '{args.csv_folder}' - check the path.")

    print(f"\nFound {len(files)} file(s):")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    all_spines_list = []
    for f in files:
        result = process_one_file(f, args.max_pair_dist)
        if result is not None:
            all_spines_list.append(result)

    if not all_spines_list:
        raise SystemExit("No files could be processed successfully.")

    all_spines = pd.concat(all_spines_list, ignore_index=True)

    n_no_region = all_spines["region"].isna().sum()
    if n_no_region > 0:
        warnings.warn(f"{n_no_region} spine(s) could not be assigned a region (filename does not "
                       f"contain any of the keywords Cortex/Hippocampus/Striatum/Amygdala).")

    # ---- Per-neuron summary ----
    per_neuron_summary = (
        all_spines.groupby(["neuron", "region", "segment_length_um"], dropna=False)
        .agg(
            n_total=("type", "size"),
            n_thin=("type", lambda s: (s == "thin").sum()),
            n_mushroom=("type", lambda s: (s == "mushroom").sum()),
            n_stubby=("type", lambda s: (s == "stubby").sum()),
            n_filopodia=("type", lambda s: (s == "filopodia").sum()),
        )
        .reset_index()
    )
    per_neuron_summary["density_per_10um"] = (
        per_neuron_summary["n_total"] / per_neuron_summary["segment_length_um"]) * 10
    per_neuron_summary["density_per_10um_no_filo"] = (
        (per_neuron_summary["n_total"] - per_neuron_summary["n_filopodia"])
        / per_neuron_summary["segment_length_um"]) * 10

    print("\n===== Summary per neuron =====")
    print(per_neuron_summary.to_string(index=False))

    # ---- Per-region summary ----
    region_summary = (
        per_neuron_summary.dropna(subset=["region"])
        .groupby("region")
        .agg(
            n_neurons=("neuron", "nunique"),
            mean_density=("density_per_10um", "mean"),
            sd_density=("density_per_10um", "std"),
            total_spines=("n_total", "sum"),
        )
        .reset_index()
    )

    print("\n===== Summary per region =====")
    print(region_summary.to_string(index=False))

    # ---- Overall summary ----
    overall = {
        "n_neurons": all_spines["neuron"].nunique(),
        "n_total": len(all_spines),
        "n_thin": (all_spines["type"] == "thin").sum(),
        "n_mushroom": (all_spines["type"] == "mushroom").sum(),
        "n_stubby": (all_spines["type"] == "stubby").sum(),
        "n_filopodia": (all_spines["type"] == "filopodia").sum(),
        "mean_density_per_10um": per_neuron_summary["density_per_10um"].mean(),
    }
    print("\n===== Overall summary =====")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # ---- Export ----
    all_spines_path = os.path.join(args.csv_folder, "all_spines_classified.csv")
    per_neuron_path = os.path.join(args.csv_folder, "summary_per_neuron.csv")
    region_path = os.path.join(args.csv_folder, "region_summary.csv")

    all_spines.to_csv(all_spines_path, index=False)
    per_neuron_summary.to_csv(per_neuron_path, index=False)
    region_summary.to_csv(region_path, index=False)

    print(f"\nSaved in '{args.csv_folder}':")
    print(f"  - all_spines_classified.csv")
    print(f"  - summary_per_neuron.csv")
    print(f"  - region_summary.csv")


if __name__ == "__main__":
    main()
