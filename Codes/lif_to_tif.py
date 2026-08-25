"""
LIF → TIF Konverter für RABIES
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import tifffile
from readlif.reader import LifFile


def convert_lif_to_tif(lif_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    print(f"\n📂 Verarbeite: {lif_path.name}")
    lif = LifFile(str(lif_path))
    series_list = list(lif.get_iter_image())
    if not series_list:
        print("   ⚠️  Keine Serien gefunden – Datei wird übersprungen.")
        return []
    print(f"   → {len(series_list)} Serie(n) gefunden")
    for series in series_list:
        serie_name = series.name.strip().replace(" ", "_").replace("/", "-")
        out_name = f"{lif_path.stem}_{serie_name}.tif"
        out_path = output_dir / out_name
        n_channels = series.channels
        n_z = series.dims.z
        n_t = series.dims.t
        n_y, n_x = series.dims.y, series.dims.x
        print(f"   Serie '{series.name}': C={n_channels}, Z={n_z}, T={n_t}, Y={n_y}, X={n_x}")
        frames = []
        for t in range(n_t):
            z_frames = []
            for z in range(n_z):
                c_frames = []
                for c in range(n_channels):
                    img = series.get_frame(z=z, t=t, c=c)
                    c_frames.append(np.array(img))
                z_frames.append(np.stack(c_frames, axis=0))
            frames.append(np.stack(z_frames, axis=0))
        volume = np.stack(frames, axis=0)
        if n_t == 1:
            volume = volume[0]
            axes = "ZCYX"
        else:
            axes = "TZCYX"
        if n_channels == 1:
            volume = volume[..., 0, :, :]
            axes = axes.replace("C", "")
        try:
            scale = series.scale
            res_x = 1.0 / (scale[0] * 1e-4) if scale[0] else None
            res_y = 1.0 / (scale[1] * 1e-4) if scale[1] else None
            res_z = scale[2] if len(scale) > 2 else None
        except Exception:
            res_x = res_y = res_z = None
        save_kwargs = dict(data=volume, imagej=True, metadata={"axes": axes})
        if res_x and res_y:
            save_kwargs["resolution"] = (res_x, res_y)
        if res_z:
            save_kwargs["metadata"]["spacing"] = res_z
        tifffile.imwrite(str(out_path), **save_kwargs)
        print(f"   ✅ Gespeichert: {out_path.name}  ({volume.shape}, dtype={volume.dtype})")
        created_files.append(out_path)
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Konvertiert LIF-Dateien in TIF-Stacks.")
    parser.add_argument("--input", "-i", default="/Volumes/PortableSSD/BACHELOR_THESIS/RABIES_secondtry")
    parser.add_argument("--output", "-o", default="/Volumes/PortableSSD/BACHELOR_THESIS/RABIES_secondtry/TIF")
    args = parser.parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists():
        print(f"❌ Fehler: Ordner nicht gefunden: {input_dir}")
        sys.exit(1)
    output_dir = Path(args.output).expanduser().resolve()
    lif_files = sorted(input_dir.glob("*.lif"))
    if not lif_files:
        print(f"⚠️  Keine .lif Dateien in {input_dir} gefunden.")
        sys.exit(0)
    print(f"🔬 LIF → TIF Konverter")
    print(f"   Eingabe : {input_dir}")
    print(f"   Ausgabe : {output_dir}")
    print(f"   Dateien : {len(lif_files)} .lif Datei(en)")
    all_created = []
    errors = []
    for lif_path in lif_files:
        try:
            created = convert_lif_to_tif(lif_path, output_dir)
            all_created.extend(created)
        except Exception as e:
            print(f"   ❌ Fehler bei {lif_path.name}: {e}")
            errors.append(lif_path.name)
    print(f"\n{'='*50}")
    print(f"✅ Fertig! {len(all_created)} TIF-Datei(en) erstellt.")
    if errors:
        print(f"⚠️  {len(errors)} Datei(en) mit Fehler: {', '.join(errors)}")
    print(f"   Ausgabe-Ordner: {output_dir}")


if __name__ == "__main__":
    main()
