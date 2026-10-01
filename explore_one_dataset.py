"""
Single Dataset Explorer
========================
Explores ONE dataset folder at a time. No editing needed —
just run the script and paste the folder path when asked.

USAGE:
    1. Install dependencies once (skip any that fail to install):
       pip install pillow pandas geopandas rasterio

    2. Run:
       python explore_one_dataset.py

    3. When it asks "Enter dataset folder path:", paste your path,
       e.g.:  D:\\MITS\\12.09.26\\Dataset\\Svamitva
       (You can copy this directly from File Explorer's address bar.)

    4. Read the report. Run the script again for the next dataset.
"""

from pathlib import Path
from collections import defaultdict, Counter

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
VECTOR_EXTS = {".shp", ".geojson", ".json", ".gpkg"}
TABLE_EXTS = {".csv", ".xlsx"}
MODEL_EXTS = {".pt", ".pth", ".h5", ".onnx", ".pb", ".weights"}

MAX_SAMPLES_PER_TYPE = 2
MAX_TREE_DEPTH = 3


def format_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def print_tree(root, max_depth=MAX_TREE_DEPTH):
    root = Path(root)
    root_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            continue
        indent = "  " * depth
        marker = "[DIR] " if path.is_dir() else ""
        print(f"{indent}{marker}{path.name}")


def inspect_image(filepath):
    info = {"file": filepath.name}
    if HAS_RASTERIO and filepath.suffix.lower() in (".tif", ".tiff"):
        try:
            with rasterio.open(filepath) as src:
                info.update({
                    "width": src.width, "height": src.height,
                    "bands": src.count, "dtype": src.dtypes[0],
                    "crs": str(src.crs) if src.crs else "None",
                    "resolution": src.res,
                })
                return info
        except Exception as e:
            info["error"] = f"rasterio failed: {e}"
    if HAS_PIL:
        try:
            with Image.open(filepath) as img:
                info["width"], info["height"] = img.size
                info["mode"] = img.mode
                return info
        except Exception as e:
            info["error"] = f"PIL failed: {e}"
    return info


def inspect_table(filepath):
    if not HAS_PANDAS:
        return {"file": filepath.name, "note": "install pandas to preview this"}
    try:
        df = pd.read_csv(filepath, nrows=5) if filepath.suffix == ".csv" else pd.read_excel(filepath, nrows=5)
        return {"file": filepath.name, "columns": list(df.columns), "sample_rows": df.head(3).to_dict(orient="records")}
    except Exception as e:
        return {"file": filepath.name, "error": str(e)}


def inspect_vector(filepath):
    if not HAS_GEOPANDAS:
        return {"file": filepath.name, "note": "install geopandas to preview this"}
    try:
        gdf = gpd.read_file(filepath)
        return {
            "file": filepath.name, "num_features": len(gdf),
            "columns": list(gdf.columns),
            "geometry_type": gdf.geom_type.unique().tolist(),
            "crs": str(gdf.crs),
        }
    except Exception as e:
        return {"file": filepath.name, "error": str(e)}


def explore(root_path):
    root = Path(root_path.strip().strip('"'))

    if not root.exists():
        print(f"\n!! This path does not exist: {root}")
        print("   Double check you copied it correctly from File Explorer's address bar.")
        return

    print(f"\n{'='*70}\nEXPLORING: {root}\n{'='*70}")

    print("\n--- Folder structure ---")
    print_tree(root)

    ext_counter = Counter()
    ext_examples = defaultdict(list)
    total_size = 0
    for path in root.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower() or "(no extension)"
            ext_counter[ext] += 1
            total_size += path.stat().st_size
            if len(ext_examples[ext]) < MAX_SAMPLES_PER_TYPE:
                ext_examples[ext].append(path)

    print("\n--- File type summary ---")
    for ext, count in sorted(ext_counter.items(), key=lambda x: -x[1]):
        print(f"  {ext:>15} : {count:>6} files")
    print(f"  TOTAL SIZE: {format_size(total_size)}")

    print("\n--- Sample file details ---")
    for ext, files in ext_examples.items():
        for f in files:
            if ext in IMAGE_EXTS:
                print(f"  {inspect_image(f)}")
            elif ext in VECTOR_EXTS:
                print(f"  {inspect_vector(f)}")
            elif ext in TABLE_EXTS:
                print(f"  {inspect_table(f)}")
            elif ext in MODEL_EXTS:
                print(f"  {f.name} -> pretrained model file ({format_size(f.stat().st_size)})")

    print("\n--- What this means for your study ---")
    has_road_hint = any(
        "road" in str(f).lower() or "highway" in str(f).lower()
        for files in ext_examples.values() for f in files
    )
    print(f"  Has images (ORI-type data): {'YES' if any(e in IMAGE_EXTS for e in ext_counter) else 'no'}")
    print(f"  Has vector labels (parcels/buildings/roads): {'YES' if any(e in VECTOR_EXTS for e in ext_counter) else 'no'}")
    print(f"  Filename/content hints at ROADS specifically: {'YES' if has_road_hint else 'not obviously - check manually'}")
    print(f"  Has tabular metadata (csv): {'YES' if any(e in TABLE_EXTS for e in ext_counter) else 'no'}")
    print(f"  Has pretrained models: {'YES' if any(e in MODEL_EXTS for e in ext_counter) else 'no'}")

    print(f"\nDone. (Tip: re-run and enter a different folder path to check another dataset.)")


if __name__ == "__main__":
    print("=" * 70)
    print("SINGLE DATASET EXPLORER")
    print("=" * 70)
    path = input("\nPaste the dataset folder path and press Enter:\n> ")
    explore(path)
    input("\nPress Enter to close...")
