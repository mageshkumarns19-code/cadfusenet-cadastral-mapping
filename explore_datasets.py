"""
Dataset Explorer for Cadastral Mapping Study
==============================================
Scans one or more dataset root folders and reports:
  - Folder structure (subfolders, depth)
  - File type counts (extensions)
  - Image properties (dimensions, bands, resolution) for raster files
  - Column/schema preview for CSV, SHP, GeoJSON files
  - A final summary table across all datasets

USAGE:
    1. Install dependencies (only what's missing):
       pip install pillow pandas geopandas rasterio tabulate

       If rasterio/geopandas fail to install (common on Windows without
       GDAL), the script will still run using Pillow-only fallbacks —
       it just won't report GeoTIFF CRS/band metadata as precisely.

    2. Edit the DATASET_PATHS dictionary below with your actual folder
       paths (copy them from File Explorer's address bar).

    3. Run:
       python explore_datasets.py

    4. Read the printed report, and check the generated
       'dataset_exploration_report.txt' file for the full log.
"""

import os
from pathlib import Path
from collections import defaultdict, Counter

# ---- Optional imports (script degrades gracefully if unavailable) ----
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


# =========================================================================
# EDIT THESE PATHS to match your folders (copy from File Explorer address bar)
# =========================================================================
DATASET_PATHS = {
    "DeepGlobe Land Cover": r"D:\MITS\12.09.26\Dataset\DeepGlobe Land Cover",
    "Svamitva":             r"D:\MITS\12.09.26\Dataset\Svamitva",
    "WHU":                  r"D:\MITS\12.09.26\Dataset\WHU",
}

# File extensions grouped by category, for the summary
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
VECTOR_EXTS = {".shp", ".geojson", ".json", ".gpkg"}
TABLE_EXTS = {".csv", ".xlsx"}
MODEL_EXTS = {".pt", ".pth", ".h5", ".onnx", ".pb", ".weights"}

MAX_SAMPLE_FILES_PER_TYPE = 2   # how many files per extension to inspect deeply
MAX_DEPTH_LISTING = 3           # how many folder levels deep to print


def format_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def print_folder_tree(root, max_depth=MAX_DEPTH_LISTING, log=None):
    root = Path(root)
    root_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            continue
        indent = "  " * depth
        marker = "[DIR] " if path.is_dir() else ""
        line = f"{indent}{marker}{path.name}"
        print(line)
        if log is not None:
            log.append(line)


def inspect_image(filepath):
    """Return a dict describing an image file (dimensions, bands, etc.)"""
    info = {"file": filepath.name}
    if HAS_RASTERIO and filepath.suffix.lower() in (".tif", ".tiff"):
        try:
            with rasterio.open(filepath) as src:
                info["width"] = src.width
                info["height"] = src.height
                info["bands"] = src.count
                info["dtype"] = src.dtypes[0]
                info["crs"] = str(src.crs) if src.crs else "None"
                info["resolution"] = src.res
                return info
        except Exception as e:
            info["error"] = f"rasterio failed: {e}"
    if HAS_PIL:
        try:
            with Image.open(filepath) as img:
                info["width"], info["height"] = img.size
                info["mode"] = img.mode
                info["bands"] = len(img.getbands())
                return info
        except Exception as e:
            info["error"] = f"PIL failed: {e}"
    return info


def inspect_table(filepath):
    """Preview a CSV/Excel file's columns and row count."""
    if not HAS_PANDAS:
        return {"file": filepath.name, "error": "pandas not installed"}
    try:
        if filepath.suffix.lower() == ".csv":
            df = pd.read_csv(filepath, nrows=5)
            full_len = sum(1 for _ in open(filepath, encoding="utf-8", errors="ignore")) - 1
        else:
            df = pd.read_excel(filepath, nrows=5)
            full_len = "unknown (Excel)"
        return {
            "file": filepath.name,
            "columns": list(df.columns),
            "approx_rows": full_len,
            "sample": df.head(3).to_dict(orient="records"),
        }
    except Exception as e:
        return {"file": filepath.name, "error": str(e)}


def inspect_vector(filepath):
    """Preview a shapefile/GeoJSON's schema and feature count."""
    if not HAS_GEOPANDAS:
        return {"file": filepath.name, "error": "geopandas not installed"}
    try:
        gdf = gpd.read_file(filepath)
        return {
            "file": filepath.name,
            "num_features": len(gdf),
            "columns": list(gdf.columns),
            "geometry_type": gdf.geom_type.unique().tolist(),
            "crs": str(gdf.crs),
        }
    except Exception as e:
        return {"file": filepath.name, "error": str(e)}


def explore_dataset(name, root_path, log):
    root = Path(root_path)
    header = f"\n{'='*70}\nDATASET: {name}\nPATH: {root}\n{'='*70}"
    print(header)
    log.append(header)

    if not root.exists():
        msg = f"  !! PATH DOES NOT EXIST — check DATASET_PATHS in the script."
        print(msg)
        log.append(msg)
        return None

    # --- 1. Folder structure ---
    sub_header = "\n--- Folder structure (top levels) ---"
    print(sub_header)
    log.append(sub_header)
    print_folder_tree(root, log=log)

    # --- 2. File type counts ---
    ext_counter = Counter()
    ext_examples = defaultdict(list)
    total_size = 0
    for path in root.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower() or "(no extension)"
            ext_counter[ext] += 1
            total_size += path.stat().st_size
            if len(ext_examples[ext]) < MAX_SAMPLE_FILES_PER_TYPE:
                ext_examples[ext].append(path)

    count_header = "\n--- File type summary ---"
    print(count_header)
    log.append(count_header)
    for ext, count in sorted(ext_counter.items(), key=lambda x: -x[1]):
        line = f"  {ext:>15} : {count:>6} files"
        print(line)
        log.append(line)
    size_line = f"  TOTAL dataset size: {format_size(total_size)}"
    print(size_line)
    log.append(size_line)

    # --- 3. Deep inspection of sample files per category ---
    deep_header = "\n--- Sample file inspection ---"
    print(deep_header)
    log.append(deep_header)

    for ext, files in ext_examples.items():
        for f in files:
            if ext in IMAGE_EXTS:
                info = inspect_image(f)
            elif ext in VECTOR_EXTS:
                info = inspect_vector(f)
            elif ext in TABLE_EXTS:
                info = inspect_table(f)
            elif ext in MODEL_EXTS:
                info = {"file": f.name, "type": "pretrained model weights",
                        "size": format_size(f.stat().st_size)}
            else:
                continue
            line = f"  {info}"
            print(line)
            log.append(line)

    return {
        "name": name,
        "ext_counts": dict(ext_counter),
        "total_size": total_size,
        "has_images": any(e in IMAGE_EXTS for e in ext_counter),
        "has_vectors": any(e in VECTOR_EXTS for e in ext_counter),
        "has_tables": any(e in TABLE_EXTS for e in ext_counter),
        "has_models": any(e in MODEL_EXTS for e in ext_counter),
    }


def main():
    log = []
    results = []

    intro = "DATASET EXPLORATION REPORT\n" + "=" * 70
    print(intro)
    log.append(intro)

    for name, path in DATASET_PATHS.items():
        result = explore_dataset(name, path, log)
        if result:
            results.append(result)

    # --- Final cross-dataset summary ---
    summary_header = f"\n\n{'='*70}\nCROSS-DATASET SUMMARY (does this cover your study's needs?)\n{'='*70}"
    print(summary_header)
    log.append(summary_header)

    checklist = [
        ("Imagery/ORI (image files present)",  "has_images"),
        ("Vector labels: parcels/buildings/roads (shp/geojson)", "has_vectors"),
        ("Tabular metadata (csv)", "has_tables"),
        ("Pretrained models included", "has_models"),
    ]
    for label, key in checklist:
        line = f"\n{label}:"
        print(line)
        log.append(line)
        for r in results:
            mark = "YES" if r[key] else "no"
            sub = f"    [{mark}]  {r['name']}"
            print(sub)
            log.append(sub)

    road_note = (
        "\nNOTE: This script cannot automatically confirm whether ROAD "
        "network labels (a required feature in your problem statement) "
        "exist. Check the 'Sample file inspection' output above for "
        "vector files (.shp/.geojson) with columns like 'road_type', "
        "'highway', or 'class' containing road-related values."
    )
    print(road_note)
    log.append(road_note)

    # Write full log to file
    report_path = Path("dataset_exploration_report.txt")
    report_path.write_text("\n".join(log), encoding="utf-8")
    print(f"\n\nFull report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
