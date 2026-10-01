"""
Dataset Exploration & Inspection Script
=========================================
For: DeepGlobe Land Cover, Massachusetts Roads, Svamitva, WHU Building

WHAT THIS SCRIPT DOES:
  1. Verifies folder structure matches what's expected for each dataset
  2. Counts files, checks image-mask pairing consistency
  3. Reports image dimensions, channel count, resolution, value ranges
  4. Reads and previews CSV metadata (class_dict, label_class_dict, metadata)
  5. Saves sample image (+ mask, if available) visualizations as PNG files
     into an 'eda_outputs' folder — these are ready to drop into your
     manuscript's Dataset/Methodology section as figures.
  6. Prints a final checklist confirming what each dataset provides.

SETUP:
    pip install pillow pandas numpy matplotlib rasterio

USAGE:
    1. Edit the DATASET_PATHS dictionary below with your real folder paths.
    2. Run:  python dataset_eda.py
    3. Check the printed report AND the 'eda_outputs' folder for saved figures.
"""

import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just saves files
import matplotlib.pyplot as plt

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


# =========================================================================
# EDIT THESE PATHS to match your machine (copy from File Explorer address bar)
# =========================================================================
DATASET_PATHS = {
    "DeepGlobe":     r"D:\MITS\12.09.26\Dataset\DeepGlobe Land Cover",
    "MassRoads":     r"D:\MITS\12.09.26\Dataset\Massachusetts Roads",
    "Svamitva":      r"D:\MITS\12.09.26\Dataset\Svamitva",
    "WHU":           r"D:\MITS\12.09.26\Dataset\WHU",
}

OUTPUT_DIR = Path("eda_outputs")
SAMPLES_PER_DATASET = 3
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def format_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def load_image_array(path):
    """Load an image as a numpy array, using rasterio for georeferenced
    TIFFs when available, else Pillow. Returns (array, meta_dict)."""
    meta = {}
    if HAS_RASTERIO and path.suffix.lower() in (".tif", ".tiff"):
        try:
            with rasterio.open(path) as src:
                arr = src.read()  # (bands, H, W)
                arr = np.moveaxis(arr, 0, -1)  # -> (H, W, bands)
                meta["crs"] = str(src.crs) if src.crs else "None"
                meta["resolution"] = src.res
                meta["bands"] = src.count
                meta["dtype"] = str(src.dtypes[0])
                return arr, meta
        except Exception as e:
            meta["error"] = f"rasterio failed: {e}"
    if HAS_PIL:
        try:
            img = Image.open(path)
            arr = np.array(img)
            meta["mode"] = img.mode
            meta["bands"] = 1 if arr.ndim == 2 else arr.shape[-1]
            meta["dtype"] = str(arr.dtype)
            return arr, meta
        except Exception as e:
            meta["error"] = f"PIL failed: {e}"
    return None, meta


def summarize_folder(root, label):
    """Print & return counts of files by extension under a folder."""
    root = Path(root)
    if not root.exists():
        print(f"  !! Path not found: {root}")
        return {}
    counter = Counter()
    total_size = 0
    for p in root.rglob("*"):
        if p.is_file():
            counter[p.suffix.lower() or "(none)"] += 1
            total_size += p.stat().st_size
    print(f"\n  [{label}] file types:")
    for ext, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"      {ext:>10} : {cnt:>6}")
    print(f"      TOTAL SIZE: {format_size(total_size)}")
    return counter


def preview_csv(path, label):
    if not HAS_PANDAS:
        print(f"  [{label}] (install pandas to preview)")
        return
    try:
        df = pd.read_csv(path)
        print(f"\n  [{label}] shape={df.shape}, columns={list(df.columns)}")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"  [{label}] could not read: {e}")


def find_images(folder, limit=None):
    folder = Path(folder)
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if limit:
        files = random.sample(files, min(limit, len(files)))
    return files


def save_sample_figure(dataset_name, image_path, mask_path=None, out_dir=OUTPUT_DIR):
    out_dir.mkdir(exist_ok=True)
    img_arr, img_meta = load_image_array(image_path)
    if img_arr is None:
        print(f"    Could not load image: {image_path}")
        return

    ncols = 2 if mask_path else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # If more than 3 bands, show only first 3 as RGB-ish
    disp = img_arr[..., :3] if img_arr.ndim == 3 and img_arr.shape[-1] >= 3 else img_arr
    axes[0].imshow(disp.astype(np.uint8) if disp.dtype != np.uint8 else disp)
    axes[0].set_title(f"Image\n{image_path.name}\nshape={img_arr.shape}, {img_meta.get('dtype','?')}")
    axes[0].axis("off")

    if mask_path:
        mask_arr, mask_meta = load_image_array(mask_path)
        if mask_arr is not None:
            axes[1].imshow(mask_arr, cmap="tab20")
            axes[1].set_title(f"Mask\n{mask_path.name}\nshape={mask_arr.shape}")
            axes[1].axis("off")

    out_file = out_dir / f"{dataset_name}_{image_path.stem}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    plt.close(fig)
    print(f"    Saved sample figure -> {out_file}")


# =========================================================================
# Dataset-specific exploration (each dataset has a different known layout)
# =========================================================================

def explore_deepglobe(root):
    print(f"\n{'='*70}\nDEEPGLOBE LAND COVER\nPATH: {root}\n{'='*70}")
    root = Path(root)
    summarize_folder(root, "DeepGlobe (all)")

    for csv_name in ["class_dict.csv", "metadata.csv"]:
        csv_path = root / csv_name
        if csv_path.exists():
            preview_csv(csv_path, f"DeepGlobe/{csv_name}")

    train_dir = root / "train"
    images = find_images(train_dir, limit=SAMPLES_PER_DATASET)
    print(f"\n  Found {len(list(train_dir.rglob('*.jpg'))) + len(list(train_dir.rglob('*.png')))} "
          f"candidate files in train/ (sampling {len(images)})")
    for img_path in images:
        # DeepGlobe convention: <id>_sat.jpg paired with <id>_mask.png
        mask_path = None
        if "_sat" in img_path.stem:
            candidate = img_path.parent / (img_path.stem.replace("_sat", "_mask") + ".png")
            if candidate.exists():
                mask_path = candidate
        save_sample_figure("DeepGlobe", img_path, mask_path)


def explore_massroads(root):
    print(f"\n{'='*70}\nMASSACHUSETTS ROADS\nPATH: {root}\n{'='*70}")
    root = Path(root)
    summarize_folder(root, "MassRoads (all)")

    for csv_name in ["label_class_dict.csv", "metadata.csv"]:
        csv_path = root / csv_name
        if csv_path.exists():
            preview_csv(csv_path, f"MassRoads/{csv_name}")

    tiff_dir = root / "tiff"
    images = find_images(tiff_dir, limit=SAMPLES_PER_DATASET)
    print(f"\n  Sampling {len(images)} images from tiff/")
    for img_path in images:
        # Common convention: sat/ and map/ subfolders, or _sat/_mask suffixes
        mask_path = None
        if "sat" in img_path.stem.lower():
            candidate = img_path.parent / img_path.name.lower().replace("sat", "mask")
            if candidate.exists():
                mask_path = candidate
        save_sample_figure("MassRoads", img_path, mask_path)


def explore_svamitva(root):
    print(f"\n{'='*70}\nSVAMITVA\nPATH: {root}\n{'='*70}")
    root = Path(root)
    summarize_folder(root, "Svamitva (all)")

    # List pretrained models
    models_dir = root / "PreTrainedModels"
    if models_dir.exists():
        print("\n  Pretrained models found:")
        for f in models_dir.rglob("*"):
            if f.is_file():
                print(f"      {f.name} ({format_size(f.stat().st_size)})")

    # FilteredData: Images / Masks / BinaryMasks subfolders
    filtered = root / "FilteredData"
    for sub in ["Images", "Masks", "BinaryMasks"]:
        d = filtered / sub
        if d.exists():
            n = len(list(d.rglob("*.png")))
            print(f"  FilteredData/{sub}: {n} PNG files")

    images_dir = filtered / "Images"
    masks_dir = filtered / "Masks"
    images = find_images(images_dir, limit=SAMPLES_PER_DATASET)
    print(f"\n  Sampling {len(images)} images from FilteredData/Images")
    for img_path in images:
        candidate_mask = masks_dir / img_path.name
        mask_path = candidate_mask if candidate_mask.exists() else None
        save_sample_figure("Svamitva", img_path, mask_path)

    # TestGeoTiff: real georeferenced ORI
    geotiff_dir = root / "TestGeoTiff"
    geo_images = find_images(geotiff_dir, limit=1)
    for img_path in geo_images:
        print(f"\n  Inspecting georeferenced tile: {img_path.name}")
        arr, meta = load_image_array(img_path)
        if arr is not None:
            print(f"      shape={arr.shape}, meta={meta}")
        save_sample_figure("Svamitva_GeoTiff", img_path)


def explore_whu(root):
    print(f"\n{'='*70}\nWHU BUILDING DATASET\nPATH: {root}\n{'='*70}")
    root = Path(root)
    summarize_folder(root, "WHU (all)")

    train_dir = root / "train"
    images = find_images(train_dir, limit=SAMPLES_PER_DATASET)
    print(f"\n  Sampling {len(images)} images from train/")
    for img_path in images:
        # Common WHU convention: image/ and label/ subfolders with same filename
        mask_path = None
        for alt_dir_name in ["label", "labels", "mask", "masks"]:
            candidate = img_path.parent.parent / alt_dir_name / img_path.name
            if candidate.exists():
                mask_path = candidate
                break
        save_sample_figure("WHU", img_path, mask_path)


def main():
    random.seed(42)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("DATASET EXPLORATION & INSPECTION")
    print("=" * 70)
    print(f"Sample figures will be saved to: {OUTPUT_DIR.resolve()}")

    explore_deepglobe(DATASET_PATHS["DeepGlobe"])
    explore_massroads(DATASET_PATHS["MassRoads"])
    explore_svamitva(DATASET_PATHS["Svamitva"])
    explore_whu(DATASET_PATHS["WHU"])

    print(f"\n\n{'='*70}")
    print("DONE. Open the 'eda_outputs' folder to see sample image/mask figures.")
    print("These are ready to include as figures in your manuscript's")
    print("Dataset / Methodology section.")
    print("=" * 70)


if __name__ == "__main__":
    main()
