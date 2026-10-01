"""
Preprocessing Pipeline
========================
Converts all 4 raw datasets into THREE unified, training-ready folders:

    preprocessed/
        building/   <- from WHU (already split) + Svamitva (custom split)
            train/images, train/masks
            val/images,   val/masks
            test/images,  test/masks
        road/       <- from Massachusetts Roads (custom split)
            train/images, train/masks
            val/images,   val/masks
            test/images,  test/masks
        landuse/    <- from DeepGlobe (already split)
            train/images, train/masks
            val/images,   val/masks
            test/images,  test/masks

WHAT IT DOES (per the decisions confirmed by the stats reports):
    1. Standardizes every image/mask to 512x512 tiles
    2. Strips alpha channels from every mask
    3. Binarizes building & road masks to clean 0/255 single-channel
    4. Converts DeepGlobe's RGB-coded land-use masks to single-channel
       class-index masks (0-6) using the dataset's own class_dict.csv
    5. Uses Svamitva's BinaryMasks (not the colored Masks folder) for
       the building task, and gives Svamitva its own 80/10/10 split
       since it doesn't ship with one
    6. Writes a manifest.csv logging every output file, its source
       dataset, split, and task - useful for your manuscript's dataset
       table and for the training script later

SETUP:
    pip install pillow numpy pandas rasterio tqdm

USAGE:
    1. Edit DATASET_PATHS and OUTPUT_ROOT below if your folders differ.
    2. IMPORTANT - test small first: set LIMIT_IMAGES_PER_SPLIT = 20
       near the top, run once, check the output folders look correct,
       THEN set it to None and run the full job (this can take a
       long time and use tens of GB of disk space at full scale).
    3. Run:  python preprocess.py
"""

import random
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
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
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # fallback: no progress bar
        return iterable


# =========================================================================
# CONFIG - edit these
# =========================================================================
DATASET_PATHS = {
    "DeepGlobe": r"D:\MITS\12.09.26\Dataset\DeepGlobe Land Cover",
    "MassRoads": r"D:\MITS\12.09.26\Dataset\Massachusetts Roads",
    "Svamitva":  r"D:\MITS\12.09.26\Dataset\Svamitva",
    "WHU":       r"D:\MITS\12.09.26\Dataset\WHU",
}
OUTPUT_ROOT = Path(r"D:\MITS\12.09.26\Dataset\preprocessed")

TILE_SIZE = 512
RANDOM_SEED = 42

# SET THIS TO A SMALL NUMBER (e.g. 20) FOR YOUR FIRST TEST RUN.
# Set to None once you've confirmed the output looks correct.
LIMIT_IMAGES_PER_SPLIT = None

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

MANIFEST_ROWS = []  # (task, split, source_dataset, image_out_path, mask_out_path)


# =========================================================================
# Generic helpers
# =========================================================================

def load_array(path):
    try:
        if HAS_RASTERIO and path.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                arr = src.read()
                return np.moveaxis(arr, 0, -1)
        if HAS_PIL:
            with Image.open(path) as img:
                return np.array(img)
    except Exception as e:
        print(f"    !! Failed to load {path.name}: {e}")
        return None
    return None


def strip_alpha(arr):
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return arr[..., :3]
    return arr


def save_png(arr, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(out_path)


def pad_to_multiple(arr, multiple):
    """Pad H and W (reflect padding) so both are divisible by `multiple`."""
    h, w = arr.shape[0], arr.shape[1]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h, pad_w = new_h - h, new_w - w
    if arr.ndim == 3:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")


def tile_array(arr, tile_size):
    """Split a (padded, multiple-of-tile_size) array into a list of tiles."""
    h, w = arr.shape[0], arr.shape[1]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tiles.append(arr[y:y + tile_size, x:x + tile_size])
    return tiles


def binarize_mask(arr, threshold=127):
    """Collapse any mask (RGB or grayscale) to clean single-channel 0/255."""
    arr = strip_alpha(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return np.where(arr > threshold, 255, 0).astype(np.uint8)


def limit_list(items):
    if LIMIT_IMAGES_PER_SPLIT is not None and len(items) > LIMIT_IMAGES_PER_SPLIT:
        return items[:LIMIT_IMAGES_PER_SPLIT]
    return items


def split_stems(stems, seed=RANDOM_SEED):
    """Random 80/10/10 split of a list of filename stems."""
    stems = list(stems)
    random.Random(seed).shuffle(stems)
    n = len(stems)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }


def write_pair(task, split, source, prefix, name, img_tile, mask_tile):
    img_out = OUTPUT_ROOT / task / split / "images" / f"{prefix}_{name}.png"
    mask_out = OUTPUT_ROOT / task / split / "masks" / f"{prefix}_{name}.png"
    save_png(img_tile, img_out)
    save_png(mask_tile, mask_out)
    MANIFEST_ROWS.append((task, split, source, str(img_out), str(mask_out)))


# =========================================================================
# TASK: BUILDING  (WHU + Svamitva -> binary building masks)
# =========================================================================

def process_whu_building(root):
    print("\n=== Building task: processing WHU (already split) ===")
    root = Path(root)
    split_map = {"train": "train", "val": "val", "test": "test"}

    for whu_split, out_split in split_map.items():
        split_dir = root / whu_split
        if not split_dir.exists():
            continue
        # stem -> list of matching files within this split only
        index = defaultdict(list)
        for p in split_dir.rglob("*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                index[p.stem].append(p)
        stems = [s for s, paths in index.items() if len(paths) == 2]
        stems = limit_list(stems)

        for stem in tqdm(stems, desc=f"WHU/{whu_split}"):
            p1, p2 = index[stem]
            arr1, arr2 = load_array(p1), load_array(p2)
            if arr1 is None or arr2 is None:
                continue
            # mask = the one with fewer unique values
            u1 = len(np.unique(arr1.reshape(-1, arr1.shape[-1]) if arr1.ndim == 3 else arr1))
            u2 = len(np.unique(arr2.reshape(-1, arr2.shape[-1]) if arr2.ndim == 3 else arr2))
            img_arr, mask_arr = (arr1, arr2) if u1 > u2 else (arr2, arr1)

            mask_arr = binarize_mask(mask_arr)
            img_arr = strip_alpha(img_arr)
            # WHU is already 512x512 - just save directly
            if img_arr.shape[0] != TILE_SIZE or img_arr.shape[1] != TILE_SIZE:
                img_arr = np.array(Image.fromarray(img_arr.astype(np.uint8)).resize((TILE_SIZE, TILE_SIZE)))
                mask_arr = np.array(Image.fromarray(mask_arr).resize((TILE_SIZE, TILE_SIZE)))
            write_pair("building", out_split, "WHU", "whu", stem, img_arr, mask_arr)


def process_svamitva_building(root):
    print("\n=== Building task: processing Svamitva (custom split + tiling) ===")
    root = Path(root)
    images_dir = root / "FilteredData" / "Images"
    masks_dir = root / "FilteredData" / "BinaryMasks"
    if not images_dir.exists() or not masks_dir.exists():
        print(f"  !! Could not find Images/BinaryMasks under {root / 'FilteredData'}")
        return

    image_stems = {p.stem: p for p in images_dir.glob("*.png")}
    mask_stems = {p.stem: p for p in masks_dir.glob("*.png")}
    common_stems = sorted(set(image_stems) & set(mask_stems))
    print(f"  Found {len(common_stems)} matched image/mask pairs.")

    splits = split_stems(common_stems)
    for split, stems in splits.items():
        stems = limit_list(stems)
        for stem in tqdm(stems, desc=f"Svamitva/{split}"):
            img_arr = load_array(image_stems[stem])
            mask_arr = load_array(mask_stems[stem])
            if img_arr is None or mask_arr is None:
                continue
            img_arr = strip_alpha(img_arr)
            mask_arr = binarize_mask(mask_arr)

            # 1024x1024 -> four 512x512 tiles (2x2, non-overlapping)
            img_padded = pad_to_multiple(img_arr, TILE_SIZE)
            mask_padded = pad_to_multiple(mask_arr, TILE_SIZE)
            img_tiles = tile_array(img_padded, TILE_SIZE)
            mask_tiles = tile_array(mask_padded, TILE_SIZE)

            for i, (it, mt) in enumerate(zip(img_tiles, mask_tiles)):
                write_pair("building", split, "Svamitva", "svamitva", f"{stem}_t{i}", it, mt)


# =========================================================================
# TASK: ROAD  (Massachusetts Roads -> binary road masks)
# =========================================================================

def process_massroads(root):
    print("\n=== Road task: processing Massachusetts Roads (custom split + tiling) ===")
    root = Path(root)
    tiff_dir = root / "tiff"

    index = defaultdict(list)
    for p in tiff_dir.rglob("*"):
        if p.suffix.lower() in (".tif", ".tiff"):
            index[p.stem].append(p)
    common_stems = sorted([s for s, paths in index.items() if len(paths) == 2])
    print(f"  Found {len(common_stems)} matched image/mask pairs.")

    splits = split_stems(common_stems)
    for split, stems in splits.items():
        stems = limit_list(stems)
        for stem in tqdm(stems, desc=f"MassRoads/{split}"):
            p1, p2 = index[stem]
            arr1, arr2 = load_array(p1), load_array(p2)
            if arr1 is None or arr2 is None:
                continue
            u1 = len(np.unique(arr1.reshape(-1, arr1.shape[-1]) if arr1.ndim == 3 else arr1))
            u2 = len(np.unique(arr2.reshape(-1, arr2.shape[-1]) if arr2.ndim == 3 else arr2))
            img_arr, mask_arr = (arr1, arr2) if u1 > u2 else (arr2, arr1)

            img_arr = strip_alpha(img_arr)
            mask_arr = binarize_mask(mask_arr)

            # 1500x1500 -> pad to 1536x1536 -> nine 512x512 tiles (3x3)
            img_padded = pad_to_multiple(img_arr, TILE_SIZE)
            mask_padded = pad_to_multiple(mask_arr, TILE_SIZE)
            img_tiles = tile_array(img_padded, TILE_SIZE)
            mask_tiles = tile_array(mask_padded, TILE_SIZE)

            for i, (it, mt) in enumerate(zip(img_tiles, mask_tiles)):
                write_pair("road", split, "MassRoads", "massroads", f"{stem}_t{i}", it, mt)


# =========================================================================
# TASK: LAND-USE  (DeepGlobe -> class-index masks using class_dict.csv)
# =========================================================================

def load_deepglobe_color_map(root):
    csv_path = Path(root) / "class_dict.csv"
    if not HAS_PANDAS or not csv_path.exists():
        print("  !! class_dict.csv not found or pandas missing - cannot decode land-use classes.")
        return None
    df = pd.read_csv(csv_path)
    # expects columns: name, r, g, b (case-insensitive fallback)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name", df.columns[0])
    r_col, g_col, b_col = cols.get("r", df.columns[1]), cols.get("g", df.columns[2]), cols.get("b", df.columns[3])
    color_map = {}
    for idx, row in df.iterrows():
        color_map[(int(row[r_col]), int(row[g_col]), int(row[b_col]))] = idx
    print(f"  Loaded {len(color_map)} classes from class_dict.csv: "
          f"{dict(zip(df[name_col], range(len(df))))}")
    return color_map


def rgb_mask_to_class_index(mask_arr, color_map):
    mask_arr = strip_alpha(mask_arr)
    h, w = mask_arr.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for color, idx in color_map.items():
        matches = np.all(mask_arr == np.array(color), axis=-1)
        out[matches] = idx
    return out


def process_deepglobe(root):
    print("\n=== Land-use task: processing DeepGlobe (custom split) ===")
    print("    NOTE: DeepGlobe's original test/ and valid/ folders have NO masks")
    print("    (they're the unlabeled Kaggle competition holdout sets), so we")
    print("    build our own 80/10/10 split from the labeled train/ folder only.")
    root = Path(root)
    color_map = load_deepglobe_color_map(root)
    if color_map is None:
        return

    train_dir = root / "train"
    if not train_dir.exists():
        print(f"  !! train folder not found under {root}")
        return

    valid_stems = []
    for sat_path in sorted(train_dir.glob("*_sat.jpg")):
        stem = sat_path.stem.replace("_sat", "")
        mask_path = train_dir / f"{stem}_mask.png"
        if mask_path.exists():
            valid_stems.append(stem)
    print(f"  Found {len(valid_stems)} labeled image/mask pairs in train/.")

    splits = split_stems(valid_stems)
    for split, stems in splits.items():
        stems = limit_list(stems)
        for stem in tqdm(stems, desc=f"DeepGlobe/{split}"):
            sat_path = train_dir / f"{stem}_sat.jpg"
            mask_path = train_dir / f"{stem}_mask.png"
            img_arr = load_array(sat_path)
            mask_arr = load_array(mask_path)
            if img_arr is None or mask_arr is None:
                continue

            img_arr = strip_alpha(img_arr)
            class_idx_mask = rgb_mask_to_class_index(mask_arr, color_map)

            # 2448x2448 -> pad to 2560x2560 -> 25 tiles of 512x512 (5x5)
            img_padded = pad_to_multiple(img_arr, TILE_SIZE)
            mask_padded = pad_to_multiple(class_idx_mask, TILE_SIZE)
            img_tiles = tile_array(img_padded, TILE_SIZE)
            mask_tiles = tile_array(mask_padded, TILE_SIZE)

            for i, (it, mt) in enumerate(zip(img_tiles, mask_tiles)):
                write_pair("landuse", split, "DeepGlobe", "deepglobe", f"{stem}_t{i}", it, mt)


# =========================================================================
def main():
    random.seed(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREPROCESSING PIPELINE")
    if LIMIT_IMAGES_PER_SPLIT is not None:
        print(f"** TEST MODE: limited to {LIMIT_IMAGES_PER_SPLIT} source images per split/dataset **")
        print("** Set LIMIT_IMAGES_PER_SPLIT = None in the script for the full run **")
    print("=" * 70)

    process_whu_building(DATASET_PATHS["WHU"])
    process_svamitva_building(DATASET_PATHS["Svamitva"])
    process_massroads(DATASET_PATHS["MassRoads"])
    process_deepglobe(DATASET_PATHS["DeepGlobe"])

    if HAS_PANDAS:
        manifest_df = pd.DataFrame(MANIFEST_ROWS,
                                    columns=["task", "split", "source_dataset", "image_path", "mask_path"])
        manifest_path = OUTPUT_ROOT / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\nManifest saved: {manifest_path}")
        print("\n--- Final tile counts per task/split/source ---")
        print(manifest_df.groupby(["task", "split", "source_dataset"]).size())
    else:
        print(f"\n(install pandas to get a manifest.csv summary - {len(MANIFEST_ROWS)} pairs written)")

    print(f"\nDONE. Preprocessed data is in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
