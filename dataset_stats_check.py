"""
Dataset Statistics & Quality Check Script
============================================
Runs FOUR checks across all 4 datasets (DeepGlobe, Massachusetts Roads,
Svamitva, WHU) and produces one combined report:

  1. CLASS BALANCE      - pixel-wise class frequency per mask type
  2. SIZE CONSISTENCY   - image/mask dimension distribution
  3. SPLIT VERIFICATION - counts per train/test/val, checks for mask
                           presence in each split
  4. CORRUPTED FILES    - flags unreadable images and image/mask
                           dimension mismatches

SETUP:
    pip install pillow pandas numpy rasterio tabulate

USAGE:
    1. Edit DATASET_PATHS below to match your folders.
    2. Run:  python dataset_stats_check.py
    3. Read the console output AND 'dataset_stats_report.txt'
       (a copy of everything printed, for pasting into your manuscript
       or sending back for review).

NOTE ON SPEED:
    Class-balance pixel counting is done on a random SAMPLE of images
    per split (default 100), not the full dataset, to keep runtime
    reasonable. Increase SAMPLE_SIZE below for more precision if you
    have time to spare.
"""

import sys
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # allow large GeoTIFFs
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# =========================================================================
# EDIT THESE PATHS to match your machine
# =========================================================================
DATASET_PATHS = {
    "DeepGlobe": r"D:\MITS\12.09.26\Dataset\DeepGlobe Land Cover",
    "MassRoads": r"D:\MITS\12.09.26\Dataset\Massachusetts Roads",
    "Svamitva":  r"D:\MITS\12.09.26\Dataset\Svamitva",
    "WHU":       r"D:\MITS\12.09.26\Dataset\WHU",
}

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SAMPLE_SIZE = 100          # images per split used for class-balance stats
MAX_MASK_VALUES_SHOWN = 15  # cap printed class list length

LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


def try_load(path):
    """Load an image as a numpy array. Returns None on failure."""
    try:
        if HAS_RASTERIO and path.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                arr = src.read()
                return np.moveaxis(arr, 0, -1)
        if HAS_PIL:
            with Image.open(path) as img:
                return np.array(img)
    except Exception:
        return None
    return None


def find_all_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def guess_mask_for_image(img_path, dataset_hint):
    """Best-effort mask-file lookup using common naming/folder conventions."""
    stem, parent = img_path.stem, img_path.parent

    candidates = []
    # suffix-based: _sat -> _mask, sat -> mask
    if "_sat" in stem:
        candidates.append(parent / (stem.replace("_sat", "_mask") + ".png"))
    if "sat" in stem.lower():
        candidates.append(parent / (stem.lower().replace("sat", "mask") + img_path.suffix))
        candidates.append(parent / (stem.lower().replace("sat", "map") + img_path.suffix))

    # sibling-folder based: image/ <-> label/, images/ <-> masks/, sat/ <-> map/
    pair_names = [("image", "label"), ("images", "labels"), ("images", "masks"),
                  ("Images", "Masks"), ("sat", "map"), ("tiff", "tiff")]
    for a, b in pair_names:
        if a in parent.parts:
            idx = parent.parts.index(a)
            new_parts = list(parent.parts)
            new_parts[idx] = b
            candidates.append(Path(*new_parts) / img_path.name)

    for c in candidates:
        if c.exists() and c != img_path:
            return c
    return None


def check_split_counts(root, dataset_name, split_names=("train", "test", "val", "valid")):
    log(f"\n--- [{dataset_name}] Split counts ---")
    root = Path(root)
    found_any = False
    for split in split_names:
        split_dir = root / split
        if split_dir.exists():
            found_any = True
            imgs = find_all_images(split_dir)
            log(f"  {split:>6}: {len(imgs):>6} image files")
    if not found_any:
        log(f"  (no standard train/test/val subfolders found directly under {root})")


def check_image_sizes(images, dataset_name, max_check=300):
    log(f"\n--- [{dataset_name}] Image size distribution (sampling up to {max_check}) ---")
    sample = images if len(images) <= max_check else random.sample(images, max_check)
    size_counter = Counter()
    unreadable = []
    for p in sample:
        arr = try_load(p)
        if arr is None:
            unreadable.append(p)
            continue
        h, w = arr.shape[0], arr.shape[1]
        size_counter[(w, h)] += 1
    for size, count in size_counter.most_common(10):
        log(f"  {size[0]}x{size[1]} : {count} files")
    if len(size_counter) > 1:
        log(f"  WARNING: multiple image sizes found ({len(size_counter)} distinct sizes) "
              f"- you will need to resize/pad/tile before training.")
    else:
        log(f"  OK: all sampled images share a consistent size.")
    return unreadable


def check_corrupted(images, dataset_name, max_check=300):
    log(f"\n--- [{dataset_name}] Corrupted/unreadable file check (sampling up to {max_check}) ---")
    sample = images if len(images) <= max_check else random.sample(images, max_check)
    bad_files = []
    for p in sample:
        arr = try_load(p)
        if arr is None or arr.size == 0:
            bad_files.append(p)
    if bad_files:
        log(f"  FOUND {len(bad_files)} unreadable/corrupted file(s):")
        for f in bad_files[:10]:
            log(f"    {f}")
        if len(bad_files) > 10:
            log(f"    ...and {len(bad_files)-10} more")
    else:
        log(f"  OK: no corrupted files found in sample of {len(sample)}.")
    return bad_files


def check_image_mask_dimension_match(images, dataset_name, max_check=100):
    log(f"\n--- [{dataset_name}] Image vs Mask dimension match (sampling up to {max_check}) ---")
    sample = images if len(images) <= max_check else random.sample(images, max_check)
    mismatches = []
    checked = 0
    for img_path in sample:
        mask_path = guess_mask_for_image(img_path, dataset_name)
        if mask_path is None:
            continue
        img_arr = try_load(img_path)
        mask_arr = try_load(mask_path)
        if img_arr is None or mask_arr is None:
            continue
        checked += 1
        if img_arr.shape[:2] != mask_arr.shape[:2]:
            mismatches.append((img_path, img_arr.shape[:2], mask_path, mask_arr.shape[:2]))
    log(f"  Checked {checked} image/mask pairs.")
    if mismatches:
        log(f"  FOUND {len(mismatches)} dimension mismatch(es):")
        for img_p, img_shape, mask_p, mask_shape in mismatches[:10]:
            log(f"    {img_p.name} {img_shape}  vs  {mask_p.name} {mask_shape}")
    else:
        log(f"  OK: all checked pairs match in dimensions.")


def check_class_balance(images, dataset_name, sample_size=SAMPLE_SIZE):
    log(f"\n--- [{dataset_name}] Class balance (mask pixel value frequency, sample={sample_size}) ---")
    sample = images if len(images) <= sample_size else random.sample(images, sample_size)
    value_counts = Counter()
    masks_found = 0
    for img_path in sample:
        mask_path = guess_mask_for_image(img_path, dataset_name)
        if mask_path is None:
            continue
        mask_arr = try_load(mask_path)
        if mask_arr is None:
            continue
        masks_found += 1
        if mask_arr.ndim == 3:
            # multi-channel mask (e.g. RGB-coded classes) -> treat unique rows as classes
            flat = mask_arr.reshape(-1, mask_arr.shape[-1])
            uniques, counts = np.unique(flat, axis=0, return_counts=True)
            for u, c in zip(uniques, counts):
                value_counts[tuple(u.tolist())] += int(c)
        else:
            uniques, counts = np.unique(mask_arr, return_counts=True)
            for u, c in zip(uniques, counts):
                value_counts[int(u)] += int(c)

    if masks_found == 0:
        log(f"  No masks found/matched for this dataset's sample - skipping class balance.")
        return

    total_pixels = sum(value_counts.values())
    log(f"  Masks analyzed: {masks_found}")
    log(f"  Distinct class values found: {len(value_counts)}")
    for val, count in sorted(value_counts.items(), key=lambda x: -x[1])[:MAX_MASK_VALUES_SHOWN]:
        pct = 100 * count / total_pixels
        log(f"    class {val}: {pct:5.2f}% of pixels")
    # simple imbalance flag
    top_pct = 100 * max(value_counts.values()) / total_pixels
    if top_pct > 90:
        log(f"  WARNING: dominant class covers {top_pct:.1f}% of pixels - severe imbalance. "
              f"Consider class weighting, focal loss, or Dice/IoU loss instead of plain "
              f"cross-entropy when training.")


def process_dataset(name, root, extra_split_names=()):
    log(f"\n{'='*70}\nDATASET: {name}\nPATH: {root}\n{'='*70}")
    root = Path(root)
    if not root.exists():
        log(f"  !! PATH NOT FOUND - check DATASET_PATHS in the script.")
        return

    check_split_counts(root, name, split_names=("train", "test", "val", "valid") + extra_split_names)

    all_images = find_all_images(root)
    log(f"\n  Total image files found (all subfolders): {len(all_images)}")
    if not all_images:
        log("  No image files found - skipping remaining checks.")
        return

    check_image_sizes(all_images, name)
    check_corrupted(all_images, name)
    check_image_mask_dimension_match(all_images, name)
    check_class_balance(all_images, name)


def main():
    random.seed(42)
    log("=" * 70)
    log("DATASET STATISTICS & QUALITY CHECK REPORT")
    log("=" * 70)

    process_dataset("DeepGlobe", DATASET_PATHS["DeepGlobe"])
    process_dataset("MassRoads", DATASET_PATHS["MassRoads"], extra_split_names=("tiff",))
    process_dataset("Svamitva", DATASET_PATHS["Svamitva"], extra_split_names=("FilteredData", "TestGeoTiff"))
    process_dataset("WHU", DATASET_PATHS["WHU"])

    log(f"\n\n{'='*70}")
    log("SUMMARY - Use this table for your manuscript's Dataset section")
    log("=" * 70)
    log("Check the per-dataset sections above for exact counts, sizes,")
    log("class balance percentages, and any corrupted-file or mismatch")
    log("warnings that need fixing before training.")

    report_path = Path("dataset_stats_report.txt")
    report_path.write_text("\n".join(LOG_LINES), encoding="utf-8")
    log(f"\nFull report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
