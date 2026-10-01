"""
Dataset Pairing Fix + Class Balance (v2)
==========================================
Fixes two issues found in the first stats report:

  1. MassRoads & WHU showed "0 pairs checked" because their subfolder
     naming didn't match my guessed conventions (image/label, sat/map).
     This version instead auto-discovers image/mask pairs by matching
     FILENAMES across the whole dataset, regardless of subfolder names,
     then tells image apart from mask using pixel-diversity (masks have
     few unique values, real images have many).

  2. Svamitva masks showed duplicate classes because some mask files
     have an alpha channel (RGBA) and others don't (RGB) - e.g.
     (85,217,48,255) and (85,217,48) were counted as different classes.
     This version strips the alpha channel from every mask before
     counting classes, so they merge correctly.

SETUP:
    pip install pillow numpy rasterio

USAGE:
    1. Edit DATASET_PATHS below if needed (already matches your folders).
    2. Run:  python dataset_pairing_fix.py
    3. Read console output + 'pairing_fix_report.txt'
"""

import random
from pathlib import Path
from collections import Counter, defaultdict

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


DATASET_PATHS = {
    "MassRoads": r"D:\MITS\12.09.26\Dataset\Massachusetts Roads",
    "WHU":       r"D:\MITS\12.09.26\Dataset\WHU",
    "Svamitva":  r"D:\MITS\12.09.26\Dataset\Svamitva",
}

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SAMPLE_SIZE = 100
UNIQUE_VALUE_THRESHOLD = 30  # fewer unique pixel values than this => likely a mask

LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


def try_load(path):
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


def strip_alpha(arr):
    """Drop a 4th alpha channel if present, so RGBA and RGB masks merge."""
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return arr[..., :3]
    return arr


def build_stem_index(root):
    """Map filename-stem -> list of full paths, across the ENTIRE dataset
    (regardless of which subfolder they're in)."""
    index = defaultdict(list)
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index[p.stem].append(p)
    return index


def classify_image_vs_mask(path_a, path_b):
    """Given two files sharing a filename stem, decide which is the
    satellite/aerial image and which is the label mask, using pixel
    diversity (masks have far fewer unique values)."""
    arr_a, arr_b = try_load(path_a), try_load(path_b)
    if arr_a is None or arr_b is None:
        return None, None

    def n_unique(arr):
        # sample a small patch for speed on large images
        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1, 1)
        if len(flat) > 20000:
            idx = np.random.choice(len(flat), 20000, replace=False)
            flat = flat[idx]
        return len(np.unique(flat, axis=0))

    ua, ub = n_unique(arr_a), n_unique(arr_b)
    if ua <= UNIQUE_VALUE_THRESHOLD and ub > UNIQUE_VALUE_THRESHOLD:
        return path_b, path_a   # (image, mask)
    if ub <= UNIQUE_VALUE_THRESHOLD and ua > UNIQUE_VALUE_THRESHOLD:
        return path_a, path_b
    return None, None  # ambiguous - both look like images or both like masks


def discover_pairs(root, dataset_name, max_pairs=SAMPLE_SIZE):
    log(f"\n--- [{dataset_name}] Auto-discovering image/mask pairs by filename ---")
    index = build_stem_index(root)
    candidate_stems = [s for s, paths in index.items() if len(paths) == 2]
    log(f"  Total unique filenames: {len(index)}")
    log(f"  Filenames appearing in exactly 2 files (candidate pairs): {len(candidate_stems)}")

    if not candidate_stems:
        log(f"  No stem-matched pairs found. Images and masks likely use "
              f"DIFFERENT filenames in this dataset (not just different folders). "
              f"Manual inspection needed.")
        return []

    sample = candidate_stems if len(candidate_stems) <= max_pairs else \
        random.sample(candidate_stems, max_pairs)

    pairs = []
    for stem in sample:
        p1, p2 = index[stem]
        img, mask = classify_image_vs_mask(p1, p2)
        if img is not None:
            pairs.append((img, mask))

    log(f"  Successfully classified {len(pairs)} / {len(sample)} sampled pairs as (image, mask).")
    if pairs:
        log(f"  Example: image={pairs[0][0].name}  <->  mask={pairs[0][1].name}")
    return pairs


def class_balance_from_pairs(pairs, dataset_name):
    log(f"\n--- [{dataset_name}] Class balance (alpha-channel stripped) ---")
    if not pairs:
        log("  No pairs available - skipping.")
        return

    value_counts = Counter()
    for _, mask_path in pairs:
        arr = try_load(mask_path)
        if arr is None:
            continue
        arr = strip_alpha(arr)
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
            uniques, counts = np.unique(flat, axis=0, return_counts=True)
            for u, c in zip(uniques, counts):
                value_counts[tuple(u.tolist())] += int(c)
        else:
            uniques, counts = np.unique(arr, return_counts=True)
            for u, c in zip(uniques, counts):
                value_counts[int(u)] += int(c)

    total = sum(value_counts.values())
    if total == 0:
        log("  Could not read any mask pixel data.")
        return

    log(f"  Distinct classes after alpha-channel fix: {len(value_counts)}")
    for val, count in sorted(value_counts.items(), key=lambda x: -x[1])[:15]:
        log(f"    class {val}: {100*count/total:5.2f}%")

    top_pct = 100 * max(value_counts.values()) / total
    if top_pct > 90:
        log(f"  WARNING: dominant class = {top_pct:.1f}% of pixels -> severe imbalance. "
              f"Use class-weighted loss (weighted cross-entropy) or Dice/IoU loss.")


def process(dataset_name, root):
    log(f"\n{'='*70}\n{dataset_name}\n{'='*70}")
    pairs = discover_pairs(root, dataset_name)
    class_balance_from_pairs(pairs, dataset_name)


def main():
    random.seed(42)
    log("=" * 70)
    log("DATASET PAIRING FIX + CLASS BALANCE (v2)")
    log("=" * 70)

    for name, path in DATASET_PATHS.items():
        process(name, path)

    Path("pairing_fix_report.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    log(f"\nFull report saved to: pairing_fix_report.txt")


if __name__ == "__main__":
    main()
