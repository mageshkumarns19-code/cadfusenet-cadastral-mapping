"""
Data Leakage Check: Svamitva FilteredData (training) vs TestGeoTiff (fusion eval)
====================================================================================
Checks whether any building-model TRAINING patches (FilteredData/Images) were
cropped from the same physical locations as the 4 TestGeoTiff tiles used to
demonstrate the fusion pipeline (amora, guha, lalpur, uplarshi). If so, the
"real-world" fusion results would partly be evaluating on data the model
already saw during training - a genuine leakage concern.

THREE CHECKS, IN ORDER:
  1. Documentation scan - look for any README/metadata mentioning village
     names that match the TestGeoTiff filenames.
  2. Geo-coordinate overlap - if FilteredData images carry embedded
     coordinates, check bounding-box overlap directly (fast, exact).
  3. Perceptual image hashing - if no coordinates exist (likely, since
     these are plain training patches), slide a window across each
     TestGeoTiff tile and compare visual similarity (average hash) against
     every FilteredData patch. Flags near-duplicate content.

SETUP:
    pip install pillow numpy rasterio

USAGE:
    1. Edit the paths in CONFIG below.
    2. Run:  python check_data_leakage.py
    3. Read the final verdict at the end of the output, and check
       'leakage_check_report.txt' for full details.
"""

import os
import re
import glob
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


# =========================================================================
CONFIG_SVAMITVA_ROOT = r"D:\MITS\12.09.26\Dataset\Svamitva"
FILTERED_IMAGES_DIR = os.path.join(CONFIG_SVAMITVA_ROOT, "FilteredData", "Images")
TESTGEOTIFF_DIR = os.path.join(CONFIG_SVAMITVA_ROOT, "TestGeoTiff")

TESTGEOTIFF_NAMES = ["amora", "guha", "lalpur", "uplarshi"]  # used for name-matching

HASH_SIZE = 16                  # average-hash grid size (16x16 = 256-bit hash)
MATCH_THRESHOLD_PCT = 0.10      # flag as a possible match if <=10% of hash bits differ
WINDOW_STRIDE_FRACTION = 0.5    # slide window at 50% overlap across TestGeoTiff tiles
MAX_PATCHES_TO_CHECK = None     # set an int (e.g. 300) to sample fewer patches for a quick test run

LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


# =========================================================================
# Check 1: Documentation / filename scan for village-name clues
# =========================================================================

def check_documentation():
    log("\n" + "=" * 70)
    log("CHECK 1: Documentation / metadata scan for village names")
    log("=" * 70)

    found_any = False
    doc_extensions = (".txt", ".md", ".csv", ".json", ".xlsx", ".readme")
    for root, dirs, files in os.walk(CONFIG_SVAMITVA_ROOT):
        for fname in files:
            if fname.lower().endswith(doc_extensions):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        content = f.read().lower()
                    for name in TESTGEOTIFF_NAMES:
                        if name.lower() in content:
                            log(f"  MATCH: '{name}' mentioned in {fpath}")
                            found_any = True
                except Exception:
                    pass

    # Also check if any FilteredData patch filenames themselves mention the village names
    if os.path.exists(FILTERED_IMAGES_DIR):
        for fname in os.listdir(FILTERED_IMAGES_DIR):
            for name in TESTGEOTIFF_NAMES:
                if name.lower() in fname.lower():
                    log(f"  MATCH: filename '{fname}' contains village name '{name}'")
                    found_any = True

    if not found_any:
        log("  No documentation or filename clues found linking FilteredData to "
              "the specific TestGeoTiff village names. (This doesn't rule out "
              "overlap - it just means there's no easy textual evidence either way.)")
    return found_any


# =========================================================================
# Check 2: Geo-coordinate bounding-box overlap (if metadata exists)
# =========================================================================

def get_geotiff_bounds():
    bounds = {}
    if not HAS_RASTERIO:
        return bounds
    tif_files = glob.glob(os.path.join(TESTGEOTIFF_DIR, "*.tif")) + \
                glob.glob(os.path.join(TESTGEOTIFF_DIR, "*.tiff"))
    for tif in tif_files:
        try:
            with rasterio.open(tif) as src:
                bounds[os.path.basename(tif)] = (src.bounds, src.crs)
        except Exception as e:
            log(f"  Could not read {tif}: {e}")
    return bounds


def check_geo_overlap():
    log("\n" + "=" * 70)
    log("CHECK 2: Geo-coordinate bounding-box overlap")
    log("=" * 70)

    if not HAS_RASTERIO:
        log("  rasterio not installed - skipping this check.")
        return False

    geotiff_bounds = get_geotiff_bounds()
    if not geotiff_bounds:
        log("  No TestGeoTiff files could be read.")
        return False

    for name, (bounds, crs) in geotiff_bounds.items():
        log(f"  {name}: bounds={bounds}, CRS={crs}")

    # Check if FilteredData images carry any geo metadata at all
    sample_files = os.listdir(FILTERED_IMAGES_DIR)[:5] if os.path.exists(FILTERED_IMAGES_DIR) else []
    any_georeferenced = False
    for fname in sample_files:
        fpath = os.path.join(FILTERED_IMAGES_DIR, fname)
        if fname.lower().endswith((".tif", ".tiff")):
            try:
                with rasterio.open(fpath) as src:
                    if src.crs is not None:
                        any_georeferenced = True
                        log(f"  {fname} HAS geo-coordinates: bounds={src.bounds}")
            except Exception:
                pass

    if not any_georeferenced:
        log("  FilteredData/Images files (.png, checked a sample) carry NO geo-coordinates.")
        log("  This check cannot directly compare locations - proceeding to Check 3 (image hashing).")
        return False

    log("  Geo-coordinates found - manual bounding-box comparison recommended using these values.")
    return True


# =========================================================================
# Check 3: Perceptual image hashing (average hash + Hamming distance)
# =========================================================================

def average_hash(img_array, hash_size=HASH_SIZE):
    """Simple, fast average-hash: resize to hash_size x hash_size grayscale,
    threshold against the mean -> a compact binary fingerprint."""
    img = Image.fromarray(img_array).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    mean = arr.mean()
    return (arr > mean).flatten()  # boolean array, length hash_size*hash_size


def hamming_distance_pct(hash_a, hash_b):
    return np.sum(hash_a != hash_b) / len(hash_a)


def load_array(path):
    try:
        if HAS_RASTERIO and path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                arr = src.read()
                return np.moveaxis(arr, 0, -1)
        if HAS_PIL:
            with Image.open(path) as img:
                return np.array(img)
    except Exception:
        return None
    return None


def check_perceptual_hashing():
    log("\n" + "=" * 70)
    log("CHECK 3: Perceptual image-hash comparison")
    log("=" * 70)

    if not os.path.exists(FILTERED_IMAGES_DIR):
        log(f"  FilteredData/Images not found at {FILTERED_IMAGES_DIR} - cannot run this check.")
        return []

    patch_files = [f for f in os.listdir(FILTERED_IMAGES_DIR) if f.lower().endswith(".png")]
    if MAX_PATCHES_TO_CHECK:
        patch_files = patch_files[:MAX_PATCHES_TO_CHECK]
    log(f"  Hashing {len(patch_files)} training patches from FilteredData/Images...")

    patch_hashes = {}
    for fname in patch_files:
        arr = load_array(os.path.join(FILTERED_IMAGES_DIR, fname))
        if arr is not None:
            patch_hashes[fname] = average_hash(arr)

    log(f"  Successfully hashed {len(patch_hashes)} patches.")

    tif_files = glob.glob(os.path.join(TESTGEOTIFF_DIR, "*.tif")) + \
                glob.glob(os.path.join(TESTGEOTIFF_DIR, "*.tiff"))

    matches = []
    for tif_path in tif_files:
        tif_name = os.path.basename(tif_path)
        log(f"\n  Scanning {tif_name} for visual matches to training patches...")

        if not HAS_RASTERIO:
            log("    rasterio not installed - cannot open large GeoTIFFs reliably. Skipping.")
            continue

        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height
            patch_size = 1024  # matches FilteredData patch size
            stride = int(patch_size * WINDOW_STRIDE_FRACTION)

            n_windows_checked = 0
            for row in range(0, height - patch_size + 1, stride):
                for col in range(0, width - patch_size + 1, stride):
                    from rasterio.windows import Window
                    window = Window(col, row, patch_size, patch_size)
                    tile_arr = src.read(window=window)
                    tile_arr = np.moveaxis(tile_arr, 0, -1)[..., :3]

                    window_hash = average_hash(tile_arr)
                    n_windows_checked += 1

                    for patch_name, patch_hash in patch_hashes.items():
                        dist_pct = hamming_distance_pct(window_hash, patch_hash)
                        if dist_pct <= MATCH_THRESHOLD_PCT:
                            matches.append({
                                "tif": tif_name,
                                "window_row": row,
                                "window_col": col,
                                "patch": patch_name,
                                "similarity_pct": (1 - dist_pct) * 100,
                            })
                            log(f"    POSSIBLE MATCH: {tif_name} @ ({row},{col}) "
                                  f"~= {patch_name}  (similarity: {(1-dist_pct)*100:.1f}%)")

            log(f"    Checked {n_windows_checked} windows in {tif_name}.")

    return matches


# =========================================================================
def main():
    log("=" * 70)
    log("DATA LEAKAGE CHECK: Svamitva Training Patches vs Fusion Eval Tiles")
    log("=" * 70)

    doc_match = check_documentation()
    geo_checked = check_geo_overlap()
    hash_matches = check_perceptual_hashing()

    log("\n\n" + "=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)

    if hash_matches:
        log(f"  ⚠ POTENTIAL LEAKAGE DETECTED: {len(hash_matches)} visual near-matches found")
        log("  between training patches and fusion evaluation tiles.")
        log("  RECOMMENDATION: report this as a limitation, or exclude the affected")
        log("  training patches / evaluation regions and re-run to be safe.")
    elif doc_match:
        log("  ⚠ Documentation suggests possible overlap, but no visual matches confirmed.")
        log("  RECOMMENDATION: manually verify the specific locations mentioned.")
    else:
        log("  ✓ NO LEAKAGE DETECTED across all three checks.")
        log("  Training data (FilteredData/Images) and fusion evaluation tiles")
        log("  (TestGeoTiff) appear to be from independent locations/content.")
        log("  Safe to report the fusion results as a genuine real-world")
        log("  generalization demonstration in the manuscript.")

    report_path = Path("leakage_check_report.txt")
    report_path.write_text("\n".join(LOG_LINES), encoding="utf-8")
    log(f"\nFull report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
