"""
CadFuseNet Fusion + Topology Validation Pipeline
====================================================
This is the actual novelty component of the study: it takes the 3
independently trained segmentation models (building, road, land-use)
and fuses their outputs into one topology-consistent, georeferenced
cadastral map.

PIPELINE:
  1. Load the 3 trained .keras models
  2. Load Svamitva's real georeferenced drone imagery (TestGeoTiff)
  3. Tile each large GeoTIFF into 512x512 windows (preserving geo-coordinates)
  4. Run all 3 models on each window -> building mask, road mask, land-use mask
  5. Convert each raster mask to GEOREFERENCED vector polygons
  6. Merge all tiles' polygons into 3 layers: buildings, roads, land-use
  7. Apply topology validation: detect building/road overlaps, clean geometries,
     flag conflicts - this is the "detection of overlapping or inconsistent
     parcel geometries" requirement from the problem statement
  8. Export everything as GeoJSON (GIS-ready) + a visual summary map

SETUP:
    pip install tensorflow rasterio geopandas shapely numpy matplotlib

USAGE:
    1. Edit the CONFIG section below: paths to your 3 .keras files and
       the Svamitva TestGeoTiff folder.
    2. Run:  python fusion_pipeline.py
    3. Check the 'fusion_outputs' folder for GeoJSON layers, a conflict
       report, and a summary map PNG.
"""

import os
import glob
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import Affine

import geopandas as gpd
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union


# =========================================================================
# CONFIG - edit these paths
# =========================================================================
BUILDING_MODEL_PATH = r"D:\MITS\12.09.26\Dataset\models\building_model_final.keras"
ROAD_MODEL_PATH = r"D:\MITS\12.09.26\Dataset\models\road_model_final.keras"
LANDUSE_MODEL_PATH = r"D:\MITS\12.09.26\Dataset\models\landuse_model_best.keras"

# Svamitva's real georeferenced drone imagery (the actual deployment target)
GEOTIFF_DIR = r"D:\MITS\12.09.26\Dataset\Svamitva\TestGeoTiff"

OUTPUT_DIR = Path("fusion_outputs")

TILE_SIZE = 512                 # must match training input size
BUILDING_THRESHOLD = 0.5        # sigmoid cutoff for "is a building"
ROAD_THRESHOLD = 0.5            # sigmoid cutoff for "is a road"
MIN_POLYGON_AREA_M2 = 4.0       # discard tiny noise polygons smaller than this
LANDUSE_CLASS_NAMES = [
    "urban_land", "agriculture_land", "rangeland",
    "forest_land", "water", "barren_land", "unknown",
]

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================================
# Step 1: Load the 3 trained models
# =========================================================================

def load_models():
    print("Loading trained models...")
    building_model = tf.keras.models.load_model(BUILDING_MODEL_PATH, compile=False)
    road_model = tf.keras.models.load_model(ROAD_MODEL_PATH, compile=False)
    landuse_model = tf.keras.models.load_model(LANDUSE_MODEL_PATH, compile=False)
    print("  Building model loaded:", building_model.input_shape, "->", building_model.output_shape)
    print("  Road model loaded:", road_model.input_shape, "->", road_model.output_shape)
    print("  Land-use model loaded:", landuse_model.input_shape, "->", landuse_model.output_shape)
    return building_model, road_model, landuse_model


# =========================================================================
# Step 2-3: Load GeoTIFFs and tile them into georeferenced 512x512 windows
# =========================================================================

def get_tile_windows(width, height, tile_size):
    """Yield (row, col, Window) covering the full image, edge tiles included
    even if smaller than tile_size (they'll be padded before inference)."""
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            yield row, col, Window(col, row, w, h)


def read_and_prepare_tile(src, window):
    """Read a window from the raster, return a (TILE_SIZE, TILE_SIZE, 3)
    float32 array in [0,1], padded if the window is smaller than TILE_SIZE,
    plus the window's own affine transform for georeferencing outputs."""
    arr = src.read(window=window)  # (bands, h, w)
    arr = np.moveaxis(arr, 0, -1)  # (h, w, bands)

    # keep only first 3 bands (RGB) - models were trained on 3-channel input
    if arr.shape[-1] >= 3:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    h, w = arr.shape[0], arr.shape[1]
    padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
    padded[:h, :w, :] = arr[:, :, :3].astype(np.float32)
    padded = padded / 255.0 if padded.max() > 1.0 else padded

    window_transform = src.window_transform(window)
    return padded, window_transform, (h, w)  # (h, w) = the REAL (unpadded) size


# =========================================================================
# Step 4-5: Run inference + convert predictions to georeferenced vectors
# =========================================================================

def mask_to_polygons(mask_2d, transform, valid_shape, label_value=1, min_area=MIN_POLYGON_AREA_M2):
    """Convert a binary/class-index 2D mask into a list of (geometry, value)
    using the tile's real geographic transform, so output polygons land in
    true map coordinates. Only the valid (unpadded) portion is used."""
    h, w = valid_shape
    mask_cropped = mask_2d[:h, :w]
    mask_binary = (mask_cropped == label_value).astype(np.uint8)

    if mask_binary.sum() == 0:
        return []

    results = []
    for geom, val in rasterio_shapes(mask_binary, mask=mask_binary.astype(bool), transform=transform):
        poly = shapely_shape(geom)
        if poly.area >= min_area:
            results.append(poly)
    return results


def run_fusion_on_geotiff(tif_path, building_model, road_model, landuse_model):
    print(f"\n=== Processing {os.path.basename(tif_path)} ===")
    building_polys = []
    road_polys = []
    landuse_polys_by_class = defaultdict(list)

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        crs = src.crs
        print(f"  Size: {width}x{height}, CRS: {crs}")

        windows = list(get_tile_windows(width, height, TILE_SIZE))
        print(f"  Tiling into {len(windows)} windows of {TILE_SIZE}x{TILE_SIZE}")

        for idx, (row, col, window) in enumerate(windows):
            tile_arr, tile_transform, valid_shape = read_and_prepare_tile(src, window)
            batch = np.expand_dims(tile_arr, axis=0)

            # --- Building prediction ---
            building_pred = building_model.predict(batch, verbose=0)[0, ..., 0]
            building_mask = (building_pred > BUILDING_THRESHOLD).astype(np.uint8)
            building_polys.extend(mask_to_polygons(building_mask, tile_transform, valid_shape))

            # --- Road prediction ---
            road_pred = road_model.predict(batch, verbose=0)[0, ..., 0]
            road_mask = (road_pred > ROAD_THRESHOLD).astype(np.uint8)
            road_polys.extend(mask_to_polygons(road_mask, tile_transform, valid_shape))

            # --- Land-use prediction (multi-class) ---
            landuse_pred = landuse_model.predict(batch, verbose=0)[0]  # (H,W,7)
            landuse_labels = np.argmax(landuse_pred, axis=-1).astype(np.uint8)
            for class_idx in range(len(LANDUSE_CLASS_NAMES)):
                polys = mask_to_polygons(landuse_labels, tile_transform, valid_shape, label_value=class_idx)
                landuse_polys_by_class[LANDUSE_CLASS_NAMES[class_idx]].extend(polys)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(windows):
                print(f"    Processed tile {idx + 1}/{len(windows)}")

    return {
        "crs": crs,
        "buildings": building_polys,
        "roads": road_polys,
        "landuse": landuse_polys_by_class,
        "source_file": os.path.basename(tif_path),
    }


# =========================================================================
# Step 6-7: Merge tiles into layers + topology validation
# =========================================================================

def polygons_to_gdf(polygons, crs, layer_name="unnamed"):
    if not polygons:
        return gpd.GeoDataFrame({"layer": [], "geometry": []}, crs=crs)
    gdf = gpd.GeoDataFrame({"layer": [layer_name] * len(polygons), "geometry": polygons}, crs=crs)
    # clean invalid geometries (self-intersections etc.) - standard buffer(0) trick
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]
    return gdf


def topology_validate(buildings_gdf, roads_gdf):
    """Detect and report conflicts between the building and road layers -
    this directly implements the problem statement's requirement for
    'detection of overlapping or inconsistent parcel geometries'."""
    print("\n=== Topology Validation ===")
    conflicts = []

    if len(buildings_gdf) == 0 or len(roads_gdf) == 0:
        print("  One or both layers empty - skipping conflict detection.")
        return gpd.GeoDataFrame({"conflict_type": [], "geometry": []}, crs=buildings_gdf.crs), buildings_gdf

    roads_union = unary_union(roads_gdf.geometry.values)

    resolved_buildings = []
    for idx, building_geom in enumerate(buildings_gdf.geometry):
        if building_geom.intersects(roads_union):
            overlap = building_geom.intersection(roads_union)
            conflicts.append({
                "conflict_type": "building_overlaps_road",
                "geometry": overlap,
                "building_idx": idx,
                "overlap_area_m2": overlap.area,
            })
            # Resolution rule: roads take priority (physical infrastructure is
            # more reliably mapped than building-edge segmentation noise) ->
            # clip the building polygon to remove the overlapping part.
            cleaned = building_geom.difference(roads_union)
            if not cleaned.is_empty:
                resolved_buildings.append(cleaned)
        else:
            resolved_buildings.append(building_geom)

    n_conflicts = len(conflicts)
    total_overlap_area = sum(c["overlap_area_m2"] for c in conflicts)
    print(f"  Buildings checked: {len(buildings_gdf)}")
    print(f"  Conflicts found (building/road overlap): {n_conflicts}")
    print(f"  Total overlap area removed: {total_overlap_area:.2f} sq. map units")

    conflicts_gdf = gpd.GeoDataFrame(conflicts, crs=buildings_gdf.crs) if conflicts else \
        gpd.GeoDataFrame({"conflict_type": [], "geometry": []}, crs=buildings_gdf.crs)

    resolved_gdf = gpd.GeoDataFrame(
        {"layer": ["building_resolved"] * len(resolved_buildings), "geometry": resolved_buildings},
        crs=buildings_gdf.crs,
    )

    return conflicts_gdf, resolved_gdf


# =========================================================================
# Step 8: Export + visualize
# =========================================================================

def export_layer(gdf, name):
    if len(gdf) == 0:
        print(f"  (skipping empty layer: {name})")
        return None
    out_path = OUTPUT_DIR / f"{name}.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"  Saved {name}: {len(gdf)} features -> {out_path}")
    return out_path


def make_summary_map(buildings_gdf, roads_gdf, landuse_gdfs, conflicts_gdf, tile_name):
    fig, ax = plt.subplots(figsize=(12, 12))

    # land-use as base layer
    colors = plt.cm.tab10(np.linspace(0, 1, len(LANDUSE_CLASS_NAMES)))
    for i, (class_name, gdf) in enumerate(landuse_gdfs.items()):
        if len(gdf) > 0:
            gdf.plot(ax=ax, color=colors[i], alpha=0.3, label=class_name)

    # roads
    if len(roads_gdf) > 0:
        roads_gdf.plot(ax=ax, color="black", linewidth=0.5, label="roads")

    # buildings (resolved, conflict-free)
    if len(buildings_gdf) > 0:
        buildings_gdf.plot(ax=ax, color="red", edgecolor="darkred", alpha=0.7, label="buildings")

    # flagged conflicts, highlighted
    if len(conflicts_gdf) > 0:
        conflicts_gdf.plot(ax=ax, color="yellow", edgecolor="orange", linewidth=1.5, label="resolved conflicts")

    ax.set_title(f"CadFuseNet Fused Cadastral Map - {tile_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()

    out_path = OUTPUT_DIR / f"summary_map_{Path(tile_name).stem}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary map -> {out_path}")


# =========================================================================
def main():
    building_model, road_model, landuse_model = load_models()

    tif_files = sorted(glob.glob(os.path.join(GEOTIFF_DIR, "*.tif")) +
                        glob.glob(os.path.join(GEOTIFF_DIR, "*.tiff")))
    if not tif_files:
        print(f"No GeoTIFF files found in {GEOTIFF_DIR}")
        return

    print(f"Found {len(tif_files)} GeoTIFF tile(s) to process: {[os.path.basename(f) for f in tif_files]}")

    all_reports = []

    for tif_path in tif_files:
        result = run_fusion_on_geotiff(tif_path, building_model, road_model, landuse_model)

        buildings_gdf = polygons_to_gdf(result["buildings"], result["crs"], "building")
        roads_gdf = polygons_to_gdf(result["roads"], result["crs"], "road")
        landuse_gdfs = {
            cls: polygons_to_gdf(polys, result["crs"], cls)
            for cls, polys in result["landuse"].items()
        }

        conflicts_gdf, resolved_buildings_gdf = topology_validate(buildings_gdf, roads_gdf)

        tile_stem = Path(result["source_file"]).stem
        print(f"\n--- Exporting layers for {tile_stem} ---")
        export_layer(resolved_buildings_gdf, f"{tile_stem}_buildings")
        export_layer(roads_gdf, f"{tile_stem}_roads")
        for cls, gdf in landuse_gdfs.items():
            export_layer(gdf, f"{tile_stem}_landuse_{cls}")
        export_layer(conflicts_gdf, f"{tile_stem}_conflicts")

        make_summary_map(resolved_buildings_gdf, roads_gdf, landuse_gdfs, conflicts_gdf, result["source_file"])

        all_reports.append({
            "tile": result["source_file"],
            "num_buildings": len(buildings_gdf),
            "num_buildings_resolved": len(resolved_buildings_gdf),
            "num_roads": len(roads_gdf),
            "num_conflicts": len(conflicts_gdf),
        })

    # Overall summary report for the manuscript's Results section
    report_path = OUTPUT_DIR / "fusion_summary_report.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n{'='*70}")
    print("FUSION PIPELINE COMPLETE")
    print(f"Summary report saved to: {report_path}")
    for r in all_reports:
        print(f"  {r['tile']}: {r['num_buildings']} buildings "
              f"({r['num_conflicts']} conflicts resolved), {r['num_roads']} roads")
    print("=" * 70)


if __name__ == "__main__":
    main()
