#!/usr/bin/env python3
"""Extract enclosed province regions from the transparent black contour PNG.

Coordinates are kept in source-image pixel space (origin top-left).
The exterior component is excluded; open/empty areas are not fabricated.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from rasterio.features import shapes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("SOURCE_PNG", ROOT / "fauxlore_map_KONTUR.png"))
OUT = ROOT / "output"
DIAG = OUT / "diagnostics"
OUT.mkdir(exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)

def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source image not found: {SOURCE}")

    rgba = np.asarray(Image.open(SOURCE).convert("RGBA"))
    height, width = rgba.shape[:2]
    if (width, height) != (10000, 10000):
        raise ValueError(f"Expected 10000x10000 source, got {width}x{height}")

    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3].astype(np.uint16)
    luminance = (rgb[:, :, 0] * 299 + rgb[:, :, 1] * 587 + rgb[:, :, 2] * 114) // 1000
    # Alpha identifies the exported contour layer; luminance suppresses stray
    # non-black opaque pixels if the PNG contains any.
    contour = (alpha >= 16) & (luminance <= 220)
    del rgba, rgb, alpha, luminance

    contour_pixels = int(contour.sum())
    if contour_pixels == 0:
        raise ValueError("No dark, non-transparent contour pixels found.")

    # 4-connectivity prevents diagonal corner-touching gaps from joining regions.
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(~contour, structure=cross)
    del contour

    counts = np.bincount(labels.ravel(), minlength=count + 1)
    border_ids = np.unique(np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])))
    exterior = set(int(v) for v in border_ids)
    region_ids = [i for i in range(1, count + 1) if i not in exterior and counts[i] > 0]
    if not region_ids:
        raise ValueError("No enclosed regions found. Check contour threshold/connectivity.")

    # Precompute component bounding boxes once (avoid rescanning the full 100M-pixel raster per region).
    objects = ndimage.find_objects(labels, max_label=count)

    # Polygonize label raster in source pixel coordinates.
    geometries: dict[int, list[dict]] = defaultdict(list)
    for geom, value in shapes(labels, mask=np.isin(labels, region_ids), connectivity=4):
        geometries[int(value)].append(geom)

    # Build GeoJSON. A label may yield multiple polygon parts; retain all parts
    # as a GeometryCollection rather than silently dropping any.
    features = []
    rows = []
    for idx, region_id in enumerate(region_ids, start=1):
        parts = geometries.get(region_id, [])
        if not parts:
            continue
        geometry = parts[0] if len(parts) == 1 else {"type": "GeometryCollection", "geometries": parts}
        bbox = objects[region_id - 1]
        if bbox is None:
            continue
        y_slice, x_slice = bbox
        min_x, min_y = int(x_slice.start), int(y_slice.start)
        max_x, max_y = int(x_slice.stop), int(y_slice.stop)
        area = int(counts[region_id])
        feature = {
            "type": "Feature",
            "id": idx,
            "properties": {
                "region_id": f"region_{idx:05d}",
                "source_component": int(region_id),
                "pixel_area": area,
                "bbox": [min_x, min_y, max_x, max_y],
            },
            "geometry": geometry,
        }
        features.append(feature)
        rows.append((idx, region_id, area, min_x, min_y, max_x, max_y, len(parts)))

    geojson = {
        "type": "FeatureCollection",
        "name": "fauxlore_enclosed_regions_pixel_coordinates",
        "crs": {"type": "name", "properties": {"name": "PIXEL:top-left-origin"}},
        "properties": {
            "source": SOURCE.name,
            "width": width,
            "height": height,
            "contour_pixels": contour_pixels,
            "connectivity": 4,
            "excluded_border_components": len(exterior),
            "note": "Coordinates are source PNG pixels, not geographic coordinates.",
        },
        "features": features,
    }
    with (OUT / "provinces_raw.geojson").open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

    with (OUT / "region_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region_index", "source_component", "pixel_area", "min_x", "min_y",
                         "max_x_exclusive", "max_y_exclusive", "polygon_parts"])
        writer.writerows(rows)

    # Compact diagnostics: contour mask and labelled-region preview at 1/4 scale.
    preview_size = (2500, 2500)
    contour_preview = Image.fromarray((np.asarray(Image.open(SOURCE).convert("RGBA"))[:, :, 3] >= 16).astype(np.uint8) * 255)
    contour_preview.resize(preview_size, Image.Resampling.NEAREST).save(DIAG / "contour_mask_preview.png")

    # Stable pseudo-colours based on component id; background/exterior stays dark.
    small = labels[::4, ::4]
    palette = np.zeros((count + 1, 3), dtype=np.uint8)
    ids = np.arange(1, count + 1, dtype=np.uint64)
    palette[1:, 0] = ((ids * 67 + 43) % 220 + 25).astype(np.uint8)
    palette[1:, 1] = ((ids * 97 + 71) % 220 + 25).astype(np.uint8)
    palette[1:, 2] = ((ids * 131 + 19) % 220 + 25).astype(np.uint8)
    preview = palette[small]
    preview[small == 0] = (0, 0, 0)
    Image.fromarray(preview, "RGB").save(DIAG / "regions_preview.png")

    areas = [r[2] for r in rows]
    summary = {
        "source": SOURCE.name,
        "dimensions": [width, height],
        "contour_pixels": contour_pixels,
        "connected_components_including_background": int(count),
        "excluded_border_components": len(exterior),
        "enclosed_regions_exported": len(features),
        "smallest_region_pixels": int(min(areas)),
        "largest_region_pixels": int(max(areas)),
        "median_region_pixels": float(np.median(areas)),
        "regions_under_25_pixels": int(sum(a < 25 for a in areas)),
        "regions_under_100_pixels": int(sum(a < 100 for a in areas)),
        "coordinate_system": "source PNG pixels; origin top-left; x right; y down",
        "warnings": [
            "This is a first-pass raster polygonization; inspect diagnostics for open contour gaps and accidental merges.",
            "Land/sea classification is not inferred from contour-only input.",
            "No geographic CRS or Leaflet transform has been applied.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
