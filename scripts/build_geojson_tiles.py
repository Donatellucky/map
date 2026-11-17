#!/usr/bin/env python3
"""Build a deterministic fixed-grid GeoJSON tile package from the validated release.

Tiles are storage/spatial partitions in source PNG pixel coordinates, not map
projection tiles. Original source_component IDs are preserved in every tile.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from shapely.geometry import box, mapping, shape
from shapely.validation import explain_validity

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "output" / "release"
SOURCE = RELEASE / "provinces.geojson"
OUT = ROOT / "output" / "tiled_release"
TILE_SIZE = 1024
WIDTH = 10000
HEIGHT = 10000
AREA_EPSILON = 1e-7


def polygonal_only(geometry):
    """Return polygonal content only, preserving disconnected clipped parts."""
    if geometry.is_empty:
        return None
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        parts = [part for part in geometry.geoms if not part.is_empty and part.area > 0]
    elif geometry.geom_type == "GeometryCollection":
        parts = []
        for item in geometry.geoms:
            extracted = polygonal_only(item)
            if extracted is None:
                continue
            if extracted.geom_type == "Polygon":
                parts.append(extracted)
            elif extracted.geom_type == "MultiPolygon":
                parts.extend(extracted.geoms)
    else:
        return None
    if not parts:
        return None
    from shapely.geometry import MultiPolygon
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Validated release not found: {SOURCE}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_manifest_path = RELEASE / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else {}
    features = source.get("features", [])
    if not features:
        raise ValueError("Validated release contains no features.")

    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("**/*.geojson"):
        path.unlink()

    tile_features = defaultdict(list)
    province_area_totals = defaultdict(float)
    source_area_totals = {}
    tile_bounds = {}
    tile_size = TILE_SIZE

    for feature in features:
        props = feature["properties"]
        source_id = int(props["source_component"])
        geom = shape(feature["geometry"])
        if geom.is_empty or not geom.is_valid:
            raise ValueError(f"Invalid source geometry for component {source_id}: {explain_validity(geom)}")
        source_area_totals[source_id] = geom.area

        minx, miny, maxx, maxy = geom.bounds
        x_start = max(0, int(minx // tile_size))
        y_start = max(0, int(miny // tile_size))
        x_end = min((WIDTH - 1) // tile_size, int(max(maxx - 1e-9, minx) // tile_size))
        y_end = min((HEIGHT - 1) // tile_size, int(max(maxy - 1e-9, miny) // tile_size))

        for ty in range(y_start, y_end + 1):
            for tx in range(x_start, x_end + 1):
                x0, y0 = tx * tile_size, ty * tile_size
                x1, y1 = min(x0 + tile_size, WIDTH), min(y0 + tile_size, HEIGHT)
                clipped = polygonal_only(geom.intersection(box(x0, y0, x1, y1)))
                if clipped is None or clipped.area <= 0:
                    continue
                if not clipped.is_valid:
                    raise ValueError(f"Invalid clipped geometry: component={source_id}, tile={tx}/{ty}")
                tile_id = f"{tx}_{ty}"
                tile_bounds[tile_id] = [x0, y0, x1, y1]
                tile_feature = {
                    "type": "Feature",
                    "id": feature.get("id", source_id),
                    "properties": {
                        **props,
                        "source_component": source_id,
                        "tile_id": tile_id,
                        "tile_clipped": not geom.equals(clipped),
                    },
                    "geometry": mapping(clipped),
                }
                tile_features[tile_id].append(tile_feature)
                province_area_totals[source_id] += clipped.area

    mismatches = []
    for source_id, source_area in source_area_totals.items():
        tiled_area = province_area_totals[source_id]
        if abs(source_area - tiled_area) > AREA_EPSILON:
            mismatches.append({
                "source_component": source_id,
                "source_area": source_area,
                "tiled_area": tiled_area,
                "difference": tiled_area - source_area,
            })
    if mismatches:
        raise ValueError(f"Tile area conservation failed for {len(mismatches)} provinces: {mismatches[:5]}")

    tile_dir = OUT / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_manifest = []
    for tile_id, tile_items in sorted(tile_features.items(), key=lambda item: tuple(map(int, item[0].split("_")))):
        tx, ty = map(int, tile_id.split("_"))
        tile_path = tile_dir / str(ty) / f"{tx}.geojson"
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        collection = {
            "type": "FeatureCollection",
            "name": f"fauxlore_tile_{tx}_{ty}",
            "properties": {
                "tile_id": tile_id,
                "tile_x": tx,
                "tile_y": ty,
                "bounds": tile_bounds[tile_id],
                "tile_size": tile_size,
                "coordinate_system": "source PNG pixels; origin top-left; x right; y down",
            },
            "features": tile_items,
        }
        tile_path.write_text(json.dumps(collection, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tile_manifest.append({
            "tile_id": tile_id,
            "path": f"tiles/{ty}/{tx}.geojson",
            "bounds": tile_bounds[tile_id],
            "feature_fragments": len(tile_items),
            "unique_source_components": len({item["properties"]["source_component"] for item in tile_items}),
        })

    manifest = {
        "package": "Fauxlore province geometry — fixed-grid tile prototype",
        "version": "0.1.0-tile-prototype",
        "source_release": "0.1.0-rc1",
        "source_dimensions": [WIDTH, HEIGHT],
        "coordinate_system": "source PNG pixels; origin top-left; x right; y down",
        "tile_scheme": {
            "type": "fixed_grid",
            "tile_size_pixels": tile_size,
            "columns": (WIDTH + tile_size - 1) // tile_size,
            "rows": (HEIGHT + tile_size - 1) // tile_size,
            "projection": None,
            "zoom_levels": None,
        },
        "source_feature_count": len(features),
        "tile_count": len(tile_manifest),
        "tile_fragment_count": sum(item["feature_fragments"] for item in tile_manifest),
        "identity_policy": "province_id is the persistent registry identity; source_component is retained as the current raster segmentation label",
        "area_conservation": {
            "status": "passed",
            "mismatched_components": 0,
            "epsilon": AREA_EPSILON,
        },
        "tiles": tile_manifest,
        "known_exceptions": source_manifest.get("known_exceptions", []),
    }
    (OUT / "tile_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "source_features": len(features),
        "tiles": len(tile_manifest),
        "tile_fragments": manifest["tile_fragment_count"],
        "area_mismatches": 0,
        "tile_size": tile_size,
    }, indent=2))


if __name__ == "__main__":
    main()
