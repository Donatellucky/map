#!/usr/bin/env python3
"""Apply and validate the persistent Fauxlore province ID registry."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "output" / "release"
REGISTRY_SOURCE = ROOT / "data" / "province_registry.csv"
GEOJSON = RELEASE / "provinces.geojson"
STATS = RELEASE / "region_stats.csv"
REGISTRY_OUTPUT = RELEASE / "province_registry.csv"


def main() -> None:
    with REGISTRY_SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    registry = {}
    province_ids = set()
    for row in rows:
        source_component = int(row["source_component"])
        province_id = row["province_id"].strip()
        if not province_id or source_component in registry or province_id in province_ids:
            raise ValueError(f"Duplicate or empty registry entry: {row}")
        registry[source_component] = province_id
        province_ids.add(province_id)

    data = json.loads(GEOJSON.read_text(encoding="utf-8"))
    features = data.get("features", [])
    feature_ids = [int(feature["properties"]["source_component"]) for feature in features]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("GeoJSON contains duplicate source_component values.")

    feature_set = set(feature_ids)
    registry_set = set(registry)
    missing = sorted(feature_set - registry_set)
    orphaned = sorted(registry_set - feature_set)
    if missing or orphaned:
        raise ValueError(
            f"Registry/geometry mismatch: missing registry entries={missing[:20]}, "
            f"orphaned registry entries={orphaned[:20]}"
        )

    for feature in features:
        source_component = int(feature["properties"]["source_component"])
        feature["properties"]["province_id"] = registry[source_component]

    GEOJSON.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    if STATS.exists():
        with STATS.open("r", encoding="utf-8-sig", newline="") as stream:
            stats_reader = csv.DictReader(stream)
            fieldnames = list(stats_reader.fieldnames or [])
            stats_rows = list(stats_reader)
        if "province_id" not in fieldnames:
            fieldnames.append("province_id")
        for row in stats_rows:
            row["province_id"] = registry[int(row["source_component"])]
        with STATS.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_rows)

    with REGISTRY_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["province_id", "source_component"])
        writer.writeheader()
        for source_component in sorted(registry):
            writer.writerow({
                "province_id": registry[source_component],
                "source_component": source_component,
            })

    manifest_path = RELEASE / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", [])
        if "province_registry.csv" not in files:
            files.append("province_registry.csv")
        manifest["canonical_id_policy"] = (
            "province_id is assigned from the committed registry; source_component remains "
            "the current raster segmentation label and is not the persistent identity."
        )
        manifest["registry_entries"] = len(registry)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "passed",
        "registry_entries": len(registry),
        "features_with_province_id": len(features),
        "unique_province_ids": len(province_ids),
        "missing_registry_entries": 0,
        "orphaned_registry_entries": 0,
        "id_example": features[0]["properties"]["province_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
