#!/usr/bin/env python3
"""Audit province topology and build adjacency/spatial lookup indexes.

This is an observational stage: it never edits source geometry or province IDs.
Exact shared line boundaries are treated as confirmed adjacency. Near-but-not-
touching geometries are reported separately as candidates, not auto-connected.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from shapely.geometry import shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "output" / "release"
TILED = ROOT / "output" / "tiled_release"
OUT = ROOT / "output" / "topology"
NEAR_DISTANCE = 1.5
AREA_EPSILON = 1e-8


def main() -> None:
    geojson_path = RELEASE / "provinces.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not features:
        raise ValueError("No province features found.")

    records = []
    geometries = []
    id_to_index = {}
    for feature in features:
        props = feature["properties"]
        province_id = props.get("province_id")
        source_component = int(props["source_component"])
        if not province_id:
            raise ValueError(f"Missing province_id for component {source_component}")
        if province_id in id_to_index:
            raise ValueError(f"Duplicate province_id: {province_id}")
        geom = shape(feature["geometry"])
        if geom.is_empty or not geom.is_valid:
            raise ValueError(f"Invalid geometry: {province_id}")
        id_to_index[province_id] = len(records)
        records.append({
            "province_id": province_id,
            "source_component": source_component,
            "province_class": props.get("province_class", "unclassified"),
            "region_role": props.get("region_role", "province"),
            "review_status": props.get("review_status", "not_reviewed"),
            "pixel_area": float(props.get("pixel_area", geom.area)),
        })
        geometries.append(geom)

    tree = STRtree(geometries)
    adjacency = defaultdict(dict)
    corner_contacts = []
    near_candidates = []
    overlaps = []

    for i, geom in enumerate(geometries):
        for raw_j in tree.query(geom):
            j = int(raw_j)
            if j <= i:
                continue
            other = geometries[j]
            distance = geom.distance(other)
            if distance <= AREA_EPSILON:
                shared = geom.boundary.intersection(other.boundary)
                shared_length = float(shared.length)
                if shared_length > AREA_EPSILON:
                    adjacency[i][j] = shared_length
                    adjacency[j][i] = shared_length
                elif geom.intersects(other):
                    corner_contacts.append({
                        "province_a": records[i]["province_id"],
                        "province_b": records[j]["province_id"],
                        "contact_type": "point_only",
                    })
                intersection_area = float(geom.intersection(other).area)
                if intersection_area > AREA_EPSILON:
                    overlaps.append({
                        "province_a": records[i]["province_id"],
                        "province_b": records[j]["province_id"],
                        "overlap_area": intersection_area,
                    })
            elif distance <= NEAR_DISTANCE:
                near_candidates.append({
                    "province_a": records[i]["province_id"],
                    "province_b": records[j]["province_id"],
                    "distance_pixels": float(distance),
                    "review": "possible_contour_gap_or_narrow_separator",
                })

    # Connected components of the confirmed exact shared-boundary graph.
    visited = set()
    graph_components = []
    for start in range(len(records)):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        members = []
        while queue:
            node = queue.popleft()
            members.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        graph_components.append(members)

    degree_counts = Counter(len(adjacency[i]) for i in range(len(records)))
    edge_rows = []
    for i, neighbors in adjacency.items():
        for j, shared_length in neighbors.items():
            if i < j:
                edge_rows.append({
                    "province_a": records[i]["province_id"],
                    "province_b": records[j]["province_id"],
                    "shared_boundary_length_px": f"{shared_length:.6f}",
                })
    edge_rows.sort(key=lambda row: (row["province_a"], row["province_b"]))

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "province_adjacency.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["province_a", "province_b", "shared_boundary_length_px"])
        writer.writeheader()
        writer.writerows(edge_rows)

    with (OUT / "near_boundary_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["province_a", "province_b", "distance_pixels", "review"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(near_candidates, key=lambda row: (row["distance_pixels"], row["province_a"], row["province_b"])))

    with (OUT / "corner_contacts.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["province_a", "province_b", "contact_type"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(corner_contacts)

    with (OUT / "province_degrees.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["province_id", "source_component", "degree", "province_class", "region_role", "review_status"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(records):
            writer.writerow({
                **{key: record[key] for key in ["province_id", "source_component", "province_class", "region_role", "review_status"]},
                "degree": len(adjacency[i]),
            })

    # Province-to-tile reverse index, derived from the actual packaged tile files.
    tile_manifest_path = TILED / "tile_manifest.json"
    tile_manifest = json.loads(tile_manifest_path.read_text(encoding="utf-8")) if tile_manifest_path.exists() else {"tiles": []}
    province_tiles = defaultdict(list)
    tile_count_check = 0
    for tile in tile_manifest.get("tiles", []):
        tile_path = TILED / tile["path"]
        tile_data = json.loads(tile_path.read_text(encoding="utf-8"))
        for feature in tile_data.get("features", []):
            pid = feature["properties"]["province_id"]
            if tile["tile_id"] not in province_tiles[pid]:
                province_tiles[pid].append(tile["tile_id"])
            tile_count_check += 1
    with (OUT / "province_tiles.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["province_id", "tile_ids", "tile_count"])
        writer.writeheader()
        for pid in sorted(province_tiles):
            tile_ids = sorted(province_tiles[pid], key=lambda value: tuple(map(int, value.split("_"))))
            writer.writerow({"province_id": pid, "tile_ids": ";".join(tile_ids), "tile_count": len(tile_ids)})

    summary = {
        "status": "passed",
        "source_features": len(records),
        "confirmed_shared_boundary_pairs": len(edge_rows),
        "point_only_contacts": len(corner_contacts),
        "near_boundary_candidates": len(near_candidates),
        "overlapping_pairs": len(overlaps),
        "connected_components_in_exact_adjacency_graph": len(graph_components),
        "largest_adjacency_component_size": max((len(group) for group in graph_components), default=0),
        "isolated_provinces_degree_zero": sum(1 for i in range(len(records)) if not adjacency[i]),
        "degree_distribution": {str(k): v for k, v in sorted(degree_counts.items())},
        "tile_count": len(tile_manifest.get("tiles", [])),
        "tile_feature_fragments_indexed": tile_count_check,
        "provinces_with_tile_membership": len(province_tiles),
        "near_boundary_threshold_pixels": NEAR_DISTANCE,
        "interpretation": {
            "confirmed_adjacency": "positive-length exact shared polygon boundary only",
            "near_boundary_candidates": "not adjacency; review candidates potentially separated by source contour pixels",
            "point_only_contacts": "corner contact only; not treated as traversable adjacency",
            "geometry_policy": "read-only audit; no automatic geometry repair or ID changes",
        },
        "known_exceptions": [
            "STF ID mapped from source_component 2848 is intentional non-gameplay space.",
            "STF ID mapped from source_component 6644 is a known merged pair pending manual split.",
            "source_component 23 remains one intentionally large sea province.",
        ],
    }
    (OUT / "topology_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "overlapping_pairs.json").write_text(json.dumps(overlaps, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "passed",
        "provinces": len(records),
        "adjacency_pairs": len(edge_rows),
        "near_boundary_candidates": len(near_candidates),
        "point_contacts": len(corner_contacts),
        "overlaps": len(overlaps),
        "graph_components": len(graph_components),
        "tile_membership_records": len(province_tiles),
    }, indent=2))


if __name__ == "__main__":
    main()
