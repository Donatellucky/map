#!/usr/bin/env python3
"""Package provisional geographic/gameplay adjacency and province spatial index.

This stage separates geography from gameplay eligibility. It does not promote
raster evidence to legally/authoritatively confirmed borders, and does not
modify source geometry or province IDs.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "output" / "release"
TOPOLOGY = ROOT / "output" / "topology"
TILED = ROOT / "output" / "tiled_release"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    geojson = json.loads((RELEASE / "provinces.geojson").read_text(encoding="utf-8"))
    features = geojson["features"]
    provinces = {}
    for feature in features:
        p = feature["properties"]
        pid = p["province_id"]
        if pid in provinces:
            raise ValueError(f"Duplicate province ID: {pid}")
        bbox = p.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Missing/invalid bbox for {pid}")
        provinces[pid] = {
            "province_id": pid,
            "source_component": int(p["source_component"]),
            "bbox": bbox,
            "province_class": p.get("province_class", "unclassified"),
            "region_role": p.get("region_role", "province"),
            "review_status": p.get("review_status", "not_reviewed"),
        }

    tile_rows = read_csv(TOPOLOGY / "province_tiles.csv")
    tile_by_province = {row["province_id"]: row["tile_ids"].split(";") if row["tile_ids"] else [] for row in tile_rows}
    if set(tile_by_province) != set(provinces):
        missing = sorted(set(provinces) - set(tile_by_province))
        extra = sorted(set(tile_by_province) - set(provinces))
        raise ValueError(f"Province/tile index mismatch. Missing={missing[:10]}, extra={extra[:10]}")

    candidate_rows = read_csv(TOPOLOGY / "provisional_adjacency_graph.csv")
    geographic_edges = []
    gameplay_edges = []
    geographic_neighbors = {pid: set() for pid in provinces}
    gameplay_neighbors = {pid: set() for pid in provinces}
    excluded_non_gameplay_edges = []

    for row in candidate_rows:
        a, b = row["province_a"], row["province_b"]
        if a not in provinces or b not in provinces:
            raise ValueError(f"Adjacency references unknown province: {a}, {b}")
        pa, pb = provinces[a], provinces[b]
        edge = {
            "province_a": a,
            "province_b": b,
            "source_component_a": pa["source_component"],
            "source_component_b": pb["source_component"],
            "min_contour_pixels": int(row["min_contour_pixels"]),
            "contact_evidence_count": int(row["contact_evidence_count"]),
            "directions": row["directions"],
            "evidence_type": "raster_contour_cardinal_candidate",
            "status": "provisional_requires_review",
        }
        geographic_edges.append(edge)
        geographic_neighbors[a].add(b)
        geographic_neighbors[b].add(a)

        if pa["region_role"] == "province" and pb["region_role"] == "province":
            gameplay_edges.append(edge)
            gameplay_neighbors[a].add(b)
            gameplay_neighbors[b].add(a)
        else:
            excluded_non_gameplay_edges.append({
                **edge,
                "excluded_endpoint_roles": ";".join(sorted({
                    role for role in (pa["region_role"], pb["region_role"]) if role != "province"
                })),
                "note": "Retained in geographic graph; excluded from gameplay graph.",
            })

    def neighbor_payload(neighbors):
        return {
            pid: sorted(values)
            for pid, values in sorted(neighbors.items())
        }

    # Compact bbox index: tile is the first lookup level, bbox is the second.
    spatial_index = {
        "schema_version": 1,
        "coordinate_system": "source PNG pixels; origin top-left; x right; y down",
        "source_dimensions": [10000, 10000],
        "lookup_strategy": [
            "resolve tile from map coordinate",
            "load candidate province IDs listed for that tile",
            "test point against candidate province bounding boxes",
            "run exact point-in-polygon against the remaining candidates",
        ],
        "provinces": {
            pid: {
                "bbox": data["bbox"],
                "tile_ids": sorted(tile_by_province[pid], key=lambda v: tuple(map(int, v.split("_")))),
                "province_class": data["province_class"],
                "region_role": data["region_role"],
                "review_status": data["review_status"],
            }
            for pid, data in sorted(provinces.items())
        },
    }

    TOPOLOGY.mkdir(parents=True, exist_ok=True)
    edge_fields = [
        "province_a", "province_b", "source_component_a", "source_component_b",
        "min_contour_pixels", "contact_evidence_count", "directions", "evidence_type", "status",
    ]
    write_csv(TOPOLOGY / "geographic_adjacency_provisional.csv", geographic_edges, edge_fields)
    write_csv(TOPOLOGY / "gameplay_adjacency_provisional.csv", gameplay_edges, edge_fields)
    write_csv(TOPOLOGY / "non_gameplay_geographic_edges.csv", excluded_non_gameplay_edges, edge_fields + [
        "excluded_endpoint_roles", "note",
    ])
    (TOPOLOGY / "geographic_neighbors_provisional.json").write_text(
        json.dumps(neighbor_payload(geographic_neighbors), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (TOPOLOGY / "gameplay_neighbors_provisional.json").write_text(
        json.dumps(neighbor_payload(gameplay_neighbors), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (TOPOLOGY / "province_spatial_index.json").write_text(
        json.dumps(spatial_index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    degrees = {pid: len(values) for pid, values in geographic_neighbors.items()}
    gameplay_degrees = {pid: len(values) for pid, values in gameplay_neighbors.items()}
    summary = {
        "status": "passed",
        "province_count": len(provinces),
        "geographic_candidate_edges": len(geographic_edges),
        "gameplay_candidate_edges": len(gameplay_edges),
        "edges_excluded_from_gameplay_due_to_non_gameplay_endpoint": len(excluded_non_gameplay_edges),
        "provinces_with_geographic_neighbors": sum(value > 0 for value in degrees.values()),
        "provinces_with_gameplay_neighbors": sum(value > 0 for value in gameplay_degrees.values()),
        "isolated_geographic_provinces": [
            {"province_id": pid, "source_component": provinces[pid]["source_component"]}
            for pid, value in degrees.items() if value == 0
        ],
        "max_geographic_degree": max(degrees.values(), default=0),
        "max_gameplay_degree": max(gameplay_degrees.values(), default=0),
        "non_gameplay_areas": [
            {"province_id": pid, "source_component": p["source_component"]}
            for pid, p in provinces.items() if p["region_role"] != "province"
        ],
        "known_pending_topology": [
            {"source_component": 6644, "province_id": next((pid for pid,p in provinces.items() if p["source_component"] == 6644), None),
             "issue": "two sea provinces currently merged; retain current single node until precise split"},
        ],
        "tile_index": {
            "tiles": len(json.loads((TILED / "tile_manifest.json").read_text(encoding="utf-8")).get("tiles", [])),
            "provinces_indexed": len(tile_by_province),
            "max_tiles_per_province": max((len(v) for v in tile_by_province.values()), default=0),
        },
        "status_policy": "All adjacency edges remain provisional raster-derived candidates, not authoritative borders.",
        "gameplay_policy": "Geographic graph retains all regions; gameplay graph excludes region_role != province. Sea provinces remain province nodes.",
        "spatial_lookup_policy": "Tile membership and bbox are a candidate filter; exact point-in-polygon is still required.",
    }
    (TOPOLOGY / "spatial_package_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
