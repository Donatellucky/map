#!/usr/bin/env python3
"""Create a grouped, review-oriented quality register for the map dataset.

The goal is to classify families of issues and exceptions, not force a manual
one-by-one review of every candidate pair. No geometry or IDs are modified.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "output" / "release"
TOPOLOGY = ROOT / "output" / "topology"
AUDIT = ROOT / "output"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    geo = json.loads((RELEASE / "provinces.geojson").read_text(encoding="utf-8"))
    features = geo["features"]
    by_component = {}
    for feature in features:
        p = feature["properties"]
        by_component[int(p["source_component"])] = p

    tiny = read_csv(AUDIT / "component_audit" / "tiny_regions.csv")
    tiny_counts = Counter(row["review_status"] for row in tiny)
    tiny_rows = []
    for row in tiny:
        component = int(row["component_id"])
        props = by_component.get(component, {})
        tiny_rows.append({
            "province_id": props.get("province_id", ""),
            "source_component": component,
            "pixel_area": int(row["pixel_area"]),
            "review_status": row["review_status"],
            "bbox": row["bbox"],
            "group": "confirmed_tiny_province" if row["review_status"] == "confirmed_province" else "probable_noise_candidate",
            "action": "retain_as_valid_province" if row["review_status"] == "confirmed_tiny_province" else "retain_flagged_do_not_delete_automatically",
        })

    degree_rows = read_csv(TOPOLOGY / "provisional_province_degrees.csv")
    isolated = [r for r in degree_rows if int(r["candidate_degree"]) == 0]
    high_degree = [r for r in degree_rows if int(r["candidate_degree"]) >= 30]
    diagonal = read_csv(TOPOLOGY / "diagonal_only_contacts.csv")
    near = read_csv(TOPOLOGY / "near_boundary_candidates.csv")
    spatial = json.loads((TOPOLOGY / "spatial_package_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((RELEASE / "validation_report.json").read_text(encoding="utf-8"))

    issue_groups = [
        {
            "group_id": "GEO-01",
            "group": "Confirmed small island/province features",
            "count": len(validation.get("confirmed_tiny_provinces", [])),
            "severity": "informational",
            "policy": "Keep as valid province geometry; never discard by pixel area alone.",
            "next_action": "No individual review required unless a later visual defect appears.",
        },
        {
            "group_id": "GEO-02",
            "group": "Probable raster noise among tiny components",
            "count": tiny_counts.get("probable_noise", 0),
            "severity": "low",
            "policy": "Retain and flag; do not auto-delete.",
            "next_action": "Handle as one batch during final cleanup, using visual/contextual rules.",
        },
        {
            "group_id": "GEO-03",
            "group": "Isolated geographic regions",
            "count": len(isolated),
            "severity": "medium",
            "policy": "Keep geometry and ID until classified; isolation alone is not grounds for deletion.",
            "next_action": "Review by category/area and map context as a group.",
        },
        {
            "group_id": "GEO-04",
            "group": "Abnormally high candidate degree",
            "count": len(high_degree),
            "severity": "medium",
            "policy": "Interpret in light of region role and area; large non-gameplay regions can naturally touch many provinces.",
            "next_action": "Exclude non-gameplay areas from gameplay graph; retain geographic links.",
        },
        {
            "group_id": "GEO-05",
            "group": "Diagonal-only raster contacts",
            "count": len(diagonal),
            "severity": "low",
            "policy": "Keep separate from cardinal candidate graph; do not treat as traversable adjacency.",
            "next_action": "No individual review unless a downstream route or adjacency issue exposes a relevant case.",
        },
        {
            "group_id": "GEO-06",
            "group": "Near-boundary pairs not found by raster cardinal scan",
            "count": len(near),
            "severity": "low",
            "policy": "Diagnostic proximity only, not adjacency.",
            "next_action": "Use as a pooled diagnostic dataset; investigate clusters/hotspots, not every pair.",
        },
        {
            "group_id": "GEO-07",
            "group": "Known merged sea region",
            "count": len(validation.get("known_merged_components_pending_split", [])),
            "severity": "medium",
            "policy": "Keep one current node until the smallest local split is visually confirmed.",
            "next_action": "Batch local repair with other confirmed contour-gap defects.",
        },
        {
            "group_id": "GEO-08",
            "group": "Intentional non-gameplay region",
            "count": len(validation.get("non_gameplay_areas", [])),
            "severity": "informational",
            "policy": "Retain in geography; exclude from gameplay systems.",
            "next_action": "No geometry change.",
        },
        {
            "group_id": "GEO-09",
            "group": "Map-edge sea provinces",
            "count": len(validation.get("retained_edge_components", [])) if isinstance(validation.get("retained_edge_components"), list) else int(validation.get("retained_edge_components", 0)),
            "severity": "informational",
            "policy": "Retain author-confirmed sea provinces clipped by image frame.",
            "next_action": "No geometry change.",
        },
    ]

    exception_rows = []
    for component in validation.get("confirmed_tiny_provinces", []):
        p = by_component.get(int(component), {})
        exception_rows.append({
            "province_id": p.get("province_id", ""),
            "source_component": component,
            "exception_group": "confirmed_tiny_province",
            "status": "confirmed_keep",
            "policy": "retain",
        })
    for component in validation.get("non_gameplay_areas", []):
        p = by_component.get(int(component), {})
        exception_rows.append({
            "province_id": p.get("province_id", ""),
            "source_component": component,
            "exception_group": "non_gameplay_area",
            "status": "confirmed_intentional",
            "policy": "geography_only",
        })
    for component in validation.get("known_merged_components_pending_split", []):
        p = by_component.get(int(component), {})
        exception_rows.append({
            "province_id": p.get("province_id", ""),
            "source_component": component,
            "exception_group": "known_merged_sea_region",
            "status": "pending_local_repair",
            "policy": "retain_until_repaired",
        })

    OUT = TOPOLOGY
    write_csv(OUT / "tiny_component_review_register.csv", tiny_rows, [
        "province_id", "source_component", "pixel_area", "review_status", "bbox", "group", "action",
    ])
    write_csv(OUT / "known_exception_register.csv", exception_rows, [
        "province_id", "source_component", "exception_group", "status", "policy",
    ])
    write_csv(OUT / "grouped_quality_issue_register.csv", issue_groups, [
        "group_id", "group", "count", "severity", "policy", "next_action",
    ])

    summary = {
        "status": "passed",
        "province_count": len(features),
        "quality_strategy": "group similar issue types; retain ambiguous geometry; repair only confirmed defect classes in batches",
        "tiny_components_total": len(tiny),
        "confirmed_tiny_provinces": tiny_counts.get("confirmed_province", 0),
        "probable_noise_retained": tiny_counts.get("probable_noise", 0),
        "isolated_geographic_regions": len(isolated),
        "high_degree_regions_30_plus": len(high_degree),
        "diagonal_only_contacts": len(diagonal),
        "near_boundary_pairs": len(near),
        "known_merged_regions": len(validation.get("known_merged_components_pending_split", [])),
        "intentional_non_gameplay_regions": len(validation.get("non_gameplay_areas", [])),
        "retained_edge_sea_provinces": validation.get("retained_edge_components", 0),
        "geometry_validation": {
            "invalid_or_empty_geometries": validation.get("invalid_or_empty_geometries"),
            "geometry_area_mismatches": validation.get("geometry_area_mismatches"),
            "out_of_bounds_geometries": validation.get("out_of_bounds_geometries"),
            "raster_round_trip_mismatched_pixels": validation.get("raster_round_trip_mismatched_pixels"),
        },
        "review_policy": [
            "Do not inspect every adjacency pair individually.",
            "Use grouped counts, clusters, outliers, and representative samples.",
            "Do not delete small regions solely because of area.",
            "Do not apply global morphological closing.",
            "Keep known intentional geography separate from defects.",
            "Apply local geometry repairs in batches only after confirming a reusable defect pattern.",
        ],
    }
    (OUT / "grouped_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
