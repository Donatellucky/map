#!/usr/bin/env python3
"""Build a reviewable graph from raster adjacency candidates.

Only candidates with at least one horizontal/vertical raster contact enter the
provisional adjacency graph. Diagonal-only pairs are kept in a separate file.
This graph is diagnostic and is not yet the authoritative gameplay graph.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = ROOT / "output" / "topology"
RELEASE = ROOT / "output" / "release"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    candidates = read_csv(TOPOLOGY / "raster_contour_adjacency_candidates.csv")
    stats = read_csv(RELEASE / "region_stats.csv")
    area_by_id = {row["province_id"]: int(float(row["pixel_area"])) for row in stats}
    ids = sorted(area_by_id)
    parent = {province_id: province_id for province_id in ids}
    degree = Counter()
    graph_edges = []
    diagonal_only = []

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in candidates:
        directions = set(filter(None, row["directions"].split(";")))
        has_cardinal = bool({"horizontal", "vertical"} & directions)
        if not has_cardinal:
            diagonal_only.append(row)
            continue
        edge = dict(row)
        edge["cardinal_evidence"] = ";".join(sorted({"horizontal", "vertical"} & directions))
        edge["diagonal_support"] = str(bool({"diagonal_down_left", "diagonal_down_right"} & directions)).lower()
        graph_edges.append(edge)
        a, b = row["province_a"], row["province_b"]
        degree[a] += 1
        degree[b] += 1
        union(a, b)

    component_members = defaultdict(list)
    for province_id in ids:
        component_members[find(province_id)].append(province_id)
    components = sorted(component_members.values(), key=lambda values: (-len(values), values[0]))
    isolated = [province_id for province_id in ids if degree[province_id] == 0]

    degree_rows = []
    for province_id in ids:
        degree_rows.append({
            "province_id": province_id,
            "pixel_area": area_by_id[province_id],
            "candidate_degree": degree[province_id],
            "review_note": (
                "isolated_small_region_review" if degree[province_id] == 0 and area_by_id[province_id] <= 100
                else "high_degree_review" if degree[province_id] >= 30
                else "ordinary_candidate_degree"
            ),
        })

    write_csv(TOPOLOGY / "provisional_adjacency_graph.csv", graph_edges, [
        "province_a", "province_b", "source_component_a", "source_component_b",
        "min_contour_pixels", "contact_evidence_count", "directions",
        "cardinal_evidence", "diagonal_support", "status",
    ])
    write_csv(TOPOLOGY / "diagonal_only_contacts.csv", diagonal_only, [
        "province_a", "province_b", "source_component_a", "source_component_b",
        "min_contour_pixels", "contact_evidence_count", "directions", "status",
    ])
    write_csv(TOPOLOGY / "provisional_province_degrees.csv", degree_rows, [
        "province_id", "pixel_area", "candidate_degree", "review_note",
    ])

    degree_distribution = Counter(degree[province_id] for province_id in ids)
    summary = {
        "status": "passed",
        "province_count": len(ids),
        "provisional_cardinal_adjacency_pairs": len(graph_edges),
        "diagonal_only_contacts_not_in_graph": len(diagonal_only),
        "connected_components": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "isolated_provinces": len(isolated),
        "isolated_province_details": [
            {"province_id": province_id, "pixel_area": area_by_id[province_id]}
            for province_id in isolated
        ],
        "max_candidate_degree": max(degree.values(), default=0),
        "high_degree_provinces_30_plus": [
            {"province_id": province_id, "degree": degree[province_id], "pixel_area": area_by_id[province_id]}
            for province_id in ids if degree[province_id] >= 30
        ],
        "degree_distribution": {str(k): v for k, v in sorted(degree_distribution.items())},
        "policy": "Provisional diagnostic graph only; not authoritative gameplay adjacency. Diagonal-only contacts are excluded.",
    }
    (TOPOLOGY / "provisional_adjacency_graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
