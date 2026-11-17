#!/usr/bin/env python3
"""Infer candidate province adjacency across raster contour strokes.

The polygon interiors are separated by the drawn contour pixels, so exact
polygon-boundary intersection is too strict for this source. This script
recreates the 4-connected component labels from the original PNG and looks
for distinct region labels separated by a short, uninterrupted contour run.
Results are candidates for review, not automatically authoritative borders.
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("SOURCE_PNG", ROOT / "fauxlore_map_KONTUR.png"))
RELEASE = ROOT / "output" / "release"
OUT = ROOT / "output" / "topology"
MAX_CONTOUR_RUN = 5
ROW_CHUNK = 512

# One representative direction from each opposite-direction pair.
DIRECTIONS = [
    (0, 1, "horizontal"),
    (1, 0, "vertical"),
    (1, 1, "diagonal_down_right"),
    (1, -1, "diagonal_down_left"),
]


def slices_for(height: int, width: int, y0: int, y1: int, dy: int, dx: int, distance: int):
    delta_y = dy * distance
    delta_x = dx * distance
    if dy == 0:
        ay = slice(y0, y1)
        by = slice(y0, y1)
    else:
        ay = slice(y0, y1)
        by = slice(y0 + delta_y, y1 + delta_y)
    if dx >= 0:
        ax = slice(0, width - delta_x)
        bx = slice(delta_x, width)
    else:
        ax = slice(-delta_x, width)
        bx = slice(0, width + delta_x)
    return (ay, ax), (by, bx)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    rgba = np.asarray(Image.open(SOURCE).convert("RGBA"))
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3].astype(np.uint16)
    luminance = (rgb[:, :, 0] * 299 + rgb[:, :, 1] * 587 + rgb[:, :, 2] * 114) // 1000
    contour = (alpha >= 16) & (luminance <= 220)
    height, width = contour.shape
    del rgba, alpha, rgb, luminance

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(~contour, structure=cross)

    registry_path = RELEASE / "province_registry.csv"
    with registry_path.open("r", encoding="utf-8-sig", newline="") as stream:
        registry_rows = list(csv.DictReader(stream))
    id_by_component = {int(row["source_component"]): row["province_id"] for row in registry_rows}
    known_components = set(id_by_component)
    registry_mask = np.zeros(count + 1, dtype=bool)
    registry_mask[list(known_components)] = True

    # Process only regions that are present in the validated release.
    exterior_candidates = set()
    for side in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        exterior_candidates.update(int(value) for value in np.unique(side) if int(value) > 0)
    touches_all = []
    for component in exterior_candidates:
        if (np.any(labels[0, :] == component) and np.any(labels[-1, :] == component)
                and np.any(labels[:, 0] == component) and np.any(labels[:, -1] == component)):
            touches_all.append(component)
    if len(touches_all) != 1:
        raise ValueError(f"Expected one exterior component; found {touches_all}")
    exterior = touches_all[0]
    if exterior in known_components:
        raise ValueError("Exterior component unexpectedly appears in province registry.")

    pairs = defaultdict(lambda: {"evidence_count": 0, "min_contour_pixels": None, "directions": set()})
    for dy, dx, direction_name in DIRECTIONS:
        max_rows = height if dy == 0 else height - (MAX_CONTOUR_RUN + 1) * dy
        for y0 in range(0, max_rows, ROW_CHUNK):
            y1 = min(max_rows, y0 + ROW_CHUNK)
            for run in range(1, MAX_CONTOUR_RUN + 1):
                distance = run + 1
                a_slice, b_slice = slices_for(height, width, y0, y1, dy, dx, distance)
                a = labels[a_slice]
                b = labels[b_slice]
                valid = (a > 0) & (b > 0) & (a != b)
                for step in range(1, distance):
                    iy0, ix0 = y0 + dy * step, 0
                    iy1 = y1 + dy * step
                    if dx > 0:
                        middle_x = slice(step, width - distance + step)
                    elif dx < 0:
                        middle_x = slice(distance - step, width - step)
                    else:
                        middle_x = slice(0, width)
                    middle_slice = (slice(iy0, iy1), middle_x)
                    valid &= contour[middle_slice]
                if not np.any(valid):
                    continue
                left = a[valid].astype(np.int64, copy=False)
                right = b[valid].astype(np.int64, copy=False)
                valid_registry = registry_mask[left] & registry_mask[right]
                if not np.any(valid_registry):
                    continue
                raw = np.column_stack((left[valid_registry], right[valid_registry]))
                raw.sort(axis=1)
                unique_pairs, counts = np.unique(raw, axis=0, return_counts=True)
                for (component_a, component_b), contact_count in zip(unique_pairs, counts):
                    key = (int(component_a), int(component_b))
                    item = pairs[key]
                    item["evidence_count"] += int(contact_count)
                    item["min_contour_pixels"] = run if item["min_contour_pixels"] is None else min(item["min_contour_pixels"], run)
                    item["directions"].add(direction_name)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (component_a, component_b), item in pairs.items():
        rows.append({
            "province_a": id_by_component[component_a],
            "province_b": id_by_component[component_b],
            "source_component_a": component_a,
            "source_component_b": component_b,
            "min_contour_pixels": item["min_contour_pixels"],
            "contact_evidence_count": item["evidence_count"],
            "directions": ";".join(sorted(item["directions"])),
            "status": "candidate_requires_visual_review",
        })
    rows.sort(key=lambda row: (row["province_a"], row["province_b"]))
    path = OUT / "raster_contour_adjacency_candidates.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "province_a", "province_b", "source_component_a", "source_component_b",
            "min_contour_pixels", "contact_evidence_count", "directions", "status",
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "passed",
        "method": "4-connected raster components; distinct region labels across uninterrupted contour runs",
        "source_dimensions": [width, height],
        "province_count": len(known_components),
        "max_contour_run_pixels": MAX_CONTOUR_RUN,
        "candidate_adjacency_pairs": len(rows),
        "candidate_contact_evidence_total": sum(row["contact_evidence_count"] for row in rows),
        "pairs_by_min_contour_pixels": {
            str(run): sum(1 for row in rows if row["min_contour_pixels"] == run)
            for run in range(1, MAX_CONTOUR_RUN + 1)
        },
        "pairs_with_multiple_directions": sum(1 for row in rows if ";" in row["directions"]),
        "policy": "Candidates only. This stage does not modify geometry, province IDs, or authoritative adjacency.",
        "note": "A short contour run between two regions is evidence of likely shared border, but diagonal/corner and junction cases still require topology review.",
    }
    (OUT / "raster_adjacency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
