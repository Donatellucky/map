#!/usr/bin/env python3
"""Measure how small morphological closing radii affect province contour topology.

This is diagnostic only: it never replaces the raw source or the first-pass GeoJSON.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("SOURCE_PNG", ROOT / "fauxlore_map_KONTUR.png"))
OUT = ROOT / "gap_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
RADII = (1, 2, 3)
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def topology(mask: np.ndarray) -> dict:
    labels, count = ndimage.label(~mask, structure=CROSS)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    enclosed = [int(areas[i]) for i in range(1, count + 1)
                if i not in border and areas[i] > 0]
    return {
        "components_including_exterior": int(count),
        "excluded_border_components": int(len(border)),
        "enclosed_regions": int(len(enclosed)),
        "small_regions_under_25px": int(sum(a < 25 for a in enclosed)),
        "small_regions_under_100px": int(sum(a < 100 for a in enclosed)),
        "median_region_area_px": float(np.median(enclosed)) if enclosed else 0,
        "largest_region_area_px": int(max(enclosed)) if enclosed else 0,
    }


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    rgba = np.asarray(Image.open(SOURCE).convert("RGBA"))
    h, w = rgba.shape[:2]
    if (w, h) != (10000, 10000):
        raise ValueError(f"Expected 10000x10000, got {w}x{h}")
    rgb = rgba[:, :, :3].astype(np.uint16)
    lum = (rgb[:, :, 0] * 299 + rgb[:, :, 1] * 587 + rgb[:, :, 2] * 114) // 1000
    contour = (rgba[:, :, 3] >= 16) & (lum <= 220)
    del rgba, rgb, lum

    base = topology(contour)
    rows = [{"radius_px": 0, "added_contour_pixels": 0, **base}]
    preview_scale = 4
    original_preview = contour[::preview_scale, ::preview_scale]
    for radius in RADII:
        closed = ndimage.binary_closing(contour, structure=disk(radius), border_value=0)
        added = closed & ~contour
        stats = topology(closed)
        rows.append({
            "radius_px": radius,
            "added_contour_pixels": int(added.sum()),
            **stats,
        })

        # Diagnostic overlay: original contour in charcoal, newly closed pixels in red.
        canvas = np.zeros((*original_preview.shape, 3), dtype=np.uint8)
        canvas[original_preview] = (45, 45, 45)
        added_preview = added[::preview_scale, ::preview_scale]
        canvas[added_preview] = (235, 45, 45)
        Image.fromarray(canvas, "RGB").save(OUT / f"gap_candidates_radius_{radius}px.png")

        # A separate binary preview makes it easier to compare the resulting contour.
        Image.fromarray((closed[::preview_scale, ::preview_scale] * 255).astype(np.uint8)) \
            .save(OUT / f"closed_contours_radius_{radius}px.png")

    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "source": SOURCE.name,
        "dimensions": [w, h],
        "method": "binary morphological closing of dark opaque contour pixels",
        "connectivity_for_region_count": 4,
        "tested_radii_pixels": list(RADII),
        "baseline": base,
        "comparisons": rows[1:],
        "interpretation": (
            "Closing is diagnostic only. A lower region count can mean gaps were sealed, "
            "but can also mean legitimate nearby regions were merged. Inspect red candidate "
            "pixels before adopting any radius. Raw geometry remains unchanged."
        ),
    }
    (OUT / "gap_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
