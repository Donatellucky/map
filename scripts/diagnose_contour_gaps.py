#!/usr/bin/env python3
"""Measure sensitivity of contour polygonization to tiny raster gaps.

This script DOES NOT modify the source PNG or the raw GeoJSON. It compares
baseline connected components with experimental morphological closing masks.
The variants are diagnostic only: a changed component count is not proof that
the corresponding gap should be repaired.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("SOURCE_PNG", ROOT / "fauxlore_map_KONTUR.png"))
OUT = ROOT / "output" / "gap_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

CROSS_1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
SQUARE_1 = np.ones((3, 3), dtype=bool)
DIAMOND_2 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=bool)

def summarize(contour: np.ndarray) -> tuple[dict, np.ndarray]:
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(~contour, structure=cross)
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    border_ids = np.unique(np.concatenate((
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    )))
    border = np.zeros(count + 1, dtype=bool)
    border[border_ids] = True
    enclosed_ids = np.flatnonzero((counts > 0) & ~border)
    enclosed_ids = enclosed_ids[enclosed_ids != 0]
    areas = counts[enclosed_ids]
    result = {
        "components_including_background": int(count),
        "border_components": int(len(border_ids)),
        "enclosed_regions": int(len(enclosed_ids)),
        "regions_under_25_pixels": int(np.sum(areas < 25)),
        "regions_under_100_pixels": int(np.sum(areas < 100)),
        "smallest_region_pixels": int(areas.min()) if len(areas) else None,
        "median_region_pixels": float(np.median(areas)) if len(areas) else None,
        "largest_region_pixels": int(areas.max()) if len(areas) else None,
    }
    return result, labels

def save_preview(labels: np.ndarray, path: Path) -> None:
    small = labels[::4, ::4]
    max_id = int(small.max())
    ids = np.arange(max_id + 1, dtype=np.uint64)
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    palette[:, 0] = ((ids * 67 + 43) % 220 + 25).astype(np.uint8)
    palette[:, 1] = ((ids * 97 + 71) % 220 + 25).astype(np.uint8)
    palette[:, 2] = ((ids * 131 + 19) % 220 + 25).astype(np.uint8)
    palette[0] = (0, 0, 0)
    Image.fromarray(palette[small], "RGB").save(path)

def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source image not found: {SOURCE}")
    rgba = np.asarray(Image.open(SOURCE).convert("RGBA"))
    h, w = rgba.shape[:2]
    if (w, h) != (10000, 10000):
        raise ValueError(f"Expected 10000x10000 source, got {w}x{h}")
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3].astype(np.uint16)
    luminance = (rgb[:, :, 0] * 299 + rgb[:, :, 1] * 587 + rgb[:, :, 2] * 114) // 1000
    baseline = (alpha >= 16) & (luminance <= 220)
    del rgba, alpha, rgb, luminance

    variants = {
        "baseline": None,
        "closing_cross_1px": CROSS_1,
        "closing_square_1px": SQUARE_1,
        "closing_diamond_2px": DIAMOND_2,
    }
    report = {
        "source": SOURCE.name,
        "dimensions": [w, h],
        "method": "binary morphological closing of contour mask; 4-connected complement labels",
        "warning": "Experimental variants are diagnostic only; do not replace source geometry automatically.",
        "variants": {},
    }
    baseline_stats, baseline_labels = summarize(baseline)
    report["variants"]["baseline"] = {
        **baseline_stats,
        "contour_pixels": int(baseline.sum()),
        "changed_contour_pixels_vs_baseline": 0,
    }
    save_preview(baseline_labels, OUT / "baseline_regions.png")
    del baseline_labels

    for name, structure in list(variants.items())[1:]:
        candidate = ndimage.binary_closing(baseline, structure=structure, border_value=0)
        changed = candidate ^ baseline
        stats, labels = summarize(candidate)
        report["variants"][name] = {
            **stats,
            "contour_pixels": int(candidate.sum()),
            "changed_contour_pixels_vs_baseline": int(changed.sum()),
            "newly_added_contour_pixels": int(np.sum(candidate & ~baseline)),
            "removed_contour_pixels": int(np.sum(baseline & ~candidate)),
            "delta_enclosed_regions": int(stats["enclosed_regions"] - baseline_stats["enclosed_regions"]),
            "delta_regions_under_25": int(stats["regions_under_25_pixels"] - baseline_stats["regions_under_25_pixels"]),
        }
        save_preview(labels, OUT / f"{name}_regions.png")
        # Red = pixels changed by the experimental operation, black = unchanged.
        diff_small = changed[::4, ::4]
        diff_rgb = np.zeros((*diff_small.shape, 3), dtype=np.uint8)
        diff_rgb[diff_small] = (255, 50, 50)
        Image.fromarray(diff_rgb, "RGB").save(OUT / f"{name}_changed_pixels.png")
        del candidate, changed, labels

    (OUT / "gap_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
