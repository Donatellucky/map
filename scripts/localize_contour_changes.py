#!/usr/bin/env python3
"""Localize experimental contour changes without modifying source geometry.

Uses the conservative 1px cross closing only to identify inspection areas.
The original contour PNG and baseline GeoJSON remain authoritative.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("SOURCE_PNG", ROOT / "fauxlore_map_KONTUR.png"))
OUT = ROOT / "output" / "localized_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
GRID = 10
TOP_N = 8
PADDING = 128


def main() -> None:
    rgba = np.asarray(Image.open(SOURCE).convert("RGBA"))
    h, w = rgba.shape[:2]
    if (w, h) != (10000, 10000):
        raise ValueError(f"Expected 10000x10000 source, got {w}x{h}")
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3].astype(np.uint16)
    luminance = (rgb[:, :, 0] * 299 + rgb[:, :, 1] * 587 + rgb[:, :, 2] * 114) // 1000
    baseline = (alpha >= 16) & (luminance <= 220)
    del rgba, alpha, rgb, luminance

    candidate = ndimage.binary_closing(baseline, structure=STRUCTURE, border_value=0)
    changed = candidate ^ baseline
    tile_h, tile_w = h // GRID, w // GRID
    rows = []
    for row in range(GRID):
        for col in range(GRID):
            y0, y1 = row * tile_h, (row + 1) * tile_h
            x0, x1 = col * tile_w, (col + 1) * tile_w
            count = int(changed[y0:y1, x0:x1].sum())
            rows.append({
                "tile_row": row,
                "tile_col": col,
                "x_min": x0,
                "y_min": y0,
                "x_max": x1,
                "y_max": y1,
                "changed_pixels": count,
            })

    ranked = sorted(rows, key=lambda item: (-item["changed_pixels"], item["tile_row"], item["tile_col"]))
    with (OUT / "tile_change_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: (item["tile_row"], item["tile_col"])))

    # A compact heatmap makes it easy to see whether hotspots cluster together.
    heat = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(heat)
    max_count = max((r["changed_pixels"] for r in rows), default=1) or 1
    for item in rows:
        x0 = item["tile_col"] * 100
        y0 = item["tile_row"] * 100
        intensity = int(235 * item["changed_pixels"] / max_count)
        draw.rectangle((x0, y0, x0 + 99, y0 + 99), fill=(255, 255 - intensity, 255 - intensity), outline="black")
        draw.text((x0 + 5, y0 + 5), f'{item["tile_row"]},{item["tile_col"]}', fill="black")
        draw.text((x0 + 5, y0 + 22), str(item["changed_pixels"]), fill="black")
    heat.save(OUT / "tile_hotspot_heatmap.png")

    crops = []
    for rank, item in enumerate(ranked[:TOP_N], start=1):
        x0 = max(0, item["x_min"] - PADDING)
        y0 = max(0, item["y_min"] - PADDING)
        x1 = min(w, item["x_max"] + PADDING)
        y1 = min(h, item["y_max"] + PADDING)
        base_crop = baseline[y0:y1, x0:x1]
        cand_crop = candidate[y0:y1, x0:x1]
        diff_crop = changed[y0:y1, x0:x1]

        # White background, black contours. Diff panel overlays changed pixels in red.
        base_rgb = np.where(base_crop[:, :, None], 0, 255).astype(np.uint8)
        cand_rgb = np.where(cand_crop[:, :, None], 0, 255).astype(np.uint8)
        diff_rgb = base_rgb.copy()
        diff_rgb[diff_crop] = (255, 0, 0)
        panels = [Image.fromarray(a, "RGB") for a in (base_rgb, cand_rgb, diff_rgb)]
        panel = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
        x = 0
        for p in panels:
            panel.paste(p, (x, 0))
            x += p.width
        filename = f"hotspot_{rank:02d}_r{item['tile_row']:02d}_c{item['tile_col']:02d}.png"
        panel.save(OUT / filename, optimize=True)
        crops.append({
            "rank": rank,
            "tile_row": item["tile_row"],
            "tile_col": item["tile_col"],
            "changed_pixels": item["changed_pixels"],
            "crop_bbox_source_pixels": [x0, y0, x1, y1],
            "file": filename,
        })
        del base_crop, cand_crop, diff_crop, base_rgb, cand_rgb, diff_rgb, panels, panel

    report = {
        "source": SOURCE.name,
        "dimensions": [w, h],
        "grid": f"{GRID}x{GRID}",
        "tile_size_pixels": [tile_w, tile_h],
        "method": "1px cross morphological closing, used only to rank inspection tiles",
        "changed_pixels_total": int(changed.sum()),
        "top_tiles": crops,
        "interpretation": "Hotspots are candidates for visual inspection, not confirmed contour defects. No source or GeoJSON geometry is modified.",
    }
    (OUT / "localized_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
