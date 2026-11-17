#!/usr/bin/env python3
"""Audit contour components omitted at the image border and tiny enclosed regions.

This is diagnostic only: it never changes the source image or polygon GeoJSON.
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
OUT = ROOT / "output" / "component_audit"
OUT.mkdir(parents=True, exist_ok=True)

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
CROP_SIZE = 640
SMALL_LIMIT = 25


def crop_box(cx: int, cy: int, width: int, height: int, size: int = CROP_SIZE):
    half = size // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(width, x0 + size), min(height, y0 + size)
    x0, y0 = max(0, x1 - size), max(0, y1 - size)
    return x0, y0, x1, y1


def main() -> None:
    rgba = np.asarray(Image.open(SOURCE).convert("RGBA"))
    h, w = rgba.shape[:2]
    if (w, h) != (10000, 10000):
        raise ValueError(f"Expected 10000x10000 source, got {w}x{h}")
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3].astype(np.uint16)
    lum = (rgb[:, :, 0] * 299 + rgb[:, :, 1] * 587 + rgb[:, :, 2] * 114) // 1000
    contour = (alpha >= 16) & (lum <= 220)
    del rgba, alpha, rgb, lum

    labels, count = ndimage.label(~contour, structure=CROSS)
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels, max_label=count)

    edge_masks = {
        "top": labels[0, :],
        "bottom": labels[-1, :],
        "left": labels[:, 0],
        "right": labels[:, -1],
    }
    edge_ids = {}
    for side, values in edge_masks.items():
        edge_ids[side] = set(int(v) for v in np.unique(values) if int(v) > 0)
    border_ids = sorted(set().union(*edge_ids.values()))

    border_rows = []
    contact_panels = []
    for component_id in border_ids:
        bbox = objects[component_id - 1]
        if bbox is None:
            continue
        ys, xs = bbox
        min_x, min_y, max_x, max_y = int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)
        sides = [side for side, ids in edge_ids.items() if component_id in ids]
        edge_positions = {}
        for side, values in edge_masks.items():
            positions = np.flatnonzero(values == component_id)
            if len(positions):
                edge_positions[side] = [int(positions.min()), int(positions.max())]
        border_rows.append({
            "component_id": component_id,
            "pixel_area": int(counts[component_id]),
            "bbox": [min_x, min_y, max_x, max_y],
            "width": max_x - min_x,
            "height": max_y - min_y,
            "touches": ";".join(sides),
            "edge_spans_json": json.dumps(edge_positions, separators=(",", ":")),
        })

        # One full-resolution crop for each touched edge, centred on its contact interval.
        for side in sides:
            positions = np.flatnonzero(edge_masks[side] == component_id)
            mid = int((positions.min() + positions.max()) // 2)
            if side == "top":
                cx, cy = mid, 0
            elif side == "bottom":
                cx, cy = mid, h - 1
            elif side == "left":
                cx, cy = 0, mid
            else:
                cx, cy = w - 1, mid
            x0, y0, x1, y1 = crop_box(cx, cy, w, h)
            local = labels[y0:y1, x0:x1] == component_id
            base = np.where(contour[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            panel = Image.fromarray(base, "L").convert("RGB")
            overlay = Image.fromarray(np.zeros((*local.shape, 4), dtype=np.uint8), "RGBA")
            overlay_arr = np.asarray(overlay).copy()
            overlay_arr[local] = (255, 40, 40, 150)
            panel = Image.alpha_composite(panel.convert("RGBA"), Image.fromarray(overlay_arr)).convert("RGB")
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width - 1, 30), fill="white")
            draw.text((8, 8), f"border component {component_id} | {side} | area={int(counts[component_id])}", fill="black")
            contact_panels.append((component_id, side, panel))

    with (OUT / "border_components.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component_id", "pixel_area", "bbox", "width", "height", "touches", "edge_spans_json"])
        writer.writeheader()
        writer.writerows(border_rows)

    # Contact sheet for border components; every crop remains at source resolution.
    thumb_w, thumb_h = 320, 320
    cols = 4
    rows = max(1, (len(contact_panels) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (cid, side, panel) in enumerate(contact_panels):
        panel.thumbnail((thumb_w, thumb_h), Image.Resampling.NEAREST)
        x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + 24)
        sheet.paste(panel, (x, y + 24))
        draw.text((x + 5, y + 5), f"component {cid} / {side}", fill="black")
    sheet.save(OUT / "border_component_contact_sheet.png")

    # Isolate enclosed components below the small-area threshold for visual audit.
    small_ids = [i for i in range(1, count + 1) if counts[i] < SMALL_LIMIT and i not in border_ids]
    small_rows = []
    small_panels = []
    for cid in small_ids:
        bbox = objects[cid - 1]
        if bbox is None:
            continue
        ys, xs = bbox
        min_x, min_y, max_x, max_y = int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)
        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
        x0, y0, x1, y1 = crop_box(cx, cy, w, h, size=192)
        local_contour = contour[y0:y1, x0:x1]
        local_region = labels[y0:y1, x0:x1] == cid
        base = np.where(local_contour, 0, 255).astype(np.uint8)
        panel = Image.fromarray(base, "L").convert("RGB")
        arr = np.asarray(panel).copy()
        arr[local_region] = (255, 80, 80)
        panel = Image.fromarray(arr)
        small_panels.append((cid, int(counts[cid]), panel))
        small_rows.append({
            "component_id": cid, "pixel_area": int(counts[cid]),
            "bbox": [min_x, min_y, max_x, max_y],
            "centroid": [cx, cy],
        })

    with (OUT / "tiny_regions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component_id", "pixel_area", "bbox", "centroid"])
        writer.writeheader()
        writer.writerows(small_rows)

    cols = 6
    thumb = 192
    cell_h = thumb + 28
    rows = max(1, (len(small_panels) + cols - 1) // cols)
    tiny_sheet = Image.new("RGB", (cols * thumb, rows * cell_h), "white")
    draw = ImageDraw.Draw(tiny_sheet)
    for i, (cid, area, panel) in enumerate(small_panels):
        x, y = (i % cols) * thumb, (i // cols) * cell_h
        tiny_sheet.paste(panel, (x, y + 24))
        draw.text((x + 4, y + 5), f"id {cid} | {area}px", fill="black")
    tiny_sheet.save(OUT / "tiny_region_contact_sheet.png")

    summary = {
        "source": SOURCE.name,
        "dimensions": [w, h],
        "connectivity": 4,
        "border_component_count": len(border_ids),
        "border_components": border_rows,
        "tiny_region_threshold_pixels": SMALL_LIMIT,
        "tiny_enclosed_regions_count": len(small_rows),
        "warning": "Diagnostic only. Border-touching components may be exterior background OR legitimate map regions; do not include or discard them without review.",
    }
    (OUT / "component_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "border_components"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
