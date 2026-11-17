#!/usr/bin/env python3
"""Create full-resolution review crops for the largest polygonized components."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("SOURCE_PNG", ROOT / "fauxlore_map_KONTUR.png"))
OUT = ROOT / "output" / "large_region_audit"
OUT.mkdir(parents=True, exist_ok=True)
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
TOP_N = 10
PADDING = 100


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
    edge_ids = set()
    for values in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        edge_ids.update(int(v) for v in np.unique(values) if int(v) > 0)
    edge_sides = {
        cid: [name for name, values in (
            ("top", labels[0, :]), ("bottom", labels[-1, :]),
            ("left", labels[:, 0]), ("right", labels[:, -1])
        ) if np.any(values == cid)]
        for cid in edge_ids
    }
    exterior = [cid for cid, sides in edge_sides.items() if len(sides) == 4]
    if len(exterior) != 1:
        raise ValueError(f"Expected one exterior component; found {len(exterior)}")

    objects = ndimage.find_objects(labels, max_label=count)
    candidates = [cid for cid in range(1, count + 1) if cid != exterior[0]]
    candidates.sort(key=lambda cid: int(counts[cid]), reverse=True)
    records = []
    thumbs = []
    for rank, cid in enumerate(candidates[:TOP_N], start=1):
        bbox = objects[cid - 1]
        if bbox is None:
            continue
        ys, xs = bbox
        x0, y0 = max(0, xs.start - PADDING), max(0, ys.start - PADDING)
        x1, y1 = min(w, xs.stop + PADDING), min(h, ys.stop + PADDING)
        local_contour = contour[y0:y1, x0:x1]
        local_region = labels[y0:y1, x0:x1] == cid

        # White background, original contour in black, candidate component in translucent red.
        base = np.where(local_contour, 0, 255).astype(np.uint8)
        rgb_panel = np.repeat(base[:, :, None], 3, axis=2)
        rgb_panel[local_region] = (255, 90, 90)
        panel = Image.fromarray(rgb_panel, "RGB")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width - 1, 34), fill="white")
        draw.text((8, 10), f"rank={rank} component={cid} area={int(counts[cid])} px", fill="black")
        name = f"largest_{rank:02d}_component_{cid}.png"
        panel.save(OUT / name)

        thumb = panel.copy()
        thumb.thumbnail((480, 360), Image.Resampling.LANCZOS)
        thumbs.append((rank, cid, int(counts[cid]), thumb))
        records.append({
            "rank_by_pixel_area": rank,
            "component_id": cid,
            "pixel_area": int(counts[cid]),
            "bbox": [int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)],
            "crop_bbox_with_padding": [x0, y0, x1, y1],
            "crop_file": name,
            "touches_image_edge": cid in edge_ids,
            "edge_sides": edge_sides.get(cid, []),
        })

    cols, cell_w, cell_h = 2, 500, 400
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (rank, cid, area, thumb) in enumerate(thumbs):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(thumb, (x + 10, y + 32))
        draw.text((x + 10, y + 8), f"#{rank} component {cid} | {area:,} px", fill="black")
    sheet.save(OUT / "largest_components_contact_sheet.png")

    summary = {
        "source": SOURCE.name,
        "dimensions": [w, h],
        "exterior_component_excluded": exterior[0],
        "top_n": TOP_N,
        "components": records,
        "note": "Red fill marks the candidate component; black lines are source contours. Large area alone does not prove a contour defect.",
    }
    (OUT / "large_region_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
