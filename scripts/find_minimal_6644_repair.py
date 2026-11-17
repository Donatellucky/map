#!/usr/bin/env python3
"""Find the minimum single contour pixel that separates merged component 6644."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get("SOURCE_PNG",ROOT/"fauxlore_map_KONTUR.png"))
OUT=ROOT/"output"/"component_6644_gap_audit"
OUT.mkdir(parents=True,exist_ok=True)
TARGET=6644
PADDING=40
CROSS=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
SQUARE=np.ones((3,3),dtype=bool)

def count_target_parts(contour, original_target):
    labels,n=ndimage.label(~contour,structure=CROSS)
    ids=np.unique(labels[original_target])
    return sorted(int(v) for v in ids if v>0), labels

def main():
    rgba=np.asarray(Image.open(SOURCE).convert("RGBA"))
    h,w=rgba.shape[:2]
    alpha=rgba[:,:,3]
    rgb=rgba[:,:,:3].astype(np.uint16)
    lum=(rgb[:,:,0]*299+rgb[:,:,1]*587+rgb[:,:,2]*114)//1000
    contour=(alpha>=16)&(lum<=220)
    del rgba,alpha,rgb,lum
    labels,n=ndimage.label(~contour,structure=CROSS)
    objects=ndimage.find_objects(labels,max_label=n)
    sl=objects[TARGET-1]
    if sl is None: raise ValueError(f"component {TARGET} not found")
    ys,xs=sl
    x0=max(0,xs.start-PADDING); x1=min(w,xs.stop+PADDING)
    y0=max(0,ys.start-PADDING); y1=min(h,ys.stop+PADDING)
    local_contour=contour[y0:y1,x0:x1].copy()
    original_target=labels[y0:y1,x0:x1]==TARGET

    closed=ndimage.binary_closing(local_contour,structure=SQUARE)
    candidates=np.argwhere(closed & ~local_contour)
    successful=[]
    for y,x in candidates:
        trial=local_contour.copy()
        trial[y,x]=True
        ids,_=count_target_parts(trial,original_target)
        if len(ids)>=2:
            areas=[int(np.count_nonzero((ndimage.label(~trial,structure=CROSS)[0]==cid)&original_target)) for cid in ids]
            successful.append({"local_xy":[int(x),int(y)],"source_xy":[int(x+x0),int(y+y0)],"parts":len(ids),"part_pixel_areas":areas})
    successful.sort(key=lambda r:(r["parts"],-min(r["part_pixel_areas"])))

    base=np.where(local_contour,0,255).astype(np.uint8)
    panel=np.repeat(base[:,:,None],3,axis=2)
    panel[original_target]=(255,90,90)
    for row in successful:
        x,y=row["local_xy"]
        panel[y,x]=(0,220,0)
    im=Image.fromarray(panel,"RGB")
    draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,im.width-1,34),fill="white")
    draw.text((7,10),f"single-pixel candidates that split component {TARGET}: {len(successful)}",fill="black")
    im.save(OUT/"single_pixel_split_candidates.png")

    result={
      "target_component":TARGET,
      "source_dimensions":[w,h],
      "crop_bbox":[x0,y0,x1,y1],
      "square_1px_added_pixels_tested":int(len(candidates)),
      "successful_single_pixel_repairs":successful,
      "recommendation":"If exactly one candidate splits the component while preserving two substantial parts, inspect its source pixel neighborhood before approving a one-pixel local repair. If none, do not apply square closing; test minimal pairs or a manually defined short segment.",
      "note":"Diagnostic only. No source PNG or production GeoJSON is modified."
    }
    (OUT/"single_pixel_split_candidates.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
