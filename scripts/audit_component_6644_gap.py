#!/usr/bin/env python3
"""Test strictly local contour-closing candidates for author-confirmed component 6644."""
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

def main():
    rgba=np.asarray(Image.open(SOURCE).convert("RGBA"))
    h,w=rgba.shape[:2]
    alpha=rgba[:,:,3]
    rgb=rgba[:,:,:3].astype(np.uint16)
    lum=(rgb[:,:,0]*299+rgb[:,:,1]*587+rgb[:,:,2]*114)//1000
    contour=(alpha>=16)&(lum<=220)
    del rgba,alpha,rgb,lum
    labels,count=ndimage.label(~contour,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8))
    objs=ndimage.find_objects(labels,max_label=count)
    sl=objs[TARGET-1]
    if sl is None: raise ValueError(f"component {TARGET} missing")
    ys,xs=sl
    x0=max(0,xs.start-PADDING); x1=min(w,xs.stop+PADDING)
    y0=max(0,ys.start-PADDING); y1=min(h,ys.stop+PADDING)
    local_contour=contour[y0:y1,x0:x1]
    local_target=labels[y0:y1,x0:x1]==TARGET
    structures={
      "cross_1px":np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool),
      "square_1px":np.ones((3,3),dtype=bool),
      "cross_2px":ndimage.generate_binary_structure(2,1),
      "square_2px":np.ones((5,5),dtype=bool),
      "diamond_2px":np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=bool),
    }
    records=[]
    panels=[]
    base=np.where(local_contour,0,255).astype(np.uint8)
    for name,structure in structures.items():
        closed=ndimage.binary_closing(local_contour,structure=structure)
        # Do not count crop exterior; count connected regions that contain original target pixels.
        labs,n=ndimage.label(~closed,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8))
        ids=np.unique(labs[local_target]); ids=[int(v) for v in ids if v>0]
        overlaps=[]
        for cid in ids:
            overlaps.append({"local_label":cid,"original_target_pixels":int(np.count_nonzero((labs==cid)&local_target))})
        changed=int(np.count_nonzero(closed!=local_contour))
        rgb_panel=np.repeat(base[:,:,None],3,axis=2)
        rgb_panel[local_target]=(255,90,90)
        # Newly added contour pixels in blue; removed contour pixels yellow.
        rgb_panel[closed & ~local_contour]=(40,100,255)
        rgb_panel[local_contour & ~closed]=(255,220,0)
        im=Image.fromarray(rgb_panel,"RGB")
        d=ImageDraw.Draw(im); d.rectangle((0,0,im.width-1,28),fill="white")
        d.text((6,8),f"{name}: target splits={len(ids)}, changed={changed}px",fill="black")
        filename=f"{name}.png"; im.save(OUT/filename)
        thumb=im.copy(); thumb.thumbnail((500,420),Image.Resampling.LANCZOS)
        panels.append((name,len(ids),changed,thumb))
        records.append({"method":name,"target_subregions_after_closing":len(ids),"changed_contour_pixels":changed,"overlap":overlaps,"image":filename})
    cols=2; cw=520; ch=460
    sheet=Image.new("RGB",(cols*cw,((len(panels)+1)//2)*ch),"white")
    d=ImageDraw.Draw(sheet)
    for i,(name,splits,changed,im) in enumerate(panels):
        x=(i%cols)*cw; y=(i//cols)*ch
        d.text((x+8,y+8),f"{name} | split regions={splits} | changes={changed}",fill="black")
        sheet.paste(im,(x+8,y+34))
    sheet.save(OUT/"candidate_closings_contact_sheet.png")
    summary={"target_component":TARGET,"source_dimensions":[w,h],"target_bbox":[xs.start,ys.start,xs.stop,ys.stop],"crop_bbox":[x0,y0,x1,y1],"methods":records,"warning":"Diagnostic only: no source pixels or production GeoJSON are modified. Choose the smallest local repair only after visual confirmation."}
    (OUT/"component_6644_gap_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
