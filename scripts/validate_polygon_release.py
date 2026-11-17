#!/usr/bin/env python3
"""Release gate: verify that exported GeoJSON exactly reconstructs the source regions."""
from __future__ import annotations
import csv, json, os, shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get("SOURCE_PNG",ROOT/"fauxlore_map_KONTUR.png"))
OUT=ROOT/"output"
RELEASE=OUT/"release"
EXPECTED_SEA={9,15,23,24,30,36,5159,6628,39,100,364,793,1409,1875,2397,3463,3569,3810,3979,4280,4441,4676}
EXPECTED_TINY={615,741,776,848,1499,4176,4328,5006,5012,5706,6635}
NON_GAMEPLAY={2848}
KNOWN_MERGED={6644}
CROSS=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

def main():
    geo_path=OUT/"provinces_raw.geojson"
    stats_path=OUT/"region_stats.csv"
    summary_path=OUT/"summary.json"
    for p in (SOURCE,geo_path,stats_path,summary_path):
        if not p.exists(): raise FileNotFoundError(p)

    data=json.loads(geo_path.read_text(encoding="utf-8"))
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    features=data.get("features",[])
    if len(features)!=summary.get("enclosed_regions_exported"):
        raise ValueError("GeoJSON feature count does not match summary")

    ids=[int(f["id"]) for f in features]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate feature IDs")
    by_id={int(f["id"]):f for f in features}
    missing_sea=sorted(EXPECTED_SEA-set(ids))
    missing_tiny=sorted(EXPECTED_TINY-set(ids))
    if missing_sea or missing_tiny: raise ValueError(f"Missing author-confirmed regions: sea={missing_sea}, tiny={missing_tiny}")

    geom_types=Counter()
    invalid=[]
    area_mismatches=[]
    bbox_errors=[]
    raster_shapes=[]
    area_sum=0
    for f in features:
        fid=int(f["id"]); props=f["properties"]; geom=shape(f["geometry"])
        geom_types[geom.geom_type]+=1
        if geom.is_empty or not geom.is_valid: invalid.append(fid)
        if geom.geom_type!="Polygon": raise ValueError(f"Feature {fid} is {geom.geom_type}, expected Polygon")
        if not (-1e-8<=geom.bounds[0] and -1e-8<=geom.bounds[1] and geom.bounds[2]<=10000+1e-8 and geom.bounds[3]<=10000+1e-8):
            bbox_errors.append(fid)
        if abs(geom.area-int(props["pixel_area"]))>1e-7:
            area_mismatches.append(fid)
        area_sum+=int(props["pixel_area"])
        raster_shapes.append((f["geometry"],fid))
    if invalid or area_mismatches or bbox_errors:
        raise ValueError(f"Geometry validation failed: invalid={invalid[:10]}, area={area_mismatches[:10]}, bbox={bbox_errors[:10]}")

    # Rebuild the raster component labels and burn every exported polygon back
    # into the original pixel grid. Exact equality is the release criterion.
    rgba=np.asarray(Image.open(SOURCE).convert("RGBA"))
    h,w=rgba.shape[:2]
    if (w,h)!=(10000,10000): raise ValueError(f"Unexpected source dimensions {(w,h)}")
    alpha=rgba[:,:,3]
    rgb=rgba[:,:,:3].astype(np.uint16)
    lum=(rgb[:,:,0]*299+rgb[:,:,1]*587+rgb[:,:,2]*114)//1000
    contour=(alpha>=16)&(lum<=220)
    del rgba,alpha,rgb,lum
    labels,count=ndimage.label(~contour,structure=CROSS)
    edge_sets=[set(int(v) for v in np.unique(a) if int(v)>0) for a in (labels[0,:],labels[-1,:],labels[:,0],labels[:,-1])]
    edge_ids=set().union(*edge_sets)
    all_sides=set.intersection(*edge_sets)
    if len(all_sides)!=1: raise ValueError(f"Expected one exterior component touching all edges, got {sorted(all_sides)}")
    exterior_id=next(iter(all_sides))
    if exterior_id!=int(data["properties"]["excluded_exterior_component"]):
        raise ValueError("GeoJSON exterior component metadata mismatch")
    expected=labels.copy()
    expected[expected==exterior_id]=0
    burned=rasterize(raster_shapes,out_shape=(h,w),transform=Affine.identity(),fill=0,dtype="int32")
    mismatch=int(np.count_nonzero(expected!=burned))
    if mismatch: raise ValueError(f"Raster round-trip mismatch: {mismatch} pixels")

    if by_id[2848]["properties"].get("province_class")!="non_gameplay":
        raise ValueError("Component 2848 must remain non_gameplay")
    if by_id[2848]["properties"].get("review_status")!="confirmed_non_gameplay":
        raise ValueError("Component 2848 must be marked confirmed_non_gameplay")
    if by_id[6644]["properties"].get("review_status")!="known_merged_components_pending_split":
        raise ValueError("Component 6644 must remain flagged as a known merge")
    if by_id[6644]["properties"].get("province_class")!="sea":
        raise ValueError("Component 6644 should remain classified as sea pending split")

    with stats_path.open(encoding="utf-8",newline="") as f:
        csv_rows=list(csv.DictReader(f))
    if len(csv_rows)!=len(features): raise ValueError("CSV row count mismatch")
    csv_ids={int(row["source_component"]) for row in csv_rows}
    if csv_ids!=set(ids): raise ValueError("CSV component IDs do not match GeoJSON")

    RELEASE.mkdir(parents=True,exist_ok=True)
    report={
      "status":"passed",
      "source":SOURCE.name,
      "dimensions":[w,h],
      "feature_count":len(features),
      "geometry_types":dict(geom_types),
      "unique_ids":len(set(ids)),
      "invalid_or_empty_geometries":len(invalid),
      "geometry_area_mismatches":len(area_mismatches),
      "out_of_bounds_geometries":len(bbox_errors),
      "raster_round_trip_mismatched_pixels":mismatch,
      "raster_component_count":int(count),
      "excluded_exterior_component":exterior_id,
      "retained_edge_components":len(edge_ids)-1,
      "confirmed_sea_components":sorted(EXPECTED_SEA),
      "confirmed_tiny_provinces":sorted(EXPECTED_TINY),
      "non_gameplay_areas":sorted(NON_GAMEPLAY),
      "known_merged_components_pending_split":sorted(KNOWN_MERGED),
      "sum_of_feature_pixel_areas":area_sum,
      "coordinate_system":"source PNG pixels; origin top-left; x right; y down",
      "notes":[
        "Component 2848 is an intentional non-gameplay area; exclude it from gameplay mechanics, not from geometry.",
        "Component 6644 represents two sea provinces currently merged by an open contour gap; retained and explicitly flagged pending a precise split.",
        "Component 23 is intentionally retained as one large sea province pending any later source-art correction.",
        "Small components tagged probable_noise are retained, not deleted.",
        "No global morphological closing has been applied."
      ]
    }
    (RELEASE/"validation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest={
      "package":"Fauxlore province geometry — pre-tile release candidate",
      "version":"0.1.0-rc1",
      "source_image":SOURCE.name,
      "source_dimensions":[w,h],
      "coordinate_system":report["coordinate_system"],
      "files":["provinces.geojson","region_stats.csv","summary.json","validation_report.json"],
      "feature_count":len(features),
      "stable_id":"source connected-component label",
      "validation":"exact raster round-trip; Shapely validity; pixel-area and bounds checks; CSV/GeoJSON ID parity",
      "scope":"geometry extraction only; no gameplay visibility, political ownership, map projection, or Leaflet transforms",
      "known_exceptions":report["notes"]
    }
    (RELEASE/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copy2(geo_path,RELEASE/"provinces.geojson")
    shutil.copy2(stats_path,RELEASE/"region_stats.csv")
    shutil.copy2(summary_path,RELEASE/"summary.json")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
