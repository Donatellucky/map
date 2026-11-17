#!/usr/bin/env python3
"""Cross-validate the geometry, stable-ID, tile and spatial-index handoff."""
import csv, hashlib, json, os, re
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"; REL=OUT/"release"; TILED=OUT/"tiled_release"; TOPO=OUT/"topology"; DOC=OUT/"integration"
def js(p): return json.loads(p.read_text(encoding="utf-8"))
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def need(ok,msg,errors):
    if not ok: errors.append(msg)
def main():
    errors=[]
    paths=[REL/"manifest.json",REL/"provinces.geojson",REL/"province_registry.csv",REL/"validation_report.json",
           TILED/"tile_manifest.json",TOPO/"province_tiles.csv",TOPO/"province_spatial_index.json",
           TOPO/"geographic_adjacency_provisional.csv",TOPO/"gameplay_adjacency_provisional.csv"]
    for p in paths: need(p.is_file(),f"Missing {p.relative_to(OUT)}",errors)
    if errors: raise SystemExit("\n".join(errors))
    m=js(REL/"manifest.json"); v=js(REL/"validation_report.json"); g=js(REL/"provinces.geojson")
    tm=js(TILED/"tile_manifest.json"); si=js(TOPO/"province_spatial_index.json")
    fs=g.get("features",[]); byid={}; bycomp={}
    need(v.get("status")=="passed","Geometry validation not passed",errors)
    for k in ("invalid_or_empty_geometries","geometry_area_mismatches","out_of_bounds_geometries","raster_round_trip_mismatched_pixels"):
        need(v.get(k)==0,f"Geometry validation {k} != 0",errors)
    need(len(fs)==m.get("feature_count")==v.get("feature_count"),"Feature counts disagree",errors)
    for f in fs:
        p=f.get("properties",{}); pid=p.get("province_id"); comp=p.get("source_component")
        need(isinstance(pid,str) and re.fullmatch(r"STF-\\d{5}",pid or "") is not None,f"Bad province_id {pid}",errors)
        need(pid not in byid,f"Duplicate province_id {pid}",errors); byid[pid]=p
        try:
            comp=int(comp); need(comp not in bycomp,f"Duplicate source_component {comp}",errors); bycomp[comp]=pid
        except (TypeError,ValueError): errors.append(f"Invalid source_component for {pid}")
        need(f.get("geometry",{}).get("type") in ("Polygon","MultiPolygon"),f"Bad geometry type {pid}",errors)
        need(isinstance(p.get("bbox"),list) and len(p["bbox"])==4,f"Bad bbox {pid}",errors)
    reg=rows(REL/"province_registry.csv")
    rp={(int(x["source_component"]),x["province_id"]) for x in reg}
    fp={(int(p["source_component"]),pid) for pid,p in byid.items()}
    need(len(rp)==len(reg) and rp==fp,"Registry and GeoJSON ID mapping differ",errors)
    entries=tm.get("tiles",[]); membership=defaultdict(set); fragments=0; tileids=set()
    need(tm.get("source_feature_count")==len(fs),"Tile source feature count mismatch",errors)
    need(tm.get("tile_count")==len(entries),"Tile count mismatch",errors)
    need(tm.get("area_conservation",{}).get("status")=="passed","Tile area conservation not passed",errors)
    for t in entries:
        tid=t["tile_id"]; tileids.add(tid); path=TILED/t["path"]
        need(path.is_file(),f"Missing tile {t['path']}",errors)
        if not path.is_file(): continue
        data=js(path); bounds=t["bounds"]; tx,ty=map(int,tid.split("_"))
        need(path.relative_to(TILED).as_posix()==f"tiles/{ty}/{tx}.geojson",f"Bad tile path {tid}",errors)
        need(data.get("properties",{}).get("tile_id")==tid and data["properties"].get("bounds")==bounds,f"Tile metadata mismatch {tid}",errors)
        items=data.get("features",[]); fragments+=len(items)
        need(len(items)==t["feature_fragments"],f"Tile fragment count mismatch {tid}",errors)
        x0,y0,x1,y1=bounds
        for f in items:
            p=f.get("properties",{}); pid=p.get("province_id")
            need(pid in byid,f"Unknown province {pid} in tile {tid}",errors)
            need(p.get("tile_id")==tid,f"Fragment tile ID mismatch {pid}/{tid}",errors)
            if pid in byid: membership[pid].add(tid)
            geom=f.get("geometry",{}); polys=[geom.get("coordinates",[])] if geom.get("type")=="Polygon" else geom.get("coordinates",[])
            for poly in polys:
                for ring in poly:
                    for x,y in ring:
                        if x<x0-1e-7 or x>x1+1e-7 or y<y0-1e-7 or y>y1+1e-7:
                            errors.append(f"Coordinate outside tile bounds: {pid}/{tid}"); break
    need(fragments==tm.get("tile_fragment_count"),"Total tile fragment count mismatch",errors)
    need(set(byid)==set(membership),"Province tile coverage incomplete",errors)
    csvidx={r["province_id"]:set(filter(None,r["tile_ids"].split(";"))) for r in rows(TOPO/"province_tiles.csv")}
    sidx=si.get("provinces",{})
    need(set(csvidx)==set(byid)==set(sidx),"Spatial index province set mismatch",errors)
    for pid,p in byid.items():
        need(csvidx.get(pid,set())==membership.get(pid,set()),f"CSV tile index mismatch {pid}",errors)
        need(set(sidx.get(pid,{}).get("tile_ids",[]))==membership.get(pid,set()),f"JSON tile index mismatch {pid}",errors)
        need(sidx.get(pid,{}).get("bbox")==p.get("bbox"),f"Spatial bbox mismatch {pid}",errors)
    need(si.get("coordinate_system")==m.get("coordinate_system"),"Coordinate system mismatch",errors)
    need(si.get("source_dimensions")==m.get("source_dimensions"),"Source dimensions mismatch",errors)
    edge_counts={}
    for name in ("geographic_adjacency_provisional.csv","gameplay_adjacency_provisional.csv"):
        rr=rows(TOPO/name); edge_counts[name]=len(rr)
        for e in rr:
            a,b=e.get("province_a"),e.get("province_b")
            need(a in byid and b in byid and a!=b,f"Invalid adjacency endpoint {a}/{b}",errors)
            need(e.get("status")=="provisional_requires_review",f"Adjacency not marked provisional in {name}",errors)
    if errors: raise SystemExit("Integration contract failed:\n- "+"\n- ".join(errors[:100]))
    DOC.mkdir(parents=True,exist_ok=True); sha=os.getenv("GITHUB_SHA","unknown")
    contract={"status":"passed","source_commit":sha,"package_version":m.get("version"),
      "source_dimensions":m.get("source_dimensions"),"coordinate_system":m.get("coordinate_system"),
      "province_count":len(byid),"tile_count":len(entries),"tile_fragment_count":fragments,
      "registry_entries":len(reg),"spatial_index_entries":len(sidx),
      "geographic_adjacency_candidates":edge_counts["geographic_adjacency_provisional.csv"],
      "gameplay_adjacency_candidates":edge_counts["gameplay_adjacency_provisional.csv"],
      "checks":{"geometry":"passed","registry_parity":"passed","tile_files_and_bounds":"passed",
      "tile_index_parity":"passed","spatial_bbox_parity":"passed","adjacency_endpoints":"passed",
      "adjacency_semantics":"provisional_only"},
      "known_exceptions":m.get("known_exceptions",[]),
      "integration_boundary":"Geometry/storage indexes only; no Leaflet transform, projection, political ownership, fog-of-war, or authoritative gameplay adjacency."}
    (DOC/"integration_contract.json").write_text(json.dumps(contract,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    guide=f"""# Fauxlore geometry package — integration guide

- Package: {m.get('version')}
- Source commit: {sha}
- Source raster: {m.get('source_image')} ({m.get('source_dimensions')} px)
- Provinces: {len(byid)}; tiles: {len(entries)}; tile fragments: {fragments}
- IDs: persistent `STF-00001` format, assigned by the committed registry.

## Coordinate contract

Coordinates are **source PNG pixels**: origin top-left, X right, Y down. This is not a geographic CRS. No Leaflet CRS or source-pixel transform is included. The main project must explicitly define and test its transform against the 10,000 × 10,000 source image.

## Package layout

- `release/provinces.geojson`: canonical full province geometries and properties.
- `release/province_registry.csv`: persistent `province_id` ↔ `source_component` mapping. Never regenerate IDs from feature order.
- `tiled_release/tile_manifest.json`: fixed-grid scheme, bounds and file paths.
- `tiled_release/tiles/{{y}}/{{x}}.geojson`: clipped fragments retaining the original province ID.
- `topology/province_spatial_index.json`: bbox and tile membership by province.
- `topology/province_tiles.csv`: tabular tile membership.
- `topology/*adjacency_provisional.csv`: raster-derived candidate graphs only.

## Runtime lookup contract

1. Transform pointer/map coordinates into source-image pixels using the separately validated project transform.
2. Resolve tile coordinates using tile size {tm.get('tile_scheme',{}).get('tile_size_pixels')} px.
3. Load candidate province IDs and filter them by bbox.
4. Run exact point-in-polygon against canonical geometry (or an equivalently validated geometry index).
5. Resolve political ownership and Terra incognita/visibility from separate state data keyed by `province_id`.

Tiles are storage partitions, not provinces. Deduplicate candidates by `province_id`; a province can occur in multiple tiles.

## Separation of concerns

Geometry defines location; the registry defines identity; tile/spatial indexes accelerate lookup. Ownership, visibility, map modes, resources and other gameplay state remain separate layers. Adjacency files are **provisional**, not authoritative borders or movement rules. Do not wire them into gameplay without a dedicated topology validation stage.

## Known exceptions

"""+ "\n".join("- "+x for x in m.get("known_exceptions",[]))+"""

## Main-project acceptance checklist

- [ ] Verify the source-pixel transform and image alignment at multiple zoom levels.
- [ ] Import and preserve canonical IDs from the registry.
- [ ] Verify a province crossing tile boundaries remains one selectable entity.
- [ ] Verify sea-province selection and default boundary visibility policy.
- [ ] Keep Terra incognita/ownership independent from geometry.
- [ ] Keep provisional adjacency out of movement/gameplay rules.
- [ ] Profile loading and point-in-polygon performance before replacing the current renderer.

This is a geometry/storage handoff candidate, not a drop-in Leaflet implementation. Main-repository changes require separate approval.
"""
    (DOC/"INTEGRATION_GUIDE.md").write_text(guide,encoding="utf-8")
    roots=(REL,TILED,TOPO,DOC); files=sorted(p for root in roots for p in root.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (DOC/"SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}\n" for p in files),encoding="utf-8")
    print(json.dumps({"status":"passed","provinces":len(byid),"tiles":len(entries),"fragments":fragments,
      "registry_entries":len(reg),"spatial_index_entries":len(sidx),"checksummed_files":len(files),
      "source_commit":sha},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
