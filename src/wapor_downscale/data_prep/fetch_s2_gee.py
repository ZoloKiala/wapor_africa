"""For each WaPOR L3 dekad, build a cloud-masked Sentinel-2 composite over the AOI and download it.

Produces per-dekad GeoTIFFs in `data/baixo/s2/` with bands B4, B8, B11 (reflectance * 10000).
Reprojection to the L3 grid happens later in the stack-builder.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import ee
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (
    abspath,
    dekads_between,
    ensure_dir,
    gcs_public_url,
    load_config,
    parse_iso,
)
from gee_utils import export_image_to_local, init_ee


def aoi_bounds_from_l3(cfg: dict, site_code: str) -> tuple[float, float, float, float]:
    bucket = cfg["wapor"]["bucket"]
    prefix = cfg["wapor"]["l3_data_prefix"]
    mapset = cfg["wapor"]["l3_target_mapset"]
    sample = f"{prefix}/WAPOR-3.{mapset}.{site_code}.2018-01-D1.tif"
    url = "/vsicurl/" + gcs_public_url(bucket, sample)
    with rasterio.open(url) as ds:
        return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)


def tile_bounds(west: float, south: float, east: float, north: float, n_cols: int, n_rows: int):
    """Yield (idx, w, s, e, n) for each sub-tile of the bbox, row-major."""
    dx = (east - west) / n_cols
    dy = (north - south) / n_rows
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            w = west + c * dx
            e = west + (c + 1) * dx
            s = south + r * dy
            n = south + (r + 1) * dy
            yield idx, w, s, e, n
            idx += 1


def s2_composite(date_start: "ee.Date", date_end: "ee.Date", region, cloud_max: float) -> "ee.Image":
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
    )
    # Mask clouds using SCL: keep classes 4,5,6,7,11 (veg, bare, water, unclassified, snow)
    def _mask(img):
        scl = img.select("SCL")
        good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
        return img.updateMask(good)

    masked = coll.map(_mask).select(["B4", "B8", "B11"])
    median = masked.median().toUint16()
    return median.clip(region)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    ap.add_argument("--site-code", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)
    ap.add_argument("--tile-cols", type=int, default=4,
                    help="Split AOI into this many columns when calling getDownloadURL.")
    ap.add_argument("--tile-rows", type=int, default=3,
                    help="Split AOI into this many rows when calling getDownloadURL.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_ee(cfg)

    site_code = args.site_code or cfg["site"].get("l3_site_code")
    if not site_code:
        sys.exit("ERROR: need site code.")

    west, south, east, north = aoi_bounds_from_l3(cfg, site_code)
    full_region = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
    start = parse_iso(args.start or cfg["time_range"]["start"])
    end = parse_iso(args.end or cfg["time_range"]["end"])
    out_dir = ensure_dir(abspath(cfg["paths"]["s2_dir"]))
    tmp_dir = ensure_dir(out_dir / "_tiles")
    window_days = cfg["sentinel2"]["composite_window_days"]
    cloud_max = cfg["sentinel2"]["cloud_max"]
    scale = cfg["sentinel2"]["scale_m"]
    n_tiles = args.tile_cols * args.tile_rows

    n_ok = n_err = 0
    for code, dstart, dend in dekads_between(start, end):
        out_path = out_dir / f"S2_{code}.tif"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        # Widen the window a bit beyond the dekad to find a cloud-free pixel.
        pad = max(0, (window_days - (dend - dstart).days - 1) // 2)
        ee_start = ee.Date(dstart.isoformat()).advance(-pad, "day")
        ee_end = ee.Date((dend + timedelta(days=1)).isoformat()).advance(pad, "day")

        try:
            tile_paths: list[Path] = []
            for idx, tw, ts, te, tn in tile_bounds(west, south, east, north, args.tile_cols, args.tile_rows):
                tile_region = ee.Geometry.Rectangle([tw, ts, te, tn], proj="EPSG:4326", geodesic=False)
                tile_path = tmp_dir / f"S2_{code}_t{idx:02d}.tif"
                if not (tile_path.exists() and tile_path.stat().st_size > 0):
                    try:
                        img = s2_composite(ee_start, ee_end, tile_region, cloud_max)
                        export_image_to_local(img, tile_region, scale, tile_path, crs="EPSG:4326")
                    except Exception as e:
                        if "no bands" not in str(e):
                            raise
                        # Fallback: cloud-pct filter caught nothing in this sub-tile. Drop it and rely on SCL pixel mask.
                        print(f"      retry tile {idx:02d} for {code} without cloud-pct filter")
                        img = s2_composite(ee_start, ee_end, tile_region, 100.0)
                        export_image_to_local(img, tile_region, scale, tile_path, crs="EPSG:4326")
                tile_paths.append(tile_path)

            # Merge tiles into the final per-dekad raster.
            srcs = [rasterio.open(p) for p in tile_paths]
            try:
                mosaic, transform = rio_merge(srcs)
                profile = srcs[0].profile.copy()
                profile.update(
                    height=mosaic.shape[1],
                    width=mosaic.shape[2],
                    transform=transform,
                    compress="deflate",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                )
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(mosaic)
            finally:
                for s in srcs:
                    s.close()
            for p in tile_paths:
                p.unlink(missing_ok=True)
            n_ok += 1
            print(f"OK    S2_{code}.tif  (merged {n_tiles} tiles)")
        except Exception as e:
            n_err += 1
            print(f"ERR   S2_{code}.tif: {e}")

    # Clean up the tile staging directory if empty.
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"\nDone. ok={n_ok}  err={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
