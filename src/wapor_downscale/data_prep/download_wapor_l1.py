"""Download WaPOR L1 (continental, 300 m) AETI dekadal tiles clipped to the L3 AOI.

Mirrors 02_download_wapor_l2.py but pulls from the L1 mapset. Used as the
coarse input for Bergaoui-style 300 m -> 100 m Stage 1 super-resolution.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import abspath, dekads_between, ensure_dir, gcs_public_url, load_config, parse_iso

os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def aoi_bbox_from_l3(cfg: dict, site_code: str) -> tuple[float, float, float, float]:
    bucket = cfg["wapor"]["bucket"]
    prefix = cfg["wapor"]["l3_data_prefix"]
    mapset = cfg["wapor"]["l3_target_mapset"]
    sample = f"{prefix}/WAPOR-3.{mapset}.{site_code}.2018-01-D1.tif"
    url = "/vsicurl/" + gcs_public_url(bucket, sample)
    with rasterio.open(url) as ds:
        return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    ap.add_argument("--site-code", default=None)
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--pad-deg", type=float, default=0.10,
                    help="Padding around AOI in degrees (L1 is coarser; pad more)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir (default: data/<site>/wapor_l1)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    site_code = args.site_code or cfg["site"].get("l3_site_code")

    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = args.bbox
    else:
        if not site_code:
            sys.exit("ERROR: need site code or explicit --bbox.")
        lon_min, lat_min, lon_max, lat_max = aoi_bbox_from_l3(cfg, site_code)

    pad = args.pad_deg
    bbox4326 = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)
    print(f"AOI (EPSG:4326): {bbox4326}")

    start = parse_iso(args.start or cfg["time_range"]["start"])
    end = parse_iso(args.end or cfg["time_range"]["end"])
    bucket = cfg["wapor"]["bucket"]
    mapset = "L1-AETI-D"
    prefix = "DATA/WAPOR-3/MAPSET/L1-AETI-D"
    if args.out_dir:
        out_dir = ensure_dir(args.out_dir)
    else:
        # Derive sibling of wapor_l2_dir
        l2_dir = abspath(cfg["paths"]["wapor_l2_dir"])
        out_dir = ensure_dir(l2_dir.parent / "wapor_l1")

    print(f"OUT: {out_dir}")

    n_ok = n_err = n_skip = 0
    for code, _, _ in dekads_between(start, end):
        fname = f"WAPOR-3.{mapset}.{code}.tif"
        rel = f"{prefix}/{fname}"
        dst = out_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            n_skip += 1
            continue
        url = "/vsicurl/" + gcs_public_url(bucket, rel)
        try:
            with rasterio.open(url) as src:
                src_bbox = transform_bounds("EPSG:4326", src.crs, *bbox4326, densify_pts=21)
                win = from_bounds(*src_bbox, transform=src.transform).round_offsets().round_lengths()
                arr = src.read(1, window=win, boundless=False)
                profile = src.profile.copy()
                profile.update(height=arr.shape[0], width=arr.shape[1],
                               transform=rasterio.windows.transform(win, src.transform),
                               compress="deflate", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(dst, "w", **profile) as dst_ds:
                dst_ds.write(arr, 1)
            n_ok += 1
            print(f"OK    {fname}  shape={arr.shape}")
        except Exception as e:
            msg = str(e)
            if "404" in msg or "HTTP response code: 404" in msg:
                n_err += 1
                print(f"MISS  {fname}")
            else:
                n_err += 1
                print(f"ERR   {fname}: {e}")

    print(f"\nDone. ok={n_ok}  skipped={n_skip}  err={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
