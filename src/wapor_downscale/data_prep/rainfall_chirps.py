"""Build CHIRPS 10-day rainfall and 10-day lag rasters per dekad."""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import ee
import rasterio
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


def aoi_geom_from_l3(cfg: dict, site_code: str) -> "ee.Geometry":
    bucket = cfg["wapor"]["bucket"]
    prefix = cfg["wapor"]["l3_data_prefix"]
    mapset = cfg["wapor"]["l3_target_mapset"]
    sample = f"{prefix}/WAPOR-3.{mapset}.{site_code}.2018-01-D1.tif"
    url = "/vsicurl/" + gcs_public_url(bucket, sample)
    with rasterio.open(url) as ds:
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
    return ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)


def sum_range(coll: "ee.ImageCollection", start: str, end_exclusive: str) -> "ee.Image":
    return coll.filterDate(start, end_exclusive).sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    ap.add_argument("--site-code", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)
    ap.add_argument("--scale-m", type=int, default=500, help="CHIRPS native ~5566 m; we resample later.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_ee(cfg)
    site_code = args.site_code or cfg["site"].get("l3_site_code")
    region = aoi_geom_from_l3(cfg, site_code)
    out_dir = ensure_dir(abspath(cfg["paths"]["rainfall_dir"]))

    start = parse_iso(args.start or cfg["time_range"]["start"])
    end = parse_iso(args.end or cfg["time_range"]["end"])
    win = cfg["rainfall"]["window_days"]
    lag_win = cfg["rainfall"]["lag_window_days"]

    coll = ee.ImageCollection(cfg["rainfall"]["source"]).select("precipitation")

    n_ok = n_err = 0
    for code, dstart, dend in dekads_between(start, end):
        rain_start = dstart.isoformat()
        rain_end_excl = (dend + timedelta(days=1)).isoformat()
        lag_start = (dstart - timedelta(days=lag_win)).isoformat()
        lag_end_excl = dstart.isoformat()

        rain = sum_range(coll, rain_start, rain_end_excl).rename("RAIN_10d")
        rain_lag = sum_range(coll, lag_start, lag_end_excl).rename("RAIN_10d_lag")
        img = rain.addBands(rain_lag).toFloat()

        out_path = out_dir / f"RAIN_{code}.tif"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        try:
            export_image_to_local(img, region, args.scale_m, out_path, crs="EPSG:4326")
            n_ok += 1
            print(f"OK    RAIN_{code}.tif")
        except Exception as e:
            n_err += 1
            print(f"ERR   RAIN_{code}.tif: {e}")

    print(f"\nDone. ok={n_ok}  err={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
