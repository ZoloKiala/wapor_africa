"""Build static layers (DEM, Slope, Aspect_sin, Aspect_cos) for the Baixo AOI via Earth Engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import abspath, ensure_dir, gcs_public_url, load_config
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
    dx = (east - west) / n_cols
    dy = (north - south) / n_rows
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            yield idx, west + c * dx, south + r * dy, west + (c + 1) * dx, south + (r + 1) * dy
            idx += 1


def export_tiled(image: "ee.Image", west, south, east, north, scale, out_path: Path, tmp_dir: Path,
                 n_cols: int, n_rows: int, tag: str) -> None:
    tile_paths: list[Path] = []
    for idx, tw, ts, te, tn in tile_bounds(west, south, east, north, n_cols, n_rows):
        tile_region = ee.Geometry.Rectangle([tw, ts, te, tn], proj="EPSG:4326", geodesic=False)
        tp = tmp_dir / f"{tag}_t{idx:02d}.tif"
        if not (tp.exists() and tp.stat().st_size > 0):
            export_image_to_local(image, tile_region, scale, tp, crs="EPSG:4326")
        tile_paths.append(tp)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    ap.add_argument("--site-code", default=None)
    ap.add_argument("--scale-m", type=int, default=20)
    ap.add_argument("--tile-cols", type=int, default=4)
    ap.add_argument("--tile-rows", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_ee(cfg)
    site_code = args.site_code or cfg["site"].get("l3_site_code")
    west, south, east, north = aoi_bounds_from_l3(cfg, site_code)
    out_dir = ensure_dir(abspath(cfg["paths"]["static_dir"]))
    tmp_dir = ensure_dir(out_dir / "_tiles")

    dem_src = cfg["static"]["dem_source"]
    dem = ee.Image(dem_src).select(0).rename("DEM")
    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope").rename("Slope")
    aspect_deg = terrain.select("aspect")
    aspect_rad = aspect_deg.multiply(3.141592653589793 / 180.0)
    asp_sin = aspect_rad.sin().rename("Aspect_sin")
    asp_cos = aspect_rad.cos().rename("Aspect_cos")

    composite = dem.addBands([slope, asp_sin, asp_cos]).toFloat()
    out_path = out_dir / f"static_DEM_SLOPE_ASPECT_{site_code}.tif"

    print(f"Exporting static layers to {out_path}  ({args.tile_cols}x{args.tile_rows} tiles)")
    if not (out_path.exists() and out_path.stat().st_size > 0):
        export_tiled(composite, west, south, east, north, args.scale_m, out_path, tmp_dir,
                     args.tile_cols, args.tile_rows, tag=f"DEM_SLOPE_ASPECT_{site_code}")

    wc_src = cfg["static"]["worldcover_source"]
    wc = ee.ImageCollection(wc_src).first().select("Map").rename("WorldCover").toUint8()
    wc_out = out_dir / f"worldcover_{site_code}.tif"
    print(f"Exporting WorldCover to {wc_out}  ({args.tile_cols}x{args.tile_rows} tiles)")
    if not (wc_out.exists() and wc_out.stat().st_size > 0):
        export_tiled(wc, west, south, east, north, args.scale_m, wc_out, tmp_dir,
                     args.tile_cols, args.tile_rows, tag=f"worldcover_{site_code}")

    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
