"""Assemble per-dekad multi-band GeoTIFF stacks the upstream training script expects.

Output: data/baixo/stacks/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1/BAIXO_<YYYY-MM-DD>.tif
with band order + descriptions:
    1  B4
    2  B8
    3  B11
    4  ETa300m         (WaPOR L2 coarse predictor, reprojected onto L3 grid)
    5  DEM
    6  Slope
    7  Aspect_sin
    8  Aspect_cos
    9  RAIN_10d
   10  RAIN_10d_lag
   11  WorldCover
   12  b1              (LABEL = WaPOR L3 at 20 m)

The L3 raster defines the spatial reference (CRS, transform, shape).
All other inputs are reprojected/resampled to match.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import abspath, dekads_between, ensure_dir, load_config, parse_iso

L3_NAME_RE = re.compile(
    r"WAPOR-3\.L3-[A-Z]+-D\.(?P<site>[A-Z0-9]+)\.(?P<year>\d{4})-(?P<month>\d{2})-D(?P<dekad>[123])\.tif$"
)


def dekad_to_date(year: int, month: int, dekad: int) -> date:
    return date(year, month, {1: 1, 2: 11, 3: 21}[dekad])


def reproject_to_ref(src_path: Path, band: int, ref_ds, resampling=Resampling.bilinear, dst_dtype="float32"):
    with rasterio.open(src_path) as src:
        arr = np.empty((ref_ds.height, ref_ds.width), dtype=dst_dtype)
        arr[:] = np.nan if "float" in dst_dtype else 0
        reproject(
            source=rasterio.band(src, band),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=resampling,
            dst_nodata=np.nan if "float" in dst_dtype else 0,
            src_nodata=src.nodata,
        )
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)
    ap.add_argument("--min-bands", action="store_true",
                    help="Build the minimal 6-band+label stack (skip NDVI/NDMI/FVC/rain/aspect/WC).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    site_code = cfg["site"]["l3_site_code"]
    site_name = cfg["site"]["name"]
    if not site_code:
        sys.exit("ERROR: site.l3_site_code is not set.")

    l3_dir = abspath(cfg["paths"]["wapor_l3_dir"])
    l2_dir = abspath(cfg["paths"]["wapor_l2_dir"])
    s2_dir = abspath(cfg["paths"]["s2_dir"])
    static_dir = abspath(cfg["paths"]["static_dir"])
    rain_dir = abspath(cfg["paths"]["rainfall_dir"])
    out_dir = ensure_dir(abspath(cfg["paths"]["stacks_dir"]))

    start = parse_iso(args.start or cfg["time_range"]["start"])
    end = parse_iso(args.end or cfg["time_range"]["end"])
    static_tif = next(static_dir.glob(f"static_DEM_SLOPE_ASPECT_{site_code}.tif"), None)
    wc_tif = next(static_dir.glob(f"worldcover_{site_code}.tif"), None)

    scale_l2 = float(cfg["wapor"].get("scale_l2", 100))

    n_ok = n_skip = n_err = 0
    for code, dstart, _ in dekads_between(start, end):
        l3_tif = l3_dir / f"WAPOR-3.{cfg['wapor']['l3_target_mapset']}.{site_code}.{code}.tif"
        l2_tif = l2_dir / f"WAPOR-3.{cfg['wapor']['l2_predictor_mapset']}.{code}.tif"
        s2_tif = s2_dir / f"S2_{code}.tif"
        rain_tif = rain_dir / f"RAIN_{code}.tif"

        if not l3_tif.exists():
            n_skip += 1
            continue
        if not s2_tif.exists() or not l2_tif.exists():
            n_skip += 1
            print(f"SKIP  {code} (missing S2 or L2)")
            continue

        out_path = out_dir / f"{site_name}_{dstart.isoformat()}.tif"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        try:
            with rasterio.open(l3_tif) as ref:
                profile = ref.profile.copy()
                ref_arr = ref.read(1).astype(np.float32)
                ref_nod = ref.nodata if ref.nodata is not None else -9999.0
                # The L3 file holds the LABEL. WaPOR L3 dekadal T is sometimes scaled.
                # Keep raw values; consumer scales via FEATURE_BANDS.

            # Read & reproject each predictor.
            with rasterio.open(s2_tif) as s2:
                # S2 has 3 bands in order B4, B8, B11 per the EE export
                # (we exported `image.select(['B4','B8','B11'])`)
                b4 = reproject_to_ref(s2_tif, 1, _ref_view(profile), resampling=Resampling.bilinear)
                b8 = reproject_to_ref(s2_tif, 2, _ref_view(profile), resampling=Resampling.bilinear)
                b11 = reproject_to_ref(s2_tif, 3, _ref_view(profile), resampling=Resampling.bilinear)

            with rasterio.open(l2_tif) as l2:
                eta300 = reproject_to_ref(l2_tif, 1, _ref_view(profile), resampling=Resampling.bilinear)
            # ETa300m scaled to match the upstream divisor of /100.
            eta300_scaled = eta300.copy()

            dem = slope = asp_sin = asp_cos = None
            if static_tif and static_tif.exists():
                dem     = reproject_to_ref(static_tif, 1, _ref_view(profile), resampling=Resampling.bilinear)
                slope   = reproject_to_ref(static_tif, 2, _ref_view(profile), resampling=Resampling.bilinear)
                asp_sin = reproject_to_ref(static_tif, 3, _ref_view(profile), resampling=Resampling.bilinear)
                asp_cos = reproject_to_ref(static_tif, 4, _ref_view(profile), resampling=Resampling.bilinear)
            else:
                H, W = ref_arr.shape
                dem = np.zeros((H, W), np.float32)
                slope = np.zeros((H, W), np.float32)
                asp_sin = np.zeros((H, W), np.float32)
                asp_cos = np.zeros((H, W), np.float32)
                print(f"WARN  static layers missing -> zero-filled (label still required: DEM, Slope)")

            rain = rain_lag = None
            if rain_tif.exists():
                rain     = reproject_to_ref(rain_tif, 1, _ref_view(profile), resampling=Resampling.bilinear)
                rain_lag = reproject_to_ref(rain_tif, 2, _ref_view(profile), resampling=Resampling.bilinear)
            else:
                H, W = ref_arr.shape
                rain = np.zeros((H, W), np.float32)
                rain_lag = np.zeros((H, W), np.float32)

            wc = None
            if wc_tif and wc_tif.exists():
                wc = reproject_to_ref(wc_tif, 1, _ref_view(profile), resampling=Resampling.nearest, dst_dtype="uint8")

            # Compose the stack.
            bands = [
                ("B4", b4), ("B8", b8), ("B11", b11),
                ("ETa300m", eta300_scaled),
                ("DEM", dem), ("Slope", slope),
                ("Aspect_sin", asp_sin), ("Aspect_cos", asp_cos),
                ("RAIN_10d", rain), ("RAIN_10d_lag", rain_lag),
            ]
            if wc is not None:
                bands.append(("WorldCover", wc.astype(np.float32)))
            bands.append(("b1", ref_arr))  # label

            out_profile = profile.copy()
            out_profile.update(
                count=len(bands),
                dtype="float32",
                nodata=-9999.0,
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            with rasterio.open(out_path, "w", **out_profile) as dst:
                for i, (name, arr) in enumerate(bands, start=1):
                    a = arr.astype(np.float32, copy=False)
                    a = np.where(np.isfinite(a), a, -9999.0)
                    dst.write(a, i)
                    dst.set_band_description(i, name)
            n_ok += 1
            print(f"OK    {out_path.name}")
        except Exception as e:
            n_err += 1
            print(f"ERR   {code}: {e}")

    print(f"\nDone. ok={n_ok}  skipped={n_skip}  err={n_err}")
    return 0 if n_err == 0 else 1


class _RefView:
    """Lightweight stand-in for an open rasterio dataset, exposing the attrs reproject() needs."""
    def __init__(self, profile):
        self.height = profile["height"]
        self.width  = profile["width"]
        self.transform = profile["transform"]
        self.crs = profile["crs"]


def _ref_view(profile):
    return _RefView(profile)


if __name__ == "__main__":
    raise SystemExit(main())
