"""Download one L3 TIF per candidate site code and report its bounding box.

Goal: identify which 3-letter code corresponds to a target site (e.g. KOG = Koga).
The bbox printed in EPSG:4326 lets you spot the matching tile against the
`site.target_bbox_hint` in the config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import gcs_download, load_config, REPO_ROOT, ensure_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bucket = cfg["wapor"]["bucket"]
    prefix = cfg["wapor"]["l3_data_prefix"]
    cands = cfg["site"]["l3_candidates"]
    hint = cfg["site"].get("target_bbox_hint")

    tmpdir = ensure_dir(REPO_ROOT / "data" / "_probe")
    print(f"Probing {len(cands)} candidates: {cands}")
    print(f"Looking for a tile inside hint bbox: {hint}\n")

    results = []
    for code in cands:
        rel = f"{prefix}/WAPOR-3.L3-T-D.{code}.2018-01-D1.tif"
        dst = tmpdir / f"{code}.tif"
        try:
            gcs_download(bucket, rel, dst)
            with rasterio.open(dst) as ds:
                left, bottom, right, top = ds.bounds
                src_crs = ds.crs
            wgs = transform_bounds(src_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)
            in_hint = (
                hint is None
                or (wgs[0] < hint[2] and wgs[2] > hint[0] and wgs[1] < hint[3] and wgs[3] > hint[1])
            )
            results.append((code, wgs, in_hint, str(src_crs)))
        except Exception as exc:
            results.append((code, None, False, f"ERR: {exc}"))

    print(f"{'CODE':<5} {'WGS84 bbox (lon_min, lat_min, lon_max, lat_max)':<60}  CRS")
    print("-" * 100)
    for code, wgs, hit, crs in results:
        if wgs is None:
            print(f"{code:<5} {crs}")
            continue
        marker = "  <-- inside hint bbox" if hit else ""
        bbox = f"({wgs[0]:8.3f}, {wgs[1]:8.3f}, {wgs[2]:8.3f}, {wgs[3]:8.3f})"
        print(f"{code:<5} {bbox:<60}  {crs}{marker}")

    print("\nNext step: set `site.l3_site_code` in configs/baixo.yaml to the matching code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
