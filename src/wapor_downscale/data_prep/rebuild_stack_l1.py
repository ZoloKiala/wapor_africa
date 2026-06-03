"""Rebuild existing 20m stacks with REAL L1 (300m -> 20m bilinear) in the ETa300m band.

The original stacks were built with L2 100m bilinearly resampled to 20m in the `ETa300m`
band (despite the misleading name). For an apples-to-apples 300m -> 20m comparison
with SR-DRN-direct, we need stacks where this band is actually L1.

Usage:
    python 07_rebuild_stack_l1.py --config baixo
    python 07_rebuild_stack_l1.py --config lamego
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import abspath, load_config


def _band_idx(ds, name):
    for i, d in enumerate(ds.descriptions or ()):
        if d == name: return i + 1
    raise KeyError(name)


def dekad_code_from_date(d):
    sub = 1 if d.day <= 10 else (2 if d.day <= 20 else 3)
    return f"{d.year:04d}-{d.month:02d}-D{sub}"


def find_l1_for_stack(stack_fp: Path, l1_dir: Path):
    """Stack filename is <SITE>_YYYY-MM-DD.tif. Look up L1 tile by dekad."""
    from datetime import datetime
    d = datetime.strptime(stack_fp.stem.split("_")[-1], "%Y-%m-%d").date()
    code = dekad_code_from_date(d)
    p = l1_dir / f"WAPOR-3.L1-AETI-D.{code}.tif"
    return p if p.exists() else None


def rebuild_one(stack_fp: Path, l1_dir: Path, out_dir: Path) -> bool:
    out_path = out_dir / stack_fp.name
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    l1_fp = find_l1_for_stack(stack_fp, l1_dir)
    if l1_fp is None:
        print(f"  MISS L1 for {stack_fp.stem}")
        return False
    with rasterio.open(stack_fp) as src:
        profile = src.profile.copy()
        n_bands = src.count
        descriptions = list(src.descriptions or [None] * n_bands)
        # Read everything into memory (all bands)
        data = src.read()  # (n_bands, H, W)
        crs = src.crs; transform = src.transform
        H, W = src.height, src.width
        eta_idx = _band_idx(src, "ETa300m")
    # Build L1-on-20m. L1 and L2 use the same int16 storage scale (raw ~= mm/dekad), and
    # the existing stack ETa300m band stores L2 raw values as float32 (no /100 applied at
    # stack-build time; the /100 happens later in unet_common.stack_to_tensors). So we
    # write L1 raw values directly into the band.
    NODATA_OUT = -9999.0
    with rasterio.open(l1_fp) as l1:
        raw = l1.read(1)
        nod = l1.nodata
        l1_valid = (raw != nod) if nod is not None else np.ones_like(raw, dtype=bool)
        l1_data = raw.astype(np.float32)
        if nod is not None:
            l1_data = np.where(raw == nod, 0.0, l1_data)
        src_crs = l1.crs; src_transform = l1.transform
    l1_on_20m = np.zeros((H, W), dtype=np.float32)
    reproject(
        source=l1_data, destination=l1_on_20m,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=transform, dst_crs=crs,
        resampling=Resampling.bilinear,
    )
    valid_on_20m = np.zeros((H, W), dtype=np.float32)
    reproject(
        source=l1_valid.astype(np.float32), destination=valid_on_20m,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=transform, dst_crs=crs,
        resampling=Resampling.nearest,
    )
    l1_on_20m = np.where(valid_on_20m > 0.5, l1_on_20m, NODATA_OUT).astype(np.float32)
    data[eta_idx - 1] = l1_on_20m

    out_dir.mkdir(parents=True, exist_ok=True)
    profile.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_path, "w", **profile) as dst:
        for bi in range(n_bands):
            dst.write(data[bi], bi + 1)
            if descriptions[bi]:
                dst.set_band_description(bi + 1, descriptions[bi])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baixo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stacks_dir = abspath(cfg["paths"]["stacks_dir"])
    l1_dir = stacks_dir.parent.parent / "wapor_l1"
    if not l1_dir.exists():
        sys.exit(f"L1 dir missing: {l1_dir}")
    # Output dir name = original with _L1 suffix
    out_dir = Path(str(stacks_dir).replace("_FULL_1", "_L1_FULL_1"))
    print(f"src stacks: {stacks_dir}")
    print(f"L1 dir:     {l1_dir}")
    print(f"out dir:    {out_dir}")

    files = sorted(p for p in stacks_dir.glob("*.tif") if p.is_file())
    print(f"stacks to rebuild: {len(files)}")
    n_ok = n_miss = 0
    for i, fp in enumerate(files):
        if rebuild_one(fp, l1_dir, out_dir):
            n_ok += 1
        else:
            n_miss += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}]  ok={n_ok}  miss={n_miss}")
    print(f"\nDone. ok={n_ok}  miss={n_miss}  out={out_dir}")


if __name__ == "__main__":
    main()
