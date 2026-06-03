import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent / "models"))
"""Predict one downscaled 20m AETI map for a single L1-stack using SwinIR + Prithvi + Ensemble.

Outputs three GeoTIFFs (single-band float32, mm/dekad, georeferenced to the input stack):
  <out-dir>/<stem>_swinir.tif
  <out-dir>/<stem>_prithvi.tif
  <out-dir>/<stem>_ensemble.tif    (= weight * swinir + (1-weight) * prithvi)

Usage:
    python scripts/predict_ensemble_one.py \\
        --swinir-ckpt models/multi7_swinir_l1_e96_w16/swinir_best.pt \\
        --prithvi-ckpt models/multi7_prithvi_v1_l1/prithvi_best.pt \\
        --stack data/mit/stacks/MIT_STACK_S2_MATCH_L3_20M_L1_FULL_1/MIT_2024-04-01.tif \\
        --out-dir models/multi7_ensemble_l1/predictions/MIT
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "unet"))
from unet_common import NODATA, _band_index_by_name, stack_to_tensors
sys.path.insert(0, str(REPO / "swinir"))
from swinir_model import SwinIRRegression
sys.path.insert(0, str(REPO / "prithvi"))
from prithvi_regression import PrithviRegression, PrithviRegressionV2, PrithviRegressionV3


def hann2d(p, device):
    w = torch.hann_window(p, periodic=False, device=device)
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


def load_swinir(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = SwinIRRegression(
        in_chans=ckpt["in_chans"], out_chans=1,
        embed_dim=ckpt["embed_dim"], depths=ckpt["depths"],
        num_heads=ckpt["num_heads"], window_size=ckpt["window_size"],
        img_size=ckpt["patch"],
    ).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    return m, ckpt["patch"]


def load_prithvi(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    img_size = ckpt.get("img_size", 256)
    freeze_until = ckpt.get("freeze_until", "22")
    try: freeze_until = int(freeze_until)
    except (TypeError, ValueError): pass
    version = ckpt.get("model_version", "v1")
    cls = {"v1": PrithviRegression, "v2": PrithviRegressionV2, "v3": PrithviRegressionV3}[version]
    m = cls(img_size=img_size, freeze_until=freeze_until, mock_backbone=False).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    return m, img_size


def predict_tiled(stack_fp, model, device, patch, overlap):
    with rasterio.open(stack_fp) as ds:
        H, W = ds.height, ds.width
    pred_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    w_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    win_k = hann2d(patch, device)
    stride = patch - overlap
    n_rows = max(1, math.ceil((H - patch) / stride) + 1)
    n_cols = max(1, math.ceil((W - patch) / stride) + 1)
    with torch.no_grad():
        for ri in range(n_rows):
            row = min(ri * stride, max(0, H - patch))
            for ci in range(n_cols):
                col = min(ci * stride, max(0, W - patch))
                win = Window(col, row, min(patch, W - col), min(patch, H - row))
                feats, _, _ = stack_to_tensors(stack_fp, win)
                ch, hh, ww = feats.shape
                if hh != patch or ww != patch:
                    padded = np.zeros((ch, patch, patch), dtype=np.float32)
                    padded[:, :hh, :ww] = feats; feats = padded
                x = torch.from_numpy(feats).unsqueeze(0).to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    yhat = model(x).squeeze(1).squeeze(0).float()
                yhat_eff = yhat[:hh, :ww]; w_eff = win_k[:hh, :ww]
                pred_sum[row:row+hh, col:col+ww] += yhat_eff * w_eff
                w_sum[row:row+hh, col:col+ww] += w_eff
    return (pred_sum / w_sum.clamp_min(1e-6)).cpu().numpy()


def write_geotiff(out_path, arr, ref_stack_fp):
    with rasterio.open(ref_stack_fp) as ds:
        profile = ds.profile.copy()
    profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.set_band_description(1, "AETI_20m_downscaled_mm_per_dekad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swinir-ckpt", required=True, type=Path)
    ap.add_argument("--prithvi-ckpt", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--weight", type=float, default=0.5,
                    help="SwinIR weight in the ensemble (Prithvi gets 1-w). Default 0.5 = mean.")
    ap.add_argument("--overlap-frac", type=float, default=0.25,
                    help="Overlap as fraction of patch size for Hann-blend tiling.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PREDICT] device={device}  stack={args.stack.name}")

    m_swin, p_swin = load_swinir(args.swinir_ckpt, device)
    m_prit, p_prit = load_prithvi(args.prithvi_ckpt, device)
    print(f"[PREDICT] SwinIR patch={p_swin}  Prithvi patch={p_prit}  weight(SwinIR)={args.weight}")

    overlap_swin = int(p_swin * args.overlap_frac)
    overlap_prit = int(p_prit * args.overlap_frac)

    t0 = time.time()
    pred_swin = predict_tiled(args.stack, m_swin, device, p_swin, overlap_swin)
    print(f"[PREDICT] SwinIR done in {time.time()-t0:.1f}s  range=[{pred_swin.min():.2f}, {pred_swin.max():.2f}]")
    t0 = time.time()
    pred_prit = predict_tiled(args.stack, m_prit, device, p_prit, overlap_prit)
    print(f"[PREDICT] Prithvi done in {time.time()-t0:.1f}s  range=[{pred_prit.min():.2f}, {pred_prit.max():.2f}]")
    pred_ens = args.weight * pred_swin + (1.0 - args.weight) * pred_prit

    # Mask out invalid label / S2-nodata pixels (use the CB-fair mask)
    with rasterio.open(args.stack) as ds:
        b4 = ds.read(_band_index_by_name(ds, "B4")).astype(np.float32)
        b8 = ds.read(_band_index_by_name(ds, "B8")).astype(np.float32)
        b11 = ds.read(_band_index_by_name(ds, "B11")).astype(np.float32)
    valid = (b4 != NODATA) & (b8 != NODATA) & (b11 != NODATA)
    for arr in (pred_swin, pred_prit, pred_ens):
        arr[~valid] = NODATA

    stem = args.stack.stem
    out_dir = args.out_dir
    out_swin = out_dir / f"{stem}_swinir.tif"
    out_prit = out_dir / f"{stem}_prithvi.tif"
    out_ens  = out_dir / f"{stem}_ensemble.tif"
    write_geotiff(out_swin, pred_swin, args.stack)
    write_geotiff(out_prit, pred_prit, args.stack)
    write_geotiff(out_ens,  pred_ens,  args.stack)
    print(f"[DONE]")
    print(f"  {out_swin}")
    print(f"  {out_prit}")
    print(f"  {out_ens}")


if __name__ == "__main__":
    main()
