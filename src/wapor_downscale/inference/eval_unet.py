"""Multi-site UNet eval producing fair_per_dekad_<tag>.csv + fair_aggregate_<tag>.json
in the standard format used by the comparison printer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.windows import Window

from wapor_downscale.models.unet_common import (
    N_CHANNELS, NODATA, _band_index_by_name, list_stack_files, split_files_by_year, stack_to_tensors,
)

OUT_DIR = Path(r"c:\Users\z.kiala\Documents\wapor_africa\models\comparisons")


def build_model(encoder, in_channels):
    return smp.Unet(encoder_name=encoder, encoder_weights=None,
                    in_channels=in_channels, classes=1, activation=None)


def hann2d(p, device):
    w = torch.hann_window(p, periodic=False, device=device)
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


def predict_file(fp, model, device, patch=256, overlap=64):
    with rasterio.open(fp) as ds:
        H, W = ds.height, ds.width
    pred_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    win_k = hann2d(patch, device)
    stride = patch - overlap
    n_rows = max(1, math.ceil((H - patch) / stride) + 1)
    n_cols = max(1, math.ceil((W - patch) / stride) + 1)
    model.eval()
    with torch.no_grad():
        for ri in range(n_rows):
            row = min(ri * stride, max(0, H - patch))
            for ci in range(n_cols):
                col = min(ci * stride, max(0, W - patch))
                win = Window(col, row, min(patch, W - col), min(patch, H - row))
                feats, _, _ = stack_to_tensors(fp, win)
                ch, hh, ww = feats.shape
                if hh != patch or ww != patch:
                    padded = np.zeros((ch, patch, patch), dtype=np.float32)
                    padded[:, :hh, :ww] = feats; feats = padded
                x = torch.from_numpy(feats).unsqueeze(0).to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    yhat = model(x).squeeze(1).squeeze(0).float()
                yhat_eff = yhat[:hh, :ww]; w_eff = win_k[:hh, :ww]
                pred_sum[row:row+hh, col:col+ww] += yhat_eff * w_eff
                weight_sum[row:row+hh, col:col+ww] += w_eff
    return (pred_sum / weight_sum.clamp_min(1e-6)).cpu().numpy()


def cb_mask(fp):
    with rasterio.open(fp) as ds:
        b4 = ds.read(_band_index_by_name(ds, "B4")).astype(np.float32)
        b8 = ds.read(_band_index_by_name(ds, "B8")).astype(np.float32)
        b11 = ds.read(_band_index_by_name(ds, "B11")).astype(np.float32)
        lab = ds.read(_band_index_by_name(ds, "b1")).astype(np.float32)
    m = (lab != NODATA) & (b4 != NODATA) & (b8 != NODATA) & (b11 != NODATA)
    return m, lab


def metrics(pred, label, mask):
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "rrmse_pct": float("nan"), "n_pix": 0}
    p = pred[mask].astype(np.float64); t = label[mask].astype(np.float64); err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2))); mae = float(np.mean(np.abs(err)))
    tmean = float(t.mean()); ss_tot = float(np.sum((t - tmean) ** 2)) + 1e-12
    return {"rmse": rmse, "mae": mae, "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot,
            "rrmse_pct": rmse / max(tmean, 1e-6) * 100.0, "n_pix": int(mask.sum())}


def acc(s, pred, label, mask):
    if not mask.any(): return
    p = pred[mask].astype(np.float64); t = label[mask].astype(np.float64); err = p - t
    s["err2"] += float(np.sum(err ** 2)); s["abs"] += float(np.sum(np.abs(err)))
    s["n"] += int(mask.sum()); s["sum_t"] += float(t.sum()); s["sum_t_sq"] += float(np.sum(t ** 2))


def finalize(s):
    if s["n"] == 0: return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_pix": 0}
    rmse = (s["err2"] / s["n"]) ** 0.5; mt = s["sum_t"] / s["n"]
    ss_tot = s["sum_t_sq"] - s["n"] * (mt ** 2)
    return {"rmse": rmse, "mae": s["abs"] / s["n"], "r2": 1.0 - s["err2"] / max(ss_tot, 1e-12), "n_pix": s["n"]}


def parse_site(s):
    parts = s.split(":")
    name = parts[0]; year_max = int(parts[-1])
    stacks = ":".join(parts[1:-1])
    return name, Path(stacks), year_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--site-spec", action="append", required=True,
                    help="NAME:STACKS_DIR:TRAIN_YEAR_MAX (repeatable)")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder = ckpt.get("encoder", "efficientnet-b0")
    in_channels = ckpt.get("in_channels", N_CHANNELS)
    patch = ckpt.get("patch", args.patch)
    model = build_model(encoder, in_channels).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"[UNET-EVAL] encoder={encoder} patch={patch} best_val_rmse={ckpt.get('best_val_rmse','?')}")

    sites = [parse_site(s) for s in args.site_spec]
    csv_path = OUT_DIR / f"fair_per_dekad_{args.out_tag}.csv"
    fields = ["site", "tag", "date", "n_pix", "rmse", "mae", "r2", "rrmse_pct"]
    per_site = {}
    comb = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, stacks_dir, year_max in sites:
            print(f"\n[UNET-EVAL] === {name} ===")
            files = list_stack_files(stacks_dir)
            _, eval_files = split_files_by_year(files, year_max)
            print(f"  hold-out dekads: {len(eval_files)}")
            ss = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}
            for fp in eval_files:
                t0 = time.time()
                mask, label = cb_mask(fp)
                if not mask.any(): continue
                pred = predict_file(fp, model, device, patch=patch, overlap=args.overlap)
                m = metrics(pred, label, mask)
                w.writerow({"site": name, "tag": fp.stem, "date": fp.stem.split("_")[-1],
                            "n_pix": m["n_pix"], "rmse": m["rmse"], "mae": m["mae"],
                            "r2": m["r2"], "rrmse_pct": m["rrmse_pct"]})
                f.flush()
                acc(ss, pred, label, mask); acc(comb, pred, label, mask)
                print(f"  {fp.stem} n={m['n_pix']:>8} | RMSE={m['rmse']:.3f}  R2={m['r2']:.3f} | {time.time()-t0:.1f}s")
            per_site[name] = finalize(ss)
            a = per_site[name]
            print(f"  [{name}] RMSE={a['rmse']:.4f} MAE={a['mae']:.4f} R2={a['r2']:.4f}")

    out = {"ckpt": str(args.ckpt), "per_site": per_site, "combined": finalize(comb)}
    jp = OUT_DIR / f"fair_aggregate_{args.out_tag}.json"
    jp.write_text(json.dumps(out, indent=2))
    c = out["combined"]
    print(f"\n[COMBINED] RMSE={c['rmse']:.4f} MAE={c['mae']:.4f} R2={c['r2']:.4f}")
    print(f"[DONE] {csv_path}")


if __name__ == "__main__":
    main()
