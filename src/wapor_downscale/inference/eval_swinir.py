"""Per-dekad hold-out evaluation of a single SwinIR checkpoint.

Same CB-fair mask + tiling + Hann blending as fair_compare_multi.py, but only
evaluates the SwinIR model given on the CLI. Used to score a retrained model
without disturbing the existing fair_compare_multi outputs.

Usage:
    python scripts/eval_swinir_one.py --ckpt models/multi_swinir_e96_w16/swinir_best.pt \
        --out-tag swin_e96_w16
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
import torch
from rasterio.windows import Window

from wapor_downscale.models.unet_common import (
    NODATA,
    list_stack_files,
    split_files_by_year,
    stack_to_tensors,
    _band_index_by_name,
)
from wapor_downscale.models.swinir_model import SwinIRRegression  # type: ignore


SITES = [
    {"name": "BAIXO",  "stacks": Path(r"c:\Users\z.kiala\Documents\wapor_africa\data\baixo\stacks\BAIXO_STACK_S2_MATCH_L3_20M_FULL_1"),
     "train_year_max": 2024},
    {"name": "LAMEGO", "stacks": Path(r"c:\Users\z.kiala\Documents\wapor_africa\data\lamego\stacks\LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1"),
     "train_year_max": 2021},
]
OUT_DIR = Path(r"c:\Users\z.kiala\Documents\wapor_africa\models\comparisons")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out-tag", required=True, help="Suffix for output CSV/JSON, e.g. 'swin_e96_w16'")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--site-spec", action="append", default=[],
                    help="Override default SITES list. Format: STACKS_DIR:TRAIN_YEAR_MAX "
                         "(eval files = those with year > TRAIN_YEAR_MAX). Site NAME is "
                         "derived from the first stack filename's prefix. Repeatable.")
    ap.add_argument("--tta", action="store_true",
                    help="Test-time augmentation: average predictions over 8 D4 symmetries (8x slower).")
    return ap.parse_args()


def _parse_site_specs(specs: list[str]) -> list[dict]:
    out = []
    for s in specs:
        stacks_dir, ymax = s.rsplit(":", 1)
        stacks_dir = Path(stacks_dir)
        first = next(stacks_dir.glob("*.tif"), None)
        if first is None:
            raise ValueError(f"No tifs in {stacks_dir} to derive site name")
        name = first.stem.split("_")[0]
        out.append({"name": name, "stacks": stacks_dir, "train_year_max": int(ymax)})
    return out


def hann2d(p: int, device: torch.device) -> torch.Tensor:
    w = torch.hann_window(p, periodic=False, device=device)
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


def _forward_tta(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Test-time augmentation: average predictions over 8 D4 augmentations (4 rotations x {orig, hflip})."""
    preds = []
    base_views = [x, torch.flip(x, dims=(-1,))]
    for base in base_views:
        for k in range(4):
            xr = torch.rot90(base, k=k, dims=(-2, -1))
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                yr = model(xr).squeeze(1).float()
            # Invert: undo rotation, then undo flip if applicable
            y = torch.rot90(yr, k=-k, dims=(-2, -1))
            if base is not x:
                y = torch.flip(y, dims=(-1,))
            preds.append(y)
    return torch.stack(preds).mean(0)


def predict(fp: Path, model, device, patch: int, overlap: int, tta: bool = False, keep_channels=None) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(fp) as ds:
        H, W = ds.height, ds.width
    pred_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    win_k = hann2d(patch, device)
    stride = patch - overlap
    n_rows = max(1, math.ceil((H - patch) / stride) + 1)
    n_cols = max(1, math.ceil((W - patch) / stride) + 1)
    label_full = None
    model.eval()
    with torch.no_grad():
        for ri in range(n_rows):
            row = min(ri * stride, max(0, H - patch))
            for ci in range(n_cols):
                col = min(ci * stride, max(0, W - patch))
                win = Window(col, row, min(patch, W - col), min(patch, H - row))
                feats, lab, _ = stack_to_tensors(fp, win)
                if label_full is None:
                    label_full = np.full((H, W), NODATA, dtype=np.float32)
                label_full[row:row + lab.shape[0], col:col + lab.shape[1]] = lab
                ch, hh, ww = feats.shape
                if hh != patch or ww != patch:
                    padded = np.zeros((ch, patch, patch), dtype=np.float32)
                    padded[:, :hh, :ww] = feats
                    feats_in = padded
                else:
                    feats_in = feats
                x = torch.from_numpy(feats_in).unsqueeze(0).to(device, non_blocking=True)
                if keep_channels is not None:
                    x = x.index_select(1, keep_channels)
                if tta:
                    yhat = _forward_tta(model, x, device).squeeze(0)
                else:
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        yhat = model(x).squeeze(1).squeeze(0).float()
                yhat_eff = yhat[:hh, :ww]
                w_eff = win_k[:hh, :ww]
                pred_sum[row:row + hh, col:col + ww]   += yhat_eff * w_eff
                weight_sum[row:row + hh, col:col + ww] += w_eff
    return (pred_sum / weight_sum.clamp_min(1e-6)).cpu().numpy(), label_full


def cb_mask(fp: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(fp) as ds:
        b4  = ds.read(_band_index_by_name(ds, "B4")).astype(np.float32)
        b8  = ds.read(_band_index_by_name(ds, "B8")).astype(np.float32)
        b11 = ds.read(_band_index_by_name(ds, "B11")).astype(np.float32)
        lab = ds.read(_band_index_by_name(ds, "b1")).astype(np.float32)
    m = (lab != NODATA) & (b4 != NODATA) & (b8 != NODATA) & (b11 != NODATA)
    return m, lab


def metrics(pred, label, mask) -> dict:
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "rrmse_pct": float("nan"), "n_pix": 0}
    p = pred[mask].astype(np.float64); t = label[mask].astype(np.float64); err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    tmean = float(t.mean())
    ss_tot = float(np.sum((t - tmean) ** 2)) + 1e-12
    return {"rmse": rmse, "mae": mae, "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot,
            "rrmse_pct": rmse / max(tmean, 1e-6) * 100.0, "n_pix": int(mask.sum())}


def acc(s, pred, label, mask):
    if not mask.any():
        return
    p = pred[mask].astype(np.float64); t = label[mask].astype(np.float64); err = p - t
    s["err2"] += float(np.sum(err ** 2)); s["abs"] += float(np.sum(np.abs(err)))
    s["n"] += int(mask.sum()); s["sum_t"] += float(t.sum())
    s["sum_t_sq"] += float(np.sum(t ** 2))


def finalize(s):
    if s["n"] == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_pix": 0}
    rmse = (s["err2"] / s["n"]) ** 0.5
    mean_t = s["sum_t"] / s["n"]
    ss_tot = s["sum_t_sq"] - s["n"] * (mean_t ** 2)
    return {"rmse": rmse, "mae": s["abs"] / s["n"], "r2": 1.0 - s["err2"] / max(ss_tot, 1e-12), "n_pix": s["n"]}


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL-ONE] device={device}  ckpt={args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = SwinIRRegression(
        in_chans=ckpt["in_chans"], out_chans=1,
        embed_dim=ckpt["embed_dim"], depths=ckpt["depths"],
        num_heads=ckpt["num_heads"], window_size=ckpt["window_size"],
        img_size=ckpt["patch"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    patch = ckpt["patch"]
    keep_channels = ckpt.get("keep_channels")
    keep_ch_tensor = torch.tensor(keep_channels, dtype=torch.long, device=device) if keep_channels else None
    print(f"[EVAL-ONE] embed={ckpt['embed_dim']} depths={ckpt['depths']} window={ckpt['window_size']} patch={patch}  keep_channels={keep_channels}")

    sites_to_eval = _parse_site_specs(args.site_spec) if args.site_spec else SITES
    print(f"[EVAL-ONE] sites: {[s['name'] for s in sites_to_eval]}")

    csv_path = OUT_DIR / f"fair_per_dekad_{args.out_tag}.csv"
    fields = ["site", "tag", "date", "n_pix", "rmse", "mae", "r2", "rrmse_pct"]
    per_site = {}
    comb = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for site in sites_to_eval:
            print(f"\n[EVAL-ONE] === site={site['name']} ===")
            files = list_stack_files(site["stacks"])
            _, eval_files = split_files_by_year(files, site["train_year_max"])
            print(f"           hold-out dekads: {len(eval_files)}")
            s = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}
            for fp in eval_files:
                t0 = time.time()
                mask, label = cb_mask(fp)
                if not mask.any():
                    continue
                pred, _ = predict(fp, model, device, patch=patch, overlap=args.overlap, tta=args.tta, keep_channels=keep_ch_tensor)
                m = metrics(pred, label, mask)
                w.writerow({"site": site["name"], "tag": fp.stem, "date": fp.stem.split("_")[-1],
                            "n_pix": m["n_pix"], "rmse": m["rmse"], "mae": m["mae"],
                            "r2": m["r2"], "rrmse_pct": m["rrmse_pct"]})
                f.flush()
                acc(s, pred, label, mask); acc(comb, pred, label, mask)
                print(f"  {fp.stem} n={m['n_pix']:>8} | RMSE={m['rmse']:.3f}  R2={m['r2']:.3f} | {time.time()-t0:.1f}s")
            per_site[site["name"]] = finalize(s)
            a = per_site[site["name"]]
            print(f"  [{site['name']} AGG] RMSE={a['rmse']:.4f}  MAE={a['mae']:.4f}  R2={a['r2']:.4f}  n_pix={a['n_pix']:,}")

    out = {"ckpt": str(args.ckpt), "per_site": per_site, "combined": finalize(comb),
           "embed_dim": ckpt["embed_dim"], "depths": list(ckpt["depths"]),
           "window_size": ckpt["window_size"]}
    json_path = OUT_DIR / f"fair_aggregate_{args.out_tag}.json"
    json_path.write_text(json.dumps(out, indent=2))
    c = out["combined"]
    print(f"\n[COMBINED] RMSE={c['rmse']:.4f}  MAE={c['mae']:.4f}  R2={c['r2']:.4f}  n_pix={c['n_pix']:,}")
    print(f"[DONE] {csv_path}\n       {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
