"""Ensemble evaluator: weighted average of a SwinIR + a Prithvi (V1/V2/V3) checkpoint per pixel.

For each hold-out dekad, runs both models with Hann-blend tiling (optionally TTA),
combines predictions as `w * swinir + (1-w) * prithvi`, and scores on the CB-fair
mask. Output format mirrors eval_swinir_one.py / prithvi_eval.py.

Usage:
    python scripts/ensemble_evaluate.py \\
        --swinir-ckpt models/multi5_swinir_e96_w16/swinir_best.pt \\
        --prithvi-ckpt models/multi5_prithvi_v1_freeze22/prithvi_best.pt \\
        --out-tag ensemble_50_50 --weight 0.5
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

REPO_ROOT = Path(__file__).resolve().parents[1]
from wapor_downscale.models.unet_common import NODATA, list_stack_files, split_files_by_year, stack_to_tensors, _band_index_by_name
from wapor_downscale.models.swinir_model import SwinIRRegression
from wapor_downscale.models.prithvi_regression import PrithviRegression, PrithviRegressionV2, PrithviRegressionV3


SITES = [
    {"name": "BAIXO",  "stacks": REPO_ROOT / "data" / "baixo" / "stacks" / "BAIXO_STACK_S2_MATCH_L3_20M_FULL_1",
     "train_year_max": 2024},
    {"name": "LAMEGO", "stacks": REPO_ROOT / "data" / "lamego" / "stacks" / "LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1",
     "train_year_max": 2021},
]
OUT_DIR = REPO_ROOT / "models" / "comparisons"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swinir-ckpt", required=True, type=Path)
    ap.add_argument("--prithvi-ckpt", required=True, type=Path)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--weight", type=float, default=0.5,
                    help="SwinIR weight in the blend (Prithvi gets 1-w). 0.5 = simple mean.")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--site-spec", action="append", default=[])
    return ap.parse_args()


def _parse_site_specs(specs):
    out = []
    for s in specs:
        sdir, ymax = s.rsplit(":", 1)
        sdir = Path(sdir)
        first = next(sdir.glob("*.tif"), None)
        if first is None:
            raise ValueError(f"No tifs in {sdir}")
        name = first.stem.split("_")[0]
        out.append({"name": name, "stacks": sdir, "train_year_max": int(ymax)})
    return out


def hann2d(p, device):
    w = torch.hann_window(p, periodic=False, device=device)
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


def _forward(model, x, device, tta):
    if not tta:
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            return model(x).squeeze(1).float()
    preds = []
    base_views = [x, torch.flip(x, dims=(-1,))]
    for base in base_views:
        for k in range(4):
            xr = torch.rot90(base, k=k, dims=(-2, -1))
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                yr = model(xr).squeeze(1).float()
            y = torch.rot90(yr, k=-k, dims=(-2, -1))
            if base is not x:
                y = torch.flip(y, dims=(-1,))
            preds.append(y)
    return torch.stack(preds).mean(0)


def predict_ensemble(fp, swin, prit, device, patch, overlap, weight, tta):
    """Hann-blend tiled ensemble prediction. Returns (pred, mask, label)."""
    with rasterio.open(fp) as ds:
        H, W = ds.height, ds.width
        b4 = ds.read(_band_index_by_name(ds, "B4")).astype(np.float32)
        b8 = ds.read(_band_index_by_name(ds, "B8")).astype(np.float32)
        b11 = ds.read(_band_index_by_name(ds, "B11")).astype(np.float32)
        lab = ds.read(_band_index_by_name(ds, "b1")).astype(np.float32)
    mask = (lab != NODATA) & (b4 != NODATA) & (b8 != NODATA) & (b11 != NODATA)

    pred_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
    win_k = hann2d(patch, device)
    stride = patch - overlap
    n_rows = max(1, math.ceil((H - patch) / stride) + 1)
    n_cols = max(1, math.ceil((W - patch) / stride) + 1)
    swin.eval(); prit.eval()
    with torch.no_grad():
        for ri in range(n_rows):
            row = min(ri * stride, max(0, H - patch))
            for ci in range(n_cols):
                col = min(ci * stride, max(0, W - patch))
                win = Window(col, row, min(patch, W - col), min(patch, H - row))
                feats, _l, _m = stack_to_tensors(fp, win)
                ch, hh, ww = feats.shape
                if hh != patch or ww != patch:
                    padded = np.zeros((ch, patch, patch), dtype=np.float32)
                    padded[:, :hh, :ww] = feats
                    feats_in = padded
                else:
                    feats_in = feats
                x = torch.from_numpy(feats_in).unsqueeze(0).to(device, non_blocking=True)
                y_swin = _forward(swin, x, device, tta).squeeze(0)
                y_prit = _forward(prit, x, device, tta).squeeze(0)
                yhat = (weight * y_swin + (1 - weight) * y_prit)
                yhat_eff = yhat[:hh, :ww]
                w_eff = win_k[:hh, :ww]
                pred_sum[row:row + hh, col:col + ww]   += yhat_eff * w_eff
                weight_sum[row:row + hh, col:col + ww] += w_eff
    pred = (pred_sum / weight_sum.clamp_min(1e-6)).cpu().numpy()
    return pred, mask, lab


def metrics(pred, label, mask):
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "rrmse_pct": float("nan"), "n_pix": 0}
    p = pred[mask].astype(np.float64); t = label[mask].astype(np.float64); err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2))); mae = float(np.mean(np.abs(err)))
    tmean = float(t.mean()); ss_tot = float(np.sum((t - tmean) ** 2)) + 1e-12
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
    mean_t = s["sum_t"] / s["n"]; ss_tot = s["sum_t_sq"] - s["n"] * (mean_t ** 2)
    return {"rmse": rmse, "mae": s["abs"] / s["n"], "r2": 1.0 - s["err2"] / max(ss_tot, 1e-12), "n_pix": s["n"]}


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENSEMBLE] device={device}  weight(swin)={args.weight}  tta={args.tta}")

    # Load SwinIR
    s_ckpt = torch.load(args.swinir_ckpt, map_location=device, weights_only=False)
    swin = SwinIRRegression(
        in_chans=s_ckpt["in_chans"], out_chans=1, embed_dim=s_ckpt["embed_dim"],
        depths=s_ckpt["depths"], num_heads=s_ckpt["num_heads"],
        window_size=s_ckpt["window_size"], img_size=s_ckpt["patch"],
    ).to(device)
    swin.load_state_dict(s_ckpt["model_state"])
    print(f"[ENSEMBLE] SwinIR loaded: embed={s_ckpt['embed_dim']} window={s_ckpt['window_size']}")

    # Load Prithvi
    p_ckpt = torch.load(args.prithvi_ckpt, map_location=device, weights_only=False)
    version = p_ckpt.get("model_version", "v1")
    p_cls = {"v1": PrithviRegression, "v2": PrithviRegressionV2, "v3": PrithviRegressionV3}[version]
    freeze = p_ckpt.get("freeze_until", "22")
    try:
        freeze = int(freeze)
    except (TypeError, ValueError):
        pass
    prit = p_cls(img_size=p_ckpt.get("img_size", args.patch),
                 freeze_until=freeze, mock_backbone=False).to(device)
    prit.load_state_dict(p_ckpt["model_state"])
    print(f"[ENSEMBLE] Prithvi loaded: version={version} freeze={freeze}")

    sites = _parse_site_specs(args.site_spec) if args.site_spec else SITES
    print(f"[ENSEMBLE] sites: {[s['name'] for s in sites]}")

    csv_path = OUT_DIR / f"fair_per_dekad_{args.out_tag}.csv"
    fields = ["site", "tag", "date", "n_pix", "rmse", "mae", "r2", "rrmse_pct"]
    per_site = {}
    comb = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for site in sites:
            print(f"\n[ENSEMBLE] === site={site['name']} ===")
            files = list_stack_files(site["stacks"])
            _, eval_files = split_files_by_year(files, site["train_year_max"])
            s = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}
            for fp in eval_files:
                t0 = time.time()
                pred, mask, lab = predict_ensemble(fp, swin, prit, device,
                                                    args.patch, args.overlap, args.weight, args.tta)
                if not mask.any():
                    continue
                m = metrics(pred, lab, mask)
                w.writerow({"site": site["name"], "tag": fp.stem, "date": fp.stem.split("_")[-1],
                            "n_pix": m["n_pix"], "rmse": m["rmse"], "mae": m["mae"],
                            "r2": m["r2"], "rrmse_pct": m["rrmse_pct"]})
                f.flush()
                acc(s, pred, lab, mask); acc(comb, pred, lab, mask)
                print(f"  {fp.stem} n={m['n_pix']:>8} | RMSE={m['rmse']:.3f}  R2={m['r2']:.3f} | {time.time()-t0:.1f}s")
            per_site[site["name"]] = finalize(s)
            a = per_site[site["name"]]
            print(f"  [{site['name']} AGG] RMSE={a['rmse']:.4f}  MAE={a['mae']:.4f}  R2={a['r2']:.4f}  n_pix={a['n_pix']:,}")

    out = {"swinir_ckpt": str(args.swinir_ckpt), "prithvi_ckpt": str(args.prithvi_ckpt),
           "weight_swinir": args.weight, "tta": args.tta,
           "per_site": per_site, "combined": finalize(comb)}
    json_path = OUT_DIR / f"fair_aggregate_{args.out_tag}.json"
    json_path.write_text(json.dumps(out, indent=2))
    c = out["combined"]
    print(f"\n[COMBINED] RMSE={c['rmse']:.4f}  MAE={c['mae']:.4f}  R2={c['r2']:.4f}  n_pix={c['n_pix']:,}")
    print(f"[DONE] {csv_path}\n       {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
