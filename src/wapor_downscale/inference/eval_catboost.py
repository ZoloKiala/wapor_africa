import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent / "models"))
"""Multi-site CatBoost per-dekad eval in the standard fair_per_dekad_<tag>.csv format.

Usage:
    python catboost_eval_multi.py --model models/multi_catboost_l1/catboost_best.joblib \
        --site-spec BAIXO:<stacks>:2024 --site-spec LAMEGO:<stacks>:2021 \
        --out-tag cb_l1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import rasterio

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "models"))
from unet_common import NODATA, list_stack_files, split_files_by_year, stack_to_tensors

# Same feature order as catboost_train_multi.py
CB_FEATURE_ORDER = [7, 8, 3, 4, 5, 6, 1, 2, 0]
CB_FEATURE_NAMES = ["SIN_DOY", "COS_DOY", "ETa300m", "NDVI", "NDMI", "FVC", "B8", "B11", "B4"]

OUT_DIR = Path(r"c:\Users\z.kiala\Documents\wapor_africa\models\comparisons")


def predict_file(fp: Path, model) -> tuple[np.ndarray, np.ndarray]:
    feats, label, _ = stack_to_tensors(fp)
    feats_cb = feats[CB_FEATURE_ORDER]  # (9, H, W)
    C, H, W = feats_cb.shape
    X = feats_cb.reshape(C, H * W).T.astype(np.float32)
    yhat = model.predict(X).astype(np.float32).reshape(H, W)
    return yhat, label


def _band_idx(ds, name):
    for i, d in enumerate(ds.descriptions or ()):
        if d == name: return i + 1
    raise KeyError(name)


def cb_mask(fp: Path):
    with rasterio.open(fp) as ds:
        b4 = ds.read(_band_idx(ds, "B4")).astype(np.float32)
        b8 = ds.read(_band_idx(ds, "B8")).astype(np.float32)
        b11 = ds.read(_band_idx(ds, "B11")).astype(np.float32)
        lab = ds.read(_band_idx(ds, "b1")).astype(np.float32)
    m = (lab != NODATA) & (b4 != NODATA) & (b8 != NODATA) & (b11 != NODATA)
    return m, lab


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
    """NAME:STACKS:YEAR_MAX (NAME used for the site column in output CSV)."""
    parts = s.split(":")
    name = parts[0]
    year_max = int(parts[-1])
    stacks = ":".join(parts[1:-1])
    return name, Path(stacks), year_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--site-spec", action="append", required=True,
                    help="NAME:STACKS_DIR:TRAIN_YEAR_MAX (repeatable)")
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CB-EVAL] loading {args.model}")
    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    print(f"[CB-EVAL] model {type(model).__name__}")

    sites = [parse_site(s) for s in args.site_spec]
    csv_path = OUT_DIR / f"fair_per_dekad_{args.out_tag}.csv"
    fields = ["site", "tag", "date", "n_pix", "rmse", "mae", "r2", "rrmse_pct"]
    per_site = {}
    comb = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, stacks_dir, year_max in sites:
            print(f"\n[CB-EVAL] === {name} ===")
            files = list_stack_files(stacks_dir)
            _, eval_files = split_files_by_year(files, year_max)
            print(f"  hold-out dekads: {len(eval_files)}")
            ss = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0, "sum_t_sq": 0.0}
            for fp in eval_files:
                t0 = time.time()
                mask, label = cb_mask(fp)
                if not mask.any(): continue
                pred, _ = predict_file(fp, model)
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

    out = {"model_path": str(args.model), "per_site": per_site, "combined": finalize(comb)}
    jp = OUT_DIR / f"fair_aggregate_{args.out_tag}.json"
    jp.write_text(json.dumps(out, indent=2))
    c = out["combined"]
    print(f"\n[COMBINED] RMSE={c['rmse']:.4f} MAE={c['mae']:.4f} R2={c['r2']:.4f}")
    print(f"[DONE] {csv_path}")


if __name__ == "__main__":
    main()
