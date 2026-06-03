import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent / "models"))
"""Retrain CatBoost on combined Baixo + Lamego training samples.

Reuses the run-3 best feature set + CB params:
    bands = [SIN_DOY, COS_DOY, ETa300m, NDVI, NDMI, FVC, B8, B11, B4]
    params = {iter=400, depth=10, lr=0.05, l2=1.0, RMSE loss, Bayesian bootstrap}

Outputs:
    models/multi_catboost/catboost_best.joblib
    models/multi_catboost/feature_importance_selected.csv
    models/multi_catboost/training_samples.npz   (for reproducibility)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "models"))
from unet_common import NODATA, list_stack_files, split_files_by_year, stack_to_tensors


DEFAULT_BAIXO_STACKS = Path(r"c:\Users\z.kiala\Documents\wapor_africa\data\baixo\stacks\BAIXO_STACK_S2_MATCH_L3_20M_FULL_1")
DEFAULT_LAMEGO_STACKS = Path(r"c:\Users\z.kiala\Documents\wapor_africa\data\lamego\stacks\LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1")
DEFAULT_OUT_DIR = Path(r"c:\Users\z.kiala\Documents\wapor_africa\models\multi_catboost")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baixo-stacks", type=Path, default=DEFAULT_BAIXO_STACKS)
    ap.add_argument("--lamego-stacks", type=Path, default=DEFAULT_LAMEGO_STACKS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--baixo-train-year-max", type=int, default=2024)
    ap.add_argument("--lamego-train-year-max", type=int, default=2021)
    return ap.parse_args()

# UNet stack channel order [B4,B8,B11,ETa,NDVI,NDMI,FVC,SIN_DOY,COS_DOY]
# CB training feature order [SIN_DOY, COS_DOY, ETa300m, NDVI, NDMI, FVC, B8, B11, B4]
CB_FEATURE_ORDER = [7, 8, 3, 4, 5, 6, 1, 2, 0]
CB_FEATURE_NAMES = ["SIN_DOY", "COS_DOY", "ETa300m", "NDVI", "NDMI", "FVC", "B8", "B11", "B4"]

PIX_PER_STACK = 500
SEED = 11


def sample_stack(fp: Path, n_pixels: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    feats, label, _ = stack_to_tensors(fp)
    mask = label != NODATA
    if not mask.any():
        return None
    flat_idx = np.flatnonzero(mask)
    if flat_idx.size <= n_pixels:
        pick = flat_idx
    else:
        pick = rng.choice(flat_idx, size=n_pixels, replace=False)
    feats_cb = feats[CB_FEATURE_ORDER]  # (9, H, W)
    C, H, W = feats_cb.shape
    X = feats_cb.reshape(C, H * W).T[pick].astype(np.float32)
    y = label.reshape(-1)[pick].astype(np.float32)
    return X, y


def main() -> int:
    args = _parse_args()
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Build train file list using args-provided paths and year cutoffs.
    baixo_train = split_files_by_year(list_stack_files(args.baixo_stacks), args.baixo_train_year_max)[0]
    lamego_train = split_files_by_year(list_stack_files(args.lamego_stacks), args.lamego_train_year_max)[0]
    print(f"[CB-TRAIN] sites=2  train files: Baixo={len(baixo_train)}  Lamego={len(lamego_train)}  total={len(baixo_train)+len(lamego_train)}")
    train_files = list(baixo_train) + list(lamego_train)

    t0 = time.time()
    Xs, ys = [], []
    for i, fp in enumerate(train_files, 1):
        res = sample_stack(fp, PIX_PER_STACK, rng)
        if res is None:
            print(f"[SKIP] {fp.name}: no valid label pixels")
            continue
        X, y = res
        Xs.append(X)
        ys.append(y)
        if i % 50 == 0 or i == len(train_files):
            print(f"[CB-TRAIN] sampled {i}/{len(train_files)} files  (cumulative rows={sum(x.shape[0] for x in Xs):,})  {time.time()-t0:.1f}s")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f"[CB-TRAIN] total samples: X={X.shape}  y={y.shape}  y mean={y.mean():.2f} std={y.std():.2f}")

    np.savez(OUT_DIR / "training_samples.npz", X=X, y=y, feature_names=np.array(CB_FEATURE_NAMES))

    cb_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": SEED,
        "thread_count": -1,
        "od_type": "Iter",
        "od_wait": 40,
        "task_type": "CPU",
        "iterations": 400,
        "depth": 10,
        "learning_rate": 0.05,
        "l2_leaf_reg": 1.0,
        "bagging_temperature": 0.0,
        "bootstrap_type": "Bayesian",
    }
    model = CatBoostRegressor(**cb_params, verbose=50)
    t_fit = time.time()
    model.fit(X, y)
    print(f"[CB-TRAIN] fit done in {time.time()-t_fit:.1f}s")

    bundle = {
        "model": model,
        "feature_bands": CB_FEATURE_NAMES,
        "feature_groups": {"doy": ["SIN_DOY","COS_DOY"], "eta300": ["ETa300m"],
                           "idx": ["NDVI","NDMI","FVC"], "s2": ["B8","B11","B4"]},
        "selected_groups": ["doy","eta300","idx","s2"],
        "selected_bands": CB_FEATURE_NAMES,
        "selected_cols": CB_FEATURE_ORDER,
        "cb_params": cb_params,
        "seed": SEED,
        "training_sites": ["BAIXO 2018-2024", "LAMEGO 2018-2021"],
        "n_train_files": len(train_files),
        "n_samples": int(X.shape[0]),
    }
    joblib.dump(bundle, OUT_DIR / "catboost_best.joblib")
    print(f"[CB-TRAIN] saved {OUT_DIR / 'catboost_best.joblib'}")

    # Feature importance.
    importances = model.feature_importances_
    import csv as _csv
    with (OUT_DIR / "feature_importance_selected.csv").open("w", newline="") as f:
        w = _csv.writer(f); w.writerow(["band", "importance"])
        for name, imp in sorted(zip(CB_FEATURE_NAMES, importances), key=lambda kv: -kv[1]):
            w.writerow([name, f"{imp}"])
            print(f"  {name:<10} {imp:.3f}")

    Path(OUT_DIR / "metadata.json").write_text(json.dumps(
        {k: v for k, v in bundle.items() if k != "model"}, indent=2, default=str))
    print(f"[DONE] total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
