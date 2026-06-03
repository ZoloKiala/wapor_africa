"""Shared dataset + feature stacking for the UNet downscaling experiment.

Channel order (matches the CatBoost run 3 feature set):
    0: B4              (S2 red, /10000)
    1: B8              (S2 NIR, /10000)
    2: B11             (S2 SWIR, /10000)
    3: ETa300m         (WaPOR L2 reprojected to 20m, /100)
    4: NDVI            (B8-B4)/(B8+B4)
    5: NDMI            (B8-B11)/(B8+B11)
    6: FVC             (NDVI-0.2)/(0.86-0.2), clipped [0,1]
    7: SIN_DOY         constant per patch
    8: COS_DOY         constant per patch

Label: WaPOR L3 AETI in raw int16 units (mm/dekad * scale_factor).
Nodata in label = -9999.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


N_CHANNELS = 9
NODATA = -9999.0
CHANNEL_NAMES = (
    "B4", "B8", "B11", "ETa300m", "NDVI", "NDMI", "FVC", "SIN_DOY", "COS_DOY",
)


def _date_from_filename(fp: Path) -> date:
    # BAIXO_2020-01-01.tif
    stem = fp.stem
    parts = stem.split("_")[-1]
    return datetime.strptime(parts, "%Y-%m-%d").date()


def _doy_features(d: date) -> tuple[float, float]:
    doy = d.timetuple().tm_yday
    angle = 2.0 * math.pi * (doy - 1) / 365.0
    return math.sin(angle), math.cos(angle)


def _band_index_by_name(ds: rasterio.io.DatasetReader, name: str) -> int:
    for i, desc in enumerate(ds.descriptions or ()):
        if desc == name:
            return i + 1  # rasterio is 1-indexed
    raise KeyError(f"Band {name!r} not found in {ds.name}")


def _safe_index(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


def stack_to_tensors(
    fp: Path,
    win: rasterio.windows.Window | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a stack file, return (features [9,H,W], label [H,W], valid_mask [H,W])."""
    with rasterio.open(fp) as ds:
        b4_idx  = _band_index_by_name(ds, "B4")
        b8_idx  = _band_index_by_name(ds, "B8")
        b11_idx = _band_index_by_name(ds, "B11")
        eta_idx = _band_index_by_name(ds, "ETa300m")
        lab_idx = _band_index_by_name(ds, "b1")

        kw = {"window": win} if win is not None else {}
        b4  = ds.read(b4_idx,  **kw).astype(np.float32)
        b8  = ds.read(b8_idx,  **kw).astype(np.float32)
        b11 = ds.read(b11_idx, **kw).astype(np.float32)
        eta = ds.read(eta_idx, **kw).astype(np.float32)
        lab = ds.read(lab_idx, **kw).astype(np.float32)

    # Nodata handling: features get clamped to 0 (will be re-scaled), label keeps nodata sentinel.
    valid_lab = lab != NODATA
    for arr in (b4, b8, b11):
        arr[arr == NODATA] = 0.0
    eta[eta == NODATA] = 0.0
    lab[~valid_lab] = NODATA  # ensure exact -9999 where invalid

    # Normalize feature scales.
    b4_n  = b4  / 10000.0
    b8_n  = b8  / 10000.0
    b11_n = b11 / 10000.0
    eta_n = eta / 100.0

    # Vegetation indices.
    ndvi = _safe_index(b8 - b4, b8 + b4)
    ndmi = _safe_index(b8 - b11, b8 + b11)
    fvc = np.clip((ndvi - 0.2) / (0.86 - 0.2), 0.0, 1.0).astype(np.float32)

    # DOY constants (broadcast later).
    sd, cd = _doy_features(_date_from_filename(fp))
    H, W = b4.shape
    sin_doy = np.full((H, W), sd, dtype=np.float32)
    cos_doy = np.full((H, W), cd, dtype=np.float32)

    feats = np.stack([b4_n, b8_n, b11_n, eta_n, ndvi, ndmi, fvc, sin_doy, cos_doy], axis=0)
    return feats, lab, valid_lab.astype(np.float32)


_DATE_TIF_RE = __import__("re").compile(r"^[A-Z]+_\d{4}-\d{2}-\d{2}\.tif$")


def list_stack_files(stacks_dir: Path) -> list[Path]:
    """Match any `<SITE>_YYYY-MM-DD.tif` so the helper works for BAIXO/LAMEGO/etc."""
    return sorted(p for p in stacks_dir.glob("*.tif")
                  if p.is_file() and _DATE_TIF_RE.match(p.name))


def split_files_by_year(files: list[Path], train_year_max: int) -> tuple[list[Path], list[Path]]:
    train, eval_ = [], []
    for fp in files:
        y = _date_from_filename(fp).year
        if y <= train_year_max:
            train.append(fp)
        else:
            eval_.append(fp)
    return train, eval_


def list_multi_site_files(specs: list[tuple[Path, int]]) -> tuple[list[Path], list[Path]]:
    """Combine multiple (stacks_dir, train_year_max) pairs into one (train, eval) pair.

    Each site's files are split by its own train_year_max; results are concatenated.
    """
    train_all: list[Path] = []
    eval_all: list[Path] = []
    for stacks_dir, year_max in specs:
        files = list_stack_files(Path(stacks_dir))
        tr, ev = split_files_by_year(files, year_max)
        train_all.extend(tr)
        eval_all.extend(ev)
    return train_all, eval_all


@dataclass
class PatchSamplerConfig:
    patch: int = 256
    n_patches_per_epoch: int = 500
    min_valid_frac: float = 0.05
    augment: bool = True
    max_resample_tries: int = 8


class StackPatchDataset(Dataset):
    """Random-patch sampler over a list of stack rasters.

    Each __getitem__ picks a random file, samples a random crop, re-tries up to
    `max_resample_tries` if the valid-label fraction is too low. Returns
    `(features, label, mask)` where `mask` is 1 on valid label pixels.
    """

    def __init__(self, files: list[Path], cfg: PatchSamplerConfig, seed: int = 7):
        self.files = list(files)
        if not self.files:
            raise ValueError("StackPatchDataset got an empty file list")
        self.cfg = cfg
        self._rng = random.Random(seed)

        # Cache raster dimensions.
        self._dims: dict[Path, tuple[int, int]] = {}
        for fp in self.files:
            with rasterio.open(fp) as ds:
                self._dims[fp] = (ds.height, ds.width)

    def __len__(self) -> int:
        return self.cfg.n_patches_per_epoch

    def _sample_window(self, fp: Path) -> rasterio.windows.Window:
        H, W = self._dims[fp]
        p = self.cfg.patch
        row = self._rng.randint(0, max(0, H - p))
        col = self._rng.randint(0, max(0, W - p))
        return rasterio.windows.Window(col, row, p, p)

    def __getitem__(self, _idx: int):
        cfg = self.cfg
        for _try in range(cfg.max_resample_tries):
            fp = self._rng.choice(self.files)
            win = self._sample_window(fp)
            feats, lab, mask = stack_to_tensors(fp, win)
            if mask.mean() >= cfg.min_valid_frac:
                break
        else:
            # accept whatever the last sample was
            pass

        if cfg.augment:
            if self._rng.random() < 0.5:
                feats = feats[:, :, ::-1].copy()
                lab = lab[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if self._rng.random() < 0.5:
                feats = feats[:, ::-1, :].copy()
                lab = lab[::-1, :].copy()
                mask = mask[::-1, :].copy()
            k = self._rng.randint(0, 3)
            if k:
                feats = np.rot90(feats, k=k, axes=(1, 2)).copy()
                lab = np.rot90(lab, k=k).copy()
                mask = np.rot90(mask, k=k).copy()

        feats_t = torch.from_numpy(feats).float()
        lab_t   = torch.from_numpy(lab).float()
        mask_t  = torch.from_numpy(mask).float()
        return feats_t, lab_t, mask_t


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 5.0) -> torch.Tensor:
    """Huber (smooth L1) loss averaged over masked pixels."""
    diff = pred - target
    abs_d = diff.abs()
    quad = torch.minimum(abs_d, torch.full_like(abs_d, delta))
    lin = abs_d - quad
    pixel = 0.5 * quad.pow(2) + delta * lin
    denom = mask.sum().clamp_min(1.0)
    return (pixel * mask).sum() / denom


def masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    m = mask > 0.5
    if not m.any():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "rrmse_pct": float("nan"), "n": 0}
    p = pred[m].detach().double()
    t = target[m].detach().double()
    err = p - t
    rmse = err.pow(2).mean().sqrt().item()
    mae = err.abs().mean().item()
    tmean = t.mean()
    ss_tot = (t - tmean).pow(2).sum().clamp_min(1e-12)
    ss_res = err.pow(2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()
    rrmse = rmse / max(float(tmean.item()), 1e-6) * 100.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "rrmse_pct": rrmse, "n": int(m.sum().item())}
