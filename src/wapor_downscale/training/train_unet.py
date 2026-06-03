import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent / "models"))
"""Train a small UNet to downscale WaPOR L3 AETI from S2 + L2 predictors.

Usage:
    python unet_train.py --epochs 10 --patches-per-epoch 500 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unet_common import (
    N_CHANNELS,
    PatchSamplerConfig,
    StackPatchDataset,
    list_multi_site_files,
    list_stack_files,
    masked_huber_loss,
    masked_metrics,
    split_files_by_year,
)


def build_model(encoder: str, in_channels: int) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,  # regression head
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks-dir", default=r"c:\Users\z.kiala\Documents\wapor_africa\data\baixo\stacks\BAIXO_STACK_S2_MATCH_L3_20M_FULL_1")
    ap.add_argument("--out-dir", default=r"c:\Users\z.kiala\Documents\wapor_africa\models\baixo_unet_run_1")
    ap.add_argument("--train-year-max", type=int, default=2024)
    ap.add_argument(
        "--extra-site",
        action="append",
        default=[],
        help="Additional site as 'stacks_dir:train_year_max'. Repeatable. "
             "Each extra site is split by its own year_max and joined to training.",
    )
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patches-per-epoch", type=int, default=500)
    ap.add_argument("--val-patches", type=int, default=128)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)  # Windows: keep 0 to avoid spawn cost
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--huber-delta", type=float, default=5.0)
    ap.add_argument("--min-valid-frac", type=float, default=0.20)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UNET] device={device}  encoder={args.encoder}  in_channels={N_CHANNELS}")

    specs: list[tuple[Path, int]] = [(Path(args.stacks_dir), args.train_year_max)]
    for spec in args.extra_site:
        sdir, ymax = spec.rsplit(":", 1)
        specs.append((Path(sdir), int(ymax)))
    train_files, eval_files = list_multi_site_files(specs)
    print(f"[FILES] sites={len(specs)}  train={len(train_files)}  eval={len(eval_files)}")
    for sdir, ymax in specs:
        site_files = list_stack_files(Path(sdir))
        site_tr, site_ev = split_files_by_year(site_files, ymax)
        print(f"        - {sdir.name}: train_year<= {ymax}  train={len(site_tr)}  eval={len(site_ev)}")

    cfg = PatchSamplerConfig(
        patch=args.patch,
        n_patches_per_epoch=args.patches_per_epoch,
        min_valid_frac=args.min_valid_frac,
        augment=True,
        max_resample_tries=20,
    )
    train_ds = StackPatchDataset(train_files, cfg, seed=args.seed)

    val_cfg = PatchSamplerConfig(
        patch=args.patch,
        n_patches_per_epoch=args.val_patches,
        min_valid_frac=args.min_valid_frac,
        augment=False,
        max_resample_tries=20,
    )
    val_ds = StackPatchDataset(train_files, val_cfg, seed=args.seed + 100)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    model = build_model(args.encoder, N_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=steps_per_epoch, epochs=args.epochs, pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_rmse = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        for feats, lab, mask in train_loader:
            feats = feats.to(device, non_blocking=True)
            lab   = lab.to(device,   non_blocking=True)
            mask  = mask.to(device,  non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(feats).squeeze(1)
                loss = masked_huber_loss(pred, lab, mask, delta=args.huber_delta)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.item())
            running_n += 1

        train_loss = running_loss / max(1, running_n)

        # --- val ---
        model.eval()
        val_metrics_acc = {"rmse_sq_w": 0.0, "abs_w": 0.0, "n": 0, "sum_t": 0.0}
        with torch.no_grad():
            for feats, lab, mask in val_loader:
                feats = feats.to(device, non_blocking=True)
                lab   = lab.to(device,   non_blocking=True)
                mask  = mask.to(device,  non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred = model(feats).squeeze(1).float()
                m = (mask > 0.5)
                if not m.any():
                    continue
                p = pred[m].double()
                t = lab[m].double()
                err = p - t
                val_metrics_acc["rmse_sq_w"] += float(err.pow(2).sum().item())
                val_metrics_acc["abs_w"]     += float(err.abs().sum().item())
                val_metrics_acc["n"]         += int(m.sum().item())
                val_metrics_acc["sum_t"]     += float(t.sum().item())
        if val_metrics_acc["n"] > 0:
            val_rmse = (val_metrics_acc["rmse_sq_w"] / val_metrics_acc["n"]) ** 0.5
            val_mae  = val_metrics_acc["abs_w"] / val_metrics_acc["n"]
            val_mean_t = val_metrics_acc["sum_t"] / val_metrics_acc["n"]
            val_rrmse = val_rmse / max(val_mean_t, 1e-6) * 100.0
        else:
            val_rmse = val_mae = val_rrmse = float("nan")

        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[EPOCH {epoch:02d}/{args.epochs}] train_loss={train_loss:.4f}  "
            f"val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}  val_rrmse%={val_rrmse:.2f}  "
            f"lr={cur_lr:.2e}  {dt:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_rmse": val_rmse, "val_mae": val_mae, "val_rrmse_pct": val_rrmse,
            "lr": cur_lr, "seconds": dt,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "encoder": args.encoder,
                "in_channels": N_CHANNELS,
                "patch": args.patch,
                "best_val_rmse": best_val_rmse,
            }, out_dir / "unet_best.pt")

    with (out_dir / "history.json").open("w") as f:
        json.dump({"history": history, "best_val_rmse": best_val_rmse, "args": vars(args)}, f, indent=2)
    print(f"[DONE] best val_rmse={best_val_rmse:.4f}  saved to {out_dir / 'unet_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
