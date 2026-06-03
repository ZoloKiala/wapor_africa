"""Train SwinIR-Lightweight on the same stacks/loss/split as the UNet final run."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# reuse the dataset / loss / metrics
from wapor_downscale.models.unet_common import (
    N_CHANNELS,
    PatchSamplerConfig,
    StackPatchDataset,
    list_multi_site_files,
    list_stack_files,
    masked_huber_loss,
    split_files_by_year,
)
from wapor_downscale.models.swinir_model import SwinIRRegression, count_params


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks-dir", default=r"c:\Users\z.kiala\Documents\wapor_africa\data\baixo\stacks\BAIXO_STACK_S2_MATCH_L3_20M_FULL_1")
    ap.add_argument("--out-dir", default=r"c:\Users\z.kiala\Documents\wapor_africa\models\baixo_swinir_final")
    ap.add_argument("--train-year-max", type=int, default=2024)
    ap.add_argument(
        "--extra-site",
        action="append",
        default=[],
        help="Additional site as 'stacks_dir:train_year_max'. Repeatable. "
             "Files from each extra site are split by its own year_max and added to training. "
             "Eval files from extras are appended after the primary site's eval files.",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patches-per-epoch", type=int, default=1500)
    ap.add_argument("--val-patches", type=int, default=256)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--huber-delta", type=float, default=5.0)
    ap.add_argument("--min-valid-frac", type=float, default=0.20)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--embed-dim", type=int, default=60)
    ap.add_argument("--depths", default="6,6,6,6")
    ap.add_argument("--num-heads", default="6,6,6,6")
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--keep-channels", default=None,
                    help="Comma-separated channel indices to keep (default: all 9). "
                         "Stack order: 0=B4, 1=B8, 2=B11, 3=ETa300m, 4=NDVI, 5=NDMI, 6=FVC, 7=SIN_DOY, 8=COS_DOY")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume an interrupted run: load weights + epoch + fast-forward LR schedule.")
    ap.add_argument("--pretrained", type=Path, default=None,
                    help="Fine-tune from a checkpoint: load weights only, restart epoch counter and "
                         "LR schedule. Use with a lower --lr for fine-tuning.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = [int(x) for x in args.depths.split(",")]
    num_heads = [int(x) for x in args.num_heads.split(",")]
    assert len(depths) == len(num_heads)
    if args.keep_channels is None:
        keep_channels = list(range(N_CHANNELS))
    else:
        keep_channels = [int(x) for x in args.keep_channels.split(",")]
    in_chans = len(keep_channels)
    print(f"[SWINIR] device={device}  in_channels={in_chans} (keep={keep_channels})  embed={args.embed_dim}  depths={depths}  heads={num_heads}  window={args.window_size}")

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
        patch=args.patch, n_patches_per_epoch=args.patches_per_epoch,
        min_valid_frac=args.min_valid_frac, augment=True, max_resample_tries=20,
    )
    train_ds = StackPatchDataset(train_files, cfg, seed=args.seed)
    val_cfg = PatchSamplerConfig(
        patch=args.patch, n_patches_per_epoch=args.val_patches,
        min_valid_frac=args.min_valid_frac, augment=False, max_resample_tries=20,
    )
    val_ds = StackPatchDataset(train_files, val_cfg, seed=args.seed + 100)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = SwinIRRegression(
        in_chans=in_chans, out_chans=1,
        embed_dim=args.embed_dim, depths=depths, num_heads=num_heads,
        window_size=args.window_size, img_size=args.patch,
    ).to(device)
    keep_ch_tensor = torch.tensor(keep_channels, dtype=torch.long, device=device)
    print(f"[MODEL] parameters: {count_params(model)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
        epochs=args.epochs, pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_rmse = float("inf")
    start_epoch_init = 1
    if args.resume is not None and args.pretrained is not None:
        raise SystemExit("Use --resume OR --pretrained, not both.")
    if args.resume is not None:
        ckpt_resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_resume["model_state"])
        prev_epoch = int(ckpt_resume.get("epoch", 0))
        start_epoch_init = prev_epoch + 1
        best_val_rmse = float(ckpt_resume.get("best_val_rmse", float("inf")))
        ff_steps = prev_epoch * steps_per_epoch
        for _ in range(ff_steps):
            scheduler.step()
        print(f"[RESUME] loaded {args.resume} (best_val_rmse={best_val_rmse:.4f}) -> "
              f"resuming at epoch {start_epoch_init}; scheduler advanced by {ff_steps} steps; "
              f"current LR={optimizer.param_groups[0]['lr']:.2e}")
    elif args.pretrained is not None:
        ckpt_pre = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_pre["model_state"])
        print(f"[FINETUNE] loaded weights from {args.pretrained}; "
              f"fresh epoch counter + LR schedule (use a lower --lr if fine-tuning)")
    history = []
    for epoch in range(start_epoch_init, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        for feats, lab, mask in train_loader:
            feats = feats.to(device, non_blocking=True)
            feats = feats.index_select(1, keep_ch_tensor)
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

        model.eval()
        acc = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0}
        with torch.no_grad():
            for feats, lab, mask in val_loader:
                feats = feats.to(device, non_blocking=True)
                feats = feats.index_select(1, keep_ch_tensor)
                lab   = lab.to(device,   non_blocking=True)
                mask  = mask.to(device,  non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred = model(feats).squeeze(1).float()
                m = (mask > 0.5)
                if not m.any():
                    continue
                p = pred[m].double(); t = lab[m].double(); err = p - t
                acc["err2"]  += float(err.pow(2).sum().item())
                acc["abs"]   += float(err.abs().sum().item())
                acc["n"]     += int(m.sum().item())
                acc["sum_t"] += float(t.sum().item())
        if acc["n"]:
            val_rmse = (acc["err2"] / acc["n"]) ** 0.5
            val_mae  = acc["abs"] / acc["n"]
            tmean = acc["sum_t"] / acc["n"]
            val_rrmse = val_rmse / max(tmean, 1e-6) * 100.0
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
            "epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse,
            "val_mae": val_mae, "val_rrmse_pct": val_rrmse, "lr": cur_lr, "seconds": dt,
        })
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "embed_dim": args.embed_dim, "depths": depths, "num_heads": num_heads,
                "window_size": args.window_size, "in_chans": in_chans, "patch": args.patch,
                "keep_channels": keep_channels,
                "best_val_rmse": best_val_rmse,
            }, out_dir / "swinir_best.pt")

    (out_dir / "history.json").write_text(json.dumps({"history": history, "best_val_rmse": best_val_rmse, "args": vars(args)}, indent=2))
    print(f"[DONE] best val_rmse={best_val_rmse:.4f}  saved to {out_dir / 'swinir_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
