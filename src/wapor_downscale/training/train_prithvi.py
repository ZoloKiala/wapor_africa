"""Train PrithviRegression on the multi-site (Baixo + Lamego) downscaling task.

Mirrors scripts/swinir/swinir_train.py: same dataset, same loss, same
val-during-training scheme. Differences:
    - PrithviRegression backbone (300M frozen ViT + 36M trainable)
    - Two-group AdamW: low lr for unfrozen backbone blocks, high lr for new layers
    - Defaults tuned for the smaller trainable budget
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Re-use the SwinIR/UNet dataset + loss.
from wapor_downscale.models.unet_common import (
    N_CHANNELS,
    PatchSamplerConfig,
    StackPatchDataset,
    list_multi_site_files,
    list_stack_files,
    masked_huber_loss,
    split_files_by_year,
)
from wapor_downscale.models.prithvi_regression import PrithviRegression, PrithviRegressionV2, PrithviRegressionV3, count_params


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks-dir", default=r"c:\Users\z.kiala\Documents\wapor_africa\data\baixo\stacks\BAIXO_STACK_S2_MATCH_L3_20M_FULL_1")
    ap.add_argument("--out-dir", default=r"c:\Users\z.kiala\Documents\wapor_africa\models\prithvi_300m_v2_freeze22")
    ap.add_argument("--model-version", choices=("v1", "v2", "v3"), default="v3",
                    help="v1 = original; v2 = sum-fusion + thin PixelShuffle (worse); "
                         "v3 = concat-fusion + thicker decoder schedule")
    ap.add_argument("--train-year-max", type=int, default=2024)
    ap.add_argument("--extra-site", action="append", default=[],
                    help="Additional site as 'stacks_dir:train_year_max'. Repeatable.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patches-per-epoch", type=int, default=1500)
    ap.add_argument("--val-patches", type=int, default=256)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr-head", type=float, default=2e-4, help="LR for new layers (side+upsample+head)")
    ap.add_argument("--lr-backbone", type=float, default=1e-5, help="LR for unfrozen backbone blocks")
    ap.add_argument("--freeze-until", default="22",
                    help="int N freezes blocks 0..N-1; 'all' freezes everything; 'none' trains everything")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--huber-delta", type=float, default=5.0)
    ap.add_argument("--min-valid-frac", type=float, default=0.20)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume an interrupted run: load weights + epoch + fast-forward LR schedule.")
    ap.add_argument("--pretrained", type=Path, default=None,
                    help="Fine-tune from a checkpoint: load weights only, restart epoch counter and "
                         "LR schedule. Use with lower --lr-head / --lr-backbone for fine-tuning.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    return ap.parse_args()


def _new_param_groups(model, lr_backbone: float, lr_head: float, wd: float) -> list[dict]:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    bb_ids = {id(p) for p in backbone_params}
    # All non-backbone trainable params go into the "head" group regardless of V1/V2 layout.
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone.") and id(p) not in bb_ids]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": wd, "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": wd, "name": "head"})
    return groups


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File lists.
    specs: list[tuple[Path, int]] = [(Path(args.stacks_dir), args.train_year_max)]
    for spec in args.extra_site:
        sdir, ymax = spec.rsplit(":", 1)
        specs.append((Path(sdir), int(ymax)))
    train_files, eval_files = list_multi_site_files(specs)
    print(f"[PRITHVI] device={device}  in_channels={N_CHANNELS}  freeze_until={args.freeze_until}")
    print(f"[FILES] sites={len(specs)}  train={len(train_files)}  eval={len(eval_files)}")
    for sdir, ymax in specs:
        site_files = list_stack_files(Path(sdir))
        site_tr, site_ev = split_files_by_year(site_files, ymax)
        print(f"        - {sdir.name}: train_year<= {ymax}  train={len(site_tr)}  eval={len(site_ev)}")

    # Datasets / loaders.
    cfg = PatchSamplerConfig(patch=args.patch, n_patches_per_epoch=args.patches_per_epoch,
                             min_valid_frac=args.min_valid_frac, augment=True, max_resample_tries=20)
    train_ds = StackPatchDataset(train_files, cfg, seed=args.seed)
    val_cfg = PatchSamplerConfig(patch=args.patch, n_patches_per_epoch=args.val_patches,
                                 min_valid_frac=args.min_valid_frac, augment=False, max_resample_tries=20)
    val_ds = StackPatchDataset(train_files, val_cfg, seed=args.seed + 100)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Model.
    freeze: str | int = args.freeze_until
    try:
        freeze = int(args.freeze_until)
    except ValueError:
        pass  # keep "all" / "none"
    model_cls = {"v1": PrithviRegression, "v2": PrithviRegressionV2, "v3": PrithviRegressionV3}[args.model_version]
    model = model_cls(img_size=args.patch, freeze_until=freeze, mock_backbone=False).to(device)
    total, trainable = count_params(model)
    print(f"[MODEL] {args.model_version.upper()}  total {total/1e6:.2f}M  trainable {trainable/1e6:.2f}M")

    optimizer = torch.optim.AdamW(_new_param_groups(model, args.lr_backbone, args.lr_head, args.weight_decay))
    steps_per_epoch = max(1, len(train_loader))
    # OneCycle scales each group's lr to max_lr; pass list to keep groups separate.
    max_lrs = [g["lr"] for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs,
        steps_per_epoch=steps_per_epoch, epochs=args.epochs, pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_rmse = float("inf")
    start_epoch = 1
    if args.resume is not None and args.pretrained is not None:
        raise SystemExit("Use --resume OR --pretrained, not both.")
    if args.resume is not None:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        prev_epoch = int(ck.get("epoch", 0))
        start_epoch = prev_epoch + 1
        best_val_rmse = float(ck.get("best_val_rmse", float("inf")))
        ff = prev_epoch * steps_per_epoch
        for _ in range(ff):
            scheduler.step()
        print(f"[RESUME] loaded {args.resume} (best_val_rmse={best_val_rmse:.4f}) -> "
              f"start at epoch {start_epoch}, scheduler advanced by {ff} steps")
    elif args.pretrained is not None:
        ck = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(f"[FINETUNE] loaded weights from {args.pretrained}; "
              f"fresh epoch counter + LR schedule")

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        for feats, lab, mask in train_loader:
            feats = feats.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(feats).squeeze(1)
                loss = masked_huber_loss(pred, lab, mask, delta=args.huber_delta)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.item())
            running_n += 1
        train_loss = running_loss / max(1, running_n)

        # Val
        model.eval()
        acc = {"err2": 0.0, "abs": 0.0, "n": 0, "sum_t": 0.0}
        with torch.no_grad():
            for feats, lab, mask in val_loader:
                feats = feats.to(device, non_blocking=True)
                lab = lab.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
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
        cur_lr_head = next((g["lr"] for g in optimizer.param_groups if g.get("name") == "head"), float("nan"))
        cur_lr_bb = next((g["lr"] for g in optimizer.param_groups if g.get("name") == "backbone"), float("nan"))
        print(
            f"[EPOCH {epoch:02d}/{args.epochs}] train_loss={train_loss:.4f}  "
            f"val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}  val_rrmse%={val_rrmse:.2f}  "
            f"lr_h={cur_lr_head:.2e}  lr_b={cur_lr_bb:.2e}  {dt:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse,
            "val_mae": val_mae, "val_rrmse_pct": val_rrmse,
            "lr_head": cur_lr_head, "lr_backbone": cur_lr_bb, "seconds": dt,
        })
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "freeze_until": args.freeze_until,
                "img_size": args.patch, "best_val_rmse": best_val_rmse,
                "model_version": args.model_version,
                "config": vars(args),
            }, out_dir / "prithvi_best.pt")

    (out_dir / "history.json").write_text(json.dumps(
        {"history": history, "best_val_rmse": best_val_rmse, "args": vars(args)}, indent=2, default=str))
    print(f"[DONE] best val_rmse={best_val_rmse:.4f}  saved to {out_dir / 'prithvi_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
