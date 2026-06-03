# Fine-tuning the 7-site Ensemble for a new AOI

This guide is the prose companion to [`notebooks/finetune_new_aoi.ipynb`](../notebooks/finetune_new_aoi.ipynb). It explains **when fine-tuning helps**, **when it hurts**, and how to choose hyperparameters.

## What the pre-trained model gives you for free

The released checkpoints — `multi7_swinir_l1_e96_w16/swinir_best.pt` and `multi7_prithvi_v1_l1/prithvi_best.pt` — were trained on seven African irrigation pilots (Baixo, Lamego, AWA, GEZ, ODN, LOU, JEN) covering sub-Saharan and Mediterranean agro-climates. Out-of-the-box transfer RMSE (mm/day, Ensemble):

| Hold-out site | Setting | Zero-shot Ensemble |
|---|---|---|
| MIT (Algeria) | Mediterranean transfer | 0.347 |
| KOGA (Ethiopia) | sub-Saharan transfer | 0.406 |
| KMW (Kenya) | mid-altitude transfer | 0.432 |
| MAL (Sri Lanka) | extra-continental | 0.679 |

If your AOI is climatically inside the African training distribution, zero-shot is usually already operationally usable. Fine-tuning is the path to chase the **last 5-15% of accuracy** — or to recover usability for extra-continental sites where the zero-shot model is shaky.

## Decision rubric: fine-tune, retrain, or zero-shot?

| Situation | Recommendation |
|---|---|
| AOI inside Africa, no L3 ground truth available | **Zero-shot** — use the 7-site Ensemble as-is. |
| AOI inside Africa, < 20 dekads of L3 ground truth | **Zero-shot** — too little data to fine-tune safely. |
| AOI inside Africa, 20-50 dekads of L3 | **Fine-tune** with this guide. Typical RMSE reduction 5-15%. |
| AOI inside Africa, > 50 dekads of L3 | Try **both** — fine-tune AND train from scratch — and keep the better one. |
| Extra-continental (Asia, LatAm), any data | **Fine-tune is the safe first move.** If RMSE is still poor, retrain from scratch. |
| New architecture (different encoder) | **Train from scratch** — pre-trained head shape won't match. |

## What the `--pretrained` flag does

`train_swinir.py` and `train_prithvi.py` both accept two mutually-exclusive checkpoint flags:

| Flag | Loads weights? | Resets epoch counter? | Resets LR schedule? | Use when |
|---|---|---|---|---|
| `--resume <ckpt>` | yes | no (continues) | no (fast-forwards) | An interrupted run died; you want to pick up exactly where it stopped. |
| `--pretrained <ckpt>` | yes | **yes** | **yes** | Fine-tuning: start a fresh schedule on top of pre-trained weights. |

Passing both raises `SystemExit` — they are conceptually different operations.

## Recommended hyperparameters

Fine-tuning needs a **smaller learning rate** and **fewer epochs** than from-scratch training. The defaults below are what we've seen work in our own transfer tests:

### SwinIR fine-tune

```bash
python -m wapor_downscale.training.train_swinir \
  --stacks-dir data/<site>/stacks/<site>_STACK_S2_MATCH_L3_20M_L1_FULL_1 \
  --out-dir    models/finetune_swinir_<site> \
  --pretrained models/multi7_swinir_l1_e96_w16/swinir_best.pt \
  --train-year-max 2023 \
  --epochs 30 \
  --lr 5e-5 \
  --embed-dim 96 --depths 4,4,4,4 --window-size 16 \
  --patches-per-epoch 1500 --val-patches 256 --patch 256 --batch-size 2
```

`--lr 5e-5` is **4× lower** than the from-scratch default (`2e-4`). The architecture flags (`--embed-dim 96 --depths 4,4,4,4 --window-size 16`) **must match the checkpoint** — these are the values used for `multi7_swinir_l1_e96_w16`.

### Prithvi-V1 fine-tune

```bash
python -m wapor_downscale.training.train_prithvi \
  --stacks-dir data/<site>/stacks/<site>_STACK_S2_MATCH_L3_20M_L1_FULL_1 \
  --out-dir    models/finetune_prithvi_<site> \
  --pretrained models/multi7_prithvi_v1_l1/prithvi_best.pt \
  --train-year-max 2023 \
  --epochs 30 \
  --lr-head 5e-5 \
  --lr-backbone 1e-6 \
  --model-version v1 --freeze-until 22 \
  --patches-per-epoch 1500 --val-patches 256 --patch 256 --batch-size 2
```

The backbone LR (`1e-6`) is **10× lower** than the from-scratch default (`1e-5`). Prithvi's pre-training (HLS / Earth-observation FM) is valuable signal — a too-aggressive LR will overwrite it.

## When you have very little data (< 20 dekads)

Even more conservative:

```
--lr 1e-5            (SwinIR)
--lr-head 1e-5  --lr-backbone 5e-7   (Prithvi)
--epochs 10-15
```

Stop early — the validation curve will likely plateau within 5-8 epochs. Save the best-val checkpoint (the trainers already do this automatically — they write `*_best.pt` whenever val RMSE improves).

## Validating that fine-tuning actually helped

Always benchmark **fine-tuned vs zero-shot on the same hold-out dekads**:

```bash
# Fine-tuned
python -m wapor_downscale.inference.eval_ensemble \
  --swinir-ckpt  models/finetune_swinir_<site>/swinir_best.pt \
  --prithvi-ckpt models/finetune_prithvi_<site>/prithvi_best.pt \
  --site-spec    data/<site>/stacks/<site>_STACK_S2_MATCH_L3_20M_L1_FULL_1:2023 \
  --out-tag      ensemble_<site>_finetuned --weight 0.5

# Zero-shot (same hold-out)
python -m wapor_downscale.inference.eval_ensemble \
  --swinir-ckpt  models/multi7_swinir_l1_e96_w16/swinir_best.pt \
  --prithvi-ckpt models/multi7_prithvi_v1_l1/prithvi_best.pt \
  --site-spec    data/<site>/stacks/<site>_STACK_S2_MATCH_L3_20M_L1_FULL_1:2023 \
  --out-tag      ensemble_<site>_zeroshot --weight 0.5
```

Both runs write to `models/comparisons/fair_aggregate_*.json`. If fine-tuning **doesn't beat zero-shot**, you have a problem — most likely:
- LR is still too high (try 5× lower)
- Too few epochs (run more, but watch val RMSE)
- Not enough training data (fall back to zero-shot)
- Hold-out dekads are systematically different from training dekads (e.g., a regime shift) — re-split

## Catastrophic forgetting

If the fine-tuned model is **worse than zero-shot** on the hold-out, you've over-adapted to the training dekads at the cost of the model's prior. Two remedies:

1. **Lower both LRs by 5×** and rerun.
2. **Reduce epochs** to 10-15 and use the best-val checkpoint (already done automatically).

## Saving and sharing your fine-tune

The fine-tuned `swinir_best.pt` / `prithvi_best.pt` files are **fully self-contained** — they include the full state dict, not just a delta. Upload them to your model hosting (HuggingFace, Zenodo, S3) and point downstream users at them directly via `predict_ensemble.py` or the inference notebook. No need to ship the original 7-site checkpoint alongside.

## Channel layout

If you fine-tune, the 9-channel input layout is **fixed**:

```
0: B4    (Sentinel-2 red)
1: B8    (Sentinel-2 NIR)
2: B11   (Sentinel-2 SWIR-1)
3: ETa300m  (WaPOR L1, bilinear-upsampled to 20m)
4: NDVI  (Savitzky-Golay smoothed)
5: NDMI
6: FVC
7: SIN_DOY
8: COS_DOY
```

If your new AOI's stack uses a different channel ordering or a subset, you must either (a) rebuild the stack to match this layout, or (b) train from scratch — the pre-trained weights expect exactly these 9 channels in this order.
