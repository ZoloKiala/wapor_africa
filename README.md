# wapor-downscale

Deep-learning downscaling of FAO WaPOR dekadal actual evapotranspiration (AETI) from **300 m (L1)** to **20 m (L3)** across African irrigation pilots, with Sentinel-2 NDVI / NDMI / FVC as high-resolution auxiliary predictors.

## What this delivers

A **7-site Ensemble (SwinIR + Prithvi-V1, 50/50 weighted average)** trained on five sub-Saharan + two Mediterranean African irrigation pilots, producing operationally-usable 20 m dekadal AETI maps anywhere on the continent without retraining.

### Headline results (RMSE in mm/day)

| Hold-out site | Setting | Ensemble-L1-7site | SwinIR-L1-7site | Prithvi-V1-L1-7site |
|---|---|---|---|---|
| Baixo + Lamego (in-domain) | Mozambique | **0.393** | 0.398 | 0.409 |
| MIT (Algeria) | Mediterranean transfer | **0.347** | 0.356 | 0.366 |
| KOGA (Ethiopia) | sub-Saharan transfer | **0.406** | 0.435 | 0.421 |
| KMW (Kenya) | mid-altitude transfer | **0.432** | 0.458 | 0.437 |
| MAL (Sri Lanka) | extra-continental | 0.679 | **0.674** | 0.709 |

Training stayed in Africa; the model generalises well across African agro-climates (RMSE 0.35-0.43 mm/d) but degrades sharply on extra-continental tropical paddy systems.

## Repo layout

```
wapor-downscale/
├── src/wapor_downscale/        # Python package
│   ├── data_prep/              # WaPOR + Sentinel-2 ingestion, stack building
│   ├── models/                 # SwinIR + Prithvi-V1 architectures + dataset
│   ├── training/               # train_swinir.py, train_prithvi.py, train_catboost.py, train_unet.py
│   └── inference/              # predict_ensemble.py + per-model evaluators
├── configs/                    # example_baixo.yaml, example_koga.yaml
├── notebooks/inference_new_aoi.ipynb   # end-to-end inference on a new AOI
├── docs/
│   ├── methodology.md          # data pipeline + training recipe
│   └── results.md              # full transferability tables
├── third_party/                # NOT committed — Prithvi-EO-2.0-300M backbone (download separately)
├── requirements.txt
├── pyproject.toml
├── environment.yml
└── LICENSE                     # MIT
```

## Install

```bash
# 1. Clone
git clone https://github.com/ZoloKiala/wapor_africa.git wapor-downscale
cd wapor-downscale

# 2. Create env (conda recommended for rasterio/torch on Windows)
conda env create -f environment.yml
conda activate wapor-downscale

# 3. Pip-install the package in editable mode
pip install -e .

# 4. Download the Prithvi-EO-2.0-300M backbone weights (one-time, ~1.3 GB)
mkdir -p third_party
huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-2.0-300M \
  --local-dir third_party/prithvi_eo_v2_300m
```

## Trained model checkpoints

Not committed (each is 500-700 MB). Download from your model hosting (Hugging Face / Zenodo / S3) and place in `models/`:

```
models/multi7_swinir_l1_e96_w16/swinir_best.pt   (~552 MB)
models/multi7_prithvi_v1_l1/prithvi_best.pt      (~1.2 GB)
```

Replace `<HOSTING_URL>` in `scripts/download_models.sh` with your chosen URL.

## Quick start — inference on a new AOI

The fastest path is the Jupyter notebook:

```bash
jupyter lab notebooks/inference_new_aoi.ipynb
```

Edit Cell 1 to set:
- `SITE_NAME` — short tag (e.g., `XYZ`)
- `L3_SITE_CODE` — WaPOR L3 site code if a pilot exists, else `None`
- `AOI_BBOX_4326` — `[lon_min, lat_min, lon_max, lat_max]`
- `DATE_START`, `DATE_END`

Run cells top to bottom. Outputs land in `models/multi7_ensemble_l1/predictions/<SITE_NAME>/`.

### CLI alternative

```bash
# 1. Download WaPOR coarse + Sentinel-2 for your AOI
python -m wapor_downscale.data_prep.download_wapor_l1 --config configs/example_koga.yaml
python -m wapor_downscale.data_prep.fetch_s2_gee     --config configs/example_koga.yaml
python -m wapor_downscale.data_prep.build_stack      --config configs/example_koga.yaml
python -m wapor_downscale.data_prep.rebuild_stack_l1 --config configs/example_koga.yaml

# 2. Run inference (per dekad)
python -m wapor_downscale.inference.predict_ensemble \
  --swinir-ckpt  models/multi7_swinir_l1_e96_w16/swinir_best.pt \
  --prithvi-ckpt models/multi7_prithvi_v1_l1/prithvi_best.pt \
  --stack        data/koga/stacks/KOGA_STACK_S2_MATCH_L3_20M_L1_FULL_1/KOGA_2024-04-01.tif \
  --out-dir      models/multi7_ensemble_l1/predictions/KOGA \
  --weight 0.5
```

## Fine-tune for your own area

If you have WaPOR L3 ground truth for some dekads of your site and want to adapt the pre-trained model to it (rather than retrain from scratch), use:

```bash
jupyter lab notebooks/finetune_new_aoi.ipynb
```

The notebook walks through: load pre-trained 7-site weights → fine-tune at a reduced LR for 20-30 epochs → evaluate fine-tuned vs zero-shot on hold-out. See [`docs/finetuning.md`](docs/finetuning.md) for guidance on when fine-tuning helps vs hurts and how much data you need.

CLI equivalent:

```bash
# Fine-tune SwinIR (note --pretrained, not --resume)
python -m wapor_downscale.training.train_swinir \
  --stacks-dir data/newsite/stacks/NEWSITE_STACK_S2_MATCH_L3_20M_L1_FULL_1 \
  --out-dir    models/finetune_swinir_newsite \
  --pretrained models/multi7_swinir_l1_e96_w16/swinir_best.pt \
  --train-year-max 2023 --epochs 30 --lr 5e-5 \
  --embed-dim 96 --depths 4,4,4,4 --window-size 16 --batch-size 2

# Fine-tune Prithvi-V1
python -m wapor_downscale.training.train_prithvi \
  --stacks-dir data/newsite/stacks/NEWSITE_STACK_S2_MATCH_L3_20M_L1_FULL_1 \
  --out-dir    models/finetune_prithvi_newsite \
  --pretrained models/multi7_prithvi_v1_l1/prithvi_best.pt \
  --train-year-max 2023 --epochs 30 \
  --lr-head 5e-5 --lr-backbone 1e-6 \
  --model-version v1 --freeze-until 22 --batch-size 2
```

## Training (if you want to retrain)

```bash
# SwinIR (compact transformer, ~1.8M params)
python -m wapor_downscale.training.train_swinir \
  --stacks-dir data/baixo/stacks/BAIXO_STACK_S2_MATCH_L3_20M_L1_FULL_1 \
  --extra-site data/lamego/stacks/LAMEGO_STACK_S2_MATCH_L3_20M_L1_FULL_1:2021 \
  --extra-site data/awa/stacks/AWA_STACK_S2_MATCH_L3_20M_L1_FULL_1:2024 \
  --extra-site data/gez/stacks/GEZ_STACK_S2_MATCH_L3_20M_L1_FULL_1:2024 \
  --extra-site data/odn/stacks/ODN_STACK_S2_MATCH_L3_20M_L1_FULL_1:2024 \
  --extra-site data/lou/stacks/LOU_STACK_S2_MATCH_L3_20M_L1_FULL_1:2024 \
  --extra-site data/jen/stacks/JEN_STACK_S2_MATCH_L3_20M_L1_FULL_1:2024 \
  --out-dir models/my_swinir_run \
  --embed-dim 96 --depths 4,4,4,4 --window-size 16 \
  --epochs 100 --batch-size 2 --lr 2e-4

# Prithvi-V1 (frozen first 22 ViT blocks)
python -m wapor_downscale.training.train_prithvi \
  --stacks-dir data/baixo/stacks/BAIXO_STACK_S2_MATCH_L3_20M_L1_FULL_1 \
  --extra-site data/lamego/stacks/...  (same as above)
  --out-dir models/my_prithvi_run \
  --model-version v1 --freeze-until 22 \
  --epochs 100 --batch-size 2 --lr-head 2e-4 --lr-backbone 1e-5
```

Wall-clock: ~10 h on RTX 5090 for 7-site SwinIR, ~3 h for Prithvi-V1.

## Architecture

- **Input** (per dekad, 20 m grid): 9 channels — B4, B8, B11 (Sentinel-2), ETa300m (L1 bilinear), NDVI, NDMI, FVC, sin/cos DOY.
- **SwinIR** — Swin Transformer for Image Restoration (Liu et al. 2021); compact config `embed_dim=96 depths=[4,4,4,4] window_size=16` (~1.8M params).
- **Prithvi-V1** — IBM/NASA Prithvi-EO-2.0-300M ViT backbone, with `S2ToHLSAdapter` mapping our 3 S2 bands into the 6-band HLS layout the backbone expects; `PrithviRegression` head freezes the first 22 transformer blocks.
- **Ensemble** — 50/50 pixel-wise mean of SwinIR + Prithvi-V1 predictions.

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for the full data pipeline and training recipe.

## Results

See [`docs/results.md`](docs/results.md) for per-site / per-dekad / per-model tables.

## Provenance

This repo distills work from a multi-week ML pilot for the IWMI WaPOR L3 AETI 20 m downscaling effort. The codebase includes:
- Bergaoui-style SR-DRN (compact U-Net) reproduction
- Multi-site retrain experiments (2-site → 5-site → 7-site)
- L1 (300 m) vs L2 (100 m) coarse-input ablations
- Transferability tests on KOGA, KMW, MIT, MAL

The 7-site Ensemble-L1 is the operational deliverable.

## Citation

If you use this code or models, please cite (placeholder):

```
@misc{wapor_downscale_2026,
  author = {Kiala, Z. and contributors},
  title  = {WaPOR L3 AETI 300m → 20m downscaling — multi-site African ensemble},
  year   = 2026,
  howpublished = {\url{https://github.com/ZoloKiala/wapor_africa}},
}
```

## License

MIT — see [LICENSE](LICENSE).
