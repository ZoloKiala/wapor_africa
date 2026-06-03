# Methodology

End-to-end description of the data pipeline and training recipe used to produce the 7-site Ensemble-L1 model.

## 1. Coarse input: WaPOR L1 (300 m)

We use the **FAO WaPOR-3 L1-AETI-D** product as the coarse ETa input — the continental Africa AETI at ~300 m, dekadal cadence. The model bilinearly upsamples each L1 tile to the 20 m target grid and learns to redistribute the total ET within each 300 m cell using high-resolution Sentinel-2 features.

(Earlier baselines used L2 100 m — see `docs/results.md` for the L1 vs L2 ablation. L1 is the genuine 300 m → 20 m task.)

## 2. Auxiliary high-resolution inputs: Sentinel-2

Pulled via Google Earth Engine. Per dekad:

1. Query `COPERNICUS/S2_SR_HARMONIZED` for scenes overlapping the AOI in the 10-day window with `CLOUDY_PIXEL_PERCENTAGE < 60`.
2. Apply **SCL cloud mask**: keep classes 4 (vegetation), 5 (bare soil), 6 (water), 7 (unclassified), 11 (snow/ice). Drop clouds, cloud shadows, cirrus, saturated, dark area.
3. **Median composite** B4, B8, B11 across all kept observations in the window.
4. Reproject to the **L3 20 m target grid** (same UTM zone as the L3 reference).

## 3. Stack assembly

Per dekad, build a multi-band GeoTIFF at 20 m containing:

| Band | Channel | Source |
|---|---|---|
| 1 | B4 | Sentinel-2 red (median composite, int16 × 10000) |
| 2 | B8 | Sentinel-2 NIR |
| 3 | B11 | Sentinel-2 SWIR1 |
| 4 | `ETa300m` | **WaPOR L1 300 m → bilinear → 20 m** (mm/dekad) |
| 5 | DEM | SRTM 1-arc-sec elevation |
| 6 | Slope | derived from DEM |
| 7 | Aspect_sin | derived from DEM |
| 8 | Aspect_cos | derived from DEM |
| 9 | RAIN_10d | CHIRPS dekadal rainfall |
| 10 | RAIN_10d_lag | CHIRPS lag (previous dekad) |
| 11 | b1 | WaPOR L3 20 m ground truth (supervised target during training; ignored at inference) |

NDVI, NDMI, FVC, and sin/cos DOY are computed on the fly during model training/inference from the B4/B8/B11 bands and the file's date — see `models/unet_common.py:stack_to_tensors`.

The 9-channel model input is therefore: `[B4_n, B8_n, B11_n, ETa300m_n, NDVI, NDMI, FVC, sin_DOY, cos_DOY]`.

## 4. Training data

- **Training sites (7)**: Baixo Limpopo (Mozambique), Lamego (Mozambique), AWA Awash (Ethiopia), GEZ Gezira (Sudan), ODN Office du Niger (Mali), LOU (Morocco), JEN (Tunisia).
- **Held-out years**: Baixo 2025 + Lamego 2022 (temporal hold-out).
- **Held-out sites (transferability)**: KOGA (Ethiopia), KMW (Kenya), MIT (Algeria), MAL (Sri Lanka — extra-continental).

## 5. Models

### SwinIR-L1-7site

- Architecture: Swin Transformer for Image Restoration (Liu et al. 2021), compact config `embed_dim=96`, `depths=[4,4,4,4]`, `num_heads=[6,6,6,6]`, `window_size=16`. ~1.80M params.
- Input: 9-channel 256×256 patch.
- Output: 1-channel 256×256 prediction at same resolution (no internal upsampling — the L1 ETa is bilinearly upsampled outside the network).
- Loss: masked Huber (δ=5.0).
- Optimizer: AdamW, max_lr 2e-4, weight_decay 1e-4, grad_clip 1.0.
- LR schedule: OneCycleLR (pct_start=0.1), 100 epochs × 1500 random patches/epoch.
- Mixed precision (AMP) on CUDA.

### Prithvi-V1-L1-7site

- Architecture: IBM/NASA **Prithvi-EO-2.0-300M** ViT backbone (300M params, pretrained on HLS), with:
  - `S2ToHLSAdapter` mapping our 3 Sentinel-2 bands (B4/B8/B11) into the 6-band HLS layout the backbone expects (Coastal Aerosol, Blue, Green → 0; Red, NIR, SWIR1 → from B4/B8/B11; SWIR2 → 0). Pretrained normalization is preserved.
  - `PrithviRegression` head: side outputs from selected ViT layers, fused via PixelShuffle decoder, finalized with a 1×1 regression head.
- Backbone frozen up to block 22 (out of 24); blocks 22-23 + head fine-tuned.
- Loss: same masked Huber.
- Optimizer: AdamW with two LR groups — head 2e-4, backbone 1e-5.

### Ensemble-L1-7site

Pixel-wise 50/50 weighted mean: `0.5 * swinir_pred + 0.5 * prithvi_pred`. Computed at eval/inference time; no separate training.

## 6. Inference

For each held-out dekad:

1. Load the 9-channel 20 m stack (with L1 in the `ETa300m` channel).
2. Tile the raster into 256×256 patches with `stride = patch − overlap` (overlap = 64).
3. For each patch, forward through SwinIR + Prithvi. Stitch back with **Hann-window blending** to avoid edge artefacts.
4. Combine: `ensemble = 0.5 * swinir + 0.5 * prithvi`.
5. Write three GeoTIFFs (swinir, prithvi, ensemble), all single-band float32 mm/dekad, georeferenced to the input stack.
6. Apply the **CB-fair mask** for metric computation: only pixels where `b4 != nodata && b8 != nodata && b11 != nodata && b1 != nodata` count.

See `src/wapor_downscale/inference/predict_ensemble.py` and `src/wapor_downscale/inference/eval_ensemble.py`.

## 7. Why this works at ×15 super-resolution

300 m → 20 m is a 15× upscaling factor (much harder than the ×4-8 typical of computer-vision SR). It works here because the model is not really doing pure SR — it is doing **NDVI-guided spatial disaggregation**:

- The coarse L1 ETa sets the **regional total** (mass balance).
- The 20 m NDVI / NDMI / FVC bands say **where within each 300 m cell** the vegetation (and thus transpiration) actually is.
- The model learns to redistribute the coarse total according to fine-scale vegetation cues.

Without the auxiliary high-resolution Sentinel-2 inputs, ×15 SR would be impossible from the L1 ETa alone.

## 8. Reproducibility caveats

- WaPOR L1/L2/L3 are continuously updated; downstream metrics may shift slightly if you re-download.
- Sentinel-2 L2A reprocessing also occurs; pin to a specific `system:time_start` if you need exact reproducibility.
- GEE service-account credentials are not committed; you need to provide your own JSON.
- The Prithvi-EO-2.0-300M backbone checkpoint is not committed (~1.3 GB); download from Hugging Face.
