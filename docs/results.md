# Results

Full transferability tables for the 7-site Ensemble-L1 model and its ablations.

All metrics computed on the **CB-fair pixel mask** (only pixels where label & B4 & B8 & B11 are all valid). RMSE / MAE in mm/dekad and mm/day side-by-side; mm/day is just mm/dekad ÷ 10.

## Headline: 7-site Ensemble-L1 across 5 hold-outs

| Hold-out | Type | n_dekads | RMSE mm/dekad | RMSE mm/day | R² (pixel-weighted) |
|---|---|---|---|---|---|
| Baixo + Lamego in-domain | temporal hold-out | 72 | 3.93 | **0.393** | 0.877 |
| MIT (Algeria) | Mediterranean transfer | 36 | 3.47 | **0.347** | 0.839 |
| KOGA (Ethiopia) | sub-Saharan transfer | 30 | 4.06 | 0.406 | 0.844 |
| KMW (Kenya) | mid-altitude transfer | 30 | 4.32 | 0.432 | 0.803 |
| MAL (Sri Lanka) | extra-continental | 31 | 6.79 | 0.679 | 0.564 |

## Site-by-site comparison: SwinIR vs Prithvi vs Ensemble (7-site, pixel-weighted RMSE mm/day)

| Hold-out | SwinIR-L1-7site | Prithvi-V1-L1-7site | Ensemble-L1-7site |
|---|---|---|---|
| Baixo + Lamego in-domain | 0.398 | 0.409 | **0.393** |
| MIT (Algeria) | 0.356 | 0.366 | **0.347** |
| KOGA (Ethiopia) | 0.435 | 0.421 | **0.406** |
| KMW (Kenya) | 0.458 | 0.437 | **0.432** |
| MAL (Sri Lanka) | **0.674** | 0.709 | 0.679 |

Ensemble wins or ties on all African hold-outs. On Sri Lanka (extra-continental tropical paddy systems with no training analog), SwinIR alone slightly beats the ensemble.

## Effect of training-set diversity (RMSE mm/day, in-domain → KOGA)

| Training set | SwinIR in-domain | SwinIR KOGA | Prithvi in-domain | Prithvi KOGA |
|---|---|---|---|---|
| 2 sites (Baixo + Lamego) | 0.395 | 0.484 | 0.402 | 0.634 |
| 5 sites (+ AWA + GEZ + ODN) | 0.416 | 0.436 | 0.404 | 0.456 |
| 7 sites (+ LOU + JEN) | 0.398 | 0.435 | 0.409 | 0.421 |

Training diversity cut KOGA RMSE by ~10% for SwinIR and ~34% for Prithvi (Prithvi was the most over-fit to Mozambique with 2 sites only). In-domain accuracy stayed roughly constant.

## L1 (300 m) vs L2 (100 m) coarse input — earlier ablation

The pre-L1 baseline mistakenly used WaPOR L2 (100 m) in the coarse channel, which made the task ×5 SR instead of ×15 SR.

| Variant | Coarse | SR factor | SwinIR in-domain RMSE mm/d | SwinIR KOGA RMSE mm/d |
|---|---|---|---|---|
| Pre-L1 SwinIR-sweep (5-site) | L2 100m | ×5 | 0.333 | 0.323 |
| **L1 SwinIR-7site (true 300m → 20m)** | L1 300m | ×15 | **0.398** | **0.435** |

Doing the genuine ×15 task costs ~20-30% accuracy but is the operationally correct setup (L1 is the always-available product; L2 is country-specific and may be discontinued).

## Two-stage cascade vs end-to-end

We also tested a Bergaoui-style 300 → 100 → 20 cascade (SwinIR Stage-1 produces L2 100m, then existing pre-L1 SwinIR Stage-2 produces L3 20m).

| Variant | in-domain RMSE mm/d | KOGA RMSE mm/d |
|---|---|---|
| 5-site direct SwinIR | 0.416 | 0.436 |
| 5-site SwinIR cascade | 0.439 | 0.466 |

The cascade is **worse** in both settings. Stage-1 training was unstable (NaN at late epochs) and error compounding outweighed any benefit from intermediate L2 supervision. Bergaoui's per-tile training (30-72 separate models per country) probably mitigates this in his setup; our global single-model cascade does not.

## Per-dekad detail (representative)

### 7-site Ensemble-L1 on Baixo 2025 (selected)

| Dekad | RMSE mm/day | R² | NRMSE% |
|---|---|---|---|
| 2025-04-21 | 0.260 | 0.826 | 13.4 |
| 2025-06-11 | 0.197 | 0.862 | 13.3 |
| 2025-08-21 | 0.274 | 0.893 | 15.8 |
| 2025-09-01 | 0.269 | 0.896 | 16.9 |
| 2025-12-21 | 0.636 | 0.642 | 24.1 |

Best dekads have R² 0.85-0.90, NRMSE 13-17% (mid-season clear-sky periods). Worst dekads (early Jan, late Dec) have NRMSE 30-40% — cloud-impaired and edges of growing season.

### 7-site Ensemble-L1 on MIT 2024 (Algeria, far-transfer)

| Dekad | RMSE mm/day | R² | NRMSE% |
|---|---|---|---|
| 2024-09-11 | 0.306 | 0.882 | 20.7 |
| 2024-10-01 | 0.267 | 0.870 | 18.6 |
| 2024-11-21 | 0.229 | 0.807 | 19.0 |
| 2024-12-21 | 0.136 | 0.767 | 21.9 |

Mediterranean winter dekads are easy (R² 0.7-0.9, NRMSE ≤22%). The MIT mean across all 36 dekads (0.347 mm/day) is **better than in-domain** because LOU+JEN gave the model close-to-Mediterranean training examples.

### Per-dekad failure mode: Sri Lanka monsoon

On MAL 2024, dekads in May-August (southwest monsoon) all blow up — RMSE 0.8-1.0 mm/day, NRMSE 30-50%, R² often negative. Tropical paddy systems with permanently inundated fields are far enough from any training site that the model cannot extrapolate. Dry/post-monsoon dekads (Mar-Apr, Oct-Nov) achieve R² 0.45-0.69 — partial success.

## Per-dekad CSV files

Per-dekad metrics for every model × hold-out combination are saved in the comparison runs as `fair_per_dekad_<model>_<hold-out>.csv`. Schema:

| Column | Meaning |
|---|---|
| site | Site code (BAIXO, LAMEGO, KOGA, KMW, MIT, MAL) |
| tag | Stack file stem (e.g., `BAIXO_2025-04-01`) |
| date | YYYY-MM-DD |
| n_pix | Number of valid pixels in the CB-fair mask |
| rmse | RMSE in mm/dekad |
| mae | MAE in mm/dekad |
| r2 | R² |
| rrmse_pct | NRMSE (= rmse / mean_target × 100) |

(In this repo template we don't ship the CSVs; they're regenerated by re-running the eval scripts.)
