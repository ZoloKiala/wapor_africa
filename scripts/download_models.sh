#!/usr/bin/env bash
# Download the trained 7-site Ensemble checkpoints + Prithvi backbone.
# Replace <HOSTING_URL> with your hosting (HuggingFace, Zenodo, S3, etc.).
#
# Usage: bash scripts/download_models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p models/multi7_swinir_l1_e96_w16
mkdir -p models/multi7_prithvi_v1_l1
mkdir -p third_party/prithvi_eo_v2_300m

echo "[1/3] SwinIR-L1-7site checkpoint (~552 MB)..."
# curl -L -o models/multi7_swinir_l1_e96_w16/swinir_best.pt \
#   <HOSTING_URL>/multi7_swinir_l1_e96_w16/swinir_best.pt

echo "[2/3] Prithvi-V1-L1-7site checkpoint (~1.2 GB)..."
# curl -L -o models/multi7_prithvi_v1_l1/prithvi_best.pt \
#   <HOSTING_URL>/multi7_prithvi_v1_l1/prithvi_best.pt

echo "[3/3] Prithvi-EO-2.0-300M backbone (~1.3 GB)..."
# huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-2.0-300M \
#   --local-dir third_party/prithvi_eo_v2_300m

echo ""
echo "All commands are commented out — edit this script to point at your model hosting URL,"
echo "then re-run."
