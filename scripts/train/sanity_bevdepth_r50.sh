#!/usr/bin/env bash
# ==============================================================================
# Script: sanity_bevdepth_r50.sh
# Purpose: Training Sanity Verification for BEVDepth-R50 (Forward, Loss, Backward, Step)
# Repo: BEVDet dev3.0
# Config: configs/bevdet/bevdet-r50-4d-depth-cbgs.py
# Dataset: nuScenes v1.0-mini
# Expected output: outputs/checkpoints/bevdepth_r50_sanity.pth
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting BEVDepth-R50 Training Sanity Verification..."
${PYTHON_BIN} -u ${PROJECT_ROOT}/scripts/train/sanity_bevdepth_r50.py
