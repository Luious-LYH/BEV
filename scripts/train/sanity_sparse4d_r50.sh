#!/usr/bin/env bash
# ==============================================================================
# Script: sanity_sparse4d_r50.sh
# Purpose: Training Sanity Verification for Sparse4Dv3-R50 (Forward, Loss, Backward, Step)
# Repo: Sparse4D
# Branch: main (v3)
# Config: projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
# Checkpoint: /public/lyh/pretrained_model/BEV/sparse4d/sparse4dv3_r50.pth
# Dataset: nuScenes v1.0-mini
# Expected output: outputs/checkpoints/sparse4d_r50_sanity.pth
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}"

echo "Starting Sparse4Dv3-R50 Training Sanity Verification..."
${PYTHON_BIN} -u ${PROJECT_ROOT}/scripts/train/sanity_sparse4d_r50.py
