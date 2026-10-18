#!/usr/bin/env bash
# ==============================================================================
# Script: run_bevdepth_r50.sh
# Purpose: Real Inference and Intermediate Extraction for BEVDepth-R50
# Repo: BEVDet
# Branch: dev3.0
# Commit: Official Latest
# Config: configs/bevdet/bevdet-r50-4d-depth-cbgs.py
# Checkpoint: /public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-4d-depth-cbgs.pth
# Dataset: nuScenes v1.0-mini
# Expected output: 7 intermediate representations extracted and saved to outputs/bevdepth_real_intermediates/
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting BEVDepth-R50 Real Inference & Intermediate Extraction..."
${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_bevdepth_real_intermediates.py
