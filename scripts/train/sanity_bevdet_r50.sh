#!/usr/bin/env bash
# ==============================================================================
# Script: sanity_bevdet_r50.sh
# Purpose: Training Sanity Verification for BEVDet-R50
# Repo: BEVDet
# Branch: dev3.0
# Commit: Official Latest
# Config: configs/bevdet/bevdet-r50-cbgs.py
# Dataset: nuScenes v1.0-mini (train split)
# Expected output: Real forward/backward, loss descent, checkpoint saved
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting BEVDet-R50 Training Sanity..."
${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_bevdet_r50.py
