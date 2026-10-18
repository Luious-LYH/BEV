#!/usr/bin/env bash
# ==============================================================================
# Script: sanity_bevformer_small.sh
# Purpose: Training Sanity Verification for BEVFormer-Small (Forward, Loss, Backward, Step)
# Repo: BEVFormer
# Branch: master
# Config: projects/configs/bevformer/bevformer_small.py
# Dataset: nuScenes v1.0-mini
# Expected output: outputs/checkpoints/bevformer_small_sanity.pth
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}"

echo "Starting BEVFormer-Small Training Sanity Verification..."
${PYTHON_BIN} -u ${PROJECT_ROOT}/scripts/train/sanity_bevformer_small.py
