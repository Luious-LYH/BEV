#!/usr/bin/env bash
# ==============================================================================
# Script: run_bevformer_small.sh
# Purpose: Real Inference and Visualization for BEVFormer-Small
# Repo: BEVFormer
# Branch: master
# Config: projects/configs/bevformer/bevformer_small.py
# Checkpoint: /public/lyh/pretrained_model/BEV/bevformer/bevformer_small_epoch_24.pth
# Dataset: nuScenes v1.0-mini
# Expected output: outputs/bevformer_real_output/bevformer_real_predictions.png
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}"

echo "Starting BEVFormer-Small Real Inference..."
${PYTHON_BIN} -u ${PROJECT_ROOT}/scripts/inference/test_bevformer_real_inference.py
