#!/usr/bin/env bash
# ==============================================================================
# Script: run_sparse4d_r50.sh
# Purpose: Real Inference and 3D Anchor Refinement Visualization for Sparse4Dv3-R50
# Repo: Sparse4D
# Branch: main (v3.0)
# Config: projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
# Checkpoint: /public/lyh/pretrained_model/BEV/sparse4d/sparse4dv3_r50.pth
# Dataset: nuScenes v1.0-mini
# Expected output: outputs/sparse4d_real_output/
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}"

echo "Starting Sparse4Dv3-R50 Real Inference..."
${PYTHON_BIN} -u ${PROJECT_ROOT}/scripts/inference/test_sparse4d_real_inference.py
