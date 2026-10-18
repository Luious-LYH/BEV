#!/usr/bin/env bash
# ==============================================================================
# Script: run_sparseocc_v11.sh
# Purpose: Real Inference and Occupancy Visualization for SparseOcc v1.1
# Repo: SparseOcc
# Branch: main (v1.1)
# Config: configs/r50_nuimg_704x256_8f.py
# Checkpoint: /public/lyh/pretrained_model/BEV/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth
# Dataset: nuScenes v1.0-mini
# Expected output: outputs/sparseocc_real_output/sparseocc_real_dense_voxel.png
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/SparseOcc:${PYTHONPATH}"

echo "Starting SparseOcc v1.1 Real Inference..."
${PYTHON_BIN} -u ${PROJECT_ROOT}/scripts/inference/test_sparseocc_real_inference.py
