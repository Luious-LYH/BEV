#!/usr/bin/env bash
# ==============================================================================
# Script: run_bevdet_occ.sh
# Purpose: Real Inference and Dense Occupancy Visualization for BEVDet-Occ
# Repo: BEVDet
# Branch: dev3.0
# Commit: Official Latest
# Config: configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py
# Checkpoint: /public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth
# Dataset: nuScenes v1.0-mini
# Expected output: Dense 3D occupancy voxel predictions, cross-section & BEV slice
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting BEVDet-Occ Real Inference..."
${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_bevdet_occ_real_inference.py
