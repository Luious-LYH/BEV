#!/usr/bin/env bash
# ==============================================================================
# Script: run_bevdet_r50.sh
# Purpose: Real Inference for BEVDet-R50 with Official Checkpoint
# Repo: BEVDet
# Branch: dev3.0
# Commit: Official Latest
# Config: configs/bevdet/bevdet-r50-cbgs.py
# Checkpoint: /public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-cbgs.pth
# Dataset: nuScenes v1.0-mini (data/nuscenes/)
# Expected output: 3D bounding box predictions, detected objects, latency & visualization
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting BEVDet-R50 Real Inference..."
${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_bevdet_real_inference.py
