#!/usr/bin/env bash
# ==============================================================================
# Script: eval_bevdepth_r50.sh
# Purpose: Evaluation of BEVDepth-R50 on nuScenes mini
# Repo: BEVDet dev3.0
# Config: configs/bevdet/bevdet-r50-4d-depth-cbgs.py
# Checkpoint: outputs/checkpoints/bevdepth_real_best.pth
# Dataset: nuScenes v1.0-mini (val split, 81 samples)
# Expected output: outputs/eval/bevdepth_r50_mini_eval.json
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting BEVDepth-R50 Evaluation..."
${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_bevdepth_r50.py
