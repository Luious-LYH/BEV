#!/usr/bin/env bash
# ==============================================================================
# Script: eval_our_subset_bevdet_r50.sh
# Purpose: Evaluation of Our Trained BEVDet-R50 Checkpoint on nuScenes mini
# Repo: BEVDet
# Branch: dev3.0
# Config: configs/bevdet/bevdet-r50-cbgs.py
# Checkpoint: /public/lyh/projects/BEV/outputs/checkpoints/bevdet_real_best.pth
# Dataset: nuScenes v1.0-mini (val split, 81 keyframes)
# Expected output: outputs/eval/bevdet_r50_our_real_eval.json
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting Evaluation of Our Trained BEVDet-R50 Subset Model..."
${PYTHON_BIN} -u ${PROJECT_ROOT}/scripts/eval/eval_our_subset_bevdet_r50.py
