#!/usr/bin/env bash
# ==============================================================================
# Script: eval_bevdet_r50.sh
# Purpose: Official Evaluation of BEVDet-R50 on nuScenes mini
# Repo: BEVDet
# Branch: dev3.0
# Commit: Official Latest
# Config: configs/bevdet/bevdet-r50-cbgs.py
# Checkpoint: /public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-cbgs.pth
# Dataset: nuScenes v1.0-mini (val split, 81 samples)
# Expected output: mAP, NDS, and per-class AP breakdown
# ==============================================================================

set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"

echo "Starting BEVDet-R50 Official Evaluation..."
${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_bevdet_r50.py
