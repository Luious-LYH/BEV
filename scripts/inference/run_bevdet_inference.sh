#!/bin/bash
# ==============================================================================
# Script: run_bevdet_inference.sh
# Purpose: Run real BEVDet-R50 inference on nuScenes mini
# ==============================================================================
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="/public/lyh/projects/BEV/third_party/BEVDet:${PYTHONPATH}"

/public/lyh/.conda/envs/bev/bin/python -u /public/lyh/projects/BEV/scripts/inference/test_bevdet_real_inference.py
