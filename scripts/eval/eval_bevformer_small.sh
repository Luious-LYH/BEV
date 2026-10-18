#!/usr/bin/env bash
# ==============================================================================
# eval_bevformer_small.sh — BEVFormer-Small Evaluation
# ==============================================================================
set -e

GPU_ID=${1:-5}
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}"

echo "=========================================================================="
echo "  📊 [EVAL] Evaluating BEVFormer-Small on Physical GPU ${GPU_ID}"
echo "=========================================================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_bevformer_small.py
