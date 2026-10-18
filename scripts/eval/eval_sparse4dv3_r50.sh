#!/usr/bin/env bash
# ==============================================================================
# eval_sparse4dv3_r50.sh — Sparse4Dv3-R50 Evaluation
# ==============================================================================
set -e

GPU_ID=${1:-6}
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}"

echo "=========================================================================="
echo "  📊 [EVAL] Evaluating Sparse4Dv3-R50 on Physical GPU ${GPU_ID}"
echo "=========================================================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_sparse4dv3_r50.py
