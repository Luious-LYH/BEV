#!/usr/bin/env bash
# ==============================================================================
# sanity_sparseocc_v11.sh — SparseOcc-v1.1 Sanity Verification with Real Occ3D GT
# ==============================================================================
set -e

GPU_ID=${1:-7}
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/SparseOcc:${PYTHONPATH}"

echo "=========================================================================="
echo "  🚀 [EXP-004] Launching SparseOcc-v1.1 Sanity on Physical GPU ${GPU_ID}"
echo "=========================================================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_sparseocc_v11.py
