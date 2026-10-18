#!/usr/bin/env bash
set -e

PROJECT_ROOT="/public/lyh/projects/BEV"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

cd "${PROJECT_ROOT}/third_party/Sparse4D/projects/mmdet3d_plugin/ops"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "Compiling Sparse4D v3 deformable_aggregation_ext..."
${PYTHON_BIN} setup.py build_ext --inplace
echo "Compilation complete!"
