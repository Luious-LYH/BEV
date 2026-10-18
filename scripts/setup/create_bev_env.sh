#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/miniconda3/bin:$PATH"

ENV_NAME="bev"
PROJECT_ROOT="/public/lyh/projects/BEV"
ENV_FILE="${PROJECT_ROOT}/configs/bev_base_env.yml"

echo "=== BEV environment bootstrap ==="
echo "Target: ${ENV_NAME}"
echo "Environment file: ${ENV_FILE}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env '${ENV_NAME}' already exists. Not recreating it."
else
  echo "[INFO] Creating clean '${ENV_NAME}' environment..."
  conda env create -f "${ENV_FILE}"
fi

echo
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo

# Verify through conda run so activation isn't required in this process.
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda runtime:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch check failed:", repr(e))
PY

cat <<'EOF'

IMPORTANT:
1) Do NOT modify CV / polyp / SSL / ubench for this project.
2) nvidia-smi may show CUDA 12.2; target PyTorch runtime remains CUDA 11.8.
3) Before compiling custom CUDA ops, inspect:
     nvcc --version
     ls -ld /usr/local/cuda*
   and prefer a CUDA 11.8 compiler/toolkit.
4) Do NOT blindly pip install a repo requirements.txt if it pins Torch/CUDA.
5) OpenMMLab packages are intentionally NOT fully pinned/installed here;
   resolve them in S0.7 after reading each repo's requirements.
EOF
