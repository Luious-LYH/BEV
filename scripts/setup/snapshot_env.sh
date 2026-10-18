#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/miniconda3/bin:$PATH"

TAG="${1:-snapshot}"
PROJECT_ROOT="/public/lyh/projects/BEV"
OUT="${PROJECT_ROOT}/logs/env"
mkdir -p "$OUT"

TS="$(date +%Y%m%d_%H%M%S)"
PREFIX="${OUT}/${TS}_${TAG}"

echo "Saving environment snapshot to ${PREFIX}_*"

{
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "CUDA_HOME=${CUDA_HOME:-}"
  echo "PATH=$PATH"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
} > "${PREFIX}_shell_env.txt"

{
  python -V
  which python
  python - <<'PY'
import sys
print("sys.executable:", sys.executable)
print("sys.prefix:", sys.prefix)
print("sys.path:")
for p in sys.path:
    print(" ", p)
PY
} > "${PREFIX}_python.txt" 2>&1 || true

python - <<'PY' > "${PREFIX}_torch.txt" 2>&1 || true
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda runtime:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("visible gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
except Exception as e:
    print("torch inspection failed:", repr(e))
PY

python - <<'PY' > "${PREFIX}_openmmlab.txt" 2>&1 || true
for name in ["numpy", "mmcv", "mmengine", "mmdet", "mmseg", "mmdet3d"]:
    try:
        m = __import__(name)
        print(name, getattr(m, "__version__", "NA"), getattr(m, "__file__", "NA"))
    except Exception as e:
        print(name, "IMPORT_FAILED", repr(e))
PY

pip freeze > "${PREFIX}_pip_freeze.txt" 2>&1 || true
conda list > "${PREFIX}_conda_list.txt" 2>&1 || true
conda env export --from-history > "${PREFIX}_conda_history.yml" 2>&1 || true
conda info > "${PREFIX}_conda_info.txt" 2>&1 || true
nvidia-smi > "${PREFIX}_nvidia_smi.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv \
  > "${PREFIX}_gpu_query.csv" 2>&1 || true
which nvcc > "${PREFIX}_nvcc.txt" 2>&1 || true
nvcc --version >> "${PREFIX}_nvcc.txt" 2>&1 || true
ls -ld /usr/local/cuda* > "${PREFIX}_cuda_dirs.txt" 2>&1 || true
gcc --version > "${PREFIX}_gcc.txt" 2>&1 || true
g++ --version > "${PREFIX}_gxx.txt" 2>&1 || true

echo "Done."
