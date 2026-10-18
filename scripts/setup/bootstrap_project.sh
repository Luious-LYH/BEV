#!/usr/bin/env bash
# ==============================================================================
# Multi-Camera BEV & Occupancy Perception Project Bootstrap Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/public/lyh/data/BEV_dataset}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-/public/lyh/pretrained_model}"
BEV_PRETRAINED_ROOT="${BEV_PRETRAINED_ROOT:-${PRETRAINED_ROOT}/BEV}"

echo "=================================================================="
echo "🚀 Bootstrapping BEV & Occupancy Perception Project"
echo "Project Root:      $PROJECT_ROOT"
echo "Dataset Root:      $DATA_ROOT"
echo "Pretrained Root:   $PRETRAINED_ROOT"
echo "=================================================================="

echo "[1/8] Verifying resource paths..."
for p in "$PROJECT_ROOT" "$DATA_ROOT" "$PRETRAINED_ROOT"; do
  if [[ -e "$p" ]]; then
    echo "  ✓ OK: $p"
  else
    echo "  ⚠️ MISSING / OPTIONAL: $p"
  fi
done

echo "[2/8] Creating standard project directories..."
mkdir -p \
  "$PROJECT_ROOT/docs" \
  "$PROJECT_ROOT/configs" \
  "$PROJECT_ROOT/scripts/data" \
  "$PROJECT_ROOT/scripts/train" \
  "$PROJECT_ROOT/scripts/eval" \
  "$PROJECT_ROOT/scripts/visualization" \
  "$PROJECT_ROOT/third_party" \
  "$PROJECT_ROOT/patches" \
  "$PROJECT_ROOT/outputs/checkpoints" \
  "$PROJECT_ROOT/outputs/experiments" \
  "$PROJECT_ROOT/artifacts/notebooks" \
  "$PROJECT_ROOT/cache" \
  "$PROJECT_ROOT/logs"

echo "[3/8] Creating safe symlinks for data and pretrained weights..."
safe_link () {
  local target="$1"
  local link="$2"
  if [[ -L "$link" ]]; then
    echo "  EXISTS (Symlink): $link -> $(readlink "$link")"
  elif [[ -d "$link" && ! -L "$link" ]]; then
    echo "  EXISTS (Directory): $link"
  elif [[ -e "$target" ]]; then
    ln -s "$target" "$link"
    echo "  CREATED: $link -> $target"
  else
    echo "  SKIP target missing: $target (Create manually or update DATA_ROOT)"
  fi
}
safe_link "$DATA_ROOT" "$PROJECT_ROOT/data"
safe_link "$BEV_PRETRAINED_ROOT" "$PROJECT_ROOT/pretrained"

echo "[4/8] Checking NVIDIA GPU and Driver..."
nvidia-smi || true

echo "[5/8] Checking CUDA compiler (nvcc)..."
which nvcc || true
nvcc --version 2>/dev/null || true

echo "[6/8] Checking Python and Conda environment..."
if command -v conda &> /dev/null; then
  if conda env list | awk '{print $1}' | grep -qx "bev"; then
    echo "  ✓ Conda environment 'bev' exists."
    conda run -n bev python - <<'PY' || true
import sys
print("  Python:", sys.version.split()[0])
try:
    import torch
    print("  PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| Available:", torch.cuda.is_available())
except Exception as e:
    print("  PyTorch unavailable:", repr(e))
PY
  else
    echo "  ⚠️ 'bev' conda environment not found. Run: bash scripts/setup/create_bev_env.sh"
  fi
else
  echo "  ⚠️ conda command not found on PATH."
fi

echo "[7/8] Checking dataset layout..."
if [[ -d "$PROJECT_ROOT/data/nuscenes" ]]; then
  echo "  ✓ nuScenes directory detected in $PROJECT_ROOT/data/nuscenes"
else
  echo "  ⚠️ Dataset not mounted in $PROJECT_ROOT/data/nuscenes. Please check docs/dataset.md"
fi

echo "[8/8] Checking Sparse4D CUDA operators..."
if python -c "import sys; sys.path.insert(0, '$PROJECT_ROOT/third_party/Sparse4D/projects/mmdet3d_plugin/ops'); import deformable_aggregation_ext" 2>/dev/null; then
  echo "  ✓ Sparse4D deformable_aggregation_ext is compiled and functional."
else
  echo "  ⚠️ Sparse4D deformable_aggregation_ext not compiled. Run: bash scripts/setup/compile_sparse4dv3_ops.sh"
fi

cat <<'EOF'
==================================================================
🎉 Bootstrap check completed!
Next Steps:
  1. Activate environment:     conda activate bev
  2. Dataset documentation:    docs/dataset.md
  3. Pretrained models:        docs/models.md
  4. Run smoke/sanity test:    bash scripts/train/sanity_bevdepth_r50.sh
  5. Launch learning report:   bash artifacts/serve_report.sh 8088
==================================================================
EOF
