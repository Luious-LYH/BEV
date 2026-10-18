#!/usr/bin/env bash
# ==============================================================================
# run_inference.sh — 自动驾驶 BEV/Occupancy 核心模型真实推理统一入口
# ==============================================================================
# 用法:
#   bash scripts/inference/run_inference.sh [MODEL] [GPU_ID]
#
# 支持的 MODEL:
#   bevdet      - BEVDet-R50 真实推理 (29.2 FPS, 官方权重, 3D 检测框)
#   bevdepth    - BEVDepth-R50 7 阶段特征流全流程截获与深度概率分布提取
#   bevformer   - BEVFormer-Small 真实推理 (SCA 立柱参考点投影 + BEV Query)
#   sparse4d    - Sparse4Dv3-R50 900 稀疏 3D 锚框与多尺度特征提取
#   bevdet_occ  - BEVDet-Occ 稠密 3D 占用栅格 (200x200x16) 推理与切片
#   sparseocc   - SparseOcc v1.1 稀疏表面体素掩码迭代推理 (32,000 活跃体素)
#   all         - 依序执行全部 6 个模型推理
#
# 示例:
#   bash scripts/inference/run_inference.sh bevdepth 0
#   bash scripts/inference/run_inference.sh all 0
# ==============================================================================

set -e

MODEL=${1:-"all"}
GPU_ID=${2:-0}

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

run_single_model() {
    local target=$1
    echo "=========================================================================="
    echo "  🚀 [INFERENCE] 正在启动模型真实推理: ${target} (GPU: ${GPU_ID})"
    echo "=========================================================================="
    
    case "${target}" in
        bevdet)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_bevdet_real_inference.py
            ;;
        bevdepth)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_bevdepth_real_intermediates.py
            ;;
        bevformer)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_bevformer_real_inference.py
            ;;
        sparse4d)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_sparse4d_real_inference.py
            ;;
        bevdet_occ)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_bevdet_occ_real_inference.py
            ;;
        sparseocc)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/SparseOcc:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/inference/test_sparseocc_real_inference.py
            ;;
        *)
            echo "❌ 未知模型: ${target}。可选: [bevdet | bevdepth | bevformer | sparse4d | bevdet_occ | sparseocc | all]"
            exit 1
            ;;
    esac
    echo "✓ 模型 ${target} 推理完成！产物已保存至 outputs/ 或 artifacts/。"
    echo ""
}

if [ "${MODEL}" == "all" ]; then
    run_single_model "bevdet"
    run_single_model "bevdepth"
    run_single_model "bevformer"
    run_single_model "sparse4d"
    run_single_model "bevdet_occ"
    run_single_model "sparseocc"
    echo "🎉 全部 6 个模型推理测试均已顺利完成！"
else
    run_single_model "${MODEL}"
fi
