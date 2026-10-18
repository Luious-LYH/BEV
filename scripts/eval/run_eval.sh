#!/usr/bin/env bash
# ==============================================================================
# run_eval.sh — 自动驾驶 BEV & Occupancy 模型评测统一入口 (Academic Dispatcher)
# ==============================================================================
# 用法:
#   bash scripts/eval/run_eval.sh [TARGET] [GPU_ID]
#
# TARGET:
#   bevdet      - BEVDet-R50 nuScenes mini 官方评测 (mAP / NDS)
#   bevdepth    - BEVDepth-R50 7 阶段特征流与深度 GT 误差评测 (NDS 0.2563)
#   bevformer   - BEVFormer-Small (our model vs official 对比评测)
#   sparse4d    - Sparse4Dv3-R50 (our model vs official 对比评测 & 22.1 FPS 审计)
#   bevdet_occ  - BEVDet-Occ (真实 Occ3D GT 18 类 mIoU 评测, 44.71% mIoU)
#   sparseocc   - SparseOcc-v1.1 (真实 Occ3D GT 18 类 mIoU 评测)
#   our_model   - 评测自主多轮训练 BEVDet checkpoint (NDS 0.1858)
#   detection   - 评测 4 个 3D Detection 模型
#   occupancy   - 评测 2 个 3D Occupancy 模型
#   benchmark   - 运行 6 模型 GPU 硬件延迟与显存审计
#   all         - 执行全部评测项与审计
#
# GPU_ID:
#   物理 GPU 编号，默认: 4 (建议使用 4, 5, 6, 7)
#
# 示例:
#   bash scripts/eval/run_eval.sh bevformer 5
#   bash scripts/eval/run_eval.sh sparse4d 6
#   bash scripts/eval/run_eval.sh bevdet_occ 7
#   bash scripts/eval/run_eval.sh benchmark 4
# ==============================================================================

set -e

TARGET=${1:-"all"}
GPU_ID=${2:-4}

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

run_single_eval() {
    local t=$1
    echo "=========================================================================="
    echo "  📊 [EVAL] 正在执行评测: ${t} (物理 GPU: ${GPU_ID})"
    echo "=========================================================================="

    case "${t}" in
        bevdet)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_bevdet_r50.py
            ;;
        bevdepth)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_bevdepth_r50.py
            ;;
        bevformer)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_bevformer_small.py
            ;;
        sparse4d)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_sparse4dv3_r50.py
            ;;
        bevdet_occ)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_bevdet_occ.py
            ;;
        sparseocc)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/SparseOcc:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_sparseocc_v11.py
            ;;
        our_model|our_subset)
            export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/eval/eval_our_subset_bevdet_r50.py
            ;;
        benchmark)
            CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/audit_benchmarks_and_memory.py
            ;;
        *)
            echo "❌ 未知评测目标: ${t}。可选: [bevdet | bevdepth | bevformer | sparse4d | bevdet_occ | sparseocc | our_subset | benchmark | detection | occupancy | all]"
            exit 1
            ;;
    esac
    echo "✓ 评测 ${t} 完成！"
    echo ""
}

if [ "${TARGET}" == "all" ]; then
    run_single_eval "bevdet"
    run_single_eval "bevdepth"
    run_single_eval "bevformer"
    run_single_eval "sparse4d"
    run_single_eval "bevdet_occ"
    run_single_eval "sparseocc"
    run_single_eval "benchmark"
    echo "🎉 全部评测任务均已执行完成！"
elif [ "${TARGET}" == "detection" ]; then
    run_single_eval "bevdet"
    run_single_eval "bevdepth"
    run_single_eval "bevformer"
    run_single_eval "sparse4d"
    echo "🎉 4 个 Detection 模型评测任务执行完成！"
elif [ "${TARGET}" == "occupancy" ]; then
    run_single_eval "bevdet_occ"
    run_single_eval "sparseocc"
    echo "🎉 2 个 Occupancy 模型评测任务执行完成！"
else
    run_single_eval "${TARGET}"
fi
