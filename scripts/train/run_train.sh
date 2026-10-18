#!/usr/bin/env bash
# ==============================================================================
# run_train.sh — 自动驾驶 BEV & Occupancy 模型训练统一入口 (Academic Dispatcher)
# ==============================================================================
# 用法:
#   bash scripts/train/run_train.sh [STAGE] [MODEL] [GPU_ID]
#
# STAGE:
#   sanity     - 健全性验证 (3~5 steps 反向传播与梯度验证，确保真实 loss 下降与显存分配正常)
#   train      - 真实多轮全样本训练 (12/10/8 epochs, 保存 best checkpoint、损失曲线与 metrics.json)
#   real       - 等价于 train (向下兼容别名)
#
# MODEL:
#   bevdet     - BEVDet-R50
#   bevdepth   - BEVDepth-R50 (带 LiDAR 深度 GT 显式监督)
#   bevformer  - BEVFormer-Small (时空交叉注意力)
#   sparse4d   - Sparse4Dv3-R50 (稀疏时序实例注意力)
#   bevdet_occ - BEVDet-Occ (Dense 3D Voxel + Occ3D 真实语义监督)
#   sparseocc  - SparseOcc-v1.1 (Sparse Mask Query + Occ3D 真实语义监督)
#   detection  - 执行 4 个 3D 目标检测模型
#   occupancy  - 执行 2 个 3D Occupancy 模型
#   all        - 执行全部 6 个模型
#
# GPU_ID:
#   物理 GPU 编号，默认: 4 (建议使用 4, 5, 6, 7)
#
# 示例:
#   bash scripts/train/run_train.sh sanity bevformer 5
#   bash scripts/train/run_train.sh train sparse4d 6
#   bash scripts/train/run_train.sh train bevdet_occ 7
#   bash scripts/train/run_train.sh train all 4
# ==============================================================================

set -e

STAGE=${1:-"sanity"}
MODEL=${2:-"all"}
GPU_ID=${3:-4}

# 规范化阶段名：subset 映射为 train 模式
if [ "${STAGE}" == "subset" ] || [ "${STAGE}" == "real" ]; then
    STAGE="train"
fi

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

run_task() {
    local stage=$1
    local model=$2
    echo "=========================================================================="
    echo "  🏋️ [TRAIN] 运行阶段: ${stage} | 模型: ${model} (物理 GPU: ${GPU_ID})"
    echo "=========================================================================="

    if [ "${stage}" == "sanity" ]; then
        case "${model}" in
            bevdet)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_bevdet_r50.py
                ;;
            bevdepth)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_bevdepth_r50.py
                ;;
            bevformer)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_bevformer_small.py
                ;;
            sparse4d)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_sparse4d_r50.py
                ;;
            bevdet_occ)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_bevdet_occ.py
                ;;
            sparseocc)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/SparseOcc:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/train/sanity_sparseocc_v11.py
                ;;
            *)
                echo "❌ Sanity 阶段不支持模型: ${model}"
                exit 1
                ;;
        esac
    elif [ "${stage}" == "train" ]; then
        case "${model}" in
            bevdet)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_bevdet_bevdepth_controlled.py --model bevdet --epochs 12 --batch-size 2 --workers 4
                ;;
            bevdepth)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_bevdet_bevdepth_controlled.py --model bevdepth --epochs 12 --batch-size 2 --workers 4
                ;;
            controlled)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_bevdet_bevdepth_controlled.py --model both --epochs 12 --batch-size 2 --workers 4
                ;;
            bevformer)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_bevformer_real.py --epochs 10 --lr 2e-5
                ;;
            sparse4d)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_sparse4d_real.py --epochs 12 --lr 2e-5
                ;;
            bevdet_occ)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_occupancy_real.py --model bevdet_occ --epochs 8 --lr 2e-5
                ;;
            sparseocc)
                export PYTHONPATH="${PROJECT_ROOT}/third_party/SparseOcc:${PYTHONPATH}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_sparseocc_real.py --epochs 8 --lr 2e-5
                ;;
            *)
                echo "❌ Train 阶段不支持模型: ${model}"
                exit 1
                ;;
        esac
    else
        echo "❌ 未知阶段: ${stage}。请选择: [sanity | train]"
        exit 1
    fi
    echo "✓ [TRAIN ${stage} - ${model}] 完成！"
    echo ""
}

if [ "${STAGE}" == "train" ] && [ "${MODEL}" == "all" ]; then
    echo "=========================================================================="
    echo "🚀 启动 [TRAIN ALL] 模式: 调度 GPU 4, 5, 6, 7 进行高占用高效并发实验！"
    echo "=========================================================================="
    bash "${PROJECT_ROOT}/scripts/experiments/launch_all_real_training.sh"
elif [ "${MODEL}" == "all" ]; then
    run_task "${STAGE}" "bevdet"
    run_task "${STAGE}" "bevdepth"
    run_task "${STAGE}" "bevformer"
    run_task "${STAGE}" "sparse4d"
    run_task "${STAGE}" "bevdet_occ"
    run_task "${STAGE}" "sparseocc"
    echo "🎉 全部 6 个模型训练任务 (${STAGE}) 执行完成！"
elif [ "${MODEL}" == "detection" ]; then
    run_task "${STAGE}" "bevdet"
    run_task "${STAGE}" "bevdepth"
    run_task "${STAGE}" "bevformer"
    run_task "${STAGE}" "sparse4d"
    echo "🎉 4 个 Detection 模型训练任务 (${STAGE}) 执行完成！"
elif [ "${MODEL}" == "occupancy" ]; then
    run_task "${STAGE}" "bevdet_occ"
    run_task "${STAGE}" "sparseocc"
    echo "🎉 2 个 Occupancy 模型训练任务 (${STAGE}) 执行完成！"
else
    run_task "${STAGE}" "${MODEL}"
fi

