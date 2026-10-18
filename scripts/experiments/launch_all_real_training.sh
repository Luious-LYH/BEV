#!/usr/bin/env bash
# ==============================================================================
# launch_all_real_training.sh — 高效并发高占用真实实验总调度器 (GPU 4, 5, 6, 7)
# ==============================================================================
# 架构设计:
#   GPU 4: [EXP-005/006] BEVDet vs BEVDepth 受控对照训练 (12 Epochs, CBGS 815 batches/epoch, 4 workers)
#   GPU 5: [EXP-001]     BEVFormer-Small 真实多轮训练 (10 Epochs, 全 323 样本/epoch, Cosine LR)
#   GPU 6: [EXP-002]     Sparse4Dv3-R50 真实多轮训练 (12 Epochs, 全 323 样本/epoch, 6-Layer Refine)
#   GPU 7: [EXP-003]     BEVDet-Occ 真实 Occ3D 语义监督训练 (8 Epochs, 全样本 18-Class Masked CE)
# ==============================================================================

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/outputs/experiments/logs"
mkdir -p "${LOG_DIR}"

PID_FILE="${PROJECT_ROOT}/outputs/experiments/running_pids.txt"
> "${PID_FILE}"

echo "=========================================================================="
echo "🚀 启动自动驾驶 BEV & Occupancy 高效并发高占用真实实验 (GPU 4 ~ 7)"
echo "=========================================================================="
echo "Python 解释器: ${PYTHON_BIN}"
echo "工作空间:      ${PROJECT_ROOT}"
echo "日志目录:      ${LOG_DIR}"
echo ""

# 1. 检查物理显卡状态
echo "🔍 检查 GPU 4, 5, 6, 7 显卡资源..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -i 4,5,6,7
echo ""

# 2. 启动 GPU 4: BEVDet & BEVDepth 受控实验
echo "▶️ [GPU 4] 启动 BEVDet vs BEVDepth 受控对照实验 (12 Epochs, CBGS)..."
CUDA_VISIBLE_DEVICES=4 PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}" \
    nohup ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_bevdet_bevdepth_controlled.py \
    --model both --epochs 12 --batch-size 2 --workers 4 > "${LOG_DIR}/gpu4_controlled.log" 2>&1 &
PID_GPU4=$!
echo "GPU 4 PID: ${PID_GPU4} (Log: ${LOG_DIR}/gpu4_controlled.log)"
echo "GPU4: ${PID_GPU4} (BEVDet-BEVDepth-Controlled)" >> "${PID_FILE}"

# 3. 启动 GPU 5: BEVFormer-Small
echo "▶️ [GPU 5] 启动 BEVFormer-Small 真实多轮训练 (10 Epochs, 全样本)..."
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="${PROJECT_ROOT}/third_party/BEVFormer:${PYTHONPATH}" \
    nohup ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_bevformer_real.py \
    --epochs 10 --lr 2e-5 > "${LOG_DIR}/gpu5_bevformer.log" 2>&1 &
PID_GPU5=$!
echo "GPU 5 PID: ${PID_GPU5} (Log: ${LOG_DIR}/gpu5_bevformer.log)"
echo "GPU5: ${PID_GPU5} (BEVFormer-Small-Real)" >> "${PID_FILE}"

# 4. 启动 GPU 6: Sparse4Dv3-R50
echo "▶️ [GPU 6] 启动 Sparse4Dv3-R50 真实多轮训练 (12 Epochs, 全样本)..."
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="${PROJECT_ROOT}/third_party/Sparse4D:${PYTHONPATH}" \
    nohup ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_sparse4d_real.py \
    --epochs 12 --lr 2e-5 > "${LOG_DIR}/gpu6_sparse4d.log" 2>&1 &
PID_GPU6=$!
echo "GPU 6 PID: ${PID_GPU6} (Log: ${LOG_DIR}/gpu6_sparse4d.log)"
echo "GPU6: ${PID_GPU6} (Sparse4Dv3-R50-Real)" >> "${PID_FILE}"

# 5. 启动 GPU 7: BEVDet-Occ
echo "▶️ [GPU 7] 启动 BEVDet-Occ 真实 Occ3D 语义监督训练 (8 Epochs, 全样本)..."
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="${PROJECT_ROOT}/third_party/BEVDet:${PYTHONPATH}" \
    nohup ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/experiments/train_occupancy_real.py \
    --model bevdet_occ --epochs 8 --lr 2e-5 > "${LOG_DIR}/gpu7_occupancy.log" 2>&1 &
PID_GPU7=$!
echo "GPU 7 PID: ${PID_GPU7} (Log: ${LOG_DIR}/gpu7_occupancy.log)"
echo "GPU7: ${PID_GPU7} (BEVDet-Occ-Real)" >> "${PID_FILE}"

echo ""
echo "=========================================================================="
echo "🎉 4 个后台长周期实验已成功部署并开始高占用并发运行！"
echo "=========================================================================="
echo "实时监控脚本: bash scripts/experiments/monitor_experiments.sh"
echo "停止任务脚本: bash scripts/experiments/kill_experiments.sh"
echo "=========================================================================="
