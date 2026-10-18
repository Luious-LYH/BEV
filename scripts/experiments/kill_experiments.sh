#!/usr/bin/env bash
# ==============================================================================
# kill_experiments.sh — 安全停止 GPU 4~7 上运行的后台训练任务
# ==============================================================================

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PID_FILE="${PROJECT_ROOT}/outputs/experiments/running_pids.txt"

echo "=========================================================================="
echo "🛑 正在停止 GPU 4~7 上的后台训练任务..."
echo "=========================================================================="

if [ -f "${PID_FILE}" ]; then
    while read -r line; do
        pid=$(echo "$line" | awk '{print $2}')
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
            echo "正在终止 PID $pid ($line)..."
            kill -15 "$pid" 2>/dev/null || true
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            echo "✓ PID $pid 已终止"
        fi
    done < "${PID_FILE}"
    rm -f "${PID_FILE}"
else
    echo "未发现 ${PID_FILE}，尝试按脚本名过滤..."
fi

# Secondary check for matching training script processes
pkill -f "train_bevdet_bevdepth_controlled.py" 2>/dev/null || true
pkill -f "train_bevformer_real.py" 2>/dev/null || true
pkill -f "train_sparse4d_real.py" 2>/dev/null || true
pkill -f "train_occupancy_real.py" 2>/dev/null || true

echo "✓ 全部相关训练任务已停止！"
