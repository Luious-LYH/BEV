#!/usr/bin/env bash
# ==============================================================================
# monitor_experiments.sh — 实时监控 GPU 4~7 实验运行状态、显存与损失曲线
# ==============================================================================

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
LOG_DIR="${PROJECT_ROOT}/outputs/experiments/logs"

clear
echo "=========================================================================="
echo "📊 自动驾驶 BEV & Occupancy 并发长周期实验实时监控看板"
echo "更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="

echo ""
echo "--- [1] 物理 GPU 4, 5, 6, 7 显存与核心负载 ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -i 4,5,6,7

echo ""
echo "--- [2] 活跃后台进程状态 ---"
if [ -f "${PROJECT_ROOT}/outputs/experiments/running_pids.txt" ]; then
    cat "${PROJECT_ROOT}/outputs/experiments/running_pids.txt" | while read line; do
        pid=$(echo $line | awk '{print $2}')
        name=$(echo $line | awk '{print $3}')
        if ps -p $pid > /dev/null 2>&1; then
            stat="🟢 RUNNING (CPU: $(ps -p $pid -o %cpu --no-headers)%, Mem: $(ps -p $pid -o %mem --no-headers)%)"
        else
            stat="⚪ FINISHED / INACTIVE"
        fi
        echo "  $line => $stat"
    done
else
    echo "  (未找到 running_pids.txt 记录)"
fi

echo ""
echo "--- [3] 各模型训练进度与最新损失追踪 (Tail) ---"

echo ""
echo "▶️ [GPU 4: BEVDet / BEVDepth 受控实验]"
if [ -f "${LOG_DIR}/gpu4_controlled.log" ]; then
    tail -n 4 "${LOG_DIR}/gpu4_controlled.log"
else
    echo "  暂无日志"
fi

echo ""
echo "▶️ [GPU 5: BEVFormer-Small 真实多轮]"
if [ -f "${LOG_DIR}/gpu5_bevformer.log" ]; then
    tail -n 4 "${LOG_DIR}/gpu5_bevformer.log"
else
    echo "  暂无日志"
fi

echo ""
echo "▶️ [GPU 6: Sparse4Dv3-R50 真实多轮]"
if [ -f "${LOG_DIR}/gpu6_sparse4d.log" ]; then
    tail -n 4 "${LOG_DIR}/gpu6_sparse4d.log"
else
    echo "  暂无日志"
fi

echo ""
echo "▶️ [GPU 7: BEVDet-Occ 真实 Occ3D]"
if [ -f "${LOG_DIR}/gpu7_occupancy.log" ]; then
    tail -n 4 "${LOG_DIR}/gpu7_occupancy.log"
else
    echo "  暂无日志"
fi

echo ""
echo "=========================================================================="
echo "提示: 可使用 'watch -n 5 bash scripts/experiments/monitor_experiments.sh' 持续刷新"
echo "=========================================================================="
