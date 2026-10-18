#!/usr/bin/env bash
# ==============================================================================
# run_visualizations.sh — 自动驾驶 BEV/Occupancy 教学可视化一键生成统一入口
# ==============================================================================
# 用法:
#   bash scripts/visualization/run_visualizations.sh [TARGET]
#
# TARGET:
#   geometry         - 运行环视相机与 3D 几何投影可视化 (visualize_geometry_and_cameras.py)
#   bev_depth        - 运行 BEV 变换、深度监督与时序对齐可视化 (visualize_bev_and_depth.py)
#   transformer_occ  - 运行 Transformer 注意力与 3D Occupancy 可视化 (visualize_transformer_and_occupancy.py)
#   nuscenes_real    - 运行真实 nuScenes 多模态数据教学可视化综合套件 (visualize_nuscenes_real.py)
#   all              - 运行全部教学绘图模块
#
# 示例:
#   bash scripts/visualization/run_visualizations.sh geometry
#   bash scripts/visualization/run_visualizations.sh all
# ==============================================================================

set -e

TARGET=${1:-"all"}

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
PYTHON_BIN="/public/lyh/.conda/envs/bev/bin/python"

if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

run_module() {
    local mod=$1
    echo "=========================================================================="
    echo "  🎨 [VIZ] 正在生成可视化模块: ${mod}"
    echo "=========================================================================="
    
    case "${mod}" in
        geometry)
            ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/visualization/visualize_geometry_and_cameras.py
            ;;
        bev_depth)
            ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/visualization/visualize_bev_and_depth.py
            ;;
        transformer_occ)
            ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/visualization/visualize_transformer_and_occupancy.py
            ;;
        nuscenes_real)
            ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/visualization/visualize_nuscenes_real.py
            ;;
        *)
            echo "❌ 未知可视化模块: ${mod}。可选: [geometry | bev_depth | transformer_occ | nuscenes_real | all]"
            exit 1
            ;;
    esac
    echo "✓ [VIZ ${mod}] 产物已同步写入 artifacts/learning_report/images/！"
    echo ""
}

if [ "${TARGET}" == "all" ]; then
    run_module "geometry"
    run_module "bev_depth"
    run_module "transformer_occ"
    run_module "nuscenes_real"
    echo "🎉 全部教学可视化图像均已成功生成！"
else
    run_module "${TARGET}"
fi
