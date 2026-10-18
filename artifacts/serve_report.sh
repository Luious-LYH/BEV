#!/bin/bash
# ==============================================================================
# One-Click Local Web Server for BEV / Occupancy Perception Interactive Textbook
# ==============================================================================

PORT=${1:-8088}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=================================================================="
echo "启动自动驾驶感知教材本地静态服务器..."
echo "根目录: $DIR"
echo "监听端口: $PORT"
echo "本地访问 URL: http://127.0.0.1:$PORT/BEV_OCC_Learning_Report.html"
echo "=================================================================="

# Use python from bev conda env or system python
if [ -f "/public/lyh/.conda/envs/bev/bin/python" ]; then
    /public/lyh/.conda/envs/bev/bin/python -m http.server $PORT --directory "$DIR"
else
    python3 -m http.server $PORT --directory "$DIR"
fi
