#!/bin/bash
set -e

ENV_DIR="/public/lyh/.conda/envs/bev"
WHEELS_DIR="/public/lyh/projects/BEV/cache/wheels"

echo "=========================================================="
echo "Installing Core Deep Learning Stack into $ENV_DIR"
echo "=========================================================="

# 1. Install PyTorch, TorchVision, TorchAudio
echo "[Step 1] Installing PyTorch 2.0.1+cu118..."
$ENV_DIR/bin/pip install \
    $WHEELS_DIR/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl \
    $WHEELS_DIR/torchvision-0.15.2+cu118-cp38-cp38-linux_x86_64.whl \
    $WHEELS_DIR/torchaudio-2.0.2+cu118-cp38-cp38-linux_x86_64.whl --no-deps

# 2. Install MMCV-Full 1.7.2
echo "[Step 2] Installing MMCV-Full 1.7.2 (cu118)..."
$ENV_DIR/bin/pip install $WHEELS_DIR/mmcv_full-1.7.2-cp38-cp38-manylinux1_x86_64.whl --no-deps

# 3. Install MMDetection and MMSegmentation
echo "[Step 3] Installing MMDetection 2.28.2 & MMSegmentation 0.30.0..."
$ENV_DIR/bin/pip install \
    $WHEELS_DIR/mmdet-2.28.2-py3-none-any.whl \
    $WHEELS_DIR/mmsegmentation-0.30.0-py3-none-any.whl --no-deps

# 4. Install nuScenes DevKit and scientific tools
echo "[Step 4] Installing nuScenes-devkit & tooling..."
$ENV_DIR/bin/pip install \
    $WHEELS_DIR/nuscenes_devkit-1.1.11-py3-none-any.whl \
    $WHEELS_DIR/pyquaternion-0.9.9-py3-none-any.whl \
    $WHEELS_DIR/ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
    $WHEELS_DIR/terminaltables-3.1.10-py2.py3-none-any.whl \
    $WHEELS_DIR/motmetrics-1.4.0-py3-none-any.whl --no-deps

# Install any essential runtime dependencies for mmdet/nuscenes
echo "[Step 5] Installing essential runtime dependencies..."
$ENV_DIR/bin/pip install scipy pyyaml opencv-python tqdm addict yapf trimesh shapely --no-deps --proxy socks5h://127.0.0.1:26001 || true

# 5. Verification
echo "[Step 6] Running Verification..."
$ENV_DIR/bin/python -c "
import torch
print('=== PyTorch Verification ===')
print('PyTorch Version:', torch.__version__)
print('CUDA Runtime Version:', torch.version.cuda)
print('CUDA Available:', torch.cuda.is_available())
print('Device Count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Testing GPU 4 allocation...')
    x = torch.randn(1000, 1000, device='cuda:4')
    print('GPU 4 Tensor Norm:', x.norm().item())

import mmcv
print('=== MMCV Verification ===')
print('MMCV Version:', mmcv.__version__)

import mmdet
print('=== MMDetection Verification ===')
print('MMDet Version:', mmdet.__version__)

import nuscenes
print('=== nuScenes Verification ===')
print('nuScenes-devkit imported successfully!')
"

echo "=========================================================="
echo "Core Stack Installation Completed Successfully!"
echo "=========================================================="
