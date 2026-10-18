# 环境配置与编译指南 (Environment & Installation Guide)

> **面向新服务器部署与二次开发的 Python / CUDA / OpenMMLab 依赖配置手册。**  
> 详细记录经过实测验证的软件版本矩阵、安装步骤、CUDA 扩展算子编译以及常见兼容性问题的解决方案。

---

## 1. 软件环境推荐矩阵 (Verified Stack)

为保证 BEVDet (dev3.0)、BEVFormer、Sparse4D (v3) 与 SparseOcc 能够和谐共存于**同一个 Conda 环境**中，请严格遵守以下经过多轮验证的版本组合：

| 核心组件 | 推荐版本 | 说明 |
|---|---|---|
| **操作系统** | Ubuntu 20.04 / 22.04 LTS | Linux x86_64 |
| **Python** | 3.8.18 ~ 3.8.20 | 兼容 OpenMMLab 1.x 规范 |
| **NVIDIA 驱动** | >= 525.60.13 | 支持 CUDA 11.8 / 12.x |
| **PyTorch** | 2.0.1+cu118 | 与 TorchVision 0.15.2+cu118 配套 |
| **CUDA Toolkit** | CUDA 11.8 (推荐) 或 12.0+ | 用于编译 C++/CUDA 扩展 |
| **MMCV-full** | 1.7.2 | OpenMMLab 统一底层计算基座 |
| **MMDetection** | 2.28.2 | 基础目标检测框架 |
| **MMSegmentation** | 0.30.0 | 语义分割基座 (OccNet / 占据网格依赖) |
| **NumPy** | 1.23.5 | 严格控制 < 1.24 以避免 Deprecation 错误 |

---

## 2. 一键快速配置脚本

如果你使用的是标准的 Linux 服务器，可以直接运行项目内置的环境创建脚本：

```bash
# 1. 运行一键环境搭建脚本
bash scripts/setup/create_bev_env.sh

# 2. 激活新建的 conda 环境
conda activate bev

# 3. 编译 Sparse4D 核心算子
bash scripts/setup/compile_sparse4dv3_ops.sh
```

---

## 3. 手动分步安装命令

如需手动配置或调试具体依赖包，可按以下标准步骤逐行执行：

### 步骤 1：创建并激活 Conda 虚拟环境
```bash
conda create -n bev python=3.8 -y
conda activate bev
```

### 步骤 2：安装 PyTorch 与对应 CUDA 运行时
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### 步骤 3：安装 MMCV-full 预编译包
```bash
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
```

### 步骤 4：安装 MMDetection 与 MMSegmentation
```bash
pip install mmdet==2.28.2 mmsegmentation==0.30.0
```

### 步骤 5：安装自动驾驶与科学计算依赖
```bash
pip install nuscenes-devkit timm scipy pyquaternion plyfile \
    open3d imageio imageio-ffmpeg scikit-image matplotlib pandas pyyaml
```

---

## 4. CUDA 核心扩展算子编译 (Sparse4D Deformable Aggregation)

Sparse4Dv3 包含自定义的多视角可变形聚合 CUDA 算子 (`deformable_aggregation_ext`)，需在目标机器上通过系统 `nvcc` 编译：

```bash
cd third_party/Sparse4D/projects/mmdet3d_plugin/ops
python setup.py build_ext --inplace
cd ../../../../../
```

测试算子是否编译并导入成功：
```bash
python -c "
import sys
sys.path.insert(0, 'third_party/Sparse4D/projects/mmdet3d_plugin/ops')
import deformable_aggregation_ext
print('✓ Sparse4D deformable_aggregation_ext compiled and imported successfully!')
"
```

---

## 5. 项目关键兼容性补丁 (Patches) 说明

在多模型联合运行时，不同上游开源库常存在命名冲突与版本检查过严的问题。本项目已直接在 `third_party/` 中预置了以下关键修复（同时在 `patches/` 目录保留了完整 diff）：

1. **MMCV 1.7.2 版本上限解除** (`patches/bevdet.patch`):
   - BEVDet 原仓库限定 `mmcv <= 1.7.0`，修改为 `<= 1.8.0` 以完美支持 PyTorch 2.0 时代的 MMCV 1.7.2。
2. **HungarianAssigner3D 重复注册修复** (`patches/bevformer.patch`):
   - BEVFormer 与 BEVDet 同时引入时存在注册名冲突，添加 `force=True` 与安全引入保护，使得两模型可在同一进程或同一测试套件下无缝共存。
3. **SparseOcc 无 CUDA 扩展时的纯 PyTorch 回退** (`patches/sparseocc.patch`):
   - 在未配置特定 CUDA 编译环境的轻量级开发机或 CPU 调试机上，提供 `msmv_sampling_pytorch` 纯 PyTorch 算子回退机制，避免硬崩溃。
