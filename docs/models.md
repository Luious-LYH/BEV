# 预训练权重与模型库指南 (Pretrained Models & Checkpoints Guide)

> **自动驾驶多视角 BEV & Occupancy 感知算法全景模型权重清单与管理指南。**  
> 详细列出项目支持的 6 个核心模型 (Tier 1) 与 4 个轻量教学扩展模型 (Tier 2) 所需的预训练骨干网络及官方 Checkpoints。

---

## 1. 模型分层体系与定位

本项目聚焦于四大主流自动驾驶纯视觉 3D 表征范式：

```text
nuScenes 6-Camera 环视图像输入
       │
       ├── 【Depth-based Dense BEV】 (显式几何投影)
       │      ├── BEVDet          ── Baseline：LSS 视锥构建与多相机投影
       │      ├── BEVDepth (主)   ── 核心：LiDAR 显式深度监督 + 深度分布预测 + Voxel Pooling
       │      ├── BEVDet4D (轻)   ── 扩展：自车运动补偿 (Ego-motion) 与时序特征叠加
       │      └── BEVStereo (轻)  ── 扩展：时序立体几何匹配与 Cost Volume 深度精细化
       │
       ├── 【Query-based Dense BEV】 (隐式注意力采样)
       │      ├── BEVFormer V1-small (主) ── 核心：BEV Query + Deformable Attention + 时序自注意力
       │      └── BEVFormer V2 (轻)       ── 演进：透视监督 (Perspective Supervision) 与优化策略
       │
       ├── 【Sparse Object-centric Perception】 (稀疏实例表征)
       │      └── Sparse4Dv3-R50 (主)     ── 核心：去密集网格化、3D 实例 Anchor 与 Instance Bank
       │
       └── 【3D Semantic Occupancy】 (全空间体素化表征)
              ├── BEVDet-Occ R50 (基线)   ── Dense OCC：Dense Volume 构建与 3D 卷积开销
              ├── SparseOcc v1.1 R50 (主) ── Sparse OCC：稀疏体素采样与 RayIoU 空间表征
              └── OccNet (轻)             ── 扩展：多层级架构设计与多样化表征思考
```

---

## 2. 官方预训练权重清单

官方 Checkpoint 用于快速执行推理 (Inference)、关键中间几何张量提取、初始化微调以及评估测试。

| 模型名称 | 角色分类 | 推荐权重文件名 | 文件体积 | 官方开源库来源 | 下载渠道与链接参考 |
|---|---|---|---|---|---|
| **ResNet-50 Backbone** | 通用特征提取底座 | `resnet50-0676ba61.pth` | ~98 MB | PyTorch Official | [PyTorch Hub Download](https://download.pytorch.org/models/resnet50-0676ba61.pth) |
| **BEVDet-R50** | Tier 1 (显式投影基线) | `bevdet-r50-cbgs.pth` | ~170 MB | `HuangJunJie2017/BEVDet` (dev3.0) | [BEVDet Releases / Google Drive](https://github.com/HuangJunJie2017/BEVDet) |
| **BEVDepth-R50** | Tier 1 (显式深度监督) | `bevdet-r50-4d-depth-cbgs.pth` | ~287 MB | `HuangJunJie2017/BEVDet` (dev3.0) | [BEVDet Releases / Google Drive](https://github.com/HuangJunJie2017/BEVDet) |
| **BEVDet4D-R50** | Tier 2 (时序自车对齐) | `bevdet-r50-4d-cbgs.pth` | ~201 MB | `HuangJunJie2017/BEVDet` (dev3.0) | [BEVDet Releases / Google Drive](https://github.com/HuangJunJie2017/BEVDet) |
| **BEVStereo-R50** | Tier 2 (时序双目深度) | `bevdet-r50-4d-stereo-cbgs.pth` | ~223 MB | `HuangJunJie2017/BEVDet` (dev3.0) | [BEVDet Releases / Google Drive](https://github.com/HuangJunJie2017/BEVDet) |
| **BEVDet-Occ-R50** | Tier 1 (Dense 占据网格) | `bevdet-occ-r50-4d-stereo_384704-24e.pth` | ~134 MB | `HuangJunJie2017/BEVDet` (dev3.0) | [BEVDet Releases / Google Drive](https://github.com/HuangJunJie2017/BEVDet) |
| **BEVFormer-Small** | Tier 1 (注意力空间查询) | `bevformer_small_epoch_24.pth` | ~680 MB | `fundamentalvision/BEVFormer` | [BEVFormer Releases](https://github.com/fundamentalvision/BEVFormer) |
| **Sparse4Dv3-R50** | Tier 1 (稀疏目标实例) | `sparse4dv3_r50.pth` | ~551 MB | `HorizonRobot/Sparse4D` (v3) | [Sparse4D Model Zoo](https://github.com/HorizonRobot/Sparse4D) |
| **SparseOcc-R50 Pretrain** | Tier 1 (NuImages 预训练底座) | `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim.pth` | ~295 MB | MMDetection / NuImages | [SparseOcc README 链接](https://github.com/MCG-NJU/SparseOcc) |
| **SparseOcc-v1.1-R50** | Tier 1 (稀疏体素采样占据) | `sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth` | ~307 MB | `MCG-NJU/SparseOcc` (v1.1) | [SparseOcc Model Zoo](https://github.com/MCG-NJU/SparseOcc) |

---

## 3. 标准存放路径与目录组织

项目通过 `pretrained/` 目录访问预训练权重（可软链接为 `ln -s /path/to/pretrained_model/BEV ./pretrained`）。

推荐的磁盘目录结构如下：

```text
BEV/
└── pretrained/
    ├── backbones/
    │   └── resnet50-0676ba61.pth
    ├── bevdet/
    │   ├── bevdet-r50-cbgs.pth
    │   ├── bevdet-r50-4d-cbgs.pth
    │   ├── bevdet-r50-4d-depth-cbgs.pth
    │   ├── bevdet-r50-4d-stereo-cbgs.pth
    │   └── bevdet-occ-r50-4d-stereo_384704-24e.pth
    ├── bevformer/
    │   └── bevformer_small_epoch_24.pth
    ├── sparse4d/
    │   └── sparse4dv3_r50.pth
    ├── sparseocc/
    │   ├── cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim.pth
    │   └── sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth
    └── occnet/
```

---

## 4. 本地训练自产 Checkpoint 说明

在执行子集训练 (`scripts/experiments/` 或 `scripts/train/run_train.sh`) 时，项目会在 `outputs/` 目录自动保存经过验证的最新最佳权重：

- `outputs/checkpoints/bevdepth_real_best.pth` (287 MB)
- `outputs/checkpoints/bevdet_real_best.pth` (170 MB)
- `outputs/checkpoints/bevformer_small_real_best.pth` (228 MB)
- `outputs/checkpoints/sparse4dv3_r50_real_best.pth` (185 MB)
- `outputs/checkpoints/bevdet_occ_real_best.pth` (134 MB)
- `outputs/checkpoints/sparseocc_v11_real_best.pth` (307 MB)

> **注意**：由于 GitHub 限制单文件不能超过 100 MB，所有 `.pth` 模型权重文件默认通过 `.gitignore` 排除，不提交至 Git 仓库。
