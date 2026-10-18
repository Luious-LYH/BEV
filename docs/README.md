# 项目技术文档索引 (Documentation Index)

欢迎查阅 **Multi-Camera BEV & Occupancy 3D Perception System** 技术文档库。本目录为在新服务器部署、环境构建、数据集配置、模型推理与二次开发提供完整指引：

---

## 1. 文档目录概览

| 模块名称 | 文档链接 | 核心内容说明 |
|---|---|---|
| **数据集准备指南** | [docs/dataset.md](dataset.md) | nuScenes (Mini/Trainval)、CAN Bus、Occ3D-nuScenes 的规模、下载地址、存放路径与索引生成 |
| **环境配置与编译** | [docs/environment.md](environment.md) | PyTorch 2.0.1、CUDA 11.8/12.x、MMCV 1.7.2、MMDet3D 环境搭建及 Sparse4D CUDA 扩展算子编译 |
| **预训练模型与权重** | [docs/models.md](models.md) | 官方 ResNet-50 骨干网络及 6 大核心模型 Checkpoints 清单、下载参考与存放位置 |
| **实验基准与硬件审计** | [docs/experiments.md](experiments.md) | 受控训练收敛曲线、mAP / NDS / mIoU 真实指标对比、RTX 4090 延迟 (Latency) 与显存 Profiling |
| **交互式教学与实战** | [docs/learning_guide.md](learning_guide.md) | 12 章离线交互式全景学习报告 (HTML) 本地部署与 4 个交互式 Jupyter Notebook 使用指南 |
| **第三方补丁说明** | [patches/README.md](../patches/README.md) | BEVDet、BEVFormer、SparseOcc 消除命名冲突与版本上限检查的 3 个核心补丁解析 |
| **高频面试与技术手册** | [KNOWLEDGE.md](../KNOWLEDGE.md) | nuScenes 空间几何坐标变换数学推导、张量形状流追踪 (Tensor Shape Tracking) 与面试考点库 |

---

## 2. 新服务器部署快速指引 (Quick Flow)

```text
1. 克隆代码仓库:
   git clone https://github.com/Luious-LYH/BEV.git && cd BEV

2. 配置 Conda 环境 (参考 docs/environment.md):
   bash scripts/setup/create_bev_env.sh
   conda activate bev
   bash scripts/setup/compile_sparse4dv3_ops.sh

3. 配置数据集与预训练权重 (参考 docs/dataset.md 与 docs/models.md):
   ln -s /path/to/BEV_dataset ./data
   ln -s /path/to/pretrained_model/BEV ./pretrained
   python scripts/data/create_bevdet_mini_infos.py

4. 运行模型 Sanity 测试或端到端推理 (参考 docs/experiments.md):
   bash scripts/train/sanity_bevdepth_r50.sh
   bash scripts/inference/run_bevdepth_r50.sh

5. 启动交互式全景学习教材 (参考 docs/learning_guide.md):
   bash artifacts/serve_report.sh 8088
```
