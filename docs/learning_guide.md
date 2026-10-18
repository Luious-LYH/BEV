# 交互式学习与教材指南 (Interactive Learning Guide)

> **面向自动驾驶算法岗位的交互式教材与 Jupyter 实战教学系统。**  
> 包含 **12 章交互式全景学习报告 (HTML)** 与 **4 个高交互式 Jupyter Notebooks** 的详细使用说明。

---

## 1. 交互式全景教材网页 (`BEV_OCC_Learning_Report.html`)

本项目将自动驾驶 3D 感知的复杂原理与实验成果凝练为一份全功能单页离线教材（SPA 架构），内置 27 项高精度几何可视化、3D 点云渲染、多帧时序交互播放器以及技术面试高频题库。

### 1.1 一键启动本地服务
在项目根目录下执行以下脚本，即可启动静态 Web 服务器：

```bash
# 启动本地服务 (默认监听 8088 端口)
bash artifacts/serve_report.sh 8088
```

启动后在浏览器打开：
```text
http://localhost:8088/BEV_OCC_Learning_Report.html
```

> **提示**：若在远程无桌面服务器上运行，可在本地电脑通过 SSH 端口转发直接查看：
> ```bash
> ssh -L 8088:127.0.0.1:8088 username@remote_server_ip
> ```
> 然后在本地浏览器访问 `http://127.0.0.1:8088/BEV_OCC_Learning_Report.html`。

### 1.2 核心章节导航 (Curriculum)

1. **Chapter 01: nuScenes 多模态数据结构与空间坐标系变换**
   - 6 相机环视布局、LiDAR 激光点云联合标定、像素到全局世界的 4 步齐次坐标变换。
2. **Chapter 02: 显式几何投影基线 (BEVDet)**
   - Lift-Splat-Shoot (LSS) 视锥几何、深度隐式预测与高并发 Voxel Pooling 算子。
3. **Chapter 03: 显式深度监督跃迁 (BEVDepth)**
   - 为什么相机需要深度？LiDAR 点云在图像平面的稀疏投影监督与 DepthNet 分布预测。
4. **Chapter 04: 自车运动补偿与时序立体几何 (BEVDet4D & BEVStereo)**
   - Ego-motion 时序特征对齐机制、Cost Volume 构建与立体几何视差精细化。
5. **Chapter 05: 空间查询注意力与时序融合 (BEVFormer V1)**
   - BEV Query 设计、3D 立柱参考点、空间交叉注意力 (SCA) 与时序自注意力 (TSA)。
6. **Chapter 06: 透视监督与架构演进 (BEVFormer V2)**
   - Perspective Supervision 引导检测器更快收敛与多尺度特征解耦。
7. **Chapter 07: 稀疏目标实例表征 (Sparse4Dv3)**
   - 去密集网格设计、900 稀疏 3D Anchor、6 层逐级 Refinement 与 Instance Bank 跨帧追踪。
8. **Chapter 08: 密集占据网格基线 (BEVDet-Occ)**
   - 为什么 Detection 不够？Dense 3D 体素构建、3D 卷积开销与几何遮挡补全。
9. **Chapter 09: 稀疏体素采样与空间表征 (SparseOcc)**
   - 针对 90% 空体素的稀疏表征哲学、RayIoU 空间度量与精细表面重构。
10. **Chapter 10: 架构演进与横向横评 (OccNet)**
    - Coarse-to-fine 多层级占据表征设计与通用几何理解。
11. **Chapter 11: 硬件 Profiling 与工程基准**
    - 6 大模型在 RTX 4090 上的显存 (VRAM)、延迟 (Latency) 与 FPS 严谨横向 Benchmark。
12. **Chapter 12: 自动驾驶感知面试高频题库与思维模型**
    - 包含 20+ 道一线自动驾驶大厂高频面试真题与标准化高分答题框架。

---

## 2. 交互式 Jupyter Notebook 实战教学

位于 `artifacts/notebooks/` 目录下，提供 4 个开箱即用的真实交互式 Jupyter Notebooks，每份 Notebook 均包含可运行的代码单元、张量维度检查与内联交互可视化：

```text
artifacts/notebooks/
├── 01_nuscenes_data_and_geometry.ipynb      # nuScenes 数据读取与坐标几何变换
├── 02_bevdepth_from_image_to_bev.ipynb      # 深度预测、视锥投影与 Voxel Pooling
├── 03_bevformer_and_sparse4d.ipynb          # Query 逆向采样与稀疏 3D Anchor 演化
└── 04_occupancy_dense_vs_sparse.ipynb       # 密集体素 vs 稀疏占据空间网格
```

### 启动 Jupyter 实验室
```bash
conda activate bev
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### Notebook 教学重点

- **Notebook 01**：加载真实 nuScenes mini 数据，提取 6 路环视相机，计算内参外参逆变换，将 32 线 LiDAR 点云准确投射到前向/侧向相机画面中。
- **Notebook 02**：加载真实图片送入 DepthNet，输出 `[6, 112, 16, 44]` 深度概率张量，手动推导视锥网格，运行 Voxel Pooling 生成 `[1, 80, 128, 128]` BEV 特征图。
- **Notebook 03**：对比 BEVFormer 的 4 层 3D 参考点投影与 Sparse4D 的 900 个 3D 实例 Anchor；追踪 6 层特征迭代中的位置修正量 $(\Delta x, \Delta y, \Delta z)$。
- **Notebook 04**：读取 Occ3D 真实语义体素 `labels.npz`，切片对比 Dense 3D 卷积占用 vs SparseOcc 稀疏表面 Query 采样效率。
