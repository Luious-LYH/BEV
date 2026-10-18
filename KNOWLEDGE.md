# BEV & Occupancy 深度技术手册 (教学与面试专用)

> **面向自动驾驶感知面试与技术复习的高信息密度知识手册。**  
> **拒绝长篇论文摘要，专注于：几何推导、张量维度流 (Tensor Shape Tracking)、核心代码入口与高频面试考点。**

---

## 1. nuScenes 数据结构与空间几何坐标系

### 1.1 数据对象拓扑关系

```text
scene (约 20 秒连续轨迹序列)
  └── sample (关键帧，2Hz 标注，对应一个唯一的全局 timestamp)
        ├── sample_data (每个传感器在具体时间点的数据，如 CAM_FRONT, LIDAR_TOP)
        │     └── sweeps (两关键帧之间的非标注中间帧，12Hz-20Hz)
        └── ego_pose (该 timestamp 下自车在全局地图坐标系的位置与姿态四元数)
```

### 1.2 几何变换链路与数学公式

```text
2D 像素坐标 (u, v)
      │
      │ 1. 反投影 (Unprojection): P_cam = d * K^(-1) * [u, v, 1]^T
      ↓
相机 3D 坐标系 (Camera Coordinate: X-Right, Y-Down, Z-Forward)
      │
      │ 2. 相机外参刚体变换: P_ego = R_cam2ego * P_cam + T_cam2ego
      ↓
自车 3D 坐标系 (Ego Coordinate: X-Right, Y-Forward, Z-Up)
      │
      │ 3. 自车位姿变换: P_global = R_ego2global * P_ego + T_ego2global
      ↓
全局高精地图坐标系 (Global Coordinate)
```

- **内参矩阵 $K$ ($3 \times 3$)**：
  $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
- **时序跨帧自车运动对齐 (Ego-motion Alignment)**：
  从时间戳 $t-1$ 的自车坐标系转换到当前时间戳 $t$ 的自车坐标系：
  $$P_{ego}^{(t)} = \left(R_{ego2global}^{(t)}\right)^{-1} \cdot \left( R_{ego2global}^{(t-1)} \cdot P_{ego}^{(t-1)} + T_{ego2global}^{(t-1)} - T_{ego2global}^{(t)} \right)$$

---

## 2. BEVDet: 经典 Lift-Splat-Shoot Baseline

### 2.1 解决什么问题
将 2D 环视多相机特征通过视锥显式投影（Lift）并聚合成密集鸟瞰图特征（Splat），摆脱单目 2D 独立检测的视差与深度模糊。

### 2.2 张量维度数据流 (Tensor Shapes)
```text
1. 环视多视角图像:
   x: [B, N_cam=6, 3, H_img=256, W_img=704]
   
2. 图像 Backbone + FPN 特征提取:
   feat: [B, N_cam, C=64, H_feat=16, W_feat=44]
   
3. DepthNet 联合预测 (特征变换 + 深度离散分布):
   depth_logits: [B, N_cam, D=118, H_feat, W_feat] (D 为深度离散 bin 数，例如 1m 到 60m)
   context_feat: [B, N_cam, C_ctx=64, H_feat, W_feat]
   depth_prob = softmax(depth_logits, dim=2): [B, N_cam, D, H_feat, W_feat]
   
4. Lift 操作 (外积构建 3D 视锥特征):
   frustum_feat = depth_prob.unsqueeze(-1) * context_feat.unsqueeze(2):
   frustum_feat: [B, N_cam, D=118, H_feat=16, W_feat=44, C=64]
   
5. 坐标网格网格化 (结合 K, R_cam2ego, T_cam2ego 计算各视锥点在 Ego 坐标系网格索引):
   geom_xyz: [B, N_cam, D, H_feat, W_feat, 3] -> 映射到网格索引 [X_idx, Y_idx, Z_idx]
   
6. Splat / Voxel Pooling (基于 Quick-cumsum 或 CUDA 算子沿 Z 轴求和/折叠):
   bev_feat: [B, C=64, Y_bev=128, X_bev=128]
   
7. BEV Encoder + 3D Detection Head:
   cls_score: [B, Num_classes=10, Y_bev, X_bev]
   reg_preds: [B, 8 (x, y, z, w, l, h, sin_yaw, cos_yaw), Y_bev, X_bev]
```

### 2.3 核心代码入口与函数
- `mmdet3d/models/detectors/bevdet.py` → `BEVDet.forward_train()`
- `mmdet3d/models/necks/view_transformer.py` → `LSSViewTransformer.create_frustum()`, `LSSViewTransformer.voxel_pooling()`
- `mmdet3d/ops/bev_pool/` → `bev_pool_kernel.cu`

---

## 3. BEVDepth: LiDAR 深度监督与加速 Pooling (Stage 1 核心主模型)

### 3.1 解决什么问题
BEVDet/LSS 中的深度估计完全是无监督的“隐式猜测”，深度分布极易发散，且传统 Pooling 速度慢。BEVDepth 引入了 **LiDAR 投影的真实深度监督** 与 **高效 Voxel Pooling 加速**，同时在 DepthNet 中编码相机内参，使得几何映射质量产生质的飞跃。

### 3.2 深度网络机制与 LiDAR 监督原理
- **相机内参感知 (Camera-awareness)**：将相机内参 $K$ 展平后经 MLP 编码，与图像特征拼接后输入 DepthNet，使卷积层感知相机的焦距与视场大小。
- **LiDAR Depth GT 生成**：
  将当前帧的点云 $P_{lidar}$ 投影到第 $i$ 个相机成像平面：
  $$p_{img} = K_i \cdot (R_{lidar2cam_i} \cdot P_{lidar} + T_{lidar2cam_i})$$
  若 $p_{img}$ 落在图像范围内且深度 $d > 0$，采用最小滤波获取密集深度图，并离散化为 One-Hot 标签 $Y_{depth} \in [0, D-1]$。
- **深度监督损失**：
  $$L_{depth} = \text{BCEWithLogitsLoss}(logits_{depth}, Y_{depth}) \quad \text{或 Focal Loss}$$

### 3.3 张量维度数据流
```text
1. 图像特征与内参拼接:
   feat: [B*N, C, H, W]
   mlp_input = [fx, fy, cx, cy, ...]: [B*N, 16] -> MLP -> [B*N, C, 1, 1]
   feat_augmented = feat + mlp_feat
   
2. 深度预测与上下文提取:
   pred_depth: [B*N, D=112, H, W]  (受 L_depth 显式监督)
   context:    [B*N, C=80, H, W]
   
3. 视锥特征构建:
   frustum = softmax(pred_depth, dim=1).unsqueeze(2) * context.unsqueeze(1)
   frustum: [B, N, D=112, C=80, H, W]
   
4. 加速 Voxel Pooling (按批次预排序与并行规约):
   bev_feat: [B, C=80, Y=128, X=128]
```

### 3.4 核心代码入口
- `mmdet3d/models/necks/view_transformer.py` → `DepthLSSTransform.get_downsampled_gt_depth()`
- `mmdet3d/models/necks/view_transformer.py` → `DepthLSSTransform.forward()`
- 深度监督计算：`loss_depth` 计算逻辑与 `bev_pool_v2` CUDA 扩展。

### 3.5 核心面试问答
- **Q1: 训练时用了 LiDAR 做深度监督，测试阶段还需要激光雷达吗？**  
  *答*：绝对不需要。测试时仅输入 6 路纯视觉 RGB 图像，经过已学习到合理空间几何投影权重的 DepthNet 直接预测深度分布，属于典型的“Privileged Information / Teacher 辅助训练”范式。
- **Q2: 为什么预测的是深度概率分布 (Distribution) 而非单值深度 (Scalar Depth)？**  
  *答*：单目深度估计具有天然的不适定性（Ill-posed）与边缘模糊。使用离散 Categorical Distribution 能够表达多模态不确定性（如物体边缘处同时包含前景与背景的可能），通过 Softmax 线性加权保留梯度，比硬截断单值深度更平滑且易于优化。

---

## 4. 时序感知扩展：BEVDet4D 与 BEVStereo

### 4.1 BEVDet4D: 显式自车运动对齐
- **核心思想**：将历史帧 $t-1$ 的 BEV 特征，通过已知自车位姿变化变换到当前帧 $t$ 的空间坐标系下，在通道维度 Concatenate：
  $$BEV_{aligned}^{(t-1)} = \text{GridSample}\left(BEV^{(t-1)}, \text{TransformGrid}(R_{t-1 \to t}, T_{t-1 \to t})\right)$$
  $$BEV_{merged}^{(t)} = \text{Conv}_{1 \times 1}\left(\left[BEV^{(t)}, BEV_{aligned}^{(t-1)}\right]\right)$$
- **作用**：让模型具备感知目标速度（Velocity）、航向角加速度及克服遮挡的能力。

### 4.2 BEVStereo: 时序视差与 Cost Volume
- **核心思想**：单帧图像只能依靠单目语义线索猜测深度。BEVStereo 利用前后两帧图像在已知相机基线运动下的视差，沿深度平面构建 **Temporal Cost Volume**，直接通过几何立体匹配精细化深度分布。
- **对比点**：单目深度线索 (Monocular Cue) vs 几何立体视差 (Stereo Parallax Cue)。

---

## 5. BEVFormer: 基于 Query 与注意力机制的密集 BEV (Stage 2 核心主模型)

### 5.1 解决什么问题
摆脱显式深度估计中的“深度预测不准则投影全错”的几何级联误差。采用 **BEV Query 反向向图像采样** 的范式：在空间中构建虚拟 3D 网格查询，通过变形注意力机制（Deformable Attention）从对应相机特征中自适应提取特征。

### 5.2 核心架构与注意力机制
1. **BEV Queries**：可学习的 2D 网格参数 $Q \in \mathbb{R}^{H_{bev} \times W_{bev} \times C}$，代表 BEV 平面上各个空间网格单元；
2. **3D 柱状参考点 (Pillar Reference Points)**：每个网格单元 $(x, y)$ 在高度方向离散化为 $N_{ref}=4$ 个 3D 点 $(x, y, z_i)$；
3. **空间交叉注意力 (Spatial Cross Attention, SCA)**：
   将 3D 点通过相机外参和内参投影到各个视角图像特征图上：
   $$p_{ij} = K_j \cdot (R_{ego2cam_j} \cdot P_i + T_{ego2cam_j})$$
   仅在命中图像有效范围的相机视角上，以投影点为中心执行 Deformable Attention 采样少量关键特征并聚合。
4. **时序自注意力 (Temporal Self Attention, TSA)**：
   利用 Ego-motion 将上一帧历史 BEV 特征对齐后，作为 Key/Value 与当前 BEV Query 进行时序自注意力交互。

### 5.3 张量维度数据流
```text
1. BEV Query 定义:
   Q: [H_bev=200, W_bev=200, C=256] -> 展平为 [B, N_q=40000, C=256]
   
2. 3D 参考点生成 (在每个 query 上均匀采样 Z 轴):
   ref_points_3d: [B, N_q=40000, N_z=4, 3]  (坐标归一化到 [-50m, 50m])
   
3. 投影到 6 个相机特征平面:
   ref_points_cam: [B, N_cam=6, N_q=40000, N_z=4, 2]  (归一化像素坐标 (u, v))
   valid_mask:     [B, N_cam=6, N_q=40000, N_z=4]    (是否落在相机视场内)
   
4. Spatial Cross Attention:
   value: 多尺度图像特征 [B, N_cam, \sum(H_l*W_l), C=256]
   sampling_offsets: [B, N_q, N_z, N_heads=8, N_levels, N_points=4, 2]
   attention_weights: [B, N_q, N_z, N_heads=8, N_levels, N_points=4]
   Q_spatial = DeformableCrossAttention(Q, value, ref_points_cam)
   
5. Temporal Self Attention:
   Q_aligned_prev = AlignByEgoMotion(BEV_prev)
   Q_temporal = DeformableSelfAttention(query=Q, key=Q_aligned_prev, value=Q_aligned_prev)
   
6. 最终输出:
   BEV_final: [B, C=256, H_bev=200, W_bev=200]
```

### 5.4 核心代码入口
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` → `SpatialCrossAttention.forward()`
- `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` → `TemporalSelfAttention.forward()`

### 5.5 BEVFormer V2 技术演进
- **V1 核心瓶颈**：主干网络（Backbone）由于通过复杂的 3D-2D 间接注意力进行反向传播，梯度难以有效流回前置骨干，导致无法直接从强大的 2D 预训练主干中充分收益。
- **V2 创新机制 (Perspective Supervision)**：引入辅助 2D 检测头（Perspective Head），直接在多相机 2D 图像特征上计算 2D 边界框损失，强力引导 Backbone 学习目标特征，从而直接赋能现代大型骨干（如 InternImage / ConvNeXt）。

---

## 6. Sparse4D: 稀疏目标实例表征 (Stage 2 核心主模型)

### 6.1 解决什么问题
自动驾驶车辆通常只关心有限的动态与静态目标（百级别），构建几万甚至几十万网格的 Dense BEV 存在巨大的无用计算浪费。Sparse4D 彻底抛弃密集网格，直接维护少量 **稀疏 3D 实例 Anchor**。

### 6.2 核心架构与 Instance Bank
- **3D 稀疏 Anchor**：每个实例由一个 11 维显式状态定义：
  $$\text{Anchor} = (x, y, z, w, l, h, \sin\theta, \cos\theta, v_x, v_y, v_z)$$
- **Deformable 4D Feature Aggregation**：在 3D Bounding Box 的中心和各个面自适应生成稀疏采样点，投影回多视角时序图像特征图进行聚合。
- **Instance Bank**：跨帧保存置信度最高的 Top-K 实例特征与 3D Anchor，依据上一帧预测的速度向量显式传播到当前帧：
  $$Anchor_{prop}^{(t)} = Anchor^{(t-1)} + \Delta t \cdot \vec{v}$$
  实现极致高效的时序目标跟踪与感知闭环。

### 6.3 张量维度数据流
```text
1. 稀疏实例 Query:
   instance_feature: [B, N_inst=900, C=256]
   anchor_boxes:     [B, N_inst=900, 11]
   
2. 4D 采样点生成 (每个 anchor 生成 7 个关键点):
   sample_points_3d = GenerateKeyPoints(anchor_boxes): [B, N_inst, N_pts=7, 3]
   
3. 投影到当前与历史多帧图像特征:
   sample_points_2d = ProjectToCams(sample_points_3d, Matrices): [B, N_frames, N_cams, N_inst, N_pts, 2]
   
4. 4D 空间时序聚合:
   aggregated_feat = DeformableAggregation(multi_scale_cam_features, sample_points_2d)
   instance_feature = FFN(instance_feature + aggregated_feat): [B, N_inst, C=256]
   
5. 边界框回归更新 (逐层微调 Anchor):
   delta_anchor = RegHead(instance_feature): [B, N_inst, 11]
   anchor_boxes = anchor_boxes + delta_anchor
```

### 6.4 核心面试问答
- **Q: 既然 BEVFormer 已经很好，为什么业界还要演进到 Sparse4D 这类稀疏方案？**  
  *答*：
  1. **计算复杂度**：Dense BEV 计算量正比于探测距离与分辨率平方 $O(W \times H)$，远距离（如 100m~200m 高速场景）显存和延迟暴增；而 Sparse 方案计算量仅正比于候选目标数量 $O(N_{inst})$，极易扩展探测距离；
  2. **端到端跟踪天然结合**：Sparse 实例通过 Instance Bank 实现了自然的物体级时序传递，天生具备连续时序跟踪属性。

---

## 7. 3D Semantic Occupancy: BEVDet-Occ vs SparseOcc

### 7.1 为什么 3D 目标检测不够？
3D Detection 只能输出规则的长方体（Bounding Box），但真实世界包含大量长尾未知障碍物（散落货物、施工锥桶、不规则车辆、翻覆路障）以及连续分布环境（路沿、植被、积水、斜坡）。**3D Semantic Occupancy** 将物理世界离散化为规则 3D 体素网格（如 $0.4m \times 0.4m \times 0.4m$），预测每个网格的占据状态与语义类别，彻底解决任意未知障碍物感知难题。

### 7.2 BEVDet-Occ (Dense Occupancy 基线)
- **原理**：将 LSS 投影出的 3D 视锥特征转换为密集 3D 体素特征卷 $V \in \mathbb{R}^{B \times C \times Z \times Y \times X}$，采用 3D 卷积神经网络进一步提炼特征并分类。
- **瓶颈**：3D Dense 卷积在分辨率提高时内存与计算量急剧立方级膨胀 $O(X \cdot Y \cdot Z)$，难以在车载芯片上维持高帧率。

### 7.3 SparseOcc v1.1 (Sparse Occupancy 核心主模型)
- **原理**：观察到 3D 空间中绝大多数体素（>90%）是空闲空间（Free Space / Air）。SparseOcc 采用**粗到细（Coarse-to-fine）与稀疏掩码机制**，只对可能存在障碍物的体素位置生成 Query 并进行多尺度图像采样。
- **RayIoU 评估指标**：传统 mIoU 对远距离不可见的遮挡区域（Occluded Regions）惩罚过于苛刻，RayIoU 沿着相机视线射线（Camera Ray）只评估首个命中可见表面或视线沿线体素，更符合车载视觉系统的物理真实性。

---

## 8. 全模型横向对比全景表

| 维度 | BEVDet | BEVDepth | BEVFormer V1 | Sparse4D v3 | BEVDet-Occ | SparseOcc v1.1 |
|---|---|---|---|---|---|---|
| **核心范式** | Explicit Depth | Explicit Depth | Query Attention | Sparse Instance | Dense 3D Volume | Sparse 3D Query |
| **表征空间** | Dense BEV | Dense BEV | Dense BEV | Sparse 3D Anchors | Dense 3D Voxel | Sparse 3D Voxel |
| **2D→3D 机制** | LSS Lift-Splat | LSS + LiDAR GT | Deformable Cross-Attn| 4D Keypoint Project | LSS Voxelization | Coarse-to-fine Sample |
| **时序建模** | 单帧 (或 4D 拼接) | 单帧 (或 4D 拼接) | 时序自注意力 (TSA) | Instance Bank 显式跟踪| 时序 3D 卷积 | 多帧时序聚合 |
| **感知任务** | 3D Detection | 3D Detection | 3D Detection | 3D Det & Track | 3D Semantic Occ | 3D Semantic Occ |
| **显存压力** | 较低 (~4-6G) | 较低 (~5-7G) | 较高 (~12-16G) | 极低 (~4-6G) | 极高 (~16-20G) | 适中 (~8-12G) |
| **推理速度** | 极快 (~30+ FPS) | 快 (~25+ FPS) | 适中 (~10-15 FPS) | 极快 (~35+ FPS) | 较慢 (~5-10 FPS) | 适中 (~12-18 FPS) |
| **核心优势** | 结构清晰简单 | 几何质量极高 | 消除深度级联误差 | 高效、易扩展远距离 | 全场景物理占据建模 | 大幅削减无效空间计算 |

---

## 9. 核心技术面试速查题库 (Top 10)

1. **BEVDepth 中深度网络为何需要加入相机内参编码？**  
   *答*：不同相机焦距不同，相同大小的物体在成像面上的像素尺度截然不同。将内参矩阵输入网络使卷积核具备尺度自适应能力，消除不同相机的几何歧变。
2. **Deformable Attention 相比传统全全局注意力的核心优势是什么？**  
   *答*：全全局注意力计算复杂度为 $O(N_q \cdot N_{pix})$，在多相机高清图像下完全不可行；Deformable Attention 仅在每个参考点预测偏移并采样固定数量（如 4 个）特征点，复杂度骤降为 $O(N_q \cdot K)$，使得密集 BEV 转换成为可能。
3. **时序多帧融合时，为什么必须进行自车运动补偿 (Ego-motion Alignment)？**  
   *答*：自动驾驶车辆处于高速动态运动中。若直接拼接前后两帧的 BEV 特征，同一静态物体在两帧特征中的网格坐标将产生严重位移与重影，导致模型产生虚假运动预测或定位发散。
4. **LiDAR 深度监督的真实标签是如何从无序点云转换到图像网格的？**  
   *答*：通过激光雷达到相机的外参矩阵将点云转到相机系，再通过内参矩阵投影到图像 $(u, v)$，若多个激光点落入同一像素网格，则保留距离相机最近的深度值，并使用 One-Hot 或高斯模糊构建离散分布目标。
5. **3D Occupancy 任务相比传统 3D Detection 的最大本质收益是什么？**  
   *答*：长尾非规则障碍物（General Obstacle Detection, GOD）感知能力。传统目标检测只能对已知预设类别的 Bounding Box 做出预测，无法应对翻倒树木、遗落货物等任意形态障碍物，而 Occupancy 能够准确描述全空间的几何占据与可通行区域。
