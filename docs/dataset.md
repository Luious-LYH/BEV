# 数据集准备与配置指南 (Dataset Preparation Guide)

> **面向自动驾驶感知学习与复现的数据集完整说明。**  
> 包含 **nuScenes (Mini / Trainval)**、**CAN Bus 扩展包** 以及 **Occ3D-nuScenes 3D 占据网格真值** 的规模说明、下载渠道、组织结构与快速初始化方法。

---

## 1. 数据集全景总览

本项目围绕自动驾驶多视角 3D 感知 (Multi-Camera BEV & Occupancy) 构建，支持 **轻量快速启动 (Mini, ~5GB)** 与 **全量扩展训练 (Trainval, ~60GB)** 两层数据规模：

| 数据集组件 | 压缩包名称 / 格式 | 下载体积 | 解压后体积 | 包含内容与规格 | 依赖模型与阶段 | 优先级 |
|---|---|---|---|---|---|---|
| **nuScenes v1.0-mini** | `v1.0-mini.tgz` | ~3.9 GB | ~5.1 GB | 10 个典型路况场景，404 个关键帧 (2Hz)，6 环视相机图像 (1600×900)，32 线 LiDAR 点云，5 毫米波雷达，18,538 个 3D 边界框标注，自车位姿与传感器内外参标定 | 全部 6 个核心模型快速验证、端到端推理、子集训练与 4 个 Notebook | **P0 (必选)** |
| **CAN Bus Expansion** | `can_bus.zip` | ~745 MB | ~1.5 GB | 852 个场景自车底盘总线数据（车速、加速度、航向角偏航率、转向角、轮速等 JSON 序列） | BEVFormer、Sparse4D 时序自车运动补偿 (Ego-motion Alignment) | **P0 (必选)** |
| **Occ3D-nuScenes GT** | `occ3d_gts.tar.gz` + `occ3d_annotations.json` | ~2.7 GB | ~3.5 GB | 852 场景 3D 体素语义占据真值 (体素网格维度 [200, 200, 16], 0.4m 分辨率，18 类语义) + 相机可见性掩码 (mask_camera) | BEVDet-Occ、SparseOcc 占据网格训练与评估 | **P0 (Occ 必选)** |
| **nuScenes K-Means 900** | `nuscenes_kmeans900.npy` | 78 KB | 78 KB | nuScenes 全场景 3D 边界框空间分布聚类得到的 900 个先验 3D Anchor 坐标与尺寸 | Sparse4Dv3 稀疏实例 Anchor 初始化 | **已内置在代码仓** |
| **nuScenes Trainval Keyframes** | `v1.0-trainval01~10_keyframes.tgz` (10包) + `v1.0-trainval_meta.tgz` | ~52 GB | ~60 GB | 850 个场景，34,149 个关键帧 6 环视相机图像与全量元数据 | 全量学术跑分与多轮扩展训练 | **P1 (进阶可选)** |

---

## 2. 官方下载渠道

### 2.1 nuScenes 官方数据 (Mini + CAN Bus)
1. 访问 nuScenes 官方网站：[https://www.nuscenes.org/download](https://www.nuscenes.org/download)
2. 免费注册并登录账号，同意数据集学术使用协议。
3. 在下载列表中找到并下载以下文件：
   - **`v1.0-mini.tgz`**（用于轻量开发与教学）
   - **`can_bus.zip`**（在 "CAN bus expansion" 分组中）
   - （如需全量训练）下载 `v1.0-trainval_meta.tgz` 与 `v1.0-trainval01_keyframes.tgz` 至 `v1.0-trainval10_keyframes.tgz`。

### 2.2 Occ3D-nuScenes 3D 占据网格真值
1. 访问清华大学 MARS Lab 开源项目主页：[https://github.com/Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)
2. 在 Readme 的 **Occ3D-nuScenes Download** 小节中：
   - 通过官方提供的 Google Drive 或 百度网盘链接，下载：
     - `occ3d_gts.tar.gz` (真实体素占据真值，约 2.6 GB)
     - `occ3d_annotations.json` (标注元数据，约 144 MB)

---

## 3. 标准目录结构与存放路径

在项目根目录下，`data/` 目录用于组织所有数据集资产（你可以将本地已有的大容量磁盘路径软链接为 `data`，例如 `ln -s /path/to/large_disk/BEV_dataset ./data`）。

解压后的标准目录结构如下：

```text
BEV/
└── data/
    └── nuscenes/
        ├── v1.0-mini/                           <-- 由 v1.0-mini.tgz 解压得到
        │   ├── attribute.json
        │   ├── calibrated_sensor.json
        │   ├── category.json
        │   ├── ego_pose.json
        │   ├── instance.json
        │   ├── log.json
        │   ├── map.json
        │   ├── sample.json
        │   ├── sample_annotation.json
        │   ├── sample_data.json
        │   ├── scene.json
        │   ├── sensor.json
        │   └── visibility.json
        ├── samples/                             <-- 6 相机关键帧图像 + LiDAR + Radar
        │   ├── CAM_FRONT/
        │   ├── CAM_FRONT_LEFT/
        │   ├── CAM_FRONT_RIGHT/
        │   ├── CAM_BACK/
        │   ├── CAM_BACK_LEFT/
        │   ├── CAM_BACK_RIGHT/
        │   ├── LIDAR_TOP/
        │   └── RADAR_*/
        ├── sweeps/                              <-- 非标注中间帧点云与图像
        │   ├── CAM_FRONT/
        │   └── LIDAR_TOP/
        ├── maps/                                <-- 地图栅格与膨胀矢量
        │   ├── basemap/
        │   └── expansion/
        ├── can_bus/                             <-- 由 can_bus.zip 解压得到 (852个 JSON)
        │   ├── scene-0061_ms_imupacket.json
        │   ├── scene-0061_pose.json
        │   └── ... (所有场景的车辆底盘状态数据)
        ├── occ3d/                               <-- Occ3D 占据网格
        │   ├── annotations.json                 <-- 来自 occ3d_annotations.json
        │   └── gts/                             <-- 来自 occ3d_gts.tar.gz
        │       ├── scene-0061/
        │       │   └── <sample_token>/
        │       │       └── labels.npz           <-- 包含 [200, 200, 16] 的 semantic 与 mask_camera
        │       └── ... (全场景 3D 体素真值)
        ├── bevdetv3-nuscenes_infos_train.pkl    <-- 预处理生成的 Mini 训练索引
        └── bevdetv3-nuscenes_infos_val.pkl      <-- 预处理生成的 Mini 验证索引
```

---

## 4. 快速解压与初始化命令

假定你已将下载好的压缩包存放在 `/path/to/downloads/`，执行以下命令快速完成组织与解压：

```bash
# 1. 创建标准数据根目录 (推荐在项目根目录外建立并软链接)
mkdir -p /path/to/BEV_dataset/nuscenes
cd /path/to/BEV_dataset/nuscenes

# 2. 解压 nuScenes mini
tar -xzf /path/to/downloads/v1.0-mini.tgz -C ./

# 3. 解压 CAN bus 数据
unzip -q /path/to/downloads/can_bus.zip -d ./
# 确保解压出来的 json 文件位于 nuscenes/can_bus/ 下

# 4. 解压 Occ3D-nuScenes GT
mkdir -p occ3d
cp /path/to/downloads/occ3d_annotations.json occ3d/annotations.json
tar -xzf /path/to/downloads/occ3d_gts.tar.gz -C occ3d/

# 5. 回到项目根目录建立软链接
cd /path/to/BEV
ln -sfn /path/to/BEV_dataset ./data
```

---

## 5. 生成预处理索引元数据 (MMDet3D Info Pickles)

在环境配置完成后（激活 `bev` conda 环境），运行以下预处理脚本，生成 BEVDet / BEVDepth / Eval 所需的 `pkl` 元数据：

```bash
# 运行一键 Mini 索引构建脚本
python scripts/data/create_bevdet_mini_infos.py
```

执行完毕后，检查 `data/nuscenes/` 下是否已成功生成：
- `bevdetv3-nuscenes_infos_train.pkl` (~6.1 MB)
- `bevdetv3-nuscenes_infos_val.pkl` (~1.8 MB)

至此，数据集已完全就绪，可直接执行所有模型的推理、训练与评测！
