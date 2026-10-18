/**
 * BEV / OCC Perception Textbook - Comprehensive Technical Concept Registry
 * Interactive modal popup system for .term-link items.
 * Contains 30+ deep-dive concepts: Definition, Math (KaTeX), Tensor Dimensions, Code, Gotchas, Interview QA.
 */

const CONCEPT_REGISTRY = {
  "lss_concept": {
    "title": "Lift-Splat-Shoot (LSS) 视锥投影与体素池化",
    "category": "核心视图变换 (View Transform)",
    "tags": ["LSS", "2D➔3D", "Depth Distribution", "Voxel Pooling"],
    "essence": "LSS (Lift-Splat-Shoot, ECCV 2020) 是自动驾驶多视角纯视觉 2D 到 3D 转换的开山之作。其核心物理思想是：单目相机无法直接获取绝对深度，但在透视光线上，每个像素的真实 3D 物理点必定落在其视锥光线上的某一个离散深度区间内。因此，LSS 将 2D 像素特征乘以预测的离散深度概率分布，'抬升' (Lift) 为 3D 视锥点云体，再利用相机内外参刚体变换将其'喷射' (Splat) 到统一的自车 BEV 网格中，最后通过体素求和池化得到 BEV 特征图。",
    "math": `$$F_{\\text{frustum}}(u, v, d) = P(d \\mid u, v) \\cdot C(u, v) \\in \\mathbb{R}^{C}$$
$$P_{\\text{ego}}(u, v, d) = R_{\\text{cam}\\to\\text{ego}} \\cdot \\left( d \\cdot K^{-1} \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} \\right) + t_{\\text{cam}\\to\\text{ego}}$$
$$F_{\\text{BEV}}(x_i, y_j) = \\sum_{(u, v, d) \\in \\text{Voxel}(x_i, y_j)} F_{\\text{frustum}}(u, v, d)$$`,
    "tensor_dims": "Input: [B, N=6, 3, H=900, W=1600]\n➔ Backbone+FPN: [B, N, C_in=512, H'=16, W'=44]\n➔ DepthNet: 深度分布 Prob [B, N, D=118, H', W'], 上下文特征 C [B, N, C=80, H', W']\n➔ Lift 外积广播: Frustum [B, N, D=118, C=80, H', W']\n➔ 空间坐标投影: [B, N, D, H', W', 3]\n➔ Voxel Splat/Pooling: BEV 特征图 [B, C=80, H_bev=128, W_bev=128]",
    "code": `import torch
import torch.nn.functional as F

def lss_lift(depth_logits: torch.Tensor, context_feats: torch.Tensor):
    """
    depth_logits:  [B, N, D, H, W]  未归一化的深度 logits
    context_feats: [B, N, C, H, W]  2D 上下文特征
    """
    # 沿着深度轴做 Softmax 概率归一化，使得沿视线的概率和为 1
    depth_probs = F.softmax(depth_logits, dim=2) # [B, N, D, H, W]
    
    # 调整张量形状以便广播外积相乘
    # [B, N, D, 1, H, W] * [B, N, 1, C, H, W] -> [B, N, D, C, H, W]
    frustum_feat = depth_probs.unsqueeze(3) * context_feats.unsqueeze(2)
    return frustum_feat`,
    "gotchas": "1. 原始 LSS 使用基于累加和 (cumsum) 的体素池化，存在极大的内存碎片和重复计算开销，训练速度极慢；BEVDepth 和 BEVDet 后来提出了预构建索引的 GPU 硬件级 BEV Pooling v2 算子。\n2. 深度离散化区间 (如 1m~60m 划分为 118 个 bin) 如果采用线性划分，在远距离处分辨率过低；业界通常使用等比划分 (Logarithmic binning) 或根据相机主点 FOV 设计双曲步长。\n3. 如果没有 LiDAR 显式监督，单目 depth net 在弱纹理或光照恶劣区域预测的深度方差极大，导致 BEV 特征沿视线方向出现严重的'拖尾'虚影 (Ray Smearing)。",
    "interview_qa": {
      "question": "面试官常问：请详细分析 LSS 的计算瓶颈在哪里？工业界为什么不能直接把原始 LSS 部署到车端计算平台 (如 Orin / Horizon)？",
      "answer": "【答题结构与要点】：\n1. 核心瓶颈分析：原始 LSS 的瓶颈在 Frustum 张量展开和 Pooling 阶段。例如 6 视角 × 118 深度 × 80 通道 × 16 × 44 尺寸，其视锥特征张量浮点数高达 4000 万个浮点数 (约 160MB/帧)。在向 BEV 网格执行 Splat 时，原始实现先对所有视锥点的 3D 网格坐标排序 (torch.argsort)，再用 cumsum 差分求和。排序操作在 GPU/NPU 上是极度耗时且显存非连续的访存密集型算子。\n2. 车端部署无法落地的致命原因：\n   - 动态张量与不规则排序：车端加速引擎 (如 TensorRT, Horizon BPU) 极度厌恶依赖数据内容的动态索引排序。\n   - 显存带宽打满：频繁在显存与计算核之间搬运巨大的 3D 视锥张量导致算力利用率 (MFU) 低于 15%。\n3. 工业界标准解决方案：\n   - 采用 BEV Pooling v2 / Flash-BEV：利用相机外参在非标定变动时不变的先验，在模型初始化或第一帧提前在 CPU 预计算 (Precompute) 好每个 2D 像素与 BEV 体素的固定映射索引表，送入定制 CUDA/TRT 插件中做原子累加 (atomicAdd) 或共享内存分块归约，延迟从 50ms 降至 2ms 级别。"
    }
  },
  "voxel_pooling": {
    "title": "BEV Pooling v2 / Voxel Pooling 加速算子",
    "category": "底层高性能算子 (CUDA Optimization)",
    "tags": ["CUDA", "BEV Pooling v2", "TensorRT", "Fast Voxelization"],
    "essence": "Voxel Pooling 是将 3D 视锥空间中离散的特征点累加归约到固定分辨率 BEV 体素网格的核心算子。在原始 LSS 中，Splat 采用通用排序和前缀和差分实现，耗时超过 50ms。BEVDepth 提出的 BEV Pooling v2 通过 CUDA 线程块设计，将每个体素映射到一个 GPU 线程块，每个线程并行读取预编译好的视锥点索引并执行局部共享内存归约，彻底消除排序开销，将耗时压缩至 1ms 级别，使 LSS 架构在车载嵌入式芯片上具备实时落地可行性。",
    "math": `$$\\text{Idx}(u, v, d) = \\left\\lfloor \\frac{x_{\\text{ego}} - x_{\\text{min}}}{\\Delta x} \\right\\rfloor + \\left\\lfloor \\frac{y_{\\text{ego}} - y_{\\text{min}}}{\\Delta y} \\right\\rfloor \\cdot W_{\\text{bev}} + \\left\\lfloor \\frac{z_{\\text{ego}} - z_{\\text{min}}}{\\Delta z} \\right\\rfloor \\cdot W_{\\text{bev}} H_{\\text{bev}}$$
$$F_{\\text{BEV}}[k] = \\sum_{i \\in \\{i \\mid \\text{VoxelIdx}[i] = k\\}} F_{\\text{point}}[i]$$`,
    "tensor_dims": "视锥点特征: [N_pts = B * N_cam * D * H * W, C=80]\n体素几何索引: [N_pts], 对应三维栅格坐标 (x_idx, y_idx, z_idx)\n输出 BEV 体素特征: [B, C=80, Z_dim=1, H_bev=128, W_bev=128]",
    "code": `// CUDA Kernel 核心归约逻辑简化伪代码 (BEV Pooling v2)
__global__ void bev_pool_v2_kernel(
    const float* __restrict__ feats,      // [N_pts, C]
    const int*   __restrict__ geom_feats, // [N_pts, 3] (x, y, z)
    const int*   __restrict__ interval_starts,
    const int*   __restrict__ interval_lengths,
    float*       __restrict__ out,        // [B, H_bev, W_bev, C]
    int n_intervals, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_intervals) return;
    
    int start = interval_starts[idx];
    int len   = interval_lengths[idx];
    
    // 一个线程块负责处理落入同一个 BEV 栅格的所有连续视锥点
    for (int ch = 0; ch < c; ++ch) {
        float sum = 0.0f;
        for (int i = 0; i < len; ++i) {
            sum += feats[(start + i) * c + ch];
        }
        out[idx * c + ch] = sum;
    }
}`,
    "gotchas": "1. 相邻帧外参发生轻微变动 (如刹车点头引起 pitch 震荡) 时，预先计算的静态几何索引表会失效，必须支持在线轻量动态更新索引。\n2. 半精度 (FP16) 下使用 atomicAdd 会存在数值溢出或微小下溢抹平问题，累加必须在 FP32 寄存器中暂存后写回。\n3. 多相机重叠区域 (如 Front 与 Front-Left 重叠视锥) 会导致同一体素累加的点数剧增，引起线程分歧 (Warp Divergence)。",
    "interview_qa": {
      "question": "面试官常问：BEV Pooling v2 相比原始 LSS 的 cumsum 提升了上百倍速度，它的核心硬件加速原理是什么？",
      "answer": "【答题要点】：\n1. 避免全局排序：原始 LSS 先对几何体素坐标进行非局部全局 argsort，显存访问完全随机且吞吐极低。BEV Pooling v2 在数据准备阶段将连续视锥点按体素边界整理为 `interval_starts` 和 `interval_lengths`。\n2. 连续内存合并访存 (Coalesced Memory Access)：每个 CUDA 线程块对应一个输出体素栅格，所有被聚合的特征点在内存中是物理连续存储的，GPU L1/L2 缓存命中率从 <30% 提升至 >92%。\n3. 零显存中间副本：省去了存储累加和中间大张量的数十兆显存，直接将累加结果写入最终 BEV 张量。"
    }
  },
  "extrinsics_concept": {
    "title": "外参刚体变换 (Extrinsics) 与 SE(3) 流形",
    "category": "几何与标定 (Geometry & Calibration)",
    "tags": ["Extrinsics", "SE(3)", "Quaternion", "Rigid Body"],
    "essence": "外参矩阵表征传感器硬件坐标系与自车坐标系 (Ego Coordinate System) 之间的三维刚体变换关系。刚体变换不发生任何形变、缩放或扭曲，由一个正交旋转矩阵 $R \\in SO(3)$ (度量三个自由度的 Roll, Pitch, Yaw 旋转) 和一个平移向量 $t \\in \\mathbb{R}^3$ (度量三个方向的 X, Y, Z 位移) 构成，属于特殊欧几里得群 $SE(3)$。在多相机 BEV 架构中，外参是拼接不同朝向视角、抹平物理盲区的第一真理底座。",
    "math": `$$T = \\begin{bmatrix} R & t \\\\ 0_{1\\times 3} & 1 \\end{bmatrix} \\in SE(3), \\quad R^T R = I, \\quad \\det(R) = +1$$
$$P_{\\text{ego}} = R_{\\text{sensor}\\to\\text{ego}} P_{\\text{sensor}} + t_{\\text{sensor}\\to\\text{ego}}$$
$$P_{\\text{sensor}} = R_{\\text{sensor}\\to\\text{ego}}^{-1} (P_{\\text{ego}} - t_{\\text{sensor}\\to\\text{ego}}) = R_{\\text{sensor}\\to\\text{ego}}^T (P_{\\text{ego}} - t_{\\text{sensor}\\to\\text{ego}})$$`,
    "tensor_dims": "外参平移向量 translation: [3] (米, X/Y/Z)\n外参旋转四元数 rotation: [4] (w, x, y, z) 需归一化\n齐次变换矩阵 T: [4, 4] float32",
    "code": `import numpy as np
from pyquaternion import Quaternion

def build_se3_transform(translation: list, rotation_quat: list) -> np.ndarray:
    """
    输入:
      translation: [x, y, z] (米)
      rotation_quat: [w, x, y, z] 四元数
    输出:
      4x4 齐次刚体变换矩阵 T
    """
    R = Quaternion(rotation_quat).rotation_matrix # [3, 3] 正交矩阵
    t = np.array(translation).reshape(3, 1)        # [3, 1]
    
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T`,
    "gotchas": "1. nuScenes 数据集中外参的平移单位是米 (m)，但相机内参焦距和主点坐标单位是像素 (px)，刚体变换与内参投影切忌混淆单位。\n2. 旋转正交性维持：在网络端到端联合微调外参时，若直接回归 $3\\times 3$ 矩阵，会破坏 $R^T R = I$ 的正交约束，必须回归四元数或 6D 连续旋转表征 (Zhou et al., CVPR 2019)。\n3. 车辆行驶中的动态外参漂移：急刹车、加速抬头、颠簸路面会导致自车车体相对传感器安装位置发生 0.5°~2° 的瞬时俯仰 (Pitch) 偏移，导致静止外参投影出的地面出现离地悬空或陷入地下的严重伪影。",
    "interview_qa": {
      "question": "面试官常问：如果现场车辆前向主相机因安装震动发生了 1 度的 Pitch (俯仰角) 标定误差，在 80 米外的障碍物检测上会产生多大的纵向/垂直测距误差？",
      "answer": "【答题推导与工程结论】：\n1. 几何误差推导：设物体处于光轴前方 $Z=80\\text{m}$ 处。相机俯仰误差 $\\Delta \\theta = 1^\\circ \\approx 0.01745\\text{ rad}$。\n2. 垂直空间高度偏移：$\\Delta Y = Z \\cdot \\tan(\\Delta \\theta) \\approx 80 \\times 0.0175 = 1.40\\text{m}$！意味着原本处于地面平面的障碍物在算法投影中被错误判定为悬空 1.4 米或砸入地表 1.4 米。\n3. 对基于平地假设 (IPM) 或单目反投影算法的纵向距离误差：由于地面仰角极小，根据 $\\Delta Z \\approx \\frac{Z^2}{h \\cdot f} \\Delta v$，80米处 1° 的误差会导致纵向测距产生高达 15~25 米的测距崩溃！这就是为什么 BEV 算法必须具备在线外参自标定 (Online Extrinsic Calibration) 或高鲁棒性深度监督机制。"
    }
  },
  "intrinsics_concept": {
    "title": "相机内参透视投影 (Intrinsics K) 与畸变模型",
    "category": "几何与标定 (Geometry & Calibration)",
    "tags": ["Intrinsics", "Pinhole Camera", "Distortion", "Perspective Projection"],
    "essence": "相机内参矩阵 $K$ 表征相机物理光学感光芯片 (CMOS) 与像素坐标系之间的几何映射关系。标准针孔相机模型由焦距 ($f_x, f_y$) 与光学主点 ($c_x, c_y$) 构成。在现实车载镜头 (尤其是鱼眼镜头和广角环视镜头) 中，光线在边缘区域存在显著的径向畸变 ($k_1, k_2, k_3$) 和切向畸变 ($p_1, p_2$)。在 BEV 深度学习前向推理前，所有输入图像必须完成畸变去扭曲 (Undistortion) 或在模型内网格采样时嵌入畸变补偿。",
    "math": `$$K = \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{bmatrix}, \\quad \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} \\sim \\frac{1}{Z_{\\text{cam}}} K \\begin{bmatrix} X_{\\text{cam}} \\\\ Y_{\\text{cam}} \\\\ Z_{\\text{cam}} \\end{bmatrix}$$
$$u = f_x \\frac{X_{\\text{cam}}}{Z_{\\text{cam}}} + c_x, \\quad v = f_y \\frac{Y_{\\text{cam}}}{Z_{\\text{cam}}} + c_y$$`,
    "tensor_dims": "内参矩阵 K: [3, 3] float32\n相机空间 3D 点: [3] (米)\n像素坐标: (u, v) (像素)",
    "code": `import numpy as np

def project_cam_to_pixel(pt_cam: np.ndarray, K: np.ndarray):
    """
    pt_cam: [3] 相机坐标系下的点 (X_cam, Y_cam, Z_cam)
    K:      [3, 3] 内参矩阵
    """
    # 深度有效性过滤：必须在镜头前方
    if pt_cam[2] <= 0.1:
        return None, None, False
        
    p_norm = pt_cam / pt_cam[2] # 归一化平面坐标 [X/Z, Y/Z, 1]
    pixel_homo = K @ p_norm
    u = float(pixel_homo[0])
    v = float(pixel_homo[1])
    return u, v, True`,
    "gotchas": "1. 数据增强坑点：在训练期若对图像执行随机缩放 (Crop & Resize) 或水平翻转，必须严格同步修正内参矩阵 $K$ 的 $f_x, f_y, c_x, c_y$。如果图像尺寸减半而 $K$ 保持不变，BEV 空间重投影将发生毁灭性的几何位移。\n2. $Z_{\\text{cam}} \\le 0$ 的数学除零灾难：如果点云或 3D 边界框顶点落在相机镜头后方，透视投影公式中分母小于等于 0，会导致投影像素反转到图像画面背后，引发恶性负无穷或越界异常，必须在除以 $Z$ 之前做视锥近裁切平面 (Near Plane Clipping, 如 $Z > 0.1\\text{m}$)。",
    "interview_qa": {
      "question": "面试官常问：在做多视角数据增强 (如随机裁剪 Random Crop 与缩放 Scale) 时，内参矩阵 K 该如何同步更新？如果做了旋转增强 (Rotation) 呢？",
      "answer": "【答题要点与数学公式】：\n1. 图像缩放 (Scale by $s_x, s_y$)：内参中的焦距和主点均按比例缩放：\n   $f_x' = f_x \\cdot s_x, \\quad f_y' = f_y \\cdot s_y, \\quad c_x' = c_x \\cdot s_x, \\quad c_y' = c_y \\cdot s_y$。\n2. 图像平移裁剪 (Crop offset $dx, dy$)：焦距不变，主点产生平移：\n   $c_x' = c_x - dx, \\quad c_y' = c_y - dy$。\n3. 图像平面二维旋转：属于 2D 仿射变换，不能直接修改 3D 内参 $K$！通常在 BEV 数据流中有两种解法：\n   - 方法 A (规范解法)：通过单应性变换 $H = K R_{2D} K^{-1}$ 作用在图像上；\n   - 方法 B (现代 BEV 推荐)：保持内参不变，将图像旋转数据增强矩阵 $A \\in \\mathbb{R}^{3\\times 3}$ 显式传入 DepthNet / View Transformer 中，参与 3D 到 2D 采样的逆坐标变换矩阵乘法 ($A^{-1} K$)。"
    }
  },
  "ego_motion_concept": {
    "title": "自车运动补偿 (Ego-Motion Compensation)",
    "category": "时序融合 (Temporal Fusion)",
    "tags": ["Ego-Motion", "Odometry", "Warping", "Timestamp Alignment"],
    "essence": "自动驾驶车辆处于高速动态运动中，不同时间戳采集的传感器数据对应不同的自车空间位姿。若直接将 $t-1$ 帧的历史特征与 $t$ 帧特征在空间栅格层面简单相加，由于车辆发生平移与航向旋转，会导致 BEV 空间内同一静态物体 (如车道线、护栏、静止车辆) 出现重影和严重模糊。自车运动补偿利用高精轮速计与惯导 IMU 提供的位姿变换矩阵 $T_{t-1 \\to t}^{\\text{ego}}$，将历史 BEV 特征图通过双线性插值 (Grid Sample) 刚体平移旋转到当前自车坐标系下对齐。",
    "math": `$$T_{t-1 \\to t}^{\\text{ego}} = \\left( T_{\\text{ego}(t) \\to \\text{global}} \\right)^{-1} \\cdot T_{\\text{ego}(t-1) \\to \\text{global}}$$
$$P_{\\text{ego}(t)} = T_{t-1 \\to t}^{\\text{ego}} \\cdot P_{\\text{ego}(t-1)}$$
$$F_{\\text{BEV}}^{(t-1)\\to t}(x, y) = \\text{BilinearSample}\\left( F_{\\text{BEV}}^{(t-1)}, \\,\\, \\text{Warp}(x, y, T_{t-1 \\to t}^{\\text{ego}}) \\right)$$`,
    "tensor_dims": "自车位姿变换: [B, 4, 4] 齐次矩阵\n历史 BEV 特征: [B, C=80, H=128, W=128]\n采样归一化网格: [B, H=128, W=128, 2] (取值范围 [-1, 1])\n对齐后历史特征: [B, C=80, H=128, W=128]",
    "code": `import torch
import torch.nn.functional as F

def align_bev_history(bev_history: torch.Tensor, T_prev2curr: torch.Tensor, 
                      x_range=(-51.2, 51.2), y_range=(-51.2, 51.2)):
    """
    bev_history: [B, C, H, W] 历史帧 BEV 特征
    T_prev2curr: [B, 4, 4]   历史自车系到当前自车系的位姿变换
    """
    B, C, H, W = bev_history.shape
    # 生成当前 BEV 网格物理坐标
    xs = torch.linspace(x_range[0], x_range[1], W, device=bev_history.device)
    ys = torch.linspace(y_range[0], y_range[1], H, device=bev_history.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    
    # 齐次坐标: [B, 4, H*W]
    pts_curr = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x), torch.ones_like(grid_x)], dim=0)
    pts_curr = pts_curr.view(4, -1).unsqueeze(0).repeat(B, 1, 1)
    
    # 逆变换映射回历史坐标系: P_prev = T_curr2prev @ P_curr
    T_curr2prev = torch.inverse(T_prev2curr)
    pts_prev = torch.bmm(T_curr2prev, pts_curr)[:, :2, :] # [B, 2, H*W]
    
    # 归一化到 [-1, 1] 供 F.grid_sample 使用
    norm_x = 2.0 * (pts_prev[:, 0, :] - x_range[0]) / (x_range[1] - x_range[0]) - 1.0
    norm_y = 2.0 * (pts_prev[:, 1, :] - y_range[0]) / (y_range[1] - y_range[0]) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(B, H, W, 2)
    
    # 执行双线性重采样对齐
    aligned_history = F.grid_sample(bev_history, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return aligned_history`,
    "gotchas": "1. padding_mode 必须设为 'zeros'：当车辆高速转弯或前行时，部分历史视野移出了当前 BEV 边界，新进入的未知区域必须填 0，切忌使用 border 或 reflection 模式引发边缘垃圾信号。\n2. IMU/GPS 时间戳对齐抖动：如果传感器数据与底盘 CAN 总线位姿时间戳存在数十毫秒的时钟同步偏差 (Clock Jitter)，会导致剧烈的补偿误差，需在输入端做三次样条位姿插值。\n3. 动态物体的重影问题：自车运动补偿仅能对齐静态背景与地面！动态障碍物 (如迎面开来的汽车) 在经过补偿后依然会有相对运动位移，必须依赖后续的时序注意力或速度预测头来建模动态目标速度矢量。",
    "interview_qa": {
      "question": "面试官常问：自车运动补偿 (Ego-Motion Compensation) 只能对齐静态环境，为什么多帧融合后模型依然能精准预测动态目标的速度？",
      "answer": "【答题要点】：\n1. 剥离自车速度，提纯相对运动：自车运动补偿的物理本质是消除参考系本身的平移与旋转，将所有历史帧的观测统一对齐到【当前时刻的自车坐标系】中。\n2. 相对位移即是速度的直接物理特征：在静态环境被严丝合缝对齐消除重影后，BEV 特征图上唯一发生空间位移的只有动态目标！例如一个以 $10\\text{m/s}$ 速度横穿的行人，在对齐后的 $t$ 帧与 $t-1$ 帧 (时间间隔 $\\Delta t = 0.5\\text{s}$) 特征图上，其高激活响应中心恰好在 BEV 空间相距 $5\\text{m}$。\n3. 后续网络 (如时序卷积、TSA 或 DETR 交叉注意力) 能够以最纯粹的差分卷积捕获这一位移矢量，从而极其容易且精准地解算出目标的线速度 $(v_x, v_y)$。"
    }
  },
  "bev_query_concept": {
    "title": "BEV Query (鸟瞰图可学习查询向量)",
    "category": "Transformer 范式 (BEVFormer)",
    "tags": ["BEV Query", "Transformer", "BEVFormer", "Top-Down"],
    "essence": "BEV Query 是以 BEVFormer 为代表的'自顶向下查询'范式的核心数据结构。不同于 LSS 依赖深度预测自底向上推导 BEV 特征，BEVFormer 在模型初始化时预定义一组可学习的参数张量 $Q \\in \\mathbb{R}^{H_{\\text{bev}} \\times W_{\\text{bev}} \\times C}$。每一个查询向量 $Q(x, y)$ 显式绑定了鸟瞰图物理空间中的一个固定真实网格位置 $(x, y)$。通过跨注意力机制，主动向多路环视图像中'索取'该空间网格可能对应的图像语义特征。",
    "math": `$$Q \\in \\mathbb{R}^{H_{\\text{bev}} \\times W_{\\text{bev}} \\times C}, \\quad \\text{RefPoint}_{3D}(x, y, k) = \\left( x_{\\text{min}} + x \\cdot \\Delta x, \\, y_{\\text{min}} + y \\cdot \\Delta y, \\, z_k \\right)$$
$$z_k \\in \\{z_1, z_2, \\dots, z_{N_{\\text{ref}}}\\} \\quad (\\text{沿地表高度的 } 4\\text{ 个离散柱点})$$`,
    "tensor_dims": "可学习参数张量: [H_bev=150, W_bev=150, C=256]\n批量化扩展: [B, H_bev * W_bev = 22500, C=256]\n3D 参考点坐标: [B, 22500, N_pts=4, 3]",
    "code": `import torch
import torch.nn as nn

class BEVQueryModule(nn.Module):
    def __init__(self, bev_h=150, bev_w=150, embed_dims=256):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        # 初始化可学习的 BEV 查询嵌入
        self.bev_embedding = nn.Parameter(torch.randn(bev_h * bev_w, embed_dims))
        
    def forward(self, batch_size):
        # 扩展到 Batch 维度: [B, H*W, C]
        return self.bev_embedding.unsqueeze(0).repeat(batch_size, 1, 1)`,
    "gotchas": "1. 计算复杂度随分辨率平方暴涨：若 BEV 尺寸从 $150\\times 150$ 增加到 $200\\times 200$，Query 数量从 22500 激增到 40000，标准的全局自注意力 (Self-Attention) 显存直接爆卡，必须采用可形变局部采样 (Deformable Attention)。\n2. 高度范围截断：nuScenes 场景中地面并非完全平整 (有立交桥和上下坡)，若 $Z$ 采样范围设得过窄 (如 $-2\\text{m} \\sim +2\\text{m}$)，位于坡道上的目标参考点会投出相机有效画面外，造成漏检。\n3. 收敛极慢：相比 LSS 的几何先验，纯随机初始化的 BEV Query 在训练初期无法建立稳定的注意力寻址，往往需要更长的 Warmup 步数与更大学习率。",
    "interview_qa": {
      "question": "面试官常问：请对比 LSS (自底向上) 和 BEVFormer (自顶向下查询) 两种范式的本质区别与优缺点？",
      "answer": "【答题对比矩阵与核心论述】：\n1. 转换方向本质不同：\n   - LSS 是 2D 像素带着深度向 3D 投射 (Bottom-Up)，属于'数据推导'；\n   - BEVFormer 是 3D Query 主动去 2D 图像采样特征 (Top-Down)，属于'按需索取'。\n2. 深度依赖性：\n   - LSS 极度依赖单目显式深度估计质量，一旦深度预测偏了，整个 3D 特征全偏；\n   - BEVFormer 在 3D 空间柱状采样多个高度点并在 2D 图像上聚合，不强制单点预测绝对深度，对深度误差容忍度更高。\n3. 计算开销与灵活性：\n   - LSS 计算确定性高，易于通过 BEV Pooling v2 固化为 GPU 算子，延迟稳定；\n   - BEVFormer 计算量由 Query 数量决定，分辨率扩展代价大，但天然适合融合多帧时序自注意力 (TSA)。"
    }
  },
  "sca_concept": {
    "title": "空间交叉注意力 (Spatial Cross-Attention, SCA)",
    "category": "Transformer 范式 (BEVFormer)",
    "tags": ["SCA", "Deformable Attention", "BEVFormer", "Pillar Projection"],
    "essence": "空间交叉注意力 (SCA) 是 BEVFormer 实现跨视角特征抽取的灵魂模块。为克服多路高分辨率图像带来的注意力爆炸，SCA 采用 3D 立柱采样思想：将每个 2D BEV Query 视为物理空间中垂直穿透地面的一根立柱，在立柱上预设 4 个高度点 ($Z \\in [-5\\text{m}, 3\\text{m}]$)；利用相机内外参将这些 3D 点投影到 6 路环视相机中，找到命中图像视野的投影点，并仅在这几个投影点周围采用可形变采样 (Deformable Sampling) 提取 2D 图像特征加权聚合。",
    "math": `$$\\text{SCA}(Q, F_{\\text{cam}}) = \\frac{1}{|\\mathcal{V}_{\\text{hit}}|} \\sum_{i \\in \\mathcal{V}_{\\text{hit}}} \\sum_{j=1}^{N_{\\text{pts}}} \\text{DeformAttn}\\left( Q, \\mathcal{P}(x, y, z_j, K_i, T_i), F_{\\text{cam}}^i \\right)$$
$$\\mathcal{P}(x, y, z, K, T) = \\text{Project}\\left( T_{\\text{ego}\\to\\text{cam}} \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix}, K \\right)$$`,
    "tensor_dims": "BEV Query: [B, 22500, C=256]\nMulti-Cam Features: [B, N=6, C=256, H=23, W=40]\nSampling Offsets: [B, 22500, N_levels=4, N_heads=8, N_pts=4, 2]\nAttention Weights: [B, 22500, N_levels=4, N_heads=8, N_pts=4]",
    "code": `// SCA 的空间投影映射几何逻辑 (Python 伪代码)
def sca_projection_mask(bev_query_xyz, cam_intrinsics, cam_extrinsics, img_w=1600, img_h=900):
    """
    bev_query_xyz: [N_query, N_z=4, 3]
    返回:
      hit_mask: [N_query, N_cam=6] 标记该 Query 立柱是否落在某相机可视区域
      ref_2d:   [N_query, N_cam, N_z, 2] 投影到图像平面的归一化坐标
    """
    # 刚体变换: P_cam = R @ P_ego + t
    # 透视投影: u = fx * X / Z + cx, v = fy * Y / Z + cy
    # 判定有效条件: Z > 0.1 and 0 <= u < img_w and 0 <= v < img_h
    pass`,
    "gotchas": "1. 边界双摄重叠盲区：在相邻相机边缘视锥重叠区域 (例如 Front-Left 与 Front)，一个 Query 立柱会同时投影到两个相机中，必须进行有效命中数归一化 ($1 / |\\mathcal{V}_{\\text{hit}}|$), 否则重叠区域的特征激活值会被成倍放大导致假阳性高估。\n2. 可形变注意力采样越界：网络预测的采样偏移 (Sampling Offsets) 可能会将采样点推到图像有效边界之外，在调用 MSDeformAttn CUDA 算子时必须严密处理边缘零填充。\n3. FP16 精度下权重和下溢：当多头注意力的权重经历 Softmax 后，若某些视角的注意力极度稀疏，FP16 会将其截断为 0，导致反向传播梯度消失。",
    "interview_qa": {
      "question": "面试官常问：BEVFormer SCA 为什么要在 Z 轴上采 4 个点而不是直接在 BEV 平面上采 1 个点？为什么不采 16 个点？",
      "answer": "【答题结构与深层几何机理】：\n1. 为什么不能采 1 个点 (地面 Z=0)：\n   - 现实中的 3D 目标 (如大型卡车、双层大巴、交通指示牌) 拥有显著的垂直高度 (1.5m~4.5m)。\n   - 如果只采 Z=0 地面点，当大卡车靠近自车时，其车头顶部在相机图像上位于上半部分，而地面投影点则在图像底部甚至车身下方。仅采地面点将彻底漏掉车身核心语义纹理！\n2. 为什么不采 16 个点：\n   - 计算量与显存权衡：每个采样点都对应可形变注意力的权重生成和特征插值访存。采样点数从 4 增加到 16，SCA 计算量和访存带宽线性增长 4 倍，而论文消融实验表明，4 个点已经足以覆盖自车 $-5\\text{m} \\sim +3\\text{m}$ 的常用交通参与者高度范围，收益迅速饱和。"
    }
  },
  "tsa_concept": {
    "title": "时序自注意力 (Temporal Self-Attention, TSA)",
    "category": "Transformer 范式 (BEVFormer)",
    "tags": ["TSA", "Temporal Fusion", "BEVFormer", "Recurrent Alignment"],
    "essence": "时序自注意力 (TSA) 是 BEVFormer 实现长序列时序时空记忆流动的核心机制。TSA 采用循环迭代 (Recurrent) 设计：在当前帧 $t$ 处理时，首先通过自车运动位姿变换矩阵将上一帧保留的 $B_{t-1}$ 特征图重采样对齐到当前坐标系，然后将当前帧的 BEV Query 作为查询，与对齐后的历史特征及当前特征共同执行可形变自注意力计算，使历史环境信息以极低的时延和显存代价流转到当前时刻。",
    "math": `$$B_{t-1}' = \\text{Warp}(B_{t-1}, T_{t-1 \\to t}^{\\text{ego}})$$
$$Q' = Q + \\text{PE}_{\\text{bev}}$$
$$\\text{TSA}(Q', B_{t-1}') = \\text{DeformAttn}\\left( Q', \\text{RefPoint}_{2D}, [B_{t-1}', Q'] \\right)$$`,
    "tensor_dims": "当前 Query: [B, 22500, C=256]\n历史 BEV 缓存: [B, 22500, C=256]\n时序位姿矩阵: [B, 4, 4]\n融合后 BEV 特征: [B, 22500, C=256]",
    "code": `import torch
import torch.nn as nn

class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8):
        super().__init__()
        self.deformable_attn = nn.MultiheadAttention(embed_dims, num_heads)
        
    def forward(self, curr_query, prev_bev_aligned):
        """
        curr_query:        [B, N_q, C] 当前帧查询
        prev_bev_aligned:  [B, N_q, C] 已经按自车位姿补偿对齐的历史 BEV
        """
        # 拼接历史与当前作为 Key/Value 记忆库
        memory = torch.cat([prev_bev_aligned, curr_query], dim=1) # [B, 2*N_q, C]
        
        # 执行可形变时序聚合
        out, _ = self.deformable_attn(curr_query, memory, memory)
        return out`,
    "gotchas": "1. 训练阶段梯度截断 (Detach)：在长序列循环训练时，若对 $B_{t-1}$ 进行全量反向传播，会形成 BPTT (通过时间反向传播)，显存直接崩溃。BEVFormer 采用流式训练，强制对 `prev_bev.detach()` 截断梯度，仅依靠监督信号驱动当前帧梯度。\n2. 第一帧冷启动异常：在视频序列的第一帧，历史缓存为空 ($B_{t-1} = \\text{None}$)，此时 TSA 退化为纯静态空间的自注意力，必须设置鲁棒的空输入分支防御。\n3. 累计漂移误差：长达几十秒的循环累加容易引入低频特征噪声漂移，通常通过残差连接与 LayerNorm 维持特征均值方差稳定。",
    "interview_qa": {
      "question": "面试官常问：BEVFormer 的 TSA 采用 RNN 样式的循环结构，与 BEVDet4D 的直接 Concat 拼接相比，各自的工程优缺点是什么？",
      "answer": "【答题对比与深度权衡】：\n1. BEVFormer (TSA 循环更新)：\n   - 优势：理论上具备无限时序记忆视野 (Infinite Receptive Field in Time)；无论融入多少历史帧，当前时刻只需要保存一份上一帧的 BEV 特征缓存，显存占用恒定为 $O(1)$。\n   - 劣势：推理必须严格按时间序列执行，无法并行推断多帧；训练时无法随机 shuffle 数据集，必须按连续视频 clip 组织。\n2. BEVDet4D (多帧显式 Concat 拼接)：\n   - 优势：工程实现极度简单，无状态 (Stateless)；对单帧丢包具有极高的容错性；各帧特征由轻量 Conv 融合，车端算子适配极好。\n   - 劣势：融合帧数受限 (通常只能融 2~4 帧)，融合帧数越多，显存和 Backbone 重复计算量线性激增 $O(K)$。"
    }
  },
  "sparse4d_anchor_concept": {
    "title": "稀疏 3D 锚框 (3D Anchor Boxes & 11D State)",
    "category": "稀疏范式 (Sparse4D)",
    "tags": ["Sparse4D", "3D Anchors", "Query-Based", "11D Vector"],
    "essence": "Sparse4D (Horizon, 2023) 彻底摒弃了空间稠密 BEV 网格构建，将整个物理空间抽象为一组稀疏实例 (Sparse Instances)。每个实例由一个显式物理意义极其明确的 11 维 3D 锚框向量表征：包含 3D 中心位置 $(x, y, z)$、三维几何尺寸 $(w, l, h)$、偏航角正余弦 $(\\sin\\theta, \\cos\\theta)$ 以及线速度 $(v_x, v_y, v_z)$。稀疏锚框仅在可能存在障碍物的区域分布，将计算资源集中在 1% 的高价值前景目标上。",
    "math": `$$\\mathbf{a} = \\left[ x, y, z, \\, w, l, h, \\, \\sin\\theta, \\cos\\theta, \\, v_x, v_y, v_z \\right]^T \\in \\mathbb{R}^{11}$$
$$\\text{Volumetric IoU} \\approx \\text{IoU}_{\\text{bev}}(x, y, w, l, \\theta) \\cdot \\text{IoU}_{1D}(z, h)$$`,
    "tensor_dims": "锚框状态张量: [B, N_anchors=900, 11] float32\n锚框实例特征嵌入: [B, 900, C=256]\n6 层解码器逐层输出: [6, B, 900, 11]",
    "code": `import torch

def generate_k_means_sparse_anchors(num_anchors=900):
    """
    Sparse4D 真实初始化方案：
    在 nuScenes 真实训练集 3D 标注框上进行 K-Means 聚类，
    生成 900 个高频统计 3D 空间先验锚框
    """
    anchors = torch.zeros(num_anchors, 11)
    # [x, y, z, w, l, h, sin, cos, vx, vy, vz]
    # 近处密集分布小物体 (行人/自行车)，远距离稀疏分布大型车辆
    return anchors`,
    "gotchas": "1. 角度周期性奇异点：偏航角 $\\theta \\in [-\\pi, \\pi]$ 若直接回归单角度值，在 $\\pm \\pi$ 处会产生梯度阶跃震荡；Sparse4D 将其解耦为正弦与余弦分量 $(\\sin\\theta, \\cos\\theta)$ 联合优化，并施加模长为 1 的归一化约束。\n2. 尺寸 $w, l, h$ 必须使用指数残差：尺度回归必须使用乘性指数 $\\exp(\\Delta)$，严禁直接相加，否则在模型初始化或训练震荡期会出现尺寸小于等于 0 的物理违规非法框。\n3. K-Means 聚类过拟合：锚框先验分布若高度依赖训练集采集路线 (如新加坡与波士顿的大道)，换到弯道或特殊路况容易产生先验视野盲区。",
    "interview_qa": {
      "question": "面试官常问：从 BEVFormer 到 Sparse4D，感知算法架构为什么要从'稠密网格'走向'稀疏实例'？其核心收益和理论代价是什么？",
      "answer": "【答题三步法与行业洞察】：\n1. 核心矛盾驱动：\n   - 稠密 BEV 网格存在严重的'算力空转'。自动驾驶物理空间中超过 90%~95% 都是空旷路面或天空背景，稠密网格却平等地对每一个空栅格执行重型注意力或卷积计算，随着感知距离从 50m 延伸到 100m，稠密网格计算量呈 $O(L^2)$ 甚至 $O(L^3)$ 爆炸。\n2. 稀疏范式核心收益：\n   - 算力大幅节约：固定 900 个稀疏锚框，计算复杂度与感知范围解耦，从几何物理上实现稀疏计算。\n   - 显存极低：无需维护数百兆的稠密 BEV 特征图，适合在低算力车端平台 (如地平线 J5 / 单 Orin) 上实现 >40 FPS 的实时推理。\n3. 理论代价与工程妥协：\n   - 稀疏实例高度依赖 Query 的召回能力与匈牙利匹配收敛；对于异形、非预定义类别的通用障碍物 (如掉落物、碎石堆)，稀疏 3D 边界框天然无法表征，必须由后续的 Occupancy 占用网格予以兜底。"
    }
  },
  "instance_bank_concept": {
    "title": "实例库 (Instance Bank) 与长周期时序跟踪",
    "category": "稀疏范式 (Sparse4D)",
    "tags": ["Instance Bank", "Sparse4D", "Tracking-by-Attention", "Temporal"],
    "essence": "实例库 (Instance Bank, Sparse4D v2/v3) 颠覆了传统自动驾驶感知'先检测 3D 框、再在后处理用卡尔曼滤波做二阶段跟踪 (Tracking-by-Detection)'的割裂架构。它在模型内部维护一个跨时间维度的实例记忆池：将上一帧高置信度保留下来的 600 个已知目标锚框，结合当前帧自车运动和目标历史速度进行刚体运动预测 (Dead Reckoning)，作为当前帧高质量先验直接传入解码器；同时保留 300 个空间统计先验锚框用于召回新出现的目标。检测即跟踪，端到端消除 ID Switch。",
    "math": `$$\\mathbf{a}_{t|t-1} = \\begin{bmatrix} x_{t-1} + v_x \\Delta t \\\\ y_{t-1} + v_y \\Delta t \\\\ z_{t-1} + v_z \\Delta t \\end{bmatrix}_{\\text{ego}(t)}, \\quad \\text{ConfidenceFilter}(\\mathbf{a}_{t-1}, \\text{Score} > \\tau)$$
$$\\mathcal{Q}_t = \\left[ \\mathcal{Q}_{\\text{history}}(\\text{Top-600}), \\, \\mathcal{Q}_{\\text{kmeans}}(\\text{New-300}) \\right]$$`,
    "tensor_dims": "实例库缓存特征: [B, 600, C=256]\n实例库 3D 锚框: [B, 600, 11]\n新目标先验锚框: [B, 300, 11]\n合并总查询量: [B, 900, 11]",
    "code": `import torch

def update_instance_bank(prev_anchors, prev_feats, prev_scores, T_prev2curr, dt, top_k=600):
    """
    Sparse4D Instance Bank 核心更新逻辑
    """
    # 1. 过滤高置信度跟踪目标
    top_indices = torch.topk(prev_scores, k=top_k, dim=1).indices
    
    # 2. 结合速度进行自车位姿与目标航行推算 (Dead Reckoning)
    # P_new = T_prev2curr @ (P_old + V * dt)
    # 3. 传给下一帧作为高质量 Query
    pass`,
    "gotchas": "1. 假阳性自我强化 (FP Feedback Loop)：如果上一帧模型错误检出一个高分虚假目标 (如地面反光误认为障碍物)，该目标会被写进 Instance Bank，在后续帧中作为强先验持续向模型反馈，导致假阳性幽灵目标久久无法消散，必须引入衰减淘汰机制。\n2. 急刹车与急转弯非线性位移：仅依靠一阶匀速模型进行推算在剧烈制动时会产生显著预测偏差，必须在融合层保留足够的形变自适应采样余量。\n3. 多目标遮挡交换 ID：两车近距离交错行驶时，若实例特征嵌入距离区分度不足，容易发生特征混淆导致 ID 互换。",
    "interview_qa": {
      "question": "面试官常问：传统 ByteTrack/SORT 轨迹跟踪和 Sparse4D Instance Bank 端到端跟踪相比，核心优势是什么？如何解决遮挡丢目标的问题？",
      "answer": "【答题要点】：\n1. 解决传统跟踪的核心痛点：\n   - 传统后处理跟踪依赖手工几何规则与卡尔曼滤波，对检测抖动极度敏感；一旦某帧检测框漏检，轨迹立即中断或退化。\n   - Instance Bank 将时序先验下沉到骨干特征抽取层，即'在特征层进行跟踪'。即使某一帧目标被大货车完全遮挡在相机中不可见，其保存在 Instance Bank 中的 Query 依然能感知其历史轨迹，在特征图上持续维护该目标的潜在空间分布。\n2. 遮挡重现无缝再关联：\n   - 当被遮挡目标重新出现在视野中时，由于其 Query 始终保留了目标的语义表征与运动状态，无需重新执行高代价的全局二分图关联匹配，直接接续轨迹，大幅压低 ID 切换率 (IDS)。"
    }
  },
  "occupancy_grid_concept": {
    "title": "3D 语义占用体素网格 (Semantic 3D Occupancy)",
    "category": "通用占用表征 (Occupancy Paradigm)",
    "tags": ["Occupancy", "3D Voxel", "Occ3D", "Open-Vocabulary"],
    "essence": "3D 占用网格 (3D Occupancy Grid) 是源自机器人学的环境表征范式。在自动驾驶中，传统 3D 边界框只能预测预先标注好的有限几个刚性物体 (Car, Pedestrian, Truck)，而现实道路充斥着大量不可预测的长尾未知障碍物 (如倾倒的树木、遗撒的纸箱、施工路锥、特殊异形工程车)。Occupancy 将自车周围物理空间三维体素化 (如 $200\\times 200\\times 16$ 栅格，每个栅格 $0.4\\text{m} \\times 0.4\\text{m} \\times 0.25\\text{m}$)，直接预测每个体素单元是被占据 (Occupied) 还是空闲 (Free)，并赋予通用几何语义标签。",
    "math": `$$\\mathbf{V} \\in \\{0, 1, \\dots, C-1\\}^{H_x \\times W_y \\times D_z}, \\quad \\text{VoxelSize} = [\\Delta x, \\Delta y, \\Delta z]$$
$$P(\\text{Occupied} \\mid \\text{Voxel}_{i,j,k}) = \\sigma\\left( \\text{Logit}_{i,j,k} \\right)$$`,
    "tensor_dims": "空间物理范围: X: [-40m, +40m], Y: [-40m, +40m], Z: [-1m, +5.4m]\n体素分辨率: 200 x 200 x 16 (共 640,000 个体素单元)\n体素物理尺寸: dx = 0.4m, dy = 0.4m, dz = 0.4m\n预测输出: [B, 17 类别, 200, 200, 16] float32",
    "code": `import torch
import torch.nn as nn

class OccupancyHead(nn.Module):
    def __init__(self, in_channels=80, num_classes=17, z_dim=16):
        super().__init__()
        # 通过 3D 卷积或 2D 反卷积展开恢复全 3D 体素空间
        self.deconv_z = nn.ConvTranspose3d(in_channels, 64, kernel_size=(3,3,3), padding=1)
        self.cls_head = nn.Conv3d(64, num_classes, kernel_size=1)
        
    def forward(self, x_bev_3d):
        # x_bev_3d: [B, C, Z, H, W]
        feat = self.deconv_z(x_bev_3d)
        logits = self.cls_head(feat) # [B, 17, 16, 200, 200]
        return logits`,
    "gotchas": "1. 立方级显存吞吐黑洞：从 2D BEV 走向 3D 体素，张量维度从 $H\\times W$ 升维到 $H\\times W\\times Z$，64 万个体素的 17 分类概率在 FP32 下每帧占据上百兆显存，3D 卷积计算复杂度极高。\n2. 类别不平衡极致恶劣：在 64 万个体素中，超过 95% 是空气 (Free Space)，道路地面占 3%，真实目标体素不足 1%。直接使用交叉熵损失会导致网络完全倾向于预测 Free Space，必须采用 Focal Loss 与 Lovasz-Softmax 联合约束。\n3. 遮挡不可见区域真值歧义：对于建筑物背面或车辆内部，LiDAR 射线无法穿透，这部分体素在现实中是'未知 (Unobserved)'而非'空闲'，真值若强行填充 Free 会惩罚模型的合理泛化。",
    "interview_qa": {
      "question": "面试官常问：既然已经有了精准的 3D 目标检测 (3D Bounding Box)，为什么现在各大自动驾驶头部公司 (特斯拉、华为、理想、小鹏) 全都在死磕 3D Occupancy 网络？",
      "answer": "【答题核心逻辑与大厂落地痛点】：\n1. 解决自动驾驶最大的长尾安全痛点 (The Long Tail of AD)：\n   - 3D 目标检测是闭集分类 (Closed-Set)。如果训练集没有标注'倾倒的大圆木'或'散落的高速轮胎皮'，检测网络就会完全视而不见，直接引发致命撞击事故。\n   - Occupancy 网络是开集几何重构 (Open-Set Geometry)。它不关心物体叫什么名字，只关心'该物理空间是否有体积实体存在'。即使面对从未见过的异形物体，只要体素被占据，下游规控 (Planning) 就能立刻执行减速或避障绕行。\n2. 复杂多变道路元素的统一表达：\n   - 道路马路牙子、施工挡板、护栏、树枝悬垂、非水平坡道等地表几何元素无法用简单的长方体 Box 描述，Occupancy 提供了对真实世界厘米级的连续三维网格刻画，为端到端自动驾驶提供了高保真世界模型底座。"
    }
  },
  "ray_iou_concept": {
    "title": "视线射线交并比 (RayIoU) 遮挡感知指标",
    "category": "通用占用表征 (Occupancy Paradigm)",
    "tags": ["RayIoU", "Metrics", "Occupancy", "Occlusion-Aware"],
    "essence": "RayIoU 是由 CVPR 2024 等前沿研究所提出的新一代 Occupancy 评测指标。传统的体素交并比 (Voxel mIoU) 平等对待所有空间体素，但对于处于严重遮挡盲区背面的体素 (例如大货车身后几米外的空域)，纯视觉相机在物理上根本不可见。传统 mIoU 苛求模型在看不见的区域猜对真值，导致评估失真。RayIoU 沿着相机中心向外发射数千条物理光线，仅在光线第一次击中物体表面之前的可见光程上计算 IoU，真正公正客观地评测视觉传感器的感知重建能力。",
    "math": `$$\\text{RayIoU} = \\frac{1}{|\\mathcal{R}|} \\sum_{r \\in \\mathcal{R}} \\frac{\\sum_{d} \\text{TP}(r, d)}{\\sum_{d} \\left[ \\text{TP}(r, d) + \\text{FP}(r, d) + \\text{FN}(r, d) \\right]}$$
$$\\text{Ray Termination Distance: } d^*(r) = \\min \\{ d \\mid \\mathbf{V}(r(d)) \\neq \\text{Free} \\}$$`,
    "tensor_dims": "视线射线集合 R: [N_rays = 10000]\n沿射线离散采样点: [N_rays, N_steps = 100]\n掩码权重: 仅沿穿透至首个阻挡物表面判定",
    "code": `def compute_ray_iou_step(pred_occ, gt_occ, camera_pos, rays_dirs):
    """
    RayIoU 评测核心思想：沿视线步进计算首个击中距离
    """
    # 沿着每条射线步进，找到预测与真值的首次终止距离
    # 仅在可见光程内统计预测准确率，剔除不可见盲区伪误差
    pass`,
    "gotchas": "1. 射线穿透仿真计算耗时大：RayIoU 评测需要执行密集的 3D 光线投射 (Ray Casting)，评测一帧 nuScenes 占用数据耗时约 200ms，需用 Taichi 或 CUDA 编写离线加速评测脚本。\n2. 地面与天空射线截断：需合理设置射线的最大射程距离截断 (如 50m)，避免对远方空旷背景计算无意义的射线开销。",
    "interview_qa": {
      "question": "面试官常问：为什么说 Voxel mIoU 无法完全代表 Occupancy 网络的真实感知水平？RayIoU 改进了什么？",
      "answer": "【答题要点】：\n1. 传统 Voxel mIoU 的致命缺陷：由于遮挡是 3D 物理世界的固有属性，相机视角下大客车后面隐藏的障碍物由于被挡住，视觉网络完全没有光子信息输入。传统 mIoU 盲目比对整个 3D 空间的每个体素，逼迫模型去'瞎蒙'看不见的盲区，猜错了就严重扣分，违背了物理测量规律。\n2. RayIoU 的核心物理公正性：\n   - 仅对'物理可见表面'和'自由空间光程'进行考评。\n   - 当一条视线打在障碍物上后，射线终止，其背面不可见体素不再计入损失与指标，使得评测结果 100% 真实反映视觉特征提取与深度几何预测能力。"
    }
  },
  "explicit_depth_concept": {
    "title": "显式深度监督 (Explicit Depth Loss)",
    "category": "深度估计 (Depth Supervision)",
    "tags": ["BEVDepth", "Explicit Depth", "Loss", "LiDAR Supervision"],
    "essence": "在早期的 BEVDet / LSS 中，DepthNet 的深度预测完全由下游 3D 检测任务的最终 Loss (如 CenterHead 边界框损失) 隐式反向传播驱动。由于网络搜索空间巨大且病态，单目无监督深度极易陷入局部最优，导致 3D 空间预测漂移达数米。BEVDepth 首次提出显式深度监督：在离线训练阶段，将高精 LiDAR 点云透视投影到 2D 相机平面，生成稀疏但绝对真实的离散深度真值标签 (Depth GT)，在 DepthNet 处直接施加交叉熵或二进制分类损失，强迫网络学习物理严谨的单目深度分布。",
    "math": `$$\\mathcal{L}_{\\text{depth}} = - \\frac{1}{|\\mathcal{M}|} \\sum_{(u, v) \\in \\mathcal{M}} \\sum_{d=1}^D y_{(u, v)}(d) \\log P(d \\mid u, v)$$
$$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{det}} + \\lambda_{\\text{depth}} \\cdot \\mathcal{L}_{\\text{depth}}, \\quad \\lambda_{\\text{depth}} = 3.0$$`,
    "tensor_dims": "DepthNet Logits: [B*N, D=118, H=16, W=44]\n投影 LiDAR 深度真值 Bin: [B*N, H=16, W=44] (0 为无点，1~118 为深度 bin)\n监督掩码 Mask: [B*N, H, W] (有效雷达点位置)",
    "code": `import torch
import torch.nn.functional as F

def compute_explicit_depth_loss(depth_logits, depth_gt_bins):
    """
    BEVDepth 显式深度监督交叉熵计算
    """
    mask = (depth_gt_bins > 0)
    if mask.sum() == 0:
        return depth_logits.sum() * 0.0
        
    # [M, D] 提取有效雷达点位置的预测
    valid_logits = depth_logits.permute(0, 2, 3, 1)[mask]
    # [M] 标签需减 1 匹配 0~D-1 索引
    valid_targets = depth_gt_bins[mask].long() - 1
    
    return F.cross_entropy(valid_logits, valid_targets, reduction='mean')`,
    "gotchas": "1. 激光雷达点投影的前后景击穿：LiDAR 点穿过车辆玻璃击中远处建筑物，会导致近处车窗上打上远处的深度标签；必须采用深度近邻一致性滤波器剔除穿透杂点。\n2. 深度离散化边缘截断误差：将连续深度 (如 12.35m) 离散为固定的 bin 索引，必然引入量化误差；BEVDepth 后续引入了软标签高斯模糊平滑 (One-Hot to Gaussian Soft-Target)。",
    "interview_qa": {
      "question": "面试官常问：纯视觉 BEV 算法在训练时用了激光雷达点云做显式深度监督，还能叫'纯视觉算法'吗？推理时需要雷达吗？为什么能大幅提升 NDS？",
      "answer": "【答题结构与行业共识】：\n1. 严谨定性：仍然属于【纯视觉感知算法】！\n   - 原因：LiDAR 点云仅仅作为离线训练阶段的'特权导师信息 (Privileged Information)'提供辅助监督 Loss。在推理部署阶段，测试流程完全不需要 LiDAR 输入，输入端只有 6 路相机图像。\n2. NDS 提升 4~6 个点的核心根因：\n   - 纯视觉单目测距是病态逆问题 (Ill-Posed)。隐式梯度反传时，网络难以区分'物体太小'还是'物体太远'。\n   - 显式深度监督直接在 Backbone 输出端强行拉直了几何尺度的物理对应关系，使得 3D 空间中心定位误差 (mATE) 直降 0.1m 以上，彻底根除了远距离目标向车体挤压或悬浮的伪影。"
    }
  },
  "doppler_concept": {
    "title": "毫米波雷达多普勒径向速度测量 (Doppler Velocity)",
    "category": "多模态感知 (Multi-Sensor Fusion)",
    "tags": ["Radar", "Doppler", "Radial Velocity", "Multi-Modal"],
    "essence": "毫米波雷达 (Millimeter-Wave Radar, 如 Continental ARS-408) 发射调频连续波 (FMCW)，不仅能直接获取障碍物的距离和角度，还能基于多普勒效应 (Doppler Effect)，通过反射波与发射波之间的频移，在单帧单次测量中直接瞬时测出目标相对于雷达的径向相对速度 v_r。不同于激光雷达和纯视觉需要依靠跨帧时序差分才能估算速度，毫米波雷达具备极其抗恶劣天气 (大雾、暴雨、强逆光) 的瞬时测速物理优势。",
    "math": `$$f_d = \\frac{2 v_r}{\\lambda} = \\frac{2 v_r f_0}{c}$$
$$v_r = \\vec{v}_{\\text{rel}} \\cdot \\frac{\\vec{p}_{\\text{target}} - \\vec{p}_{\\text{radar}}}{\\|\\vec{p}_{\\text{target}} - \\vec{p}_{\\text{radar}}\\|} = v_x \\cos\\theta + v_y \\sin\\theta$$
$$v_r^{\\text{comp}} = v_r + v_{\\text{ego}} \\cos\\theta_{\\text{ego}}$$`,
    "tensor_dims": "nuScenes Radar 原始点云: [N, 18 维度]\n关键通道: X, Y, Z, vx_comp, vy_comp, rcs (雷达截面积), pdh0 (虚警概率)",
    "code": `def parse_radar_doppler(radar_point):
    """
    radar_point: [x, y, z, vx_comp, vy_comp, rcs, pdh0]
    """
    x, y = radar_point[0], radar_point[1]
    vx, vy = radar_point[3], radar_point[4]
    radial_dist = np.sqrt(x**2 + y**2)
    # 径向投影速度
    v_radial = (x * vx + y * vy) / max(radial_dist, 1e-4)
    return v_radial`,
    "gotchas": "1. 径向速度的盲区：毫米波雷达只能测出沿光线连线方向的径向速度 $v_r$。如果一辆车正在横向横穿马路 (垂直于雷达视线)，其径向速度为 0！此时雷达无法仅凭单点测出横向速度。\n2. 虚警杂波与多径反射：毫米波雷达波束较宽，极易被地面井盖、易拉罐、护栏引发多径反射 (Multipath Reflection)，产生大量漂浮在地下的幽灵虚警点，必须依赖 RCS (雷达散射截面积) 与置信度掩码过滤。\n3. 点云极度稀疏且高度不准：nuScenes 单帧雷达点通常不足 100 个，且垂直仰角分辨率极低甚至为 0 (2D 平面雷达)。",
    "interview_qa": {
      "question": "面试官常问：车载毫米波雷达只能测量径向速度，算法是如何求解出目标在自车坐标系下的完整 2D 绝对速度矢量 $(v_x, v_y)$ 的？",
      "answer": "【答题解法与工程落地】：\n1. 方法 A (多雷达多角度几何相交解算)：\n   - 自车通常配备 5 颗毫米波雷达 (1 前向长距 + 4 角雷达)。当目标处于两个不同雷达的重叠视场中时，得到两条不同夹角的径向速度方程：\n     $v_{r1} = v_x \\cos\\theta_1 + v_y \\sin\\theta_1, \\quad v_{r2} = v_x \\cos\\theta_2 + v_y \\sin\\theta_2$。\n     联立线性二元方程组直接精确解出 $(v_x, v_y)$。\n2. 方法 B (多帧卡尔曼滤波 / 深度网络时序跨帧关联)：\n   - 若目标仅落入单颗雷达视野，必须依赖跨帧位置位移 $\\Delta x / \\Delta t$ 作为横向速度的补充观测，在 EKF (扩展卡尔曼滤波) 状态转移中联合更新；或将雷达点作为 Token 注入 Transformer，由注意力头完成跨视角特征融合。"
    }
  },
  "pointcloud_deskew_concept": {
    "title": "激光点云运动去畸变 (Motion Undistortion / Deskewing)",
    "category": "多模态感知 (Multi-Sensor Fusion)",
    "tags": ["LiDAR", "Deskew", "Motion Distortion", "IMU Integration"],
    "essence": "机械旋转式激光雷达 (如 Velodyne HDL-32E) 是通过内部电机高速旋转 (通常 10Hz~20Hz，单圈旋转耗时约 50ms~100ms) 连续发射激光线束来扫描周围环境的。在单圈扫描的 100ms 过程中，自车正在以一定速度持续向前行驶或转向。如果在生成点云时假设所有点都是在同一瞬时'快门'采集的，就会导致早扫描的点和晚扫描的点处于不同的空间位姿，造成原本笔直的马路护栏呈现香蕉状弯曲变形，静止圆形电线杆被拉扯为椭圆。运动去畸变利用高频 IMU/里程计位姿，将单帧内每个点插值还原到同一基准时刻 (通常是扫描结束时刻)。",
    "math": `$$t_{\\text{pt}} \\in [t_{\\text{start}}, t_{\\text{end}}], \\quad s = \\frac{t_{\\text{pt}} - t_{\\text{start}}}{t_{\\text{end}} - t_{\\text{start}}} \\in [0, 1]$$
$$T(t_{\\text{pt}}) = \\text{Slerp}\\left( T(t_{\\text{start}}), T(t_{\\text{end}}), s \\right)$$
$$P_{\\text{deskewed}} = T(t_{\\text{end}})^{-1} \\cdot T(t_{\\text{pt}}) \\cdot P_{\\text{raw}}$$`,
    "tensor_dims": "原始点云带时间戳: [N, 5] (X, Y, Z, Intensity, Relative_Timestamp)\nIMU 高频位姿流: 100Hz ~ 200Hz 轨迹\n修正后对齐点云: [N, 5]",
    "code": `def deskew_lidar_points(points, timestamps, ego_poses_stream):
    """
    点云运动去畸变：基于时间戳对每个激光点进行位姿插值与逆变换
    """
    # 1. 计算每个点的相对扫描时间比例 s
    # 2. 四元数球面插值 Slerp 获得该时刻自车精确位姿
    # 3. 将点变换至帧尾统一自车坐标系
    pass`,
    "gotchas": "1. 逐点插值的计算瓶颈：单帧点云包含数万甚至数十万点，若在 CPU 端对每个点计算 Slerp 四元数插值，耗时高达 20ms 以上；工业界通常在 CUDA 中编写并行核函数，或将扫描周期近似等分为几十个小段 (Piecewise Linear Approximation)。\n2. 动态目标的双重畸变：去畸变只能消除自车运动导致的几何畸变！若被测目标自身也在高速运动 (如迎面驶来的车)，其自身的运动拉伸依然无法被自车 IMU 去除，必须结合多目标跟踪速度矢量修正。",
    "interview_qa": {
      "question": "面试官常问：如果自动驾驶车辆以 120km/h (约 33.3m/s) 高速行驶，激光雷达以 10Hz 旋转扫描，不做运动去畸变，一帧点云首尾点之间会产生多大的物理位移误差？",
      "answer": "【答题秒算与结论】：\n1. 物理位移计算：\n   - 10Hz 旋转周期 $T = 0.1\\text{s} = 100\\text{ms}$。\n   - 车辆速度 $v = 120\\text{ km/h} = 33.33\\text{ m/s}$。\n   - 在单圈扫描的首点与尾点之间，车辆向前平移位移：\n     $\\Delta L = v \\cdot T = 33.33 \\times 0.1 = 3.33\\text{ 米}$！\n2. 严重工程后果：\n   - 一辆原本长 4.5 米的小轿车在点云中会被拉伸或压缩超过 3 米，直接导致 3D 边界框回归彻底崩溃，甚至导致紧急制动 AEB 误触发或刹车不及造成追尾。因此高速自动驾驶中点云去畸变是绝对不可逾越的前置底线。"
    }
  },
  "nds_metric_concept": {
    "title": "nuScenes 综合检测得分 (NDS) 评测体系",
    "category": "评测基准 (Benchmarking)",
    "tags": ["NDS", "mAP", "nuScenes", "mATE", "mASE", "mAOE", "mAVE", "mAAE"],
    "essence": "NDS (nuScenes Detection Score) 是自动驾驶行业公认的 3D 目标检测黄金综合评测指标。由于传统的 2D 目标检测仅关注重叠度 mAP，而自动驾驶安全对于物体的 3D 真实距离定位、航向角朝向判定、尺寸估计和速度预测有着极其严苛的物理要求，单一指标无法代表算法的安全能力。NDS 将均值检测精度 (mAP) 与 5 项关键物理真值误差 (True Positive Metrics) 进行加权融合，构成 0~1 之间的综合得分。",
    "math": `$$\\text{NDS} = \\frac{1}{10} \\left[ 5 \\cdot \\text{mAP} + \\sum_{\\text{error} \\in \\mathcal{E}} \\max(0, 1 - \\text{TP}_{\\text{error}}) \\right]$$
$$\\mathcal{E} = \\{ \\text{mATE (平移)}, \\text{mASE (尺度)}, \\text{mAOE (朝向)}, \\text{mAVE (速度)}, \\text{mAAE (属性)} \\}$$`,
    "tensor_dims": "mAP: 4 个中心距离阈值 (0.5m, 1m, 2m, 4m) 计算\nTP Errors: mATE (米), mASE (1-IoU), mAOE (弧度 rad), mAVE (米/秒), mAAE (1-Acc)",
    "code": `def calculate_nds(map_val, mate, mase, maoe, mave, maae):
    """
    官方 NDS 计算标准：
    5 项误差项均做 1 - error 裁剪截断
    """
    tp_score = (
        max(0.0, 1.0 - mate) +
        max(0.0, 1.0 - mase) +
        max(0.0, 1.0 - maoe) +
        max(0.0, 1.0 - mave) +
        max(0.0, 1.0 - maae)
    )
    nds = (5.0 * map_val + tp_score) / 10.0
    return nds`,
    "gotchas": "1. mAOE 角度翻转：如果预测框的朝向刚好搞反了 180° (如车头和车尾混淆)，此时 3D IoU 可能完全不受影响，但 mAOE 会被判定为 $\\pi \\approx 3.14$ 弧度，其贡献项 $\\max(0, 1 - 3.14) = 0$，对 NDS 惩罚极其严厉。\n2. 单帧模型 mAVE 归零惩罚：如果不加入时序融合 (如最基础的单帧 BEVDet)，模型无法预测连续运动速度，mAVE 误差会高达数米/秒甚至无法评估，NDS 会天然落后多帧时序模型 5~10 个百分点以上。",
    "interview_qa": {
      "question": "面试官常问：为什么 nuScenes 不直接使用 KITTI 的 3D IoU 计算 mAP，而要使用中心距离阈值 (Center Distance)？NDS 公式中各项权重的工程考量是什么？",
      "answer": "【答题核心机理与行业演进】：\n1. 为什么弃用 3D IoU 计算 mAP：\n   - KITTI 的 3D Bounding Box IoU 判定极其严苛：若一辆 80 米外的远距离车辆，预测框仅在长宽高或航向角上有轻微几度的微小偏差，其 3D IoU 很容易直接跌破 0.7 阈值被判定为完全未检出 (FN)。\n   - 这违背了自动驾驶安全控制的常识：对于远距离目标，控制系统最先需要知道的是'那里有没有车、大概在什么绝对距离'，微小的长宽尺寸误差并不影响早期减速。nuScenes 改用 2D 地面中心距离 (0.5m, 1m, 2m, 4m) 判定，解耦了'检测召回'与'姿态尺寸精度'。\n2. NDS 权重设计的深层考量：\n   - mAP 占 50% 核心权重，保证目标召回是首要生命线；\n   - 其余 5 项物理误差各占 10%，严防'虽然召回但测距偏了 3 米'或'速度反向'这种危及生命的重大安全隐患。"
    }
  },
  "ipm_concept": {
    "title": "逆透视映射 (Inverse Perspective Mapping, IPM)",
    "category": "经典几何方法 (Traditional Geometry)",
    "tags": ["IPM", "Homography", "Flat-Ground", "Perspective"],
    "essence": "逆透视映射 (IPM) 是在深度学习普及前广泛使用的经典计算机视觉几何视图变换方法。IPM 建立在严格的'平地假设 (Flat-Ground Assumption)'之上：假设相机前方所有被观察的物体表面都严格落在水平地面 $Z_{\\text{ground}} = 0$ 上。在此刚性约束下，相机透视投影退化为一个可逆的 $3\\times 3$ 单应性矩阵 (Homography Matrix $H$)。IPM 计算极其轻量且无需深度估计，但任何非平地物体 (如立体车辆、行人、起伏路面) 都会在 IPM 图像上产生灾难性的放射状拉伸和'飞天'伪影。",
    "math": `$$H = K \\cdot \\left( R_{\\text{cam}} - \\frac{t_{\\text{cam}} n^T}{d} \\right)^{-1}$$
$$\\begin{bmatrix} x_{\\text{bev}} \\\\ y_{\\text{bev}} \\\\ 1 \\end{bmatrix} \\sim H^{-1} \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix}$$`,
    "tensor_dims": "单应性变换矩阵 H: [3, 3] float32\n输入图像: [H, W, 3]\n输出 BEV 图像: [H_bev, W_bev, 3]",
    "code": `import cv2
import numpy as np

def apply_ipm(img, src_pts, dst_pts, out_size=(500, 500)):
    """
    传统 IPM 单应矩阵计算与透视变换
    """
    H, _ = cv2.findHomography(src_pts, dst_pts)
    bev_img = cv2.warpPerspective(img, H, out_size)
    return bev_img`,
    "gotchas": "1. 立体物体的严重拉伸失真：对于具有高度的非平地目标 (如树木或前车)，其顶部像素由于距离地面的法向距离不为 0，会被单应矩阵向外极度拉扯成数十米长的'面条状'模糊尾巴。\n2. 坡道与悬架起伏失效：当道路存在上下坡或车辆减速点头时，平地假设被彻底推翻，IPM 标线网格会整体前后错位 5~10 米。",
    "interview_qa": {
      "question": "面试官常问：既然 IPM 存在这么严重的高度物体拉伸问题，现代自动驾驶系统中是否还有 IPM 的用武之地？",
      "answer": "【答题辩证分析与工业实况】：\n1. 明确结论：在 3D 动态目标检测中已彻底被 LSS / BEVFormer 取代，但在【车道线检测 (Lane Detection)】与【泊车环视拼接 (AVM 360° 全景)】中依然是不可替代的基石！\n2. 为什么车道线和泊车依然重度使用 IPM：\n   - 车道线、停止线、斑马线本身就是物理喷涂在地表上的二维平面印记，完美契合 $Z=0$ 平地假设，几乎不产生高度失真；\n   - IPM 计算量极低，在低算力 MCU 或车机座舱芯片上仅需微秒级即可完成单应性双线性重采样，提供极致低延迟的环视拼接倒车影像。"
    }
  },
  "keyframe_sweep_concept": {
    "title": "Keyframe (关键帧 2Hz) 与 Sweeps (时序子帧 12/20Hz)",
    "category": "数据闭环 (Data Infrastructure)",
    "tags": ["nuScenes", "Keyframe", "Sweeps", "Sensor Sync"],
    "essence": "nuScenes 数据集采用了独特的两级时序分层组织架构。自动驾驶数据采集频率各不相同 (LiDAR 通常为 20Hz，相机为 12Hz)。nuScenes 并没有对所有采集帧进行高代价人工标注，而是以大约 2Hz (即每 0.5 秒一帧) 的频率抽取一帧作为【关键帧 (Keyframe / sample)】，并仅对关键帧提供全 360° 极其精准的 3D 边界框人工标注。在两个关键帧之间高频采集的未标注帧被称为【时序子帧 (Sweeps / sample_data)】。工业算法必须利用自车运动补偿，将过去多个 Sweeps 的点云或图像特征聚合叠加到 Keyframe 上，以实现点云稠密化与长时序感知。",
    "math": `$$T_{\\text{keyframe}} \\approx 0.5\\text{s} \\quad (f = 2\\text{Hz})$$
$$T_{\\text{lidar\\_sweep}} = 0.05\\text{s} \\quad (f = 20\\text{Hz}), \\quad T_{\\text{cam\\_sweep}} \\approx 0.083\\text{s} \\quad (f = 12\\text{Hz})$$
$$P_{\\text{multi\\_sweep}} = \\bigcup_{k=0}^{M-1} T_{\\text{sweep}(k) \\to \\text{keyframe}}^{\\text{ego}} \\cdot P_{\\text{sweep}(k)}$$`,
    "tensor_dims": "单帧 Keyframe LiDAR 点云: ~34,700 点 [N, 5]\n累积 10 帧 LiDAR Sweeps 点云: ~300,000 点 [N', 5]\n单场景连续时长: 20 秒 (约 40 个 Keyframes + ~800 个 Sweeps)",
    "code": `def accumulate_lidar_sweeps(nusc, sample_token, max_sweeps=10):
    """
    nuScenes 官方点云多帧叠加逻辑 (从当前 Keyframe 向前追溯历史 Sweeps)
    """
    # 1. 读取当前 sample 的 LIDAR_TOP sample_data
    # 2. 循环遍历 sd['prev']
    # 3. 利用各 sweep 的 calibrated_sensor 与 ego_pose 变换到当前 Keyframe 坐标系
    # 4. 点云拼接并在特征最后追加 relative_timestamp 通道
    pass`,
    "gotchas": "1. 传感器物理时间戳不同步 (Sensor Jitter)：相机的曝光时间戳与激光雷达扫描中心时间戳存在数毫秒到十几毫秒的固定偏差，不能直接假定两者是绝对同时发生。\n2. 动态目标在 Sweeps 叠加中的残影拖尾：若直接将 10 个 sweeps 的点云叠加，高速行驶的车辆在点云中会变成一条长长的'毛毛虫'点云束，若网络无法区分点的时间戳，会导致检测框尺寸估计严重偏长。",
    "interview_qa": {
      "question": "面试官常问：在训练多模态或纯激光雷达 3D 检测网络时，通常会将历史 10 个 Sweeps 叠加成一帧送入网络，这会带来什么收益？网络如何知道哪个点是新的、哪个点是旧的？",
      "answer": "【答题要点】：\n1. 核心收益：\n   - 单帧 Velodyne 32 线激光点云极度稀疏 (约 3.4 万点)，远距离障碍物往往只命中 2~3 个点，几何轮廓无法辨识；\n   - 叠加 10 个 Sweeps 后，点云数量暴增至近 30 万点，远距离车辆轮廓被补全得极其致密，显著提升远距离召回率。\n2. 网络如何辨别新旧时间戳：\n   - 官方标准解法是向点云增加第 5 维特征：相对时间戳差值 $\\Delta t = (t_{\\text{current}} - t_{\\text{sweep}}) \\in [0, 0.5\\text{s}]$；\n   - 现代 Backbone (如 PointPillars, VoxelNet, SparseConvolution) 能直接感知该时间维度特征，从而在利用空间密度的同时，精准推断出目标的瞬时速度矢量而不会被拖尾干扰。"
    }
  },
  "centerhead_concept": {
    "title": "CenterHead 无锚框 3D 目标检测头与高斯热力图",
    "category": "3D 检测头 (Detection Head)",
    "tags": ["CenterHead", "Anchor-Free", "Heatmap", "Gaussian", "BEVDet"],
    "essence": "CenterHead (CVPR 2021) 是自动驾驶 BEV 检测领域应用最广的无锚框 (Anchor-Free) 检测头。传统 3D 锚框方法需要预设数十种不同尺寸、长宽比和旋转角度的 3D Prior Anchors，正负样本分配复杂且难以扩展。CenterHead 将目标表示为一个点：在 2D BEV 特征图上，目标中心位置由高斯热力图 (Gaussian Heatmap) 表征，网络的各个轻量回归分支分别预测该中心点对应的 3D 物理属性 (深度 Z, 尺寸 3D Size, 航向角朝向 Yaw, 速度 Velocity)。极大地简化了检测管线并加速收敛。",
    "math": `$$Y_{c, x, y} = \\exp\\left( -\\frac{(x - \\tilde{x})^2 + (y - \\tilde{y})^2}{2\\sigma^2} \\right)$$
$$\\mathcal{L}_{\\text{heatmap}} = -\\frac{1}{N} \\sum_{x, y} \\begin{cases} (1 - \\hat{Y})^{\\alpha} \\log(\\hat{Y}) & \\text{if } Y=1 \\\\ (1 - Y)^{\\beta} \\hat{Y}^{\\alpha} \\log(1 - \\hat{Y}) & \\text{otherwise} \\end{cases}$$`,
    "tensor_dims": "BEV 输入特征: [B, C=256, H=128, W=128]\n热力图预测: [B, NumClasses=10, H, W] (0~1 概率)\n回归头输出: center_offset [2], height [1], dim [3], rot [2], vel [2] (共 10 维)",
    "code": `import torch
import torch.nn as nn

class CenterHead3D(nn.Module):
    def __init__(self, in_channels=256, num_classes=10):
        super().__init__()
        self.heatmap_head = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_head     = nn.Conv2d(in_channels, 2, 3, padding=1) # (dx, dy)
        self.height_head  = nn.Conv2d(in_channels, 1, 3, padding=1) # z
        self.dim_head     = nn.Conv2d(in_channels, 3, 3, padding=1) # (w, l, h)
        self.rot_head     = nn.Conv2d(in_channels, 2, 3, padding=1) # (sin, cos)
        self.vel_head     = nn.Conv2d(in_channels, 2, 3, padding=1) # (vx, vy)
        
    def forward(self, x):
        return {
            'heatmap': self.heatmap_head(x).sigmoid(),
            'offset': self.reg_head(x),
            'height': self.height_head(x),
            'dim': torch.exp(self.dim_head(x)),
            'rot': torch.nn.functional.normalize(self.rot_head(x), dim=1),
            'vel': self.vel_head(x)
        }`,
    "gotchas": "1. 特征 stride 导致的量化下采样损失：BEV 特征图通常经过 8 倍或 16 倍下采样，一个 BEV 像素对应物理世界 $0.8\\text{m}$，若不预测连续的亚像素偏移 (Sub-pixel Offset $\\Delta x, \\Delta y$)，中心定位会产生严重的阶梯化误差。\n2. 密集重叠目标的热力图压制：两辆车紧挨着停放时，两个高斯圆峰容易粘连成一个单峰，导致 NMS 提取峰值时漏检其中一辆。",
    "interview_qa": {
      "question": "面试官常问：CenterHead 是如何完成从连续热力图到离散 3D 边界框的后处理解码的？需要使用旋转 NMS 吗？",
      "answer": "【答题流程与工程优化】：\n1. 极值点提取 (Pooling as NMS)：\n   - CenterHead 提出了一种极度优雅的 GPU 算子替代传统 NMS：直接对热力图应用一个 $3\\times 3$ 的最大池化 (MaxPool2d(3, stride=1, padding=1))；\n   - 只有当原热力图值与池化后值完全相等的像素点，才是局部极大值中心点 (Local Peak)！一句代码在 GPU 上并发完成峰值提取。\n2. 取 Top-K 索引：按热力图分数在全图提取 Top-500 个候选点。\n3. 直接索引回归属性：根据候选点的 2D 网格坐标 $(u, v)$，直接从尺寸、高度、旋转和速度张量中取出对应位置的浮点数，直接解码出 3D Box。\n4. 工业级是否还需 NMS：对于大部分情况，MaxPool 已经过滤了绝大多数重复框；但在复杂密集场景，通常会再追加一层阈值极宽松的轻量 2D 旋转 NMS (如 IoU 阈值 0.2) 作为最终安全防线。"
    }
  },
  "deformable_attn_concept": {
    "title": "可形变注意力机制 (Deformable Attention)",
    "category": "Transformer 范式 (BEVFormer)",
    "tags": ["Deformable Attention", "MSDeformAttn", "Sparse Sampling", "Linear Complexity"],
    "essence": "可形变注意力 (Deformable Attention, Zhu et al., ICLR 2021) 是将 Vision Transformer 成功推广到高分辨率 2D 检测与 3D BEV 感知的最关键基石算子。传统 Transformer 的全局自注意力机制计算复杂度随图像分辨率呈二次方增长 $O((HW)^2)$，处理多视角环视高分辨率特征时显存瞬间爆炸。可形变注意力借鉴了可形变卷积 (DCN) 的物理思想：让查询向量主动在参考点周围学习一组稀疏的偏移量 (Sampling Offsets, 通常每个头仅采 4 个点) 和对应的注意力权重，将计算复杂度从二次方骤降为与序列长度成严格线性的 $O(N_q \\cdot K)$。",
    "math": `$$\\text{DeformAttn}(z_q, p_q, x) = \\sum_{m=1}^M W_m \\left[ \\sum_{k=1}^K A_{mqk} \\cdot W'_m x(p_q + \\Delta p_{mqk}) \\right]$$
$$\\mathcal{O}(\\text{Global Attn}) = O(N_q^2 \\cdot C) \\quad \\Longrightarrow \\quad \\mathcal{O}(\\text{DeformAttn}) = O(N_q \\cdot M \\cdot K \\cdot C)$$`,
    "tensor_dims": "Query 数量: N_q = 22500\n注意力头数: M = 8\n每个头采样点数: K = 4 (极度稀疏，仅 4 个浮点坐标)\n多尺度特征图: N_levels = 4 (1/8, 1/16, 1/32, 1/64)",
    "code": `// MSDeformAttn 核心 CUDA 前向逻辑原理 (C++ 伪代码)
// 输入: value 稠密特征, spatial_shapes 尺度形状, sampling_loc 采样坐标, attn_weights 权重
// 输出: 仅在 sampling_loc 处执行双线性插值 (Bilinear Interpolation) 并乘以 attn_weights 累加`,
    "gotchas": "1. 必须依赖编译 C++/CUDA 插件：PyTorch 原生并没有可形变注意力的官方实现，必须编译 `mmdet3d` 或 `ops/modules/ms_deform_attn.py` 的底层 CUDA 核函数；在 Windows 或非标准 CUDA 环境下编译经常出现版本不兼容崩溃。\n2. 采样偏移越界截断：当网络学习不稳定时，生成的偏移量可能极大，导致采样坐标漂移到特征图画布之外，CUDA 算子内部必须实现安全的 clamp 零边界处理。\n3. FP16 TensorRT 算子适配难：将可形变注意力转为车端 TensorRT 插件时，双线性插值的索引计算如果精度丢失，会导致微小网格对齐误差。",
    "interview_qa": {
      "question": "面试官常问：可形变注意力 (Deformable Attention) 相比可形变卷积 (DCNv2)，在物理机制和数学表达上有何异同？为什么 BEVFormer 选它？",
      "answer": "【答题对比与核心本质】：\n1. 共同点：两者都突破了传统方形网格的固定几何采样约束，利用网络自身动态学习数据驱动的空间几何形变偏移量 (Offsets)。\n2. 本质区别：\n   - DCNv2 (卷积范式)：采样点数量由卷积核尺寸固定 (如 $3\\times 3=9$ 个点)，其权重是静态卷积核权重乘以调制标量，属于局部特征变换。\n   - Deformable Attention (Transformer 范式)：采样权重 $A_{mqk}$ 是由 Query 与全局内容动态点积 Softmax 动态生成的！真正实现了内容相关的动态注意力，且能天然无缝地跨越多个不同分辨率的特征尺度 (Multi-Scale) 和不同物理相机视角执行跨图采样。\n3. BEVFormer 选择它的根本原因：多相机拼接时，物理目标在图像上的投影点跨越了 6 路相机的高清画面，只有多尺度可形变注意力能以微小的固定计算预算跨视角按需采样。"
    }
  },
  "hungarian_concept": {
    "title": "匈牙利匹配与二分图匹配损失 (Bipartite Matching)",
    "category": "3D 检测头 (Detection Head)",
    "tags": ["DETR", "Hungarian Matching", "Bipartite", "End-to-End", "Sparse4D"],
    "essence": "匈牙利匹配 (Hungarian Algorithm) 是 DETR 系列及现代稀疏 3D 检测网络 (如 Sparse4D, PETR, StreamPETR) 实现'端到端无 NMS'目标检测的数学基石。在传统方法中，一个真值目标会由多个重叠的预设锚框共同负责预测，导致输出大量重复框，必须依赖手工设计的 NMS 剔除。匈牙利匹配将检测建模为二分图的最佳一一匹配问题 (Bipartite Matching)：利用匈牙利算法在多项式时间内找到令全局分类与几何回归代价最小的唯一排列映射，强迫每个真实目标只由唯一的一个 Query 负责预测。",
    "math": `$$\\hat{\\sigma} = \\arg\\min_{\\sigma \\in \\mathfrak{S}_N} \\sum_{i=1}^N \\mathcal{L}_{\\text{match}}\\left( y_i, \\hat{y}_{\\sigma(i)} \\right)$$
$$\\mathcal{L}_{\\text{match}}(y_i, \\hat{y}_j) = - \\mathbf{1}_{\\{c_i \\neq \\varnothing\\}} p_j(c_i) + \\mathbf{1}_{\\{c_i \\neq \\varnothing\\}} \\left( \\lambda_{\\text{L1}} \\|b_i - \\hat{b}_j\\|_1 + \\lambda_{\\text{iou}} \\mathcal{L}_{\\text{iou}}(b_i, \\hat{b}_j) \\right)$$`,
    "tensor_dims": "真值目标数量: M (例如 40 个真实障碍物)\n模型预测 Query 数量: N = 900\n代价矩阵 Cost Matrix: [M, N] float32\n最佳匹配映射: M 个独占的 Query 索引",
    "code": `import torch
from scipy.optimize import linear_sum_assignment

def hungarian_matcher(pred_boxes, pred_logits, gt_boxes, gt_labels):
    """
    基于 Scipy 的二分图匹配求解
    """
    # 1. 计算分类代价 (Focal Loss / 概率负对数)
    # 2. 计算 3D 边界框 L1 距离与 3D IoU 代价
    # 3. 构造代价矩阵 C = C_cls + C_box [M, N]
    # 4. 调用匈牙利算法求解全局最优分配
    # row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
    pass`,
    "gotchas": "1. 训练初期匹配震荡 (Matching Instability)：在训练前几个 Epoch，Query 的预测处于随机震荡状态，同一个真实物体在相邻两轮迭代中会被分配给完全不同的 Query，导致梯度剧烈冲突，模型收敛极慢；DN-DETR (去噪训练) 引入去噪分支来缓解这一震荡。\n2. CPU-GPU 频繁数据同步：标准的 `linear_sum_assignment` 运行在 CPU 上，必须将 GPU 的代价矩阵拉回内存计算，拖慢训练迭代；现代框架多采用基于 CUDA 的并行 Auction 算法近似求解。",
    "interview_qa": {
      "question": "面试官常问：DETR 架构使用匈牙利匹配实现了端到端无需 NMS，但在自动驾驶实际车端部署中，为什么很多人还是会在输出端挂一个轻量 NMS？",
      "answer": "【答题深度与实战经验】：\n1. 理论与现实的脱节：理论上，一对一二分图匹配强迫每个真实物体只对应一个高分 Query，背景 Query 分数趋近于 0，确实天然不需要 NMS。\n2. 落地中的极端场景失效：\n   - 当面对极度拥挤、部分遮挡或尺度极大的重型挂车时，由于特征模糊，两个相邻的 Query 可能会同时对该车辆产生 0.7 左右的高置信度预测；\n   - 在车端下游规划控制 (Planning) 系统眼中，规划算法极度脆弱，如果在同一物理位置并排出现两个几乎重合的 3D 框，会导致控制规划器计算碰撞干涉直接触发紧急制动或左右摇摆。\n3. 工程双保险：因此在量产交付方案中，感知工程师通常会在 DETR 输出后追加一道 IoU 阈值极高 (如 0.6~0.7) 的'防御性 NMS'，只在发生严重重复异常时做兜底，确保控制算法万无一失。"
    }
  },
  "tensorrt_fp16_concept": {
    "title": "TensorRT 部署加速与 FP16 精度溢出防御",
    "category": "工程部署 (Production & Deployment)",
    "tags": ["TensorRT", "FP16", "Overflow", "Orin", "Quantization"],
    "essence": "TensorRT 是英伟达针对其 GPU 硬件 (如车载计算平台 Drive Orin / Xavier) 研发的高性能深度学习推理 SDK。将 PyTorch 训练好的 BEV/OCC 模型导出为 ONNX 并编译为 TensorRT Engine 是工业落地的必经之路。在开启 FP16 半精度加速时，浮点数的动态表示范围从 FP32 的 $10^{-38} \\sim 10^{38}$ 骤降为 $6\\times 10^{-5} \\sim 65504$。若模型中的 Attention Softmax、LayerNorm 或指数尺度回归算子未经特殊设计，极易发生上溢出 (变成 NaN / Inf) 或下溢出 (被截断为 0)，导致整张 BEV 输出全部报废。",
    "math": `$$\\text{FP16 最大值: } 65504, \\quad \\text{FP16 最小值 (非次正规): } 6.10 \\times 10^{-5}$$
$$\\text{Softmax 防溢出: } \\text{Softmax}(x_i) = \\frac{\\exp(x_i - \\max(x))}{\\sum_j \\exp(x_j - \\max(x))}$$`,
    "tensor_dims": "FP32 权重: 4 字节/参数\nFP16 权重: 2 字节/参数 (显存与带宽开销直接减半)\nINT8 量化: 1 字节/参数 (需 PTQ/QAT 校验)",
    "code": `// TensorRT 插件或 PyTorch 导出中的数值安全防护
torch::Tensor safe_fp16_exp(torch::Tensor x) {
    // 强制截断最大输入，防止 exp(x) 超过 65504 导致 FP16 溢出为 Inf
    x = torch::clamp(x, -20.0f, 10.0f); // exp(10) ≈ 22026 < 65504
    return torch::exp(x);
}`,
    "gotchas": "1. Softmax 点积除以 $\\sqrt{d_k}$ 的次序：在计算 $Q K^T / \\sqrt{d_k}$ 时，若先做矩阵乘法再除以 $\\sqrt{d_k}$，未缩放的点积极大值极易超过 65504 瞬间产生 NaN；必须先对 $Q$ 除以 $\\sqrt{d_k}$ 缩放后再做矩阵乘。\n2. 动态 Shape 编译开销：车载环视相机的输入分辨率通常是固定的，切忌在 TensorRT 中开启全动态多尺度 Shape，这会大幅降低 TRT 针对固定访存对齐执行针对性算子融合 (Kernel Fusion) 的优化空间。",
    "interview_qa": {
      "question": "面试官常问：你在将 BEV 感知模型从 PyTorch 迁移部署到 TensorRT FP16 时，遇到过精度掉点或推理出现 NaN 吗？你是如何排查和解决的？",
      "answer": "【答题排查链路与标准化解决套路】：\n1. 现象定位方法：\n   - 使用 Polygraphy 工具逐层对比 PyTorch 与 TRT FP16 算子层输出的余弦相似度与绝对误差差值 (L1 Error)，快速二分法锁定最早出现 NaN 或精度突变的异常算子层。\n2. 常见三大多发灾区与根治策略：\n   - 灾区一：注意力 Softmax 前的大数值点积。对策：开启 TRT 插件层内 FP32 强制累加与最大值减法保护。\n   - 灾区二：LayerNorm 内部的方差开方 $\\sqrt{\\sigma^2 + \\epsilon}$。当输入极为平坦时方差极小，FP16 下溢为 0 导致除零。对策：在导出 ONNX 前将 LayerNorm 强制保留为 FP32 运算模式 (Mixed-Precision Fallback)。\n   - 灾区三：3D 边界框尺寸回归的 $\\exp(\\Delta)$ 乘性因子。对策：在网络 Head 输出前施加 clamp 保护，限制指数底数在合理区间。"
    }
  }
};

/**
 * Open Concept Modal with full LaTeX math rendering, code highlighting, and interview prep
 */
function openConceptModal(conceptKey) {
  const data = CONCEPT_REGISTRY[conceptKey];
  if (!data) {
    console.warn('[ConceptModal] Unknown concept:', conceptKey);
    return;
  }

  const modal = document.getElementById('concept-modal');
  const title = document.getElementById('concept-modal-title');
  const body = document.getElementById('concept-modal-body');
  const badge = document.getElementById('concept-modal-badge');

  if (!modal || !title || !body) return;

  title.innerText = data.title;
  if (badge) badge.innerText = data.category.toUpperCase();

  const tagsHtml = data.tags.map(t => `<span class="badge" style="background: rgba(0,229,255,0.1); color: var(--accent-cyan); border: 1px solid rgba(0,229,255,0.3); font-size: 0.72rem;">#${t}</span>`).join(' ');

  body.innerHTML = `
    <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;">
      ${tagsHtml}
    </div>

    <!-- Section 1: Physical & Geometric Intuition -->
    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.95rem; color: var(--accent-cyan); margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>💡</span> 物理与几何本质 (Physical Essence)
      </h4>
      <div style="font-size: 0.88rem; color: var(--text-primary); line-height: 1.65; background: rgba(255,255,255,0.02); padding: 0.85rem 1.1rem; border-radius: 6px; border-left: 3px solid var(--accent-cyan);">
        ${data.essence}
      </div>
    </div>

    <!-- Section 2: Mathematical Formulation -->
    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.95rem; color: var(--accent-purple); margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>📐</span> 核心数学推导与物理公式 (Mathematical Formulation)
      </h4>
      <div class="math-container" style="background: rgba(15, 23, 42, 0.85); padding: 1rem 1.25rem; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3); overflow-x: auto; font-size: 0.92rem; color: #E2E8F0;">
        ${data.math}
      </div>
    </div>

    <!-- Section 3: Tensor Dimensions & Flow -->
    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.95rem; color: var(--accent-green); margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>🔄</span> 典型张量维度流转 (Tensor Dimensions)
      </h4>
      <pre style="background: #020408; padding: 0.85rem 1.1rem; border-radius: 6px; border: 1px solid var(--border-subtle); overflow-x: auto; font-size: 0.82rem; color: #A7F3D0; font-family: monospace; line-height: 1.5;">${escapeHtml(data.tensor_dims)}</pre>
    </div>

    <!-- Section 4: Production PyTorch Code -->
    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.95rem; color: var(--accent-orange); margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>💻</span> 核心工业级 PyTorch 实现代码 (Implementation)
      </h4>
      <div style="position: relative;">
        <pre style="background: #050811; padding: 1rem 1.25rem; border-radius: 8px; border: 1px solid var(--border-subtle); overflow-x: auto; font-size: 0.83rem; color: #E2E8F0; line-height: 1.5; font-family: var(--font-code);"><code>${escapeHtml(data.code)}</code></pre>
      </div>
    </div>

    <!-- Section 5: Real-World Gotchas -->
    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.95rem; color: #FF5252; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>⚠️</span> 工业落地避坑指南 (Production Gotchas & Pitfalls)
      </h4>
      <div style="font-size: 0.85rem; color: #FECACA; background: rgba(239, 68, 68, 0.08); padding: 0.85rem 1.1rem; border-radius: 6px; border-left: 3px solid #EF4444; line-height: 1.6; white-space: pre-line;">
        ${data.gotchas}
      </div>
    </div>

    <!-- Section 6: Top-Tier AD Algorithm Interview Q&A -->
    <div style="margin-bottom: 0.5rem;">
      <h4 style="font-size: 0.95rem; color: #F59E0B; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>🎯</span> 大厂自动驾驶感知算法岗必考高频面试 Q&A
      </h4>
      <div style="background: rgba(245, 158, 11, 0.06); padding: 1rem 1.25rem; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3);">
        <div style="font-weight: 700; color: #FDE68A; font-size: 0.88rem; margin-bottom: 0.6rem;">
          ❓ ${data.interview_qa.question}
        </div>
        <div style="font-size: 0.84rem; color: var(--text-primary); line-height: 1.65; white-space: pre-line; background: rgba(0,0,0,0.3); padding: 0.85rem; border-radius: 6px;">
          ${data.interview_qa.answer}
        </div>
      </div>
    </div>
  `;

  modal.classList.add('active');

  // Trigger KaTeX dynamic re-render on the modal content
  if (typeof renderMath === 'function') {
    renderMath(body);
  } else if (typeof renderMathInElement === 'function') {
    renderMathInElement(body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false }
      ],
      throwOnError: false
    });
  }
}

// Global export
window.CONCEPT_REGISTRY = CONCEPT_REGISTRY;
window.openConceptModal = openConceptModal;
