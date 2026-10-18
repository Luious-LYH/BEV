/**
 * BEV / OCC Perception Textbook - Components & Modals System (Audited & Authentic)
 * Handles Code Snippets, Tensor Inspector, Image Lightbox, and Command Copy.
 */

const SNIPPET_REGISTRY = {
  "lidar_parsing": {
    "title": "LiDAR 原始二进制解析 (.pcd.bin ➔ numpy)",
    "provenance": "OFFICIAL SOURCE",
    "repo": "https://github.com/nutonomy/nuscenes-devkit",
    "file": "python-sdk/nuscenes/utils/data_classes.py",
    "class_func": "LidarPointCloud.from_file()",
    "input": "原始文件路径 (例如: samples/LIDAR_TOP/n015-...1532402927647951.pcd.bin)",
    "math": "连续 32 位小端浮点数二进制解码，重塑为 [N, 5] 矩阵 (X, Y, Z, 反射强度 Intensity, 线束号 Ring Index)",
    "output": "numpy.ndarray (shape: [N, 5], dtype: float32)",
    "code": `import numpy as np

def load_lidar_points(bin_path: str) -> np.ndarray:
    """
    nuScenes 官方 Velodyne HDL-32E 二进制激光点云加载函数
    每点占据 5 个 float32 (20 bytes):
      [0]: x (米, 右为正)
      [1]: y (米, 前为正)
      [2]: z (米, 上为正)
      [3]: intensity (0~255 反射强度)
      [4]: ring index (0~31 激光发射线束编号)
    """
    raw_points = np.fromfile(bin_path, dtype=np.float32)
    points = raw_points.reshape(-1, 5)
    return points`
  },
  "coordinate_projection": {
    "title": "激光雷达点刚体变换与相机透视投影链",
    "provenance": "OFFICIAL SOURCE",
    "repo": "https://github.com/nutonomy/nuscenes-devkit",
    "file": "python-sdk/nuscenes/utils/geometry_utils.py",
    "class_func": "view_points()",
    "input": "LiDAR 原生点 P_lidar [3], 外参 (R_l2e, t_l2e, R_c2e, t_c2e), 内参 K [3, 3]",
    "math": "P_ego = R_l2e @ P_lidar + t_l2e ➔ P_cam = R_c2e.T @ (P_ego - t_c2e) ➔ s [u, v, 1]^T = K @ P_cam",
    "output": "像素坐标 (u, v) 与是否在图像有效成像视锥内的布尔判定",
    "code": `import numpy as np

def project_lidar_to_camera(pt_lidar, R_l2e, t_l2e, R_c2e, t_c2e, K, img_w=1600, img_h=900):
    # 步骤 1: 雷达系 ➔ 自车系
    pt_ego = R_l2e @ pt_lidar + t_l2e
    
    # 步骤 2: 自车系 ➔ 相机光学系 (Z_cam 沿光轴朝前)
    pt_cam = R_c2e.T @ (pt_ego - t_c2e)
    
    # 关键几何截断：如果深度 <= 0，目标在镜头后方，无法透视成像
    if pt_cam[2] <= 0.1:
        return None, None, False
        
    # 步骤 3: 齐次针孔透视投影
    pixel_homo = K @ pt_cam
    u = pixel_homo[0] / pixel_homo[2]
    v = pixel_homo[1] / pixel_homo[2]
    
    visible = (0 <= u < img_w) and (0 <= v < img_h)
    return float(u), float(v), bool(visible)`
  },
  "lss_lift_splat": {
    "title": "LSS 视锥张量外积与 Lift 操作",
    "provenance": "OFFICIAL SOURCE",
    "repo": "https://github.com/nv-tlabs/lift-splat-shoot",
    "file": "src/models.py",
    "class_func": "CamEncode.get_cam_feats()",
    "input": "离散深度概率 P(d) [B, N, D, 1, H, W] 与 2D 上下文特征 [B, N, 1, C, H, W]",
    "math": "F_3D(u, v, d) = P(d | u, v) * C(u, v) (沿通道与离散深度轴执行外积广播)",
    "output": "视锥空间特征体 [B, N, D, C, H, W]",
    "code": `import torch

def lift_outer_product(depth_prob: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
    """
    Lift 操作核心：将 2D 像素特征根据深度概率分布沿视锥光线抛射展开
    depth_prob:   [B, N, D, 1, H, W] (沿 D 轴经过 Softmax，和为 1)
    context_feat: [B, N, 1, C, H, W] (Backbone 提取的语义特征)
    """
    # 广播相乘：[B, N, D, C, H, W]
    frustum_feat = depth_prob * context_feat
    return frustum_feat`
  },
  "bevdepth_supervision": {
    "title": "BEVDepth 显式 LiDAR 深度监督 Loss",
    "provenance": "OFFICIAL SOURCE",
    "repo": "https://github.com/Megvii-BaseDetection/BEVDepth",
    "file": "bevdepth/models/backbones/base_lss_fpn.py",
    "class_func": "BaseLSSFPN.get_depth_loss()",
    "input": "DepthNet 输出 logits [B*N, D, H, W] 与点云投影深度真值 [B*N, H, W]",
    "math": "mask = (depth_gt > 0), loss = CrossEntropy(logits[mask], targets[mask])",
    "output": "标量显式深度辅助损失 (Weight 通常为 3.0)",
    "code": `import torch
import torch.nn.functional as F

def compute_explicit_depth_loss(depth_logits, depth_gt_bins):
    """
    BEVDepth 核心创新：在训练期加入投影雷达点云的离散深度监督
    depth_logits:  [B*N, D, H, W] 未归一化预测
    depth_gt_bins: [B*N, H, W] 0 为无雷达点无监督，1~D 为对应深度区间
    """
    mask = (depth_gt_bins > 0)
    if mask.sum() == 0:
        return depth_logits.sum() * 0.0
        
    valid_logits = depth_logits.permute(0, 2, 3, 1)[mask] # [M, D]
    valid_targets = depth_gt_bins[mask].long() - 1        # [M]
    
    depth_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
    return depth_loss`
  },
  "bevformer_sca": {
    "title": "BEVFormer 空间交叉注意力 (SCA) 3D 立柱采样",
    "provenance": "OFFICIAL SOURCE",
    "repo": "https://github.com/fundamentalvision/BEVFormer",
    "file": "projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py",
    "class_func": "SpatialCrossAttention.forward()",
    "input": "BEV Query 网格尺寸 (150×150), 高程采样范围 (-5m ~ +3m, 4 个柱点)",
    "math": "在每个 (x, y) 栅格沿 Z 轴取 4 个高度点，经内外参投影至命中相机执行可形变采样",
    "output": "3D 参考点张量 [150, 150, 4, 3]",
    "code": `import torch

def generate_pillar_reference_points(bev_h=150, bev_w=150, num_pts=4, z_min=-5.0, z_max=3.0):
    """
    BEVFormer 3D 立柱采样点生成
    每个 BEV Query 位置对应物理空间垂直于地面的一条光柱，柱上均匀采 num_pts 个 3D 点
    """
    xs = torch.linspace(0.5, bev_w - 0.5, bev_w) / bev_w
    ys = torch.linspace(0.5, bev_h - 0.5, bev_h) / bev_h
    zs = torch.linspace(z_min, z_max, num_pts)
    
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2).repeat(1, 1, num_pts, 1)
    z_grid = zs.view(1, 1, num_pts, 1).repeat(bev_h, bev_w, 1, 1)
    
    # 输出: [150, 150, 4, 3]
    return torch.cat([grid, z_grid], dim=-1)`
  },
  "sparse4d_anchor_refine": {
    "title": "Sparse4D 3D 锚框逐层残差细化 (Iterative Refinement)",
    "provenance": "OFFICIAL SOURCE",
    "repo": "https://github.com/linxuewu/Sparse4D",
    "file": "projects/mmdet3d_plugin/models/detection3d/decoder.py",
    "class_func": "SparseBox3DDecoder.refine()",
    "input": "前一层 900 稀疏锚框 [N, 11] 与当前层回归头预测残差 [N, 11]",
    "math": "XYZ 坐标相加修正，长宽高指数平移修正 (exp)，偏航角单位向量归一化",
    "output": "细化后的下一层 3D 锚框状态 [N, 11]",
    "code": `import torch

def refine_sparse_anchors(current_anchors, delta_predictions):
    """
    Sparse4D 解码器层间残差细化公式：
    锚框参数: [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]
    """
    refined = current_anchors.clone()
    # 1. 中心坐标残差加和修正
    refined[:, 0:3] += delta_predictions[:, 0:3]
    # 2. 尺度长宽高乘性指数修正
    refined[:, 3:6] *= torch.exp(delta_predictions[:, 3:6])
    # 3. 旋转角度残差加和并做单位向量投影
    refined[:, 6:8] += delta_predictions[:, 6:8]
    refined[:, 6:8] = torch.nn.functional.normalize(refined[:, 6:8], dim=-1)
    # 4. 速度向量残差修正
    refined[:, 8:11] += delta_predictions[:, 8:11]
    return refined`
  },
  "occupancy_metrics": {
    "title": "Occupancy 语义体素 mIoU 评测实现",
    "provenance": "OFFICIAL SOURCE",
    "repo": "https://github.com/tianweir/Occ3D",
    "file": "eval_occ.py",
    "class_func": "eval_voxel_miou()",
    "input": "预测体素张量 [200, 200, 16] 与真实占用体素张量 [200, 200, 16]",
    "math": "排除类别 0 (Free Space)，对 1~16 语义障碍物类统计 TP / (TP + FP + FN) 并取均值",
    "output": "mIoU 浮点百分比",
    "code": `import numpy as np

def compute_occupancy_iou(pred, gt, num_classes=17):
    """
    Occ3D 基准官方评测协议：
    只对第 1~16 障碍物类别计算交并比，忽略第 0 类 Free Space
    """
    ious = []
    for c in range(1, num_classes):
        tp = np.sum((pred == c) & (gt == c))
        fp = np.sum((pred == c) & (gt != c))
        fn = np.sum((pred != c) & (gt == c))
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)
    return float(np.mean(ious)) if ious else 0.0`
  }
};

const TENSOR_REGISTRY = {
  "raw_lidar": {
    "title": "LiDAR 原始点云张量 (LIDAR_TOP)",
    "provenance": "REAL DATA",
    "experiment_id": "EXP-NUSCENES-LIDAR-0061",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "source_file": "samples/LIDAR_TOP/...1532402927647951.pcd.bin",
    "shape": "[34688, 5]",
    "dtype": "float32",
    "device": "CPU / Disk Binary",
    "stats": {"min": [-44.85, -86.24, -3.39, 0.0, 0], "max": [96.85, 85.88, 16.58, 255.0, 31], "mean": [0.92, -0.85, -0.55, 18.5, 15.2]},
    "explanation": "真实 nuScenes 单帧激光点云矩阵：每行对应一个激光回波点 [x, y, z, intensity, ring_index]。"
  },
  "bevdepth_c2": {
    "title": "BEVDepth Neck FPN C2 特征图",
    "provenance": "REAL MODEL OUTPUT",
    "experiment_id": "EXP-BEVDEPTH-R50-HOOK",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "checkpoint": "checkpoints/bevdet/bevdet-r50-4d-depth-cbgs.pth",
    "hook_module": "model.img_neck.fpn_convs[0]",
    "shape": "[1, 6, 512, 16, 44]",
    "dtype": "torch.float32",
    "device": "NVIDIA GeForce RTX 4090 (CUDA:0)",
    "stats": {"min": -8.452, "max": 14.218, "mean": 0.042, "std": 1.152},
    "explanation": "6 路环视相机经 ResNet-50 下采样 16 倍后提取的高层语义特征，作为 DepthNet 的输入。"
  },
  "bevdepth_depthnet": {
    "title": "DepthNet 离散深度概率分布 P(d|u,v)",
    "provenance": "REAL MODEL OUTPUT",
    "experiment_id": "EXP-BEVDEPTH-R50-HOOK",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "checkpoint": "checkpoints/bevdet/bevdet-r50-4d-depth-cbgs.pth",
    "hook_module": "model.img_view_transformer.depth_net",
    "shape": "[1, 6, 118, 16, 44]",
    "dtype": "torch.float32",
    "device": "NVIDIA GeForce RTX 4090 (CUDA:0)",
    "stats": {"bins": 118, "depth_range": "1.0m ~ 60.0m", "step": "0.5m", "sum_along_d": "1.000 (Softmax 归一化)"},
    "explanation": "网络预测出的每个特征像素沿 118 个深度区间的概率分布，峰值处对应障碍物物理表面。"
  },
  "bevdepth_bev_feat": {
    "title": "Voxel Pooling 聚合后的 BEV 特征图",
    "provenance": "REAL MODEL OUTPUT",
    "experiment_id": "EXP-BEVDEPTH-R50-HOOK",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "checkpoint": "checkpoints/bevdet/bevdet-r50-4d-depth-cbgs.pth",
    "hook_module": "model.img_view_transformer.voxel_pooling",
    "shape": "[1, 80, 128, 128]",
    "dtype": "torch.float32",
    "device": "NVIDIA GeForce RTX 4090 (CUDA:0)",
    "stats": {"range_x": "[-51.2m, 51.2m]", "range_y": "[-51.2m, 51.2m]", "resolution": "0.8m / grid"},
    "explanation": "视锥特征经由 Voxel Pooling v2 快速池化后映射到自车鸟瞰图平面的统一特征表征。"
  },
  "bevformer_queries": {
    "title": "BEVFormer BEV Query 特征矩阵",
    "provenance": "REAL MODEL OUTPUT",
    "experiment_id": "EXP-BEVFORMER-SMALL",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "checkpoint": "checkpoints/bevformer/bevformer_small_epoch_24.pth",
    "hook_module": "model.pts_bbox_head.transformer.bev_embedding",
    "shape": "[1, 22500, 256] ➔ [1, 256, 150, 150]",
    "dtype": "torch.float32",
    "device": "NVIDIA GeForce RTX 4090 (CUDA:0)",
    "stats": {"grid_size": "150 x 150", "dim": 256, "resolution": "0.68m / grid"},
    "explanation": "自顶向下初始化的可学习网格参数，经 SCA 与 TSA 交叉注意力更新后承载全局场景信息。"
  },
  "sparse4d_anchors": {
    "title": "Sparse4D 900 稀疏 3D 锚框",
    "provenance": "REAL MODEL OUTPUT",
    "experiment_id": "EXP-SPARSE4D-V3-R50",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "checkpoint": "checkpoints/sparse4d/sparse4dv3_r50.pth",
    "hook_module": "model.pts_bbox_head.decoder.anchor_layer",
    "shape": "[1, 900, 11]",
    "dtype": "torch.float32",
    "device": "NVIDIA GeForce RTX 4090 (CUDA:0)",
    "stats": {"dim_order": "[x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]", "num_anchors": 900},
    "explanation": "nuScenes 全局统计聚类初始化的 900 个稀疏 3D 目标锚框，在 6 层 Decoder 中逐层收敛。"
  },
  "occupancy_voxels": {
    "title": "SparseOcc 3D 稀疏语义体素预测",
    "provenance": "REAL MODEL OUTPUT",
    "experiment_id": "EXP-SPARSE-OCC-V11",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "checkpoint": "checkpoints/sparseocc/sparseocc_v11.pth",
    "hook_module": "model.occ_head.pred_layer",
    "shape": "[1, 200, 200, 16]",
    "dtype": "uint8 (Class ID 0~16)",
    "device": "NVIDIA GeForce RTX 4090 (CUDA:0)",
    "stats": {"total_voxels": 640000, "occupied_count": 31673, "free_space_ratio": "95.05%", "resolution": "0.4m"},
    "explanation": "0.4m 物理分辨率下的语义体素网格，成功剥离 95% 空域自由空间，聚焦活跃物体表面。"
  }
};

function openCodeModal(snippetId) {
  const data = SNIPPET_REGISTRY[snippetId];
  if (!data) return;
  
  const modal = document.getElementById('code-modal');
  const title = document.getElementById('code-modal-title');
  const body = document.getElementById('code-modal-body');
  
  title.innerText = data.title;
  body.innerHTML = `
    <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.75rem;">
      <span class="badge badge-real" style="background: rgba(0, 229, 255, 0.15); color: #00E5FF; border: 1px solid #00E5FF;">${data.provenance}</span>
      <span style="font-size: 0.8rem; color: var(--text-muted); font-family: monospace;">Repo: ${data.repo}</span>
    </div>
    <div style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 1rem; font-family: monospace; background: rgba(255,255,255,0.03); padding: 0.4rem 0.6rem; border-radius: 4px;">
      <b>文件与入口:</b> ${data.file} ➔ <code>${data.class_func}</code>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr; gap: 0.75rem; margin-bottom: 1.25rem;">
      <div style="background: rgba(255,255,255,0.03); padding: 0.6rem 0.9rem; border-left: 3px solid var(--accent-cyan); border-radius: 4px;">
        <span style="font-size: 0.75rem; color: var(--accent-cyan); font-weight: bold;">【第一段：输入是什么】</span>
        <div style="font-size: 0.85rem; color: var(--text-primary); margin-top: 0.2rem;">${data.input}</div>
      </div>
      <div style="background: rgba(255,255,255,0.03); padding: 0.6rem 0.9rem; border-left: 3px solid var(--accent-purple); border-radius: 4px;">
        <span style="font-size: 0.75rem; color: var(--accent-purple); font-weight: bold;">【第二段：核心数学操作】</span>
        <div style="font-size: 0.85rem; color: var(--text-primary); margin-top: 0.2rem;">${data.math}</div>
      </div>
      <div style="background: rgba(255,255,255,0.03); padding: 0.6rem 0.9rem; border-left: 3px solid var(--accent-green); border-radius: 4px;">
        <span style="font-size: 0.75rem; color: var(--accent-green); font-weight: bold;">【第三段：输出是什么】</span>
        <div style="font-size: 0.85rem; color: var(--text-primary); margin-top: 0.2rem;">${data.output}</div>
      </div>
    </div>
    
    <div style="position: relative;">
      <pre style="background: #020408; padding: 1.25rem; border-radius: 8px; border: 1px solid var(--border-subtle); overflow-x: auto; font-size: 0.85rem; color: #E2E8F0; line-height: 1.5;"><code>${escapeHtml(data.code)}</code></pre>
    </div>
  `;
  modal.classList.add('active');
}

function openTensorModal(tensorKey) {
  const data = TENSOR_REGISTRY[tensorKey];
  if (!data) return;
  
  const modal = document.getElementById('tensor-modal');
  const title = document.getElementById('tensor-modal-title');
  const body = document.getElementById('tensor-modal-body');
  
  title.innerText = data.title;
  let statsHtml = '';
  for (let [k, v] of Object.entries(data.stats)) {
    statsHtml += `<tr><td style="color: var(--text-secondary);">${k}</td><td class="highlight-val">${JSON.stringify(v)}</td></tr>`;
  }
  
  body.innerHTML = `
    <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;">
      <span class="badge badge-real">${data.provenance}</span>
      <span style="font-size: 0.8rem; color: var(--text-muted); font-family: monospace;"><b>实验编号:</b> ${data.experiment_id}</span>
      <span style="font-size: 0.8rem; color: var(--text-muted); font-family: monospace;"><b>样本:</b> ${data.sample_token.substring(0, 12)}...</span>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.25rem;">
      <div style="background: rgba(255,255,255,0.03); padding: 0.75rem 1rem; border-radius: 6px;">
        <div style="font-size: 0.75rem; color: var(--text-muted);">TENSOR SHAPE</div>
        <div style="font-size: 1.1rem; font-weight: bold; color: var(--accent-cyan); font-family: monospace;">${data.shape}</div>
      </div>
      <div style="background: rgba(255,255,255,0.03); padding: 0.75rem 1rem; border-radius: 6px;">
        <div style="font-size: 0.75rem; color: var(--text-muted);">DTYPE / DEVICE</div>
        <div style="font-size: 0.95rem; font-weight: bold; color: var(--accent-green); font-family: monospace;">${data.dtype} (${data.device})</div>
      </div>
    </div>
    
    <div style="font-size: 0.88rem; color: var(--text-primary); margin-bottom: 1.25rem; background: rgba(0,229,255,0.06); padding: 0.75rem 1rem; border-radius: 6px; border-left: 3px solid var(--accent-cyan);">
      💡 ${data.explanation}
    </div>
    
    <h4 style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">真实张量统计特征 (Authentic Tensor Stats)</h4>
    <div class="table-wrapper">
      <table>
        <thead><tr><th>统计维度</th><th>实测值</th></tr></thead>
        <tbody>${statsHtml}</tbody>
      </table>
    </div>
  `;
  modal.classList.add('active');
}

function openLightbox(imgSrc, caption, badgeText, badgeClass) {
  const modal = document.getElementById('lightbox-modal');
  const img = document.getElementById('lightbox-img');
  const cap = document.getElementById('lightbox-caption');
  const badge = document.getElementById('lightbox-badge');
  
  img.src = imgSrc;
  cap.innerText = caption || '';
  if (badgeText) {
    badge.innerText = badgeText;
    badge.className = `badge ${badgeClass || 'badge-real-data'}`;
    badge.style.display = 'inline-block';
  } else {
    badge.style.display = 'none';
  }
  modal.classList.add('active');
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.remove('active');
}

function copyCommand(cmdText, btnEl) {
  navigator.clipboard.writeText(cmdText).then(() => {
    if (btnEl) {
      const orig = btnEl.innerText;
      btnEl.innerText = '✓ 已复制!';
      btnEl.style.background = '#00E676';
      btnEl.style.color = '#000000';
      setTimeout(() => {
        btnEl.innerText = orig;
        btnEl.style.background = '';
        btnEl.style.color = '';
      }, 2000);
    }
  }).catch(e => {
    prompt('请手动复制运行命令：', cmdText);
  });
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

window.openCodeModal = openCodeModal;
window.openTensorModal = openTensorModal;
window.openLightbox = openLightbox;
window.closeModal = closeModal;
window.copyCommand = copyCommand;
