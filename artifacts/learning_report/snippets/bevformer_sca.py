# ==============================================================================
# BEVFormer Spatial Cross-Attention Reference Point Projection
# File: projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
# ==============================================================================
import torch

def generate_pillar_reference_points(bev_h=150, bev_w=150, num_points_in_pillar=4, z_min=-5.0, z_max=3.0):
    """
    Input:
        bev_h, bev_w: BEV 网格分辨率 (150x150)
        num_points_in_pillar: 每个 3D 立柱内部的高程采样点数 (4)
        z_min, z_max: 高程采样范围 (-5.0m 到 +3.0m)
    Math Operation:
        1. 在 BEV 正交平面生成归一化网格坐标 (x_i, y_j)
        2. 在 Z 轴均匀划分 4 个离散高度: z_k = z_min + (k + 0.5) * (z_max - z_min) / 4
        3. 构建 3D 物理坐标 [X, Y, Z] 并准备投射至 6 路相机视角
    Output:
        ref_points: shape [bev_h, bev_w, num_points_in_pillar, 3]
    """
    xs = torch.linspace(0.5, bev_w - 0.5, bev_w) / bev_w
    ys = torch.linspace(0.5, bev_h - 0.5, bev_h) / bev_h
    zs = torch.linspace(z_min, z_max, num_points_in_pillar)
    
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2).repeat(1, 1, num_points_in_pillar, 1)
    z_grid = zs.view(1, 1, num_points_in_pillar, 1).repeat(bev_h, bev_w, 1, 1)
    ref_points_3d = torch.cat([grid, z_grid], dim=-1)
    return ref_points_3d
