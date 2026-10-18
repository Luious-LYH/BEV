# ==============================================================================
# LSS Lift & Frustum Outer Product
# File: mmdet3d/models/necks/view_transformer.py -> LSSViewTransformer
# ==============================================================================
import torch

def lift_splat_outer_product(depth_prob, context_feat):
    """
    Input:
        depth_prob:   DepthNet 离散深度概率分布 [B, N, D, H, W]
        context_feat: 2D 上下文图像特征 [B, N, C, H, W]
    Math Operation:
        F_3D(u, v, d) = P(d | u, v) * F_2D(u, v)
        在通道维度与深度维度执行外积广播乘法
    Output:
        frustum_feat: 视锥 3D 特征体 [B, N, D, C, H, W]
    """
    # 扩展维度以进行外积
    # depth_prob:    [B, N, D, 1, H, W]
    # context_feat:  [B, N, 1, C, H, W]
    frustum_feat = depth_prob.unsqueeze(3) * context_feat.unsqueeze(2)
    return frustum_feat
