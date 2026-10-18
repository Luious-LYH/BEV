# ==============================================================================
# BEVDepth LiDAR Point Depth Supervision Loss
# File: mmdet3d/models/necks/bevdepth.py -> DepthNet
# ==============================================================================
import torch
import torch.nn.functional as F

def compute_depth_supervision_loss(depth_logits, depth_gt_bins):
    """
    Input:
        depth_logits:  DepthNet 预测的未归一化 logits [B*N, D, H, W]
        depth_gt_bins: 真实 LiDAR 投射并离散化的深度标签 [B*N, H, W]
                       (有效值为 1 ~ D，0 代表无点云击中的无效区域)
    Math Operation:
        1. 筛选前景掩码 mask = (depth_gt_bins > 0)
        2. 将离散标签转化为 one-hot 概率目标
        3. 对有效像素计算多类别交叉熵 Cross-Entropy 或 Binary Cross-Entropy
    Output:
        depth_loss: 深度回归损失标量 (参与反向传播，梯度回传至 Image Backbone 与 DepthNet)
    """
    mask = (depth_gt_bins > 0)
    valid_logits = depth_logits.permute(0, 2, 3, 1)[mask] # [num_valid, D]
    valid_targets = depth_gt_bins[mask].long() - 1        # [num_valid]
    
    depth_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
    return depth_loss
