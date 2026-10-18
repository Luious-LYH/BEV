# ==============================================================================
# Sparse4D Iterative Anchor Refinement
# File: projects/mmdet3d_plugin/models/detection3d/decoder.py -> SparseBox3DDecoder
# ==============================================================================
import torch

def refine_sparse_anchors(current_anchors, delta_predictions):
    """
    Input:
        current_anchors:   当前层 3D 锚框 [N, 11] (x, y, z, w, l, h, sin, cos, vx, vy, vz)
        delta_predictions: 当前 Decoder 层预测的残差增量 [N, 11]
    Math Operation:
        1. 位置修正: x_new = x + Δx, y_new = y + Δy, z_new = z + Δz
        2. 尺度修正: w_new = w * exp(Δw), l_new = l * exp(Δl), h_new = h * exp(Δh)
        3. 速度与朝向按残差累加并归一化
    Output:
        refined_anchors: 作为下一 Decoder 层的输入基准，实现 Coarse-to-Fine 迭代
    """
    refined_anchors = current_anchors.clone()
    refined_anchors[:, 0:3] += delta_predictions[:, 0:3]
    refined_anchors[:, 3:6] *= torch.exp(delta_predictions[:, 3:6])
    refined_anchors[:, 6:8] += delta_predictions[:, 6:8]
    refined_anchors[:, 6:8] = torch.nn.functional.normalize(refined_anchors[:, 6:8], dim=-1)
    refined_anchors[:, 8:11] += delta_predictions[:, 8:11]
    return refined_anchors
