# ==============================================================================
# Occupancy Metrics: mIoU & RayIoU
# File: mmdet3d/core/evaluation/occ_metrics.py
# ==============================================================================
import numpy as np

def compute_occupancy_iou(pred_voxels, gt_voxels, num_classes=17, ignore_index=255):
    """
    Input:
        pred_voxels: 预测体素语义类别 [X, Y, Z]
        gt_voxels:   真值体素语义类别 [X, Y, Z]
        num_classes: 语义类别总数 (17)
    Math Operation:
        对每个类别 c:
        TP_c = sum((pred == c) & (gt == c))
        FP_c = sum((pred == c) & (gt != c))
        FN_c = sum((pred != c) & (gt == c))
        IoU_c = TP_c / (TP_c + FP_c + FN_c + 1e-6)
        mIoU = mean(IoU_c) (排除自由空间与 ignore 类别)
    Output:
        class_ious: 字典形式各类别 IoU
        miou:       全空间语义平均交并比
    """
    valid_mask = (gt_voxels != ignore_index)
    pred = pred_voxels[valid_mask]
    gt = gt_voxels[valid_mask]
    
    ious = []
    # 类别 0 通常为 free space，评估通常聚焦 1~16 障碍物类
    for c in range(1, num_classes):
        tp = np.sum((pred == c) & (gt == c))
        fp = np.sum((pred == c) & (gt != c))
        fn = np.sum((pred != c) & (gt == c))
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)
    
    miou = np.mean(ious) if len(ious) > 0 else 0.0
    return miou
