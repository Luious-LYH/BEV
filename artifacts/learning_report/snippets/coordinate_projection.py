# ==============================================================================
# Rigid Transform & Camera Perspective Projection
# File: nuscenes/utils/geometry_utils.py -> view_points
# ==============================================================================
import numpy as np

def project_lidar_to_camera(pt_lidar, R_l2e, t_l2e, R_c2e, t_c2e, K, img_w=1600, img_h=900):
    """
    Input:
        pt_lidar: 激光雷达坐标点 [3]
        R_l2e, t_l2e: LiDAR 到自车系旋转与平移 (3x3, 3)
        R_c2e, t_c2e: 相机到自车系旋转与平移 (3x3, 3)
        K: 相机内参标定矩阵 (3x3)
    Math Operation:
        1. 刚体变换至自车系: P_ego = R_l2e @ P_lidar + t_l2e
        2. 刚体变换至相机系: P_cam = R_c2e.T @ (P_ego - t_c2e)
        3. 透视投影至像素系: s [u, v, 1]^T = K @ P_cam
    Output:
        (u, v): 像素坐标
        visible: 是否在有效视锥内 (Z_cam > 0 且落在 0<=u<W, 0<=v<H)
    """
    pt_ego = R_l2e @ pt_lidar + t_l2e
    pt_cam = R_c2e.T @ (pt_ego - t_c2e)
    if pt_cam[2] <= 0:
        return None, None, False
    pixel_homo = K @ pt_cam
    u = pixel_homo[0] / pixel_homo[2]
    v = pixel_homo[1] / pixel_homo[2]
    visible = (0 <= u < img_w) and (0 <= v < img_h)
    return u, v, visible
