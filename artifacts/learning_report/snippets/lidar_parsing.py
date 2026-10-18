# ==============================================================================
# LiDAR Raw Binary Parsing (.pcd.bin -> numpy array)
# File: nuscenes/utils/data_classes.py -> LidarPointCloud
# ==============================================================================
import numpy as np

def load_lidar_points(bin_path: str) -> np.ndarray:
    """
    Input:
        bin_path: 激光雷达二进制文件路径 (.pcd.bin)
    Math Operation:
        读取连续 32 位浮点数流并重塑为 [N, 5] 矩阵:
        - X, Y, Z (米, 激光雷达坐标系)
        - Intensity (0~255 反射强度)
        - Ring Index (0~31 线束序号)
    Output:
        points: shape [N, 5], dtype float32
    """
    raw_points = np.fromfile(bin_path, dtype=np.float32)
    points = raw_points.reshape(-1, 5)
    return points
