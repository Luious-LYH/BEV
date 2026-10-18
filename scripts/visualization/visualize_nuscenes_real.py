#!/usr/bin/env python3
"""
visualize_nuscenes_real.py — 真实 nuScenes 多传感器教学可视化

从 nuScenes v1.0-mini 真实数据生成 10+ 类教学可视化：
1. 6 Camera 环视图
2. LiDAR BEV 投影 (高度编码)
3. Radar BEV 可视化
4. GT 3D BBox BEV
5. LiDAR → Camera 投影
6. 3D Box → Camera 投影
7. 传感器布局 3D 图
8. 坐标变换链可视化
9. 时间关系图
10. 标定参数展示

所有图标注 [REAL DATA]，使用真实 nuScenes mini 数据。
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
from pyquaternion import Quaternion

# nuScenes devkit
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

# ── Config ──────────────────────────────────────────────────────────────
NUSCENES_ROOT = '/public/lyh/data/BEV_dataset/nuscenes'
OUTPUT_DIR = '/public/lyh/projects/BEV/artifacts/learning_report/images/real'
SCENE_NAME = 'scene-0061'  # Classic Boston scene

# Camera order for surround view
CAM_CHANNELS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'
]

RADAR_CHANNELS = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
]

# nuScenes 类别颜色映射
CATEGORY_COLORS = {
    'car': '#FF6B6B',
    'truck': '#4ECDC4',
    'bus': '#45B7D1',
    'trailer': '#96CEB4',
    'construction_vehicle': '#FFEAA7',
    'pedestrian': '#DDA0DD',
    'motorcycle': '#98D8C8',
    'bicycle': '#F7DC6F',
    'traffic_cone': '#E74C3C',
    'barrier': '#85929E',
}

def get_color_for_category(name):
    """根据类别名返回颜色"""
    for key, color in CATEGORY_COLORS.items():
        if key in name:
            return color
    return '#AAAAAA'


def setup():
    """初始化 nuScenes 并找到目标 scene/sample"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading nuScenes v1.0-mini from {NUSCENES_ROOT}...")
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_ROOT, verbose=True)

    # 找到目标 scene
    target_scene = None
    for scene in nusc.scene:
        if scene['name'] == SCENE_NAME:
            target_scene = scene
            break
    if target_scene is None:
        # 使用第一个 scene
        target_scene = nusc.scene[0]
        print(f"Warning: {SCENE_NAME} not found, using {target_scene['name']}")

    # 获取第一个 sample (keyframe)
    first_sample_token = target_scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    print(f"Scene: {target_scene['name']} ({target_scene['description']})")
    print(f"Sample token: {sample['token']}")
    print(f"Timestamp: {sample['timestamp']}")

    return nusc, target_scene, sample


def viz_01_surround_cameras(nusc, sample):
    """[REAL DATA] 6 Camera 环视图 — 同一 sample 同步采集"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('[REAL DATA] nuScenes 6-Camera Surround View\n'
                 f'Sample: {sample["token"][:16]}... | Timestamp: {sample["timestamp"]}',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, cam in enumerate(CAM_CHANNELS):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        cam_data = nusc.get('sample_data', sample['data'][cam])
        img_path = os.path.join(NUSCENES_ROOT, cam_data['filename'])
        img = Image.open(img_path)

        # 获取标定信息
        cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = np.array(cs['camera_intrinsic'])

        ax.imshow(img)
        ax.set_title(f'{cam}\n'
                     f'fx={intrinsic[0,0]:.1f} fy={intrinsic[1,1]:.1f} '
                     f'cx={intrinsic[0,2]:.1f} cy={intrinsic[1,2]:.1f}',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, 'real_surround_cameras.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[1/10] Saved: {save_path}")
    return save_path


def viz_02_lidar_bev(nusc, sample):
    """[REAL DATA] LiDAR BEV 俯视图 — 真实点云高度编码"""
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc_path = os.path.join(NUSCENES_ROOT, lidar_data['filename'])
    pc = LidarPointCloud.from_file(pc_path)  # (4, N): x, y, z, intensity

    points = pc.points  # (4, N)
    x, y, z = points[0], points[1], points[2]
    intensity = points[3]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'[REAL DATA] LIDAR_TOP Point Cloud BEV\n'
                 f'{points.shape[1]} points | {pc_path.split("/")[-1]}',
                 fontsize=14, fontweight='bold')

    # BEV 高度编码
    ax = axes[0]
    scatter = ax.scatter(x, y, c=z, cmap='turbo', s=0.3, alpha=0.8,
                         vmin=-3, vmax=3)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m) [→ Forward]', fontsize=11)
    ax.set_ylabel('Y (m) [← Left]', fontsize=11)
    ax.set_title('Height Encoded', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Height (m)', shrink=0.8)
    ax.plot(0, 0, 'r*', markersize=15, label='Ego Vehicle')
    ax.legend(fontsize=9)

    # BEV 强度编码
    ax = axes[1]
    scatter = ax.scatter(x, y, c=intensity, cmap='hot', s=0.3, alpha=0.8,
                         vmin=0, vmax=50)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m) [→ Forward]')
    ax.set_ylabel('Y (m) [← Left]')
    ax.set_title('Intensity Encoded', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.8)
    ax.plot(0, 0, 'r*', markersize=15)

    # BEV 距离编码
    ax = axes[2]
    dist = np.sqrt(x**2 + y**2)
    scatter = ax.scatter(x, y, c=dist, cmap='viridis', s=0.3, alpha=0.8,
                         vmin=0, vmax=60)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m) [→ Forward]')
    ax.set_ylabel('Y (m) [← Left]')
    ax.set_title('Distance Encoded', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Distance (m)', shrink=0.8)
    ax.plot(0, 0, 'r*', markersize=15)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'real_lidar_bev.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[2/10] Saved: {save_path}")
    return save_path


def viz_03_radar_bev(nusc, sample):
    """[REAL DATA] 5 Radar BEV — 真实雷达点 + 速度向量"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    all_x, all_y, all_vx, all_vy = [], [], [], []
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for idx, radar_ch in enumerate(RADAR_CHANNELS):
        radar_data = nusc.get('sample_data', sample['data'][radar_ch])
        radar_path = os.path.join(NUSCENES_ROOT, radar_data['filename'])

        try:
            pc = RadarPointCloud.from_file(radar_path)
            pts = pc.points  # (18, N)

            # Radar 坐标 → ego 坐标
            cs = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
            rot = Quaternion(cs['rotation']).rotation_matrix
            trans = np.array(cs['translation'])

            pts_xyz = pts[:3, :]  # (3, N)
            pts_ego = rot @ pts_xyz + trans.reshape(3, 1)

            x_ego, y_ego = pts_ego[0], pts_ego[1]

            # 速度补偿 (compensated velocities)
            vx_comp = pts[8, :]  # vx_comp
            vy_comp = pts[9, :]  # vy_comp

            ax.scatter(x_ego, y_ego, c=colors_list[idx], s=30, alpha=0.8,
                       label=f'{radar_ch} ({pts.shape[1]} pts)', zorder=5)

            # 速度矢量
            for j in range(len(x_ego)):
                if abs(vx_comp[j]) > 0.1 or abs(vy_comp[j]) > 0.1:
                    vx_ego = rot[0,0]*vx_comp[j] + rot[0,1]*vy_comp[j]
                    vy_ego = rot[1,0]*vx_comp[j] + rot[1,1]*vy_comp[j]
                    ax.arrow(x_ego[j], y_ego[j], vx_ego*0.5, vy_ego*0.5,
                             head_width=0.3, head_length=0.15,
                             fc=colors_list[idx], ec=colors_list[idx], alpha=0.5)
        except Exception as e:
            print(f"  Warning: {radar_ch} load error: {e}")

    ax.plot(0, 0, 'r*', markersize=20, label='Ego Vehicle', zorder=10)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m) [→ Forward]', fontsize=12)
    ax.set_ylabel('Y (m) [← Left]', fontsize=12)
    ax.set_title('[REAL DATA] 5-Radar BEV with Velocity Vectors\n'
                 'Arrows show compensated velocity direction',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, 'real_radar_bev.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[3/10] Saved: {save_path}")
    return save_path


def viz_04_gt_bbox_bev(nusc, sample):
    """[REAL DATA] GT 3D BBox BEV — 真实标注框鸟瞰"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # 加载 LiDAR 点云作为背景
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc_path = os.path.join(NUSCENES_ROOT, lidar_data['filename'])
    pc = LidarPointCloud.from_file(pc_path)

    # LiDAR → ego
    cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
    pc.translate(np.array(cs['translation']))

    ax.scatter(pc.points[0], pc.points[1], c='#333333', s=0.2, alpha=0.3)

    # 绘制 GT boxes
    ann_count = 0
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

        # Global → ego
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        # 获取颜色
        color = get_color_for_category(ann['category_name'])

        # 绘制 2D BEV 框 (x, y, yaw)
        corners = box.bottom_corners()[:2, :]  # (2, 4)
        corners = np.concatenate([corners, corners[:, :1]], axis=1)  # close
        ax.plot(corners[0], corners[1], color=color, linewidth=2)

        # 前向箭头
        center = box.center[:2]
        front = (corners[:, 0] + corners[:, 1]) / 2
        ax.annotate('', xy=front, xytext=center,
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # 标注类名
        short_name = ann['category_name'].split('.')[-1]
        ax.text(center[0], center[1]+0.5, short_name,
                fontsize=6, ha='center', color=color, fontweight='bold')
        ann_count += 1

    ax.plot(0, 0, 'r*', markersize=20, label='Ego', zorder=10)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m) [→ Forward]', fontsize=12)
    ax.set_ylabel('Y (m) [← Left]', fontsize=12)
    ax.set_title(f'[REAL DATA] GT 3D BBox BEV — {ann_count} Annotations\n'
                 f'Arrows indicate heading direction',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # 图例
    legend_handles = []
    for cat, color in CATEGORY_COLORS.items():
        legend_handles.append(patches.Patch(color=color, label=cat))
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right', ncol=2)

    save_path = os.path.join(OUTPUT_DIR, 'real_gt_bbox_bev.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[4/10] Saved: {save_path}")
    return save_path


def viz_05_lidar_to_camera(nusc, sample):
    """[REAL DATA] LiDAR → Camera 投影 — 点云深度着色投影到 6 个相机"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('[REAL DATA] LiDAR Points Projected onto 6 Cameras\n'
                 'Color = depth (near=red → far=blue)',
                 fontsize=14, fontweight='bold', y=0.98)

    # 加载 LiDAR
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc_path = os.path.join(NUSCENES_ROOT, lidar_data['filename'])
    pc_orig = LidarPointCloud.from_file(pc_path)

    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    for idx, cam in enumerate(CAM_CHANNELS):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        cam_data = nusc.get('sample_data', sample['data'][cam])
        img_path = os.path.join(NUSCENES_ROOT, cam_data['filename'])
        img = Image.open(img_path)
        ax.imshow(img)

        # 复制点云
        pc = LidarPointCloud(pc_orig.points.copy())

        # LiDAR sensor → ego
        pc.rotate(Quaternion(lidar_cs['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_cs['translation']))

        # ego → global
        pc.rotate(Quaternion(lidar_ego['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_ego['translation']))

        # global → cam ego
        cam_ego = nusc.get('ego_pose', cam_data['ego_pose_token'])
        pc.translate(-np.array(cam_ego['translation']))
        pc.rotate(Quaternion(cam_ego['rotation']).inverse.rotation_matrix)

        # cam ego → cam sensor
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        pc.translate(-np.array(cam_cs['translation']))
        pc.rotate(Quaternion(cam_cs['rotation']).inverse.rotation_matrix)

        # 过滤：只保留相机前方 (z > 1) 的点
        depths = pc.points[2, :]
        mask = depths > 1.0
        pts = pc.points[:3, mask]
        d = depths[mask]

        # 投影到像素
        intrinsic = np.array(cam_cs['camera_intrinsic'])
        proj = view_points(pts, intrinsic, normalize=True)

        # 过滤在图像范围内的点
        w, h = img.size
        img_mask = (proj[0] > 0) & (proj[0] < w) & (proj[1] > 0) & (proj[1] < h)
        proj = proj[:, img_mask]
        d = d[img_mask]

        scatter = ax.scatter(proj[0], proj[1], c=d, cmap='turbo',
                            s=1.0, alpha=0.7, vmin=0, vmax=50)
        ax.set_title(f'{cam} ({len(d)} pts)', fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, 'real_lidar_to_camera.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[5/10] Saved: {save_path}")
    return save_path


def viz_06_3dbox_to_camera(nusc, sample):
    """[REAL DATA] 3D Box → Camera 投影 — GT 框透视投影回相机"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('[REAL DATA] 3D GT Bounding Boxes Projected onto Cameras',
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, cam in enumerate(CAM_CHANNELS):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        cam_data = nusc.get('sample_data', sample['data'][cam])
        img_path = os.path.join(NUSCENES_ROOT, cam_data['filename'])
        img = Image.open(img_path)
        ax.imshow(img)

        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_ego = nusc.get('ego_pose', cam_data['ego_pose_token'])
        intrinsic = np.array(cam_cs['camera_intrinsic'])

        box_count = 0
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

            # Global → ego
            box.translate(-np.array(cam_ego['translation']))
            box.rotate(Quaternion(cam_ego['rotation']).inverse)

            # Ego → camera
            box.translate(-np.array(cam_cs['translation']))
            box.rotate(Quaternion(cam_cs['rotation']).inverse)

            # 检查是否在相机前方
            if box.center[2] <= 1:
                continue

            # 投影 8 corners
            corners_3d = box.corners()  # (3, 8)
            corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2]

            # 检查至少部分角点在图像内
            w, h = img.size
            in_img = ((corners_2d[0] > 0) & (corners_2d[0] < w) &
                      (corners_2d[1] > 0) & (corners_2d[1] < h))
            if not np.any(in_img):
                continue

            color = get_color_for_category(ann['category_name'])

            # 绘制 12 条边
            # Bottom face: 0-1-2-3, Top face: 4-5-6-7
            edges = [(0,1),(1,2),(2,3),(3,0),  # bottom
                     (4,5),(5,6),(6,7),(7,4),  # top
                     (0,4),(1,5),(2,6),(3,7)]  # pillars
            for e0, e1 in edges:
                ax.plot([corners_2d[0,e0], corners_2d[0,e1]],
                       [corners_2d[1,e0], corners_2d[1,e1]],
                       color=color, linewidth=1.5, alpha=0.8)
            box_count += 1

        ax.set_title(f'{cam} ({box_count} boxes)', fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, 'real_3dbox_camera_projection.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[6/10] Saved: {save_path}")
    return save_path


def viz_07_sensor_layout(nusc, sample):
    """[REAL DATA] 传感器布局 — 6 Camera + LiDAR + 5 Radar 相对 ego 3D 位置"""
    fig = plt.figure(figsize=(14, 12))

    # 使用两个子图：上面是 BEV, 下面是侧视图
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    ax_bev = fig.add_subplot(gs[0, :])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    sensors = {}

    # 收集所有传感器位置
    for channel in CAM_CHANNELS + ['LIDAR_TOP'] + RADAR_CHANNELS:
        sd = nusc.get('sample_data', sample['data'][channel])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pos = np.array(cs['translation'])
        rot = Quaternion(cs['rotation'])
        sensors[channel] = {'pos': pos, 'rot': rot, 'cs': cs}

    # BEV 视图 (X-Y plane)
    for name, info in sensors.items():
        x, y = info['pos'][0], info['pos'][1]

        if 'CAM' in name:
            color = '#2196F3'
            marker = '^'
            size = 150
        elif 'LIDAR' in name:
            color = '#F44336'
            marker = 'o'
            size = 200
        else:  # RADAR
            color = '#4CAF50'
            marker = 's'
            size = 120

        ax_bev.scatter(x, y, c=color, marker=marker, s=size, zorder=5,
                       edgecolors='black', linewidth=1)

        # 绘制朝向箭头
        fwd = info['rot'].rotate(np.array([1, 0, 0]))  # 前向
        ax_bev.annotate('', xy=(x + fwd[0]*0.5, y + fwd[1]*0.5), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

        short_name = name.replace('CAM_', 'C:').replace('RADAR_', 'R:').replace('LIDAR_', 'L:')
        ax_bev.text(x, y + 0.15, short_name, fontsize=7, ha='center',
                    fontweight='bold', color=color)

    # 绘制 ego 车辆轮廓 (简化)
    car_w, car_l = 0.9, 2.2
    car_rect = patches.FancyBboxPatch((-car_l/2, -car_w/2), car_l, car_w,
                                       boxstyle="round,pad=0.1",
                                       facecolor='#FFD700', edgecolor='black',
                                       alpha=0.5, linewidth=2)
    ax_bev.add_patch(car_rect)
    ax_bev.text(0, 0, 'EGO', fontsize=10, ha='center', va='center', fontweight='bold')

    ax_bev.set_xlim(-2.5, 3.0)
    ax_bev.set_ylim(-2.0, 2.0)
    ax_bev.set_aspect('equal')
    ax_bev.set_xlabel('X (m) [→ Forward]', fontsize=11)
    ax_bev.set_ylabel('Y (m) [← Left]', fontsize=11)
    ax_bev.set_title('[REAL DATA] Sensor Layout (BEV)\n'
                     '▲ Camera  ● LiDAR  ■ Radar',
                     fontsize=13, fontweight='bold')
    ax_bev.grid(True, alpha=0.3)

    # 侧视图 (X-Z plane)
    for name, info in sensors.items():
        x, z = info['pos'][0], info['pos'][2]
        if 'CAM' in name:
            color, marker = '#2196F3', '^'
        elif 'LIDAR' in name:
            color, marker = '#F44336', 'o'
        else:
            color, marker = '#4CAF50', 's'
        ax_side.scatter(x, z, c=color, marker=marker, s=80, zorder=5,
                       edgecolors='black', linewidth=1)

    ax_side.set_xlabel('X (m) [→ Forward]', fontsize=10)
    ax_side.set_ylabel('Z (m) [↑ Up]', fontsize=10)
    ax_side.set_title('Side View (X-Z)', fontsize=11, fontweight='bold')
    ax_side.grid(True, alpha=0.3)
    ax_side.set_aspect('equal')

    # 传感器参数表
    ax_table.axis('off')
    table_data = []
    for name in ['LIDAR_TOP'] + CAM_CHANNELS[:3] + RADAR_CHANNELS[:2]:
        info = sensors[name]
        pos = info['pos']
        table_data.append([
            name.replace('CAM_', 'C:').replace('RADAR_', 'R:').replace('LIDAR_', 'L:'),
            f'({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})'
        ])

    table = ax_table.table(cellText=table_data,
                          colLabels=['Sensor', 'Position (x,y,z)'],
                          loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax_table.set_title('Sensor Positions (meters)', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'real_sensor_layout.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[7/10] Saved: {save_path}")
    return save_path


def viz_08_coordinate_transform(nusc, sample):
    """[REAL DATA] 坐标变换链 — 真实矩阵数值展示"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    fig.suptitle('[REAL DATA] Coordinate Transform Chain: Sensor → Ego → Global\n'
                 'Real calibration matrices from nuScenes',
                 fontsize=14, fontweight='bold')

    # CAM_FRONT
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_ego = nusc.get('ego_pose', cam_data['ego_pose_token'])

    # LiDAR
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # Camera Intrinsic
    ax = axes[0][0]
    ax.axis('off')
    K = np.array(cam_cs['camera_intrinsic'])
    text = "CAM_FRONT Intrinsic K (3×3)\n\n"
    text += f"┌ {K[0,0]:8.2f}  {K[0,1]:8.2f}  {K[0,2]:8.2f} ┐\n"
    text += f"│ {K[1,0]:8.2f}  {K[1,1]:8.2f}  {K[1,2]:8.2f} │\n"
    text += f"└ {K[2,0]:8.2f}  {K[2,1]:8.2f}  {K[2,2]:8.2f} ┘\n\n"
    text += f"fx = {K[0,0]:.1f}  fy = {K[1,1]:.1f}\n"
    text += f"cx = {K[0,2]:.1f}  cy = {K[1,2]:.1f}"
    ax.text(0.5, 0.5, text, fontsize=10, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    ax.set_title('Camera Intrinsic', fontsize=12, fontweight='bold')

    # Camera → Ego (extrinsic)
    ax = axes[0][1]
    ax.axis('off')
    R = Quaternion(cam_cs['rotation']).rotation_matrix
    t = np.array(cam_cs['translation'])
    text = "CAM_FRONT → Ego (Extrinsic)\n\n"
    text += "Rotation (3×3):\n"
    for i in range(3):
        text += f"  [{R[i,0]:7.4f} {R[i,1]:7.4f} {R[i,2]:7.4f}]\n"
    text += f"\nTranslation:\n  [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"
    ax.text(0.5, 0.5, text, fontsize=9, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
    ax.set_title('Sensor → Ego', fontsize=12, fontweight='bold')

    # Ego → Global
    ax = axes[0][2]
    ax.axis('off')
    R_ego = Quaternion(cam_ego['rotation']).rotation_matrix
    t_ego = np.array(cam_ego['translation'])
    text = "Ego → Global (Ego Pose)\n\n"
    text += "Rotation (3×3):\n"
    for i in range(3):
        text += f"  [{R_ego[i,0]:7.4f} {R_ego[i,1]:7.4f} {R_ego[i,2]:7.4f}]\n"
    text += f"\nTranslation (global position):\n  [{t_ego[0]:.2f}, {t_ego[1]:.2f}, {t_ego[2]:.2f}]"
    ax.text(0.5, 0.5, text, fontsize=9, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))
    ax.set_title('Ego → Global', fontsize=12, fontweight='bold')

    # LiDAR → Ego
    ax = axes[1][0]
    ax.axis('off')
    R_lid = Quaternion(lidar_cs['rotation']).rotation_matrix
    t_lid = np.array(lidar_cs['translation'])
    text = "LIDAR_TOP → Ego\n\n"
    text += "Rotation (3×3):\n"
    for i in range(3):
        text += f"  [{R_lid[i,0]:7.4f} {R_lid[i,1]:7.4f} {R_lid[i,2]:7.4f}]\n"
    text += f"\nTranslation:\n  [{t_lid[0]:.4f}, {t_lid[1]:.4f}, {t_lid[2]:.4f}]"
    ax.text(0.5, 0.5, text, fontsize=9, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#FCE4EC', alpha=0.8))
    ax.set_title('LiDAR → Ego', fontsize=12, fontweight='bold')

    # 变换链流程图
    ax = axes[1][1]
    ax.axis('off')
    text = ("┌─────────────┐    ┌──────────┐    ┌──────────┐\n"
            "│ Pixel (u,v) │ ←─ │ Camera   │ ←─ │ Ego      │\n"
            "│             │ K  │ (x,y,z)  │ T  │ (x,y,z)  │\n"
            "└─────────────┘    └──────────┘    └──────────┘\n"
            "                                        ↕ T_ego\n"
            "┌─────────────┐    ┌──────────┐    ┌──────────┐\n"
            "│ LiDAR       │ ──→│ Ego      │ ──→│ Global   │\n"
            "│ (x,y,z)     │ T  │ (x,y,z)  │ T  │ (x,y,z)  │\n"
            "└─────────────┘    └──────────┘    └──────────┘")
    ax.text(0.5, 0.5, text, fontsize=9, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8))
    ax.set_title('Transform Chain Overview', fontsize=12, fontweight='bold')

    # Timestamp 信息
    ax = axes[1][2]
    ax.axis('off')
    text = "Timestamps & Metadata\n\n"
    text += f"Sample timestamp:\n  {sample['timestamp']}\n\n"
    text += f"CAM_FRONT data:\n"
    text += f"  file: {cam_data['filename'].split('/')[-1]}\n"
    text += f"  timestamp: {cam_data['timestamp']}\n"
    text += f"  width×height: {cam_data['width']}×{cam_data['height']}\n\n"
    text += f"LIDAR_TOP data:\n"
    text += f"  file: {lidar_data['filename'].split('/')[-1]}\n"
    text += f"  timestamp: {lidar_data['timestamp']}\n"
    text += f"  n_points: {lidar_data.get('nbr_points', 'N/A')}"
    ax.text(0.5, 0.5, text, fontsize=9, family='monospace',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#FFFDE7', alpha=0.8))
    ax.set_title('Sample Metadata', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'real_coordinate_transforms.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[8/10] Saved: {save_path}")
    return save_path


def viz_09_temporal_structure(nusc, scene):
    """[REAL DATA] 时间关系图 — scene → sample → sample_data → sweep"""
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    # 上图：sample timeline
    ax = axes[0]
    sample_token = scene['first_sample_token']
    timestamps = []
    sample_tokens = []
    idx = 0

    while sample_token != '':
        s = nusc.get('sample', sample_token)
        timestamps.append(s['timestamp'])
        sample_tokens.append(sample_token)
        sample_token = s['next']
        idx += 1

    timestamps = np.array(timestamps)
    t_relative = (timestamps - timestamps[0]) / 1e6  # seconds

    ax.plot(t_relative, np.ones_like(t_relative), 'ro-', markersize=8,
            linewidth=2, label=f'Keyframes ({len(timestamps)})')

    for i, t in enumerate(t_relative):
        ax.text(t, 1.03, f'KF{i}', fontsize=7, ha='center', rotation=45)

    # 获取第一个 sample 的 CAM_FRONT sweeps
    first_sample = nusc.get('sample', sample_tokens[0])
    cam_sd_token = first_sample['data']['CAM_FRONT']
    sweep_timestamps = []

    # 向后追踪 sweeps
    sd = nusc.get('sample_data', cam_sd_token)
    while sd['next'] != '':
        sd = nusc.get('sample_data', sd['next'])
        if not sd['is_key_frame']:
            sweep_timestamps.append(sd['timestamp'])
        if len(sweep_timestamps) > 50:
            break

    if sweep_timestamps:
        sweep_rel = (np.array(sweep_timestamps) - timestamps[0]) / 1e6
        ax.plot(sweep_rel, np.ones_like(sweep_rel) * 0.97, 'b|',
                markersize=12, label=f'CAM_FRONT sweeps ({len(sweep_timestamps)})')

    ax.set_xlim(-0.5, t_relative[-1] + 0.5)
    ax.set_ylim(0.9, 1.1)
    ax.set_xlabel('Time (seconds from scene start)', fontsize=12)
    ax.set_title(f'[REAL DATA] {scene["name"]} Timeline — {len(timestamps)} Keyframes\n'
                 f'Duration: {t_relative[-1]:.1f}s | Avg interval: {np.mean(np.diff(t_relative)):.3f}s',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yticks([])

    # 下图：数据结构示意
    ax = axes[1]
    ax.axis('off')

    # 展示 scene → sample → sample_data 层级
    struct_text = (
        f"Scene: {scene['name']}\n"
        f"  ├── Description: {scene['description']}\n"
        f"  ├── Log: {scene['log_token'][:12]}...\n"
        f"  ├── Keyframes: {len(timestamps)}\n"
        f"  ├── Duration: {t_relative[-1]:.1f} seconds\n"
        f"  │\n"
        f"  ├── Sample[0] (keyframe)\n"
        f"  │     ├── token: {sample_tokens[0][:16]}...\n"
        f"  │     ├── timestamp: {timestamps[0]}\n"
        f"  │     ├── data:\n"
        f"  │     │     ├── CAM_FRONT → sample_data → [keyframe + sweeps]\n"
        f"  │     │     ├── CAM_FRONT_LEFT → ...\n"
        f"  │     │     ├── CAM_FRONT_RIGHT → ...\n"
        f"  │     │     ├── CAM_BACK → ...\n"
        f"  │     │     ├── CAM_BACK_LEFT → ...\n"
        f"  │     │     ├── CAM_BACK_RIGHT → ...\n"
        f"  │     │     ├── LIDAR_TOP → ...\n"
        f"  │     │     ├── RADAR_FRONT → ...\n"
        f"  │     │     └── ... (5 radars total)\n"
        f"  │     └── anns: [{len(first_sample['anns'])} annotations]\n"
        f"  │\n"
        f"  ├── Sample[1] → ...\n"
        f"  └── Sample[{len(timestamps)-1}] (last)\n"
    )

    ax.text(0.05, 0.95, struct_text, fontsize=10, family='monospace',
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))
    ax.set_title('Data Structure Hierarchy', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'real_temporal_structure.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[9/10] Saved: {save_path}")
    return save_path


def viz_10_combined_bev(nusc, sample):
    """[REAL DATA] 综合 BEV — LiDAR + GT Boxes + Radar 叠加"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # LiDAR
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc_path = os.path.join(NUSCENES_ROOT, lidar_data['filename'])
    pc = LidarPointCloud.from_file(pc_path)

    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(lidar_cs['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_cs['translation']))

    ax.scatter(pc.points[0], pc.points[1], c=pc.points[2], cmap='Greys_r',
               s=0.2, alpha=0.3, vmin=-2, vmax=2)

    # GT Boxes
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        color = get_color_for_category(ann['category_name'])
        corners = box.bottom_corners()[:2, :]
        corners = np.concatenate([corners, corners[:, :1]], axis=1)
        ax.plot(corners[0], corners[1], color=color, linewidth=2, alpha=0.8)

    # Radar
    for idx_r, radar_ch in enumerate(RADAR_CHANNELS):
        try:
            radar_data = nusc.get('sample_data', sample['data'][radar_ch])
            radar_path = os.path.join(NUSCENES_ROOT, radar_data['filename'])
            rpc = RadarPointCloud.from_file(radar_path)
            rcs = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
            rot = Quaternion(rcs['rotation']).rotation_matrix
            trans = np.array(rcs['translation'])
            pts_ego = rot @ rpc.points[:3, :] + trans.reshape(3, 1)
            ax.scatter(pts_ego[0], pts_ego[1], c='lime', s=15, alpha=0.6,
                       marker='D', zorder=6)
        except:
            pass

    # Ego vehicle
    ax.plot(0, 0, 'r*', markersize=20, zorder=10)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m) [→ Forward]', fontsize=12)
    ax.set_ylabel('Y (m) [← Left]', fontsize=12)
    ax.set_title('[REAL DATA] Combined BEV: LiDAR + GT 3D Boxes + Radar\n'
                 'Grey=LiDAR points | Colored boxes=GT annotations | Green ◆=Radar',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # 图例
    legend_handles = [patches.Patch(color='lime', label='Radar')]
    for cat, color in list(CATEGORY_COLORS.items())[:6]:
        legend_handles.append(patches.Patch(color=color, label=cat))
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

    save_path = os.path.join(OUTPUT_DIR, 'real_combined_bev.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[10/10] Saved: {save_path}")
    return save_path


def main():
    print("=" * 70)
    print("[REAL DATA] nuScenes Multi-Sensor Teaching Visualization")
    print("=" * 70)

    nusc, scene, sample = setup()

    results = {}
    results['surround'] = viz_01_surround_cameras(nusc, sample)
    results['lidar_bev'] = viz_02_lidar_bev(nusc, sample)
    results['radar_bev'] = viz_03_radar_bev(nusc, sample)
    results['gt_bbox'] = viz_04_gt_bbox_bev(nusc, sample)
    results['lidar_proj'] = viz_05_lidar_to_camera(nusc, sample)
    results['box_proj'] = viz_06_3dbox_to_camera(nusc, sample)
    results['sensor_layout'] = viz_07_sensor_layout(nusc, sample)
    results['coord_transform'] = viz_08_coordinate_transform(nusc, sample)
    results['temporal'] = viz_09_temporal_structure(nusc, scene)
    results['combined_bev'] = viz_10_combined_bev(nusc, sample)

    print("\n" + "=" * 70)
    print("[COMPLETE] All 10 real-data visualizations generated:")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
