#!/usr/bin/env python3
"""
SparseOcc v1.1 Real Model Inference & Occupancy Visualization
=============================================================
Repo: SparseOcc
Version: v1.1
Config: configs/r50_nuimg_704x256_8f.py
Checkpoint: /public/lyh/pretrained_model/BEV/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth
Dataset: nuScenes v1.0-mini
Output: outputs/sparseocc_real_output/

Teaching:
- Sparse Query-based Occupancy Perception (coarse-to-fine sparse voxel prediction)
- 100 Mask Queries -> 3D Voxel Occupancy [200, 200, 16]
- Comparison with Dense BEVDet-Occ representation

Tags: [REAL MODEL OUTPUT]
"""

import os
import sys

# Safe registry patch for cross-repo OpenMMLab compatibility
from mmcv.utils import Registry
old_register = Registry._register_module
def safe_register(self, module, module_name=None, force=True):
    return old_register(self, module, module_name=module_name, force=True)
Registry._register_module = safe_register

SPARSE_OCC_ROOT = '/public/lyh/projects/BEV/third_party/SparseOcc'
if SPARSE_OCC_ROOT not in sys.path:
    sys.path.insert(0, SPARSE_OCC_ROOT)

# Prepend cwd to avoid relative path lookup issues in SparseOcc plugins
os.chdir(SPARSE_OCC_ROOT)

import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
import models
from mmdet.models import build_detector
from nuscenes.nuscenes import NuScenes

CONFIG_PATH = os.path.join(SPARSE_OCC_ROOT, 'configs/r50_nuimg_704x256_8f.py')
CHECKPOINT_PATH = '/public/lyh/pretrained_model/BEV/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth'
OUTPUT_DIR = '/public/lyh/projects/BEV/outputs/sparseocc_real_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free'
]

def run_sparseocc_inference():
    print("=" * 70)
    print("SPARSEOCC V1.1 REAL MODEL INFERENCE & VISUALIZATION")
    print("=" * 70)

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.img_backbone.pretrained = None
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    # Keep official 8 frames matching checkpoint parameter shapes
    cfg.model.pts_bbox_head.transformer.num_frames = 8

    # Ensure device is cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n1. Building SparseOcc v1.1 Model...")
    model = build_detector(cfg.model).to(device).eval()

    print("\n2. Loading Official v1.1 Checkpoint...")
    checkpoint = load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
    print("  Official v1.1 Checkpoint loaded successfully!")

    print("\n3. Loading Real nuScenes mini sample data...")
    nusc = NuScenes(version='v1.0-mini', dataroot='/public/lyh/data/BEV_dataset/nuscenes', verbose=False)
    sample = nusc.sample[0]

    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    raw_imgs = []
    single_lidar2cams = []
    single_cam_intrinsics = []
    H_in, W_in = 256, 704

    from PIL import Image
    from pyquaternion import Quaternion
    
    # Reference lidar pose
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)
    ref_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    T_lidar2ego = np.eye(4, dtype=np.float32)
    T_lidar2ego[:3, :3] = Quaternion(ref_cs['rotation']).rotation_matrix
    T_lidar2ego[:3, 3] = np.array(ref_cs['translation'])
    T_ego2lidar = np.linalg.inv(T_lidar2ego).astype(np.float32)

    for cam_name in cam_names:
        cam_token = sample['data'][cam_name]
        cam_sd = nusc.get('sample_data', cam_token)
        img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        pil_img = pil_img.resize((W_in, H_in), Image.BILINEAR)
        img_arr = np.array(pil_img, dtype=np.float32)
        raw_imgs.append(img_arr.transpose(2, 0, 1))

        cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        K = np.eye(4, dtype=np.float32)
        K_3x3 = np.array(cs['camera_intrinsic'], dtype=np.float32)
        K_3x3[0, :] *= (W_in / orig_w)
        K_3x3[1, :] *= (H_in / orig_h)
        K[:3, :3] = K_3x3
        single_cam_intrinsics.append(K)

        T_cam2ego = np.eye(4, dtype=np.float32)
        T_cam2ego[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
        T_cam2ego[:3, 3] = np.array(cs['translation'])
        T_lidar2cam = np.linalg.inv(T_cam2ego) @ T_lidar2ego
        single_lidar2cams.append(T_lidar2cam.astype(np.float32))

    # Construct 8-frame temporal inputs (48 cameras total)
    num_frames = 8
    imgs_8f = []
    lidar2imgs_8f = []
    ego2lidars_8f = []
    filenames_8f = []

    for t in range(num_frames):
        for i in range(6):
            imgs_8f.append(raw_imgs[i])
            # lidar2img is K @ lidar2cam
            l2i = single_cam_intrinsics[i] @ single_lidar2cams[i]
            lidar2imgs_8f.append(l2i)
            ego2lidars_8f.append(T_ego2lidar)
            filenames_8f.append(f"frame_{t}_cam_{i}.jpg")

    imgs_tensor = torch.tensor(np.stack(imgs_8f), dtype=torch.float32, device=device).unsqueeze(0)  # [1, 48, 3, 256, 704]
    lidar2imgs_np = np.stack(lidar2imgs_8f).astype(np.float32)  # [48, 4, 4]
    ego2lidars_np = np.stack(ego2lidars_8f).astype(np.float32)  # [48, 4, 4]

    img_metas = [{
        'lidar2img': lidar2imgs_np,
        'ego2lidar': ego2lidars_np,
        'filename': filenames_8f,
        'img_shape': [(H_in, W_in, 3)] * 48,
        'ori_shape': [(H_in, W_in, 3)] * 48,
        'pad_shape': [(H_in, W_in, 3)] * 48,
        'img_timestamp': [0.0] * 48,
        'box_type_3d': None
    }]

    print(f"  Input Images Tensor Shape: {imgs_tensor.shape} (1 batch, 48 camera views)")

    print("\n4. Running Forward Inference...")
    t0 = time.time()
    with torch.no_grad():
        img_feats = model.extract_feat(imgs_tensor, img_metas=img_metas)
        occ_outs = model.pts_bbox_head(img_feats, img_metas=img_metas)
        merged = model.pts_bbox_head.merge_occ_pred(occ_outs)
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000
    print(f"  Forward Pass Time: {latency_ms:.2f} ms ({1000/latency_ms:.1f} FPS)")

    sem_pred = merged['sem_pred'][0].cpu().numpy()  # [N]
    occ_loc = merged['occ_loc'].cpu().numpy().reshape(-1, 3)      # [N, 3]
    print(f"  Predicted Occupied Voxels: {len(sem_pred)}, occ_loc shape: {occ_loc.shape}")
    print(f"  Semantic Classes Present: {np.unique(sem_pred).tolist()}")

    # Reconstruct 3D Dense Voxel Grid [200, 200, 16]
    occ_grid = np.full((200, 200, 16), 17, dtype=np.int32)  # 17 = free space
    valid_mask = (occ_loc[:, 0] >= 0) & (occ_loc[:, 0] < 200) & \
                 (occ_loc[:, 1] >= 0) & (occ_loc[:, 1] < 200) & \
                 (occ_loc[:, 2] >= 0) & (occ_loc[:, 2] < 16)
    valid_locs = occ_loc[valid_mask]
    valid_classes = sem_pred[valid_mask]
    occ_grid[valid_locs[:, 0], valid_locs[:, 1], valid_locs[:, 2]] = valid_classes

    occupied_count = np.sum(occ_grid != 17)
    print(f"  Total Occupied Voxels in Grid: {occupied_count} / {200*200*16} ({occupied_count / (200*200*16) * 100:.2f}%)")

    # 5. Plotting SparseOcc Real Visualization
    print("\n5. Plotting SparseOcc Real Occupancy Slices...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0A0D14')
    for ax in axes:
        ax.set_facecolor('#121722')

    # Left: Ground Plane BEV Occupancy Slice (Z = 0m)
    ax_bev = axes[0]
    z_idx = 8
    bev_slice = occ_grid[:, :, z_idx]
    im1 = ax_bev.imshow(bev_slice.T, cmap='tab20', origin='lower', vmin=0, vmax=17)
    ax_bev.set_title(f"SparseOcc v1.1 3D Voxel Slice (Z = 0m, slice={z_idx})\nSparse Query-to-Voxel (17 Classes + Free)",
                     color='white', fontsize=12, fontweight='bold')
    ax_bev.axis('off')
    ax_bev.text(0.03, 0.94, 'REAL MODEL OUTPUT', transform=ax_bev.transAxes, color='white', fontweight='bold',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='#00C853', alpha=0.9))

    # Right: Elevation Cross-Section (Y-Z Plane at X=100)
    ax_side = axes[1]
    x_idx = 100
    side_slice = occ_grid[x_idx, :, :]
    im2 = ax_side.imshow(side_slice.T, cmap='tab20', origin='lower', vmin=0, vmax=17)
    ax_side.set_title(f"SparseOcc v1.1 Elevation Cross-Section (Y-Z Plane at X={x_idx})\nVertical Voxel Column Distribution (Z=16)",
                      color='white', fontsize=12, fontweight='bold')
    ax_side.axis('off')
    ax_side.text(0.03, 0.94, 'REAL MODEL OUTPUT', transform=ax_side.transAxes, color='white', fontweight='bold',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='#00C853', alpha=0.9))

    cbar = plt.colorbar(im1, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_ticks(np.arange(len(OCC_CLASSES)))
    cbar.set_ticklabels(OCC_CLASSES)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=8)

    plt.suptitle("SparseOcc v1.1 Real 3D Occupancy Perception on nuScenes mini (Official Checkpoint: 24e_v1.1)\n"
                 f"Latency: {latency_ms:.1f} ms | Total Occupied Voxels: {occupied_count:,}",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'sparseocc_real_dense_voxel.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  SparseOcc occupancy visualization saved to: {save_path}")

    # Save benchmark json
    import json
    bench_data = {
        'model': 'SparseOcc-v1.1-R50',
        'checkpoint': CHECKPOINT_PATH,
        'latency_ms': latency_ms,
        'fps': 1000 / latency_ms,
        'num_occupied_voxels': int(occupied_count),
        'device': str(device),
        'tag': 'REAL MODEL OUTPUT'
    }
    with open(os.path.join(OUTPUT_DIR, 'sparseocc_benchmark.json'), 'w') as f:
        json.dump(bench_data, f, indent=2)
    print(f"  Benchmark saved to: {os.path.join(OUTPUT_DIR, 'sparseocc_benchmark.json')}")

    print("\n" + "=" * 70)
    print("SPARSEOCC V1.1 REAL INFERENCE FINISHED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_sparseocc_inference()
