#!/usr/bin/env python3
"""
BEVFormer-Small Real Model Inference & Spatial Cross-Attention Visualization
=============================================================================
Repo: BEVFormer
Branch: master
Config: projects/configs/bevformer/bevformer_small.py
Checkpoint: /public/lyh/pretrained_model/BEV/bevformer/bevformer_small_epoch_24.pth
Dataset: nuScenes v1.0-mini
Output: outputs/bevformer_real_output/

Teaching:
- BEV Query & 3D Pillar Reference Points (pc_range [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
- Camera Projection Matrix (lidar2img) mapping 3D reference points into 2D camera views
- Spatial Cross Attention (SCA) & Deformable Attention sampling
- BEV Embedding Grid [150, 150, 256] & CenterHead 3D Bounding Box Predictions

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

BEVFORMER_ROOT = '/public/lyh/projects/BEV/third_party/BEVFormer'
if BEVFORMER_ROOT not in sys.path:
    sys.path.insert(0, BEVFORMER_ROOT)
os.chdir(BEVFORMER_ROOT)

import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
import projects.mmdet3d_plugin
from mmdet.models import build_detector
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from nuscenes.nuscenes import NuScenes

CONFIG_PATH = os.path.join(BEVFORMER_ROOT, 'projects/configs/bevformer/bevformer_small.py')
CHECKPOINT_PATH = '/public/lyh/pretrained_model/BEV/bevformer/bevformer_small_epoch_24.pth'
OUTPUT_DIR = '/public/lyh/projects/BEV/outputs/bevformer_real_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

def run_bevformer_inference():
    print("=" * 70)
    print("BEVFORMER-SMALL REAL MODEL INFERENCE & VISUALIZATION")
    print("=" * 70)

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.pretrained = None
    if 'img_backbone' in cfg.model and 'pretrained' in cfg.model.img_backbone:
        cfg.model.img_backbone.pretrained = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n1. Building BEVFormer Detector...")
    model = build_detector(cfg.model).to(device).eval()

    print("\n2. Loading Official Pretrained Checkpoint...")
    checkpoint = load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
    print("  BEVFormer-Small official checkpoint loaded successfully!")

    print("\n3. Loading Real nuScenes mini sample data...")
    nusc = NuScenes(version='v1.0-mini', dataroot='/public/lyh/data/BEV_dataset/nuscenes', verbose=False)
    sample = nusc.sample[0]

    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    raw_pil_imgs = []
    imgs = []
    lidar2imgs = []
    
    # Scale 0.8: 1600*0.8 = 1280, 900*0.8 = 720
    H_in, W_in = 720, 1280

    from PIL import Image
    from pyquaternion import Quaternion

    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)
    ref_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    T_lidar2ego = np.eye(4, dtype=np.float32)
    T_lidar2ego[:3, :3] = Quaternion(ref_cs['rotation']).rotation_matrix
    T_lidar2ego[:3, 3] = np.array(ref_cs['translation'])

    for cam_name in cam_names:
        cam_token = sample['data'][cam_name]
        cam_sd = nusc.get('sample_data', cam_token)
        img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
        pil_img = Image.open(img_path).convert('RGB')
        raw_pil_imgs.append(pil_img)
        orig_w, orig_h = pil_img.size
        pil_resized = pil_img.resize((W_in, H_in), Image.BILINEAR)
        img_arr = np.array(pil_resized, dtype=np.float32)

        # BEVFormer normalization: mean=[103.530, 116.280, 123.675] (BGR order in config style='caffe')
        # to_rgb=False: RGB to BGR
        img_bgr = img_arr[:, :, ::-1]
        img_norm = (img_bgr - np.array([103.530, 116.280, 123.675])) / np.array([1.0, 1.0, 1.0])
        imgs.append(img_norm.transpose(2, 0, 1))

        cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        K_3x3 = np.array(cs['camera_intrinsic'], dtype=np.float32)
        K_3x3[0, :] *= (W_in / orig_w)
        K_3x3[1, :] *= (H_in / orig_h)
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:3, :3] = K_3x3

        T_cam2ego = np.eye(4, dtype=np.float32)
        T_cam2ego[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
        T_cam2ego[:3, 3] = np.array(cs['translation'])
        T_lidar2cam = np.linalg.inv(T_cam2ego) @ T_lidar2ego
        lidar2img = viewpad @ T_lidar2cam
        lidar2imgs.append(lidar2img)

    img_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device).unsqueeze(0)  # [1, 6, 3, 720, 1280]
    lidar2imgs_np = np.stack(lidar2imgs).astype(np.float32)  # [6, 4, 4]

    img_metas = [{
        'lidar2img': lidar2imgs_np,
        'can_bus': np.zeros(18, dtype=np.float32),
        'img_shape': [(H_in, W_in, 3)] * 6,
        'ori_shape': [(900, 1600, 3)] * 6,
        'pad_shape': [(H_in, W_in, 3)] * 6,
        'box_type_3d': LiDARInstance3DBoxes
    }]

    print(f"  Input Image Tensor Shape: {img_tensor.shape}")

    print("\n4. Running Forward Pass...")
    t0 = time.time()
    with torch.no_grad():
        img_feats = model.extract_feat(img=img_tensor, img_metas=img_metas)
        outs = model.pts_bbox_head(img_feats, img_metas, prev_bev=None)
        bbox_list = model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000
    print(f"  Forward Pass Time: {latency_ms:.2f} ms ({1000/latency_ms:.1f} FPS)")

    # BEV Embedding: [150*150, 1, 256] -> [150, 150, 256]
    bev_embed = outs['bev_embed']
    bev_h, bev_w = cfg.bev_h_, cfg.bev_w_
    bev_map = bev_embed.reshape(bev_h, bev_w, -1).norm(dim=-1).cpu().numpy()
    print(f"  BEV Feature Map Shape: {bev_map.shape} (H={bev_h}, W={bev_w})")

    # 3D Bounding Boxes
    bboxes_3d, scores, labels = bbox_list[0]
    bboxes_np = bboxes_3d.tensor.cpu().numpy()
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    conf_thresh = 0.25
    pos_mask = scores_np > conf_thresh
    filt_boxes = bboxes_np[pos_mask]
    filt_scores = scores_np[pos_mask]
    filt_labels = labels_np[pos_mask]
    print(f"  Predicted 3D Instances (> {conf_thresh}): {len(filt_boxes)} / {len(bboxes_np)}")

    # 5. Extract Reference Points Projection for Spatial Cross Attention Teaching
    # pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    # Pillar points along Z: 4 points per (x, y) grid cell
    ref_points_3d = []
    grid_step = 10  # sub-sample for clear visualization
    for gx in np.linspace(-40, 40, 9):
        for gy in np.linspace(5, 45, 9):
            for gz in [-3.0, -1.0, 1.0]:
                ref_points_3d.append([gx, gy, gz, 1.0])
    ref_points_3d = np.array(ref_points_3d)

    # Project to CAM_FRONT
    l2i_front = lidar2imgs_np[0]
    pts_cam = (l2i_front @ ref_points_3d.T).T
    valid_cam = pts_cam[:, 2] > 0.5
    u = pts_cam[valid_cam, 0] / pts_cam[valid_cam, 2]
    v = pts_cam[valid_cam, 1] / pts_cam[valid_cam, 2]
    in_view = (u >= 0) & (u < W_in) & (v >= 0) & (v < H_in)
    u_proj = u[in_view]
    v_proj = v[in_view]

    # 6. Plotting Comprehensive BEVFormer Teaching Figure
    print("\n5. Plotting BEVFormer Real Teaching Visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='#0A0D14')
    for ax in axes:
        ax.set_facecolor('#121722')

    # Panel 1: CAM_FRONT + 3D Pillar Reference Points (Spatial Cross-Attention)
    ax_cam = axes[0]
    cam_front_rgb = raw_pil_imgs[0].resize((W_in, H_in))
    ax_cam.imshow(cam_front_rgb)
    ax_cam.scatter(u_proj, v_proj, c='#00E5FF', s=35, edgecolors='white', linewidth=0.8,
                   alpha=0.9, label='BEV 3D Pillar Reference Points')
    ax_cam.set_title("Spatial Cross Attention (SCA) Reference Point Projection\nCAM_FRONT: 3D Pillar Queries -> 2D Feature Sampling",
                     color='white', fontsize=11, fontweight='bold')
    ax_cam.axis('off')
    ax_cam.legend(loc='lower left', facecolor='#1A2234', edgecolor='#00E5FF', labelcolor='white')
    ax_cam.text(0.03, 0.94, 'REAL MODEL OUTPUT', transform=ax_cam.transAxes, color='white', fontweight='bold',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='#00C853', alpha=0.9))

    # Panel 2: Authentic BEV Feature Map [150, 150, 256]
    ax_bev = axes[1]
    im_bev = ax_bev.imshow(bev_map.T, cmap='magma', origin='lower', extent=[-51.2, 51.2, -51.2, 51.2])
    ax_bev.plot(0, 0, marker='^', markersize=10, color='#00E5FF', label='Ego Vehicle')
    ax_bev.set_title(f"Authentic BEV Query Feature Map [{bev_h} x {bev_w} x 256]\nSpatial Activation Energy (L2-Norm across Channels)",
                     color='white', fontsize=11, fontweight='bold')
    ax_bev.set_xlabel("X (Ego-Right, meters)", color='#8C9BAE')
    ax_bev.set_ylabel("Y (Ego-Front, meters)", color='#8C9BAE')
    ax_bev.tick_params(colors='#8C9BAE')
    ax_bev.legend(loc='lower left', facecolor='#1A2234', edgecolor='#00E5FF', labelcolor='white')
    ax_bev.text(0.03, 0.94, 'REAL MODEL OUTPUT', transform=ax_bev.transAxes, color='white', fontweight='bold',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='#00C853', alpha=0.9))

    # Panel 3: 3D Detection Predictions Top-Down View
    ax_det = axes[2]
    ax_det.set_xlim(-51.2, 51.2)
    ax_det.set_ylim(-51.2, 51.2)
    ax_det.grid(True, linestyle='--', color='#2A364F', alpha=0.6)
    ax_det.plot(0, 0, marker='^', markersize=12, color='#00E5FF', label='Ego Vehicle')

    # Draw range circles
    for r in [15, 30, 45]:
        circle = plt.Circle((0, 0), r, fill=False, edgecolor='#2A364F', linestyle=':')
        ax_det.add_artist(circle)
        ax_det.text(r * 0.707, r * 0.707, f"{r}m", color='#5A6E85', fontsize=8)

    palette = {'car': '#00E5FF', 'truck': '#FFB300', 'pedestrian': '#FF1744', 'bus': '#76FF03'}
    for box, score, label in zip(filt_boxes, filt_scores, filt_labels):
        cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else 'obj'
        col = palette.get(cls_name, '#E040FB')
        cx, cy, cz, w, l, h, yaw = box[:7]
        rect = plt.Rectangle((cx - w/2, cy - l/2), w, l, angle=np.degrees(yaw),
                             fill=False, edgecolor=col, linewidth=2)
        ax_det.add_patch(rect)
        ax_det.text(cx, cy, f"{cls_name} {score:.2f}", color=col, fontsize=8, fontweight='bold')

    ax_det.set_title(f"BEVFormer-Small 3D Predictions ({len(filt_boxes)} Detections)\nCenterHead on Refined Object Queries",
                     color='white', fontsize=11, fontweight='bold')
    ax_det.set_xlabel("X (Ego-Right, meters)", color='#8C9BAE')
    ax_det.set_ylabel("Y (Ego-Front, meters)", color='#8C9BAE')
    ax_det.tick_params(colors='#8C9BAE')
    ax_det.legend(loc='lower left', facecolor='#1A2234', edgecolor='#00E5FF', labelcolor='white')
    ax_det.text(0.03, 0.94, 'REAL MODEL OUTPUT', transform=ax_det.transAxes, color='white', fontweight='bold',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='#00C853', alpha=0.9))

    plt.suptitle("BEVFormer-Small Real Model Output & Spatial Cross Attention on nuScenes mini\n"
                 f"Latency: {latency_ms:.1f} ms ({1000/latency_ms:.1f} FPS) | Grid: 150x150 | Queries: 900",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'bevformer_real_predictions.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  BEVFormer visualization saved to: {save_path}")

    # Benchmark json
    import json
    bench_data = {
        'model': 'BEVFormer-Small',
        'checkpoint': CHECKPOINT_PATH,
        'latency_ms': latency_ms,
        'fps': 1000 / latency_ms,
        'num_predictions': int(len(filt_boxes)),
        'device': str(device),
        'tag': 'REAL MODEL OUTPUT'
    }
    with open(os.path.join(OUTPUT_DIR, 'bevformer_benchmark.json'), 'w') as f:
        json.dump(bench_data, f, indent=2)
    print(f"  Benchmark saved to: {os.path.join(OUTPUT_DIR, 'bevformer_benchmark.json')}")

    print("\n" + "=" * 70)
    print("BEVFORMER-SMALL REAL INFERENCE FINISHED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_bevformer_inference()
