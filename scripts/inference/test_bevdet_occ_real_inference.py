#!/usr/bin/env python3
"""
BEVDet-Occ Real Model Inference & Dense Occupancy Visualization
================================================================
Repo: BEVDet
Branch: dev3.0
Config: configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py
Checkpoint: /public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth
Dataset: nuScenes v1.0-mini
Output: outputs/bevdet_occ_real_output/

Tags: [REAL MODEL OUTPUT]
"""

import os
import sys

BEVDET_ROOT = '/public/lyh/projects/BEV/third_party/BEVDet'
sys.path.insert(0, BEVDET_ROOT)

import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

CONFIG_PATH = os.path.join(BEVDET_ROOT, 'configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py')
CHECKPOINT_PATH = '/public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth'
OUTPUT_DIR = '/public/lyh/projects/BEV/outputs/bevdet_occ_real_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free'
]

def run_occ_inference():
    print("=" * 70)
    print("BEVDET-OCC REAL MODEL INFERENCE ON NUSCENES MINI")
    print("=" * 70)
    
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.data_root = '/public/lyh/data/BEV_dataset/nuscenes/'
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = os.path.join(cfg.data_root, 'bevdetv3-nuscenes_infos_val.pkl')
    cfg.model.train_cfg = None
    cfg.model.img_backbone.pop('pretrained', None)
    cfg.model.img_backbone.init_cfg = dict(
        type='Pretrained',
        checkpoint='/public/lyh/pretrained_model/BEV/backbones/resnet50-11ad3fa6.pth'
    )
    
    print("\n1. Building BEVDet-Occ model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    print("\n2. Loading official checkpoint...")
    checkpoint = load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        model = model.to(device)
    model.eval()
    print("  BEVDet-Occ model loaded successfully!")
    
    print("\n3. Building dataset...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    
    print("\n4. Running occupancy inference on sample...")
    for i, data_batch in enumerate(data_loader):
        t0 = time.time()
        with torch.no_grad():
            res = model(return_loss=False, rescale=True, **data_batch)
        t1 = time.time()
        latency = (t1 - t0) * 1000
        print(f"  [Sample {i+1}] Latency: {latency:.2f} ms")
        
        occ_pred = res[0]
        if isinstance(occ_pred, dict) and 'occ' in occ_pred:
            occ_tensor = occ_pred['occ']
        elif isinstance(occ_pred, (list, tuple)):
            occ_tensor = occ_pred[0]
        else:
            occ_tensor = occ_pred
            
        if isinstance(occ_tensor, torch.Tensor):
            occ_np = occ_tensor.cpu().numpy()
        else:
            occ_np = np.array(occ_tensor)
            
        print(f"  Occupancy prediction shape: {occ_np.shape}")
        
        # 5. Visualizing Dense Occupancy BEV slice & 3D Voxel Slice
        print("\n5. Plotting Dense Occupancy Visualization...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0A0D14')
        for ax in axes:
            ax.set_facecolor('#121722')
            
        # Left: BEV height-projected occupancy (ground plane / driveable)
        ax_bev = axes[0]
        if occ_np.ndim == 3:
            # [X, Y, Z] semantics
            bev_slice = occ_np[:, :, occ_np.shape[2] // 2]
        elif occ_np.ndim == 4:
            # [C, X, Y, Z] logits -> argmax
            bev_slice = np.argmax(occ_np, axis=0)[:, :, occ_np.shape[3] // 2]
        else:
            bev_slice = np.squeeze(occ_np)
            if bev_slice.ndim > 2:
                bev_slice = bev_slice[:, :, 0]
                
        im1 = ax_bev.imshow(bev_slice.T, cmap='tab20', origin='lower')
        ax_bev.set_title("BEVDet-Occ Dense 3D Voxel Slice (Z = 0m Ground Plane)\nSemantics: 17 Classes + Free Space",
                         color='white', fontsize=12, fontweight='bold')
        ax_bev.axis('off')
        ax_bev.text(0.03, 0.94, 'REAL MODEL OUTPUT', transform=ax_bev.transAxes, color='white', fontweight='bold',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='#00C853', alpha=0.9))

        # Right: Side view (X-Z elevation slice)
        ax_side = axes[1]
        if occ_np.ndim == 3:
            side_slice = occ_np[occ_np.shape[0] // 2, :, :]
        elif occ_np.ndim == 4:
            side_slice = np.argmax(occ_np, axis=0)[occ_np.shape[1] // 2, :, :]
        else:
            side_slice = bev_slice
            
        im2 = ax_side.imshow(side_slice.T, cmap='tab20', origin='lower')
        ax_side.set_title("BEVDet-Occ Elevation Cross-Section (Y-Z Plane)\nVertical Voxel Column Distribution",
                          color='white', fontsize=12, fontweight='bold')
        ax_side.axis('off')
        ax_side.text(0.03, 0.94, 'REAL MODEL OUTPUT', transform=ax_side.transAxes, color='white', fontweight='bold',
                     fontsize=9, bbox=dict(boxstyle='round', facecolor='#00C853', alpha=0.9))

        plt.suptitle("BEVDet-Occ Real Dense 3D Occupancy Perception on nuScenes mini (Official Checkpoint: 24e_384704)",
                     color='white', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'bevdet_occ_real_dense_voxel.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Occupancy visualization saved to: {save_path}")
        break
        
    print("\n" + "=" * 70)
    print("BEVDET-OCC REAL INFERENCE FINISHED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_occ_inference()
