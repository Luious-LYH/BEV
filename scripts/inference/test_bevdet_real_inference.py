#!/usr/bin/env python3
"""
Real Model Inference & Validation for BEVDet-R50
================================================
Repo: BEVDet
Branch: dev3.0
Config: configs/bevdet/bevdet-r50-cbgs.py
Checkpoint: /public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-cbgs.pth
Dataset: nuScenes v1.0-mini
Output: outputs/bevdet_real_output/

Tags: [REAL MODEL OUTPUT]
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

BEVDET_ROOT = '/public/lyh/projects/BEV/third_party/BEVDet'
sys.path.insert(0, BEVDET_ROOT)

CONFIG_PATH = os.path.join(BEVDET_ROOT, 'configs/bevdet/bevdet-r50-cbgs.py')
CHECKPOINT_PATH = '/public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-cbgs.pth'
OUTPUT_DIR = '/public/lyh/projects/BEV/outputs/bevdet_real_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_inference():
    print("=" * 70)
    print("BEVDET-R50 REAL MODEL INFERENCE ON NUSCENES MINI")
    print("=" * 70)
    
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.data_root = '/public/lyh/data/BEV_dataset/nuscenes/'
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = os.path.join(cfg.data_root, 'bevdetv3-nuscenes_infos_val.pkl')
    cfg.model.train_cfg = None
    
    print("\n1. Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    print("\n2. Loading official checkpoint...")
    checkpoint = load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        from mmcv.parallel import MMDataParallel
        model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully on {device} with MMDataParallel!")
    
    print("\n3. Building nuScenes mini test dataset...")
    dataset = build_dataset(cfg.data.test)
    print(f"  Dataset size: {len(dataset)} samples")
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )
    
    print("\n4. Running real model inference on first 3 samples...")
    results = []
    latencies = []
    
    with torch.no_grad():
        for i, data_batch in enumerate(data_loader):
            if i >= 3:
                break
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            # Forward pass
            result = model(return_loss=False, rescale=True, **data_batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)
            
            results.append(result)
            pts_bbox = result[0]['pts_bbox']
            boxes_3d = pts_bbox['boxes_3d']
            scores_3d = pts_bbox['scores_3d']
            labels_3d = pts_bbox['labels_3d']
            
            # Filter by confidence
            mask = scores_3d > 0.25
            filtered_boxes = boxes_3d[mask]
            filtered_scores = scores_3d[mask]
            filtered_labels = labels_3d[mask]
            
            print(f"  [Sample {i+1}] Latency: {latencies[-1]:.2f} ms | Detected: {len(filtered_boxes)} boxes (conf > 0.25)")
            if len(filtered_scores) > 0:
                print(f"    Top score: {filtered_scores[0].item():.3f}, label: {filtered_labels[0].item()}")
                
    avg_latency = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
    print(f"\n  Average Model-only Latency: {avg_latency:.2f} ms ({1000/avg_latency:.1f} FPS)")
    
    # Save inference visualization
    print("\n5. Generating REAL MODEL OUTPUT visualization...")
    vis_path = os.path.join(OUTPUT_DIR, 'bevdet_real_predictions.png')
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0A0D14')
    ax.set_facecolor('#121722')
    
    # Draw ego vehicle
    ego_rect = plt.Rectangle((-1.0, -2.0), 2.0, 4.0, fill=True, color='#00E5FF', alpha=0.9, zorder=5)
    ax.add_patch(ego_rect)
    ax.text(0, -3.0, 'EGO', color='#00E5FF', fontweight='bold', ha='center', fontsize=9)
    
    # Draw range rings
    for r in [10, 20, 30, 40, 50]:
        circle = plt.Circle((0, 0), r, fill=False, color='#26334D', linestyle='--', linewidth=0.8)
        ax.add_patch(circle)
        ax.text(r*0.707, r*0.707, f'{r}m', color='#64748B', fontsize=8)
        
    # Draw detected boxes from Sample 0
    if len(results) > 0:
        pts = results[0][0]['pts_bbox']
        b_3d = pts['boxes_3d']
        s_3d = pts['scores_3d']
        l_3d = pts['labels_3d']
        
        mask = s_3d > 0.20
        b_sel = b_3d[mask].tensor.cpu().numpy()
        s_sel = s_3d[mask].cpu().numpy()
        l_sel = l_3d[mask].cpu().numpy()
        
        class_names = cfg.class_names
        colors = ['#FF3366', '#00FF66', '#FFB833', '#A066FF', '#33CCFF', '#FF9933', '#FF66CC', '#66FFCC', '#FFFF33', '#CCCCCC']
        
        for box, score, label in zip(b_sel, s_sel, l_sel):
            x, y, z, dx, dy, dz, yaw = box[:7]
            c = colors[int(label) % len(colors)]
            c_name = class_names[int(label)] if int(label) < len(class_names) else f'C{label}'
            
            # Rotate box corners
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            corners = np.array([[-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2]])
            corners_rot = np.dot(corners, rot.T) + np.array([x, y])
            
            poly = plt.Polygon(corners_rot, fill=True, facecolor=c, alpha=0.35, edgecolor=c, linewidth=1.5, zorder=4)
            ax.add_patch(poly)
            ax.text(x, y, f'{c_name}\n{score:.2f}', color='white', fontsize=7, ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))
            
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_xlabel('X (meters, Right)', color='#94A3B8', fontsize=11)
    ax.set_ylabel('Y (meters, Forward)', color='#94A3B8', fontsize=11)
    ax.tick_params(colors='#94A3B8')
    ax.grid(True, color='#1E283D', linestyle='-', linewidth=0.5)
    
    # REAL MODEL OUTPUT badge
    ax.text(0.03, 0.96, 'REAL MODEL OUTPUT', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#00C853', alpha=0.95))
            
    plt.title(f'BEVDet-R50 Real Model Inference on nuScenes mini\nOfficial Checkpoint: bevdet-r50-cbgs.pth | Latency: {avg_latency:.1f}ms',
              color='white', fontsize=13, fontweight='bold', pad=15)
              
    plt.tight_layout()
    plt.savefig(vis_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to: {vis_path}")
    print("\n" + "=" * 70)
    print("BEVDET REAL INFERENCE FINISHED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_inference()
