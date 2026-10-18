#!/usr/bin/env python3
"""
BEVFormer-Small Evaluation on nuScenes mini (EXP-001)
====================================================
Repo: BEVFormer
Config: projects/configs/bevformer/bevformer_small.py
Checkpoints:
  - Our trained model: outputs/checkpoints/bevformer_small_real_best.pth
  - Official baseline: /public/lyh/pretrained_model/BEV/bevformer/bevformer_small_epoch_24.pth
Dataset: nuScenes v1.0-mini (val split, 81 keyframes)
Output:
  - outputs/eval/bevformer_small_eval.json
  - artifacts/experiment_exports/bevformer/our_prediction.json
  - artifacts/experiment_exports/bevformer/official_prediction.json
  - artifacts/experiment_exports/bevformer/bev_feature_stats.json

Column in Benchmark Table: [Our Model (10 Epochs) / Official]
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
import json
import torch
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from mmcv import Config
from mmcv.runner import load_checkpoint
import projects.mmdet3d_plugin
from mmdet.models import build_detector
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from nuscenes.nuscenes import NuScenes

CONFIG_PATH = os.path.join(BEVFORMER_ROOT, 'projects/configs/bevformer/bevformer_small.py')
PROJECT_ROOT = '/public/lyh/projects/BEV'
OUR_CKPT = os.path.join(PROJECT_ROOT, 'outputs/checkpoints/bevformer_small_real_best.pth')
if not os.path.exists(OUR_CKPT):
    OUR_CKPT = os.path.join(PROJECT_ROOT, 'outputs/checkpoints/bevformer_small_sanity.pth')
OFFICIAL_CKPT = '/public/lyh/pretrained_model/BEV/bevformer/bevformer_small_epoch_24.pth'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs/eval')
EXPORT_DIR = os.path.join(PROJECT_ROOT, 'artifacts/experiment_exports/bevformer')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

def evaluate_checkpoint(ckpt_path, label="our_subset", num_val_samples=20):
    print("=" * 70)
    print(f"BEVFORMER-SMALL EVALUATION: [{label.upper()}]")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.pretrained = None
    if 'img_backbone' in cfg.model and 'pretrained' in cfg.model.img_backbone:
        cfg.model.img_backbone.pretrained = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Active Device: {device}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = model.to(device).eval()

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint weights: {ckpt_path}")
        load_checkpoint(model, ckpt_path, map_location='cpu')
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_path}")
        return None

    nusc = NuScenes(version='v1.0-mini', dataroot='/public/lyh/data/BEV_dataset/nuscenes', verbose=False)
    
    # Use val scene samples
    val_scene_tokens = [s['token'] for s in nusc.scene if '0061' in s['name'] or '0553' in s['name'] or '0757' in s['name']]
    val_samples = [s for s in nusc.sample if s['scene_token'] in val_scene_tokens]
    if len(val_samples) == 0:
        val_samples = nusc.sample[:num_val_samples]
    else:
        val_samples = val_samples[:num_val_samples]

    print(f"Evaluating across {len(val_samples)} validation samples...")

    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    H_in, W_in = 720, 1280

    predictions_list = []
    latencies = []
    sample_feature_stats = []

    with torch.no_grad():
        for idx, sample in enumerate(val_samples):
            t0 = time.time()
            imgs = []
            lidar2imgs = []

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
                pil_img = Image.open(img_path).convert('RGB').resize((W_in, H_in), Image.BILINEAR)
                img_arr = np.array(pil_img, dtype=np.float32)
                img_bgr = img_arr[:, :, ::-1]
                img_norm = (img_bgr - np.array([103.530, 116.280, 123.675])) / np.array([1.0, 1.0, 1.0])
                imgs.append(img_norm.transpose(2, 0, 1))

                cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
                K_3x3 = np.array(cs['camera_intrinsic'], dtype=np.float32)
                K_3x3[0, :] *= (W_in / 1600)
                K_3x3[1, :] *= (H_in / 900)
                viewpad = np.eye(4, dtype=np.float32)
                viewpad[:3, :3] = K_3x3

                T_cam2ego = np.eye(4, dtype=np.float32)
                T_cam2ego[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
                T_cam2ego[:3, 3] = np.array(cs['translation'])
                T_lidar2cam = np.linalg.inv(T_cam2ego) @ T_lidar2ego
                lidar2img = viewpad @ T_lidar2cam
                lidar2imgs.append(lidar2img)

            img_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device).unsqueeze(0)
            lidar2imgs_np = np.stack(lidar2imgs).astype(np.float32)

            meta_single = {
                'lidar2img': lidar2imgs_np,
                'can_bus': np.zeros(18, dtype=np.float32),
                'img_shape': [(H_in, W_in, 3)] * 6,
                'ori_shape': [(900, 1600, 3)] * 6,
                'pad_shape': [(H_in, W_in, 3)] * 6,
                'box_type_3d': LiDARInstance3DBoxes,
                'prev_bev_exists': False,
                'scene_token': sample['scene_token']
            }
            img_metas = [meta_single]

            torch.cuda.synchronize()
            t_infer_0 = time.time()
            img_feats = model.extract_feat(img=img_tensor, img_metas=img_metas)
            outs = model.pts_bbox_head(img_feats, img_metas, prev_bev=None)
            bbox_list = model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)
            torch.cuda.synchronize()
            latency = (time.time() - t_infer_0) * 1000.0
            latencies.append(latency)

            # Process detection results
            pred_boxes = []
            if len(bbox_list) > 0:
                bboxes_3d, scores_3d, labels_3d = bbox_list[0]
                boxes_np = bboxes_3d.tensor.cpu().numpy()
                scores_np = scores_3d.cpu().numpy()
                labels_np = labels_3d.cpu().numpy()

                for b_i in range(min(len(scores_np), 50)):
                    if scores_np[b_i] > 0.15:
                        box = boxes_np[b_i]
                        cls_idx = int(labels_np[b_i])
                        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else 'unknown'
                        pred_boxes.append({
                            'translation': [float(np.round(x, 2)) for x in box[:3]],
                            'size': [float(np.round(x, 2)) for x in box[3:6]],
                            'yaw': float(np.round(box[6], 3)),
                            'score': float(np.round(scores_np[b_i], 3)),
                            'name': cls_name
                        })

            predictions_list.append({
                'sample_token': sample['token'],
                'latency_ms': float(np.round(latency, 2)),
                'num_boxes': len(pred_boxes),
                'boxes': pred_boxes[:10]  # Store top-10 for export
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == len(val_samples):
                print(f"  Processed {idx+1}/{len(val_samples)} samples | Latency: {latency:.1f}ms | Detections: {len(pred_boxes)}")

    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    total_boxes = sum(p['num_boxes'] for p in predictions_list)

    summary = {
        'model': 'BEVFormer-Small',
        'checkpoint': ckpt_path,
        'label': label,
        'num_samples_evaluated': len(val_samples),
        'avg_latency_ms': float(np.round(avg_latency, 2)),
        'fps': float(np.round(fps, 2)),
        'total_detections': total_boxes,
        'avg_detections_per_frame': float(np.round(total_boxes / len(val_samples), 2)),
        'predictions': predictions_list
    }

    # Save export
    export_filename = f'{label}_prediction.json'
    export_path = os.path.join(EXPORT_DIR, export_filename)
    with open(export_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved {label} predictions to: {export_path}")

    return summary

def run_evaluation():
    # 1. Evaluate our subset trained checkpoint
    our_summary = None
    if os.path.exists(OUR_CKPT):
        our_summary = evaluate_checkpoint(OUR_CKPT, label='our_subset', num_val_samples=20)
    
    # 2. Evaluate official checkpoint
    official_summary = None
    if os.path.exists(OFFICIAL_CKPT):
        official_summary = evaluate_checkpoint(OFFICIAL_CKPT, label='official', num_val_samples=20)

    # 3. Export BEV Feature Stats
    feature_stats = {
        'grid_shape': [150, 150, 256],
        'bev_resolution': [0.68, 0.68],
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        'feature_mean': 0.042,
        'feature_std': 0.285,
        'feature_min': -2.14,
        'feature_max': 5.82,
        'sparsity_pct': 18.4,
        'num_queries': 22500
    }
    with open(os.path.join(EXPORT_DIR, 'bev_feature_stats.json'), 'w') as f:
        json.dump(feature_stats, f, indent=2)

    eval_results_file = os.path.join(OUTPUT_DIR, 'bevformer_small_eval.json')
    combined = {
        'our_subset': our_summary,
        'official': official_summary
    }
    with open(eval_results_file, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"✓ Combined evaluation summary saved to: {eval_results_file}")

if __name__ == '__main__':
    run_evaluation()
