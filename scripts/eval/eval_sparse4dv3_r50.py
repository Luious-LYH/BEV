#!/usr/bin/env python3
"""
Sparse4Dv3-R50 Evaluation on nuScenes mini (EXP-002)
===================================================
Repo: Sparse4D (Version: v3)
Config: projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
Checkpoints:
  - Our trained model: outputs/checkpoints/sparse4dv3_r50_real_best.pth
  - Official baseline: /public/lyh/pretrained_model/BEV/sparse4d/sparse4dv3_r50.pth
Dataset: nuScenes v1.0-mini (val split, 81 keyframes)
Output:
  - outputs/eval/sparse4dv3_r50_eval.json
  - artifacts/experiment_exports/sparse4d/our_prediction.json
  - artifacts/experiment_exports/sparse4d/official_prediction.json
  - artifacts/experiment_exports/sparse4d/metrics.json

Column in Benchmark Table: [Our Model (12 Epochs) / Official]
"""

import os
import sys

# Safe registry patch for cross-repo OpenMMLab compatibility
from mmcv.utils import Registry
old_register = Registry._register_module
def safe_register(self, module, module_name=None, force=True):
    return old_register(self, module, module_name=module_name, force=True)
Registry._register_module = safe_register

SPARSE4D_ROOT = '/public/lyh/projects/BEV/third_party/Sparse4D'
if SPARSE4D_ROOT not in sys.path:
    sys.path.insert(0, SPARSE4D_ROOT)
os.chdir(SPARSE4D_ROOT)

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
from nuscenes.nuscenes import NuScenes

CONFIG_PATH = os.path.join(SPARSE4D_ROOT, 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py')
PROJECT_ROOT = '/public/lyh/projects/BEV'
OUR_CKPT = os.path.join(PROJECT_ROOT, 'outputs/checkpoints/sparse4dv3_r50_real_best.pth')
if not os.path.exists(OUR_CKPT):
    OUR_CKPT = os.path.join(PROJECT_ROOT, 'outputs/checkpoints/sparse4d_r50_sanity.pth')
OFFICIAL_CKPT = '/public/lyh/pretrained_model/BEV/sparse4d/sparse4dv3_r50.pth'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs/eval')
EXPORT_DIR = os.path.join(PROJECT_ROOT, 'artifacts/experiment_exports/sparse4d')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def evaluate_checkpoint(ckpt_path, label="our_subset", num_val_samples=20):
    print("=" * 70)
    print(f"SPARSE4DV3 EVALUATION: [{label.upper()}]")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.use_deformable_func = False
    cfg.model.head.deformable_model.use_deformable_func = False
    cfg.model.img_backbone.pretrained = None
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    anchor_path = '/public/lyh/projects/BEV/nuscenes_kmeans900.npy'
    cfg.model.head.instance_bank.anchor = anchor_path

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
    H_in, W_in = 256, 704

    predictions_list = []
    latencies = []

    with torch.no_grad():
        for idx, sample in enumerate(val_samples):
            imgs = []
            projection_mats = []

            lidar_token = sample['data']['LIDAR_TOP']
            lidar_sd = nusc.get('sample_data', lidar_token)
            ref_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            ref_trans = np.array(ref_pose['translation'])
            ref_rot = Quaternion(ref_pose['rotation']).rotation_matrix

            for cam_name in cam_names:
                cam_token = sample['data'][cam_name]
                cam_sd = nusc.get('sample_data', cam_token)
                img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
                pil_img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = pil_img.size
                pil_img = pil_img.resize((W_in, H_in), Image.BILINEAR)
                img_arr = np.array(pil_img, dtype=np.float32)
                img_arr = (img_arr - np.array([123.675, 116.280, 103.530])) / np.array([58.395, 57.120, 57.375])
                imgs.append(img_arr.transpose(2, 0, 1))

                cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
                cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
                K = np.eye(4, dtype=np.float32)
                K_3x3 = np.array(cam_cs['camera_intrinsic'], dtype=np.float32)
                K_3x3[0, :] *= (W_in / orig_w)
                K_3x3[1, :] *= (H_in / orig_h)
                K[:3, :3] = K_3x3

                T_cam2ego = np.eye(4, dtype=np.float32)
                T_cam2ego[:3, :3] = Quaternion(cam_cs['rotation']).rotation_matrix
                T_cam2ego[:3, 3] = np.array(cam_cs['translation'])

                T_ego2global = np.eye(4, dtype=np.float32)
                T_ego2global[:3, :3] = Quaternion(cam_pose['rotation']).rotation_matrix
                T_ego2global[:3, 3] = np.array(cam_pose['translation'])

                T_ref_global2ego = np.eye(4, dtype=np.float32)
                T_ref_global2ego[:3, :3] = ref_rot.T
                T_ref_global2ego[:3, 3] = -ref_rot.T @ ref_trans

                T_ref2cam = np.linalg.inv(T_cam2ego) @ np.linalg.inv(T_ego2global) @ np.linalg.inv(T_ref_global2ego)
                proj_mat = K @ T_ref2cam
                projection_mats.append(proj_mat)

            imgs_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            proj_tensor = torch.tensor(np.stack(projection_mats), dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            image_wh_tensor = torch.tensor([[W_in, H_in]] * 6, dtype=torch.float32, device=device).unsqueeze(0).contiguous()

            timestamp = torch.tensor([0.0], dtype=torch.float32, device=device)
            img_metas = [{'T_global': np.eye(4, dtype=np.float32), 'T_global_inv': np.eye(4, dtype=np.float32)}]

            data = {
                'projection_mat': proj_tensor,
                'image_wh': image_wh_tensor,
                'timestamp': timestamp,
                'img_metas': img_metas
            }

            torch.cuda.synchronize()
            t0 = time.time()
            result = model(img=imgs_tensor, return_loss=False, **data)
            torch.cuda.synchronize()
            latency = (time.time() - t0) * 1000.0
            latencies.append(latency)

            pred_boxes = []
            if len(result) > 0 and 'pts_bbox' in result[0]:
                pts_res = result[0]['pts_bbox']
                boxes_3d = pts_res['boxes_3d'].tensor.cpu().numpy()
                scores_3d = pts_res['scores_3d'].cpu().numpy()
                labels_3d = pts_res['labels_3d'].cpu().numpy()

                for b_i in range(min(len(scores_3d), 50)):
                    if scores_3d[b_i] > 0.2:
                        box = boxes_3d[b_i]
                        cls_idx = int(labels_3d[b_i])
                        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else 'unknown'
                        pred_boxes.append({
                            'translation': [float(np.round(x, 2)) for x in box[:3]],
                            'size': [float(np.round(x, 2)) for x in box[3:6]],
                            'yaw': float(np.round(box[6], 3)),
                            'score': float(np.round(scores_3d[b_i], 3)),
                            'name': cls_name
                        })

            predictions_list.append({
                'sample_token': sample['token'],
                'latency_ms': float(np.round(latency, 2)),
                'num_boxes': len(pred_boxes),
                'boxes': pred_boxes[:10]
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == len(val_samples):
                print(f"  Processed {idx+1}/{len(val_samples)} samples | Latency: {latency:.1f}ms | Detections: {len(pred_boxes)}")

    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    total_boxes = sum(p['num_boxes'] for p in predictions_list)

    summary = {
        'model': 'Sparse4Dv3-R50',
        'checkpoint': ckpt_path,
        'label': label,
        'num_samples_evaluated': len(val_samples),
        'avg_latency_ms': float(np.round(avg_latency, 2)),
        'fps': float(np.round(fps, 2)),
        'total_detections': total_boxes,
        'avg_detections_per_frame': float(np.round(total_boxes / len(val_samples), 2)),
        'predictions': predictions_list
    }

    export_filename = f'{label}_prediction.json'
    export_path = os.path.join(EXPORT_DIR, export_filename)
    with open(export_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved {label} predictions to: {export_path}")

    return summary

def run_evaluation():
    our_summary = None
    if os.path.exists(OUR_CKPT):
        our_summary = evaluate_checkpoint(OUR_CKPT, label='our_subset', num_val_samples=20)
    
    official_summary = None
    if os.path.exists(OFFICIAL_CKPT):
        official_summary = evaluate_checkpoint(OFFICIAL_CKPT, label='official', num_val_samples=20)

    eval_results_file = os.path.join(OUTPUT_DIR, 'sparse4dv3_r50_eval.json')
    combined = {
        'our_subset': our_summary,
        'official': official_summary
    }
    with open(eval_results_file, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"✓ Combined evaluation summary saved to: {eval_results_file}")

if __name__ == '__main__':
    run_evaluation()
