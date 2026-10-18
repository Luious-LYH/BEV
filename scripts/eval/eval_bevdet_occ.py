#!/usr/bin/env python3
"""
BEVDet-Occ Ground Truth Evaluation with Real Occ3D Dataset (EVAL-OCC)
=====================================================================
Repo: BEVDet (Branch: dev3.0)
Model: BEVDet-Occ (ResNet50-4D-Stereo)
Config: configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py
Occ3D GT: /public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/

Evaluates:
1. Our trained checkpoint: outputs/checkpoints/bevdet_occ_real_best.pth
2. Official checkpoint: /public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth

Outputs:
- artifacts/experiment_exports/bevdet_occ/eval_metrics.json
- artifacts/experiment_exports/bevdet_occ/bevdet_occ_pred.npz
- artifacts/experiment_exports/bevdet_occ/viewer_bevdet_occ.json

Tag: [EVALUATION]
"""

import os
import sys

BEVDET_ROOT = '/public/lyh/projects/BEV/third_party/BEVDet'
if BEVDET_ROOT not in sys.path:
    sys.path.insert(0, BEVDET_ROOT)
os.chdir(BEVDET_ROOT)

import time
import json
import glob
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset
from nuscenes.nuscenes import NuScenes

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]

CONFIG_PATH = os.path.join(BEVDET_ROOT, 'configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py')
PROJECT_ROOT = '/public/lyh/projects/BEV'
EXPORT_DIR = os.path.join(PROJECT_ROOT, 'artifacts/experiment_exports/bevdet_occ')
os.makedirs(EXPORT_DIR, exist_ok=True)

OUR_CKPT = os.path.join(PROJECT_ROOT, 'outputs/checkpoints/bevdet_occ_real_best.pth')
if not os.path.exists(OUR_CKPT):
    OUR_CKPT = os.path.join(PROJECT_ROOT, 'outputs/checkpoints/bevdet_occ_sanity.pth')
OFFICIAL_CKPT = '/public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth'

def compute_occ_iou(preds, gts, masks, num_classes=18):
    """
    Compute IoU per class over masked voxels.
    preds, gts, masks: numpy arrays of shape [N_voxels]
    """
    valid = masks > 0
    p = preds[valid]
    g = gts[valid]

    ious = {}
    for c in range(num_classes):
        intersection = np.logical_and(p == c, g == c).sum()
        union = np.logical_or(p == c, g == c).sum()
        if union == 0:
            ious[c] = float('nan')
        else:
            ious[c] = float(intersection / union)
    return ious

def run_eval_bevdet_occ(num_samples=15):
    print("=" * 80)
    print("🔬 [EVAL-OCC] BEVDET-OCC EVALUATION AGAINST REAL OCC3D GROUND TRUTH")
    print(f"   Validation Samples: {num_samples} | Metric: Occ3D mIoU / Camera-Masked IoU")
    print("=" * 80)

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.data_root = '/public/lyh/data/BEV_dataset/nuscenes/'
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = os.path.join(cfg.data_root, 'bevdetv3-nuscenes_infos_val.pkl')
    cfg.model.train_cfg = None
    cfg.model.img_backbone.pop('pretrained', None)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg')).to(device).eval()

    # NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='/public/lyh/data/BEV_dataset/nuscenes', verbose=False)
    sample_to_scene = {}
    for sc in nusc.scene:
        cur_token = sc['first_sample_token']
        while cur_token:
            sample_to_scene[cur_token] = sc['name']
            s = nusc.get('sample', cur_token)
            cur_token = s['next']

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    loader_iter = iter(data_loader)

    eval_results = {}

    ckpts = [
        ('our_subset', OUR_CKPT),
        ('official', OFFICIAL_CKPT)
    ]

    for ckpt_tag, ckpt_path in ckpts:
        print(f"\n---> Evaluating Checkpoint: {ckpt_tag} ({ckpt_path})")
        if not os.path.exists(ckpt_path):
            print(f"  Warning: Checkpoint {ckpt_path} not found. Skipping.")
            continue

        raw = torch.load(ckpt_path, map_location='cpu')
        state_dict = raw['state_dict'] if 'state_dict' in raw else raw
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        all_preds = []
        all_gts = []
        all_masks = []
        sample_latencies = []

        # Reset loader
        loader_iter = iter(data_loader)

        first_sample_pred = None
        first_sample_gt = None
        first_sample_mask = None

        with torch.no_grad():
            for i in range(num_samples):
                try:
                    data_batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(data_loader)
                    data_batch = next(loader_iter)

                img_metas = data_batch['img_metas'][0].data[0]
                sample_token = img_metas[0].get('sample_idx', None)
                if not sample_token:
                    sample_token = nusc.sample[i % len(nusc.sample)]['token']

                scene_name = sample_to_scene.get(sample_token, 'scene-0061')
                gt_path = f'/public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/{scene_name}/{sample_token}/labels.npz'
                if not os.path.exists(gt_path):
                    candidates = glob.glob(f'/public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/*/{sample_token}/labels.npz')
                    gt_path = candidates[0] if len(candidates) > 0 else glob.glob('/public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/*/*/*.npz')[0]

                occ_gt = np.load(gt_path)
                voxel_semantics = occ_gt['semantics']
                mask_camera = occ_gt['mask_camera']

                img_inputs = [t.to(device) for t in data_batch['img_inputs'][0]]

                torch.cuda.synchronize()
                t0 = time.time()
                img_feats, _, _ = model.extract_feat(None, img=img_inputs, img_metas=img_metas)
                occ_pred = model.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
                if hasattr(model, 'use_predicter') and model.use_predicter:
                    occ_pred = model.predicter(occ_pred)
                torch.cuda.synchronize()
                sample_latencies.append(time.time() - t0)

                pred_classes = torch.argmax(occ_pred, dim=-1).squeeze(0).cpu().numpy().astype(np.uint8)

                if i == 0:
                    first_sample_pred = pred_classes
                    first_sample_gt = voxel_semantics
                    first_sample_mask = mask_camera

                all_preds.append(pred_classes.reshape(-1))
                all_gts.append(voxel_semantics.reshape(-1))
                all_masks.append(mask_camera.reshape(-1))

        # Flatten and compute metrics
        flat_preds = np.concatenate(all_preds)
        flat_gts = np.concatenate(all_gts)
        flat_masks = np.concatenate(all_masks)

        ious = compute_occ_iou(flat_preds, flat_gts, flat_masks, num_classes=18)
        valid_ious = [v for k, v in ious.items() if not np.isnan(v) and k < 17] # exclude free
        miou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0

        per_class_table = {}
        for c_idx, c_name in enumerate(CLASS_NAMES):
            val = ious.get(c_idx, float('nan'))
            per_class_table[c_name] = round(val * 100, 2) if not np.isnan(val) else None

        avg_latency = float(np.mean(sample_latencies))
        fps = float(1.0 / avg_latency) if avg_latency > 0 else 0.0

        print(f"  Result [{ckpt_tag}]: Occ3D mIoU: {miou*100:.2f}% | Latency: {avg_latency*1000:.2f} ms | FPS: {fps:.2f}")

        eval_results[ckpt_tag] = {
            'checkpoint': ckpt_path,
            'miou_percent': round(miou * 100, 2),
            'camera_masked_iou': per_class_table,
            'avg_latency_ms': round(avg_latency * 1000, 2),
            'fps': round(fps, 2),
            'samples_evaluated': num_samples
        }

        # If this is our subset or official, export viewer format
        if ckpt_tag == 'our_subset' or (ckpt_tag == 'official' and 'our_subset' not in eval_results):
            # Save raw npz
            npz_export = os.path.join(EXPORT_DIR, 'bevdet_occ_pred.npz')
            np.savez_compressed(
                npz_export,
                pred_semantics=first_sample_pred,
                gt_semantics=first_sample_gt,
                mask_camera=first_sample_mask
            )
            print(f"  ✓ Saved 3D voxel array: {npz_export}")

            # Export viewer subsampled points
            # Extract non-free voxels (c < 17) within camera frustum
            vis_indices = np.where((first_sample_mask > 0) & (first_sample_pred < 17))
            N_vis = len(vis_indices[0])
            step = max(1, N_vis // 5000)
            xs = ((vis_indices[0][::step] - 100) * 0.4).tolist()
            ys = ((vis_indices[1][::step] - 100) * 0.4).tolist()
            zs = ((vis_indices[2][::step] - 8) * 0.5).tolist()
            sems = first_sample_pred[vis_indices][::step].tolist()

            viewer_payload = {
                'model': 'BEVDet-Occ',
                'checkpoint': ckpt_path,
                'total_occupied_voxels': N_vis,
                'sampled_voxels': len(xs),
                'classes': CLASS_NAMES,
                'voxels': {
                    'x': xs,
                    'y': ys,
                    'z': zs,
                    'semantic': sems
                }
            }
            viewer_json_path = os.path.join(EXPORT_DIR, 'viewer_bevdet_occ.json')
            with open(viewer_json_path, 'w') as f:
                json.dump(viewer_payload, f)
            print(f"  ✓ Saved 3D viewer JSON: {viewer_json_path} ({len(xs)} voxels)")

    # Save summary eval json
    out_json = os.path.join(EXPORT_DIR, 'eval_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"  ✓ Saved evaluation summary to: {out_json}")

    print("\n" + "=" * 80)
    print("🎉 BEVDet-Occ Ground Truth Evaluation Completed Successfully!")
    print("=" * 80)

if __name__ == '__main__':
    run_eval_bevdet_occ(num_samples=15)
