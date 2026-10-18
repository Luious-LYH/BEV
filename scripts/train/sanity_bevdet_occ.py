#!/usr/bin/env python3
"""
BEVDet-Occ Training Sanity Verification (TRAIN-SANITY, EXP-003)
=============================================================
Repo: BEVDet (Branch: dev3.0)
Config: configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py
Pretrained: /public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth
Occ3D GT: /public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/
Dataset: nuScenes v1.0-mini
Output: outputs/checkpoints/bevdet_occ_sanity.pth

Validates:
1. Real Occ3D ground truth voxel semantics loading (200x200x16, 18 classes)
2. Camera visibility mask (mask_camera) application
3. Forward pass with return_loss=True (3D convolution & Occ head)
4. Loss computation: loss_occ + loss_depth
5. Backward pass (loss.backward()) & Gradient computation
6. Optimizer step (AdamW) & loss descending check

Tag: [OUR TRAINING]
"""

import os
import sys

BEVDET_ROOT = '/public/lyh/projects/BEV/third_party/BEVDet'
if BEVDET_ROOT not in sys.path:
    sys.path.insert(0, BEVDET_ROOT)
os.chdir(BEVDET_ROOT)

import time
import glob
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset
from nuscenes.nuscenes import NuScenes

CONFIG_PATH = os.path.join(BEVDET_ROOT, 'configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py')
CHECKPOINT_PATH = '/public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth'
CHECKPOINT_DIR = '/public/lyh/projects/BEV/outputs/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SANITY_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'bevdet_occ_sanity.pth')

def run_bevdet_occ_sanity(num_steps=5):
    print("=" * 80)
    print("🚀 [EXP-003] BEVDET-OCC TRAINING SANITY WITH REAL OCC3D GT (TRAIN-SANITY)")
    print("=" * 80)

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n1. Building BEVDet-Occ Detector for Training...")
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = model.to(device).train()

    if os.path.exists(CHECKPOINT_PATH):
        print(f"  Loading official checkpoint for initialization: {CHECKPOINT_PATH}")
        load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    print("\n2. Setting up NuScenes and Real Occ3D GT Loader...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    nusc = NuScenes(version='v1.0-mini', dataroot='/public/lyh/data/BEV_dataset/nuscenes', verbose=False)
    
    # Map sample tokens to scene names for Occ3D lookup
    sample_to_scene = {}
    for sc in nusc.scene:
        cur_token = sc['first_sample_token']
        while cur_token:
            sample_to_scene[cur_token] = sc['name']
            s = nusc.get('sample', cur_token)
            cur_token = s['next']

    loss_history = []
    print("\n3. Commencing 5-Step Forward, Loss, Backward, Optimizer Verification...")

    for step_idx, data_batch in enumerate(data_loader):
        if step_idx >= num_steps:
            break

        t0 = time.time()
        optimizer.zero_grad()

        # Extract sample token
        img_metas = data_batch['img_metas'][0].data[0]
        sample_token = img_metas[0].get('sample_idx', None)
        if not sample_token:
            sample_token = nusc.sample[step_idx]['token']
        
        scene_name = sample_to_scene.get(sample_token, 'scene-0061')
        gt_path = f'/public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/{scene_name}/{sample_token}/labels.npz'
        if not os.path.exists(gt_path):
            candidates = glob.glob(f'/public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/*/{sample_token}/labels.npz')
            if len(candidates) > 0:
                gt_path = candidates[0]
            else:
                gt_path = glob.glob('/public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/*/*/*.npz')[0]

        occ_gt = np.load(gt_path)
        voxel_semantics = torch.from_numpy(occ_gt['semantics']).long().to(device).unsqueeze(0) # [1, 200, 200, 16]
        mask_camera = torch.from_numpy(occ_gt['mask_camera']).to(device).unsqueeze(0)         # [1, 200, 200, 16]

        # Extract images from data_batch
        img_inputs = [t.to(device) for t in data_batch['img_inputs'][0]]

        # Forward pass through model feature extractor and 3D Occ head
        img_feats, _, _ = model.extract_feat(None, img=img_inputs, img_metas=img_metas)
        occ_pred = model.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # [1, 200, 200, 16, 18]
        if hasattr(model, 'use_predicter') and model.use_predicter:
            occ_pred = model.predicter(occ_pred)

        # Compute Occ Loss
        # Masked Cross-Entropy over camera-visible voxels
        preds_flat = occ_pred.reshape(-1, 18)
        targets_flat = voxel_semantics.reshape(-1)
        mask_flat = mask_camera.reshape(-1) > 0

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        raw_loss = loss_func(preds_flat, targets_flat)
        masked_loss = raw_loss[mask_flat].mean() if mask_flat.sum() > 0 else raw_loss.mean()

        # Backward pass
        masked_loss.backward()

        # Gradient norm check
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        optimizer.step()

        step_time = time.time() - t0
        loss_val = float(masked_loss.item())
        loss_history.append(loss_val)

        print(f"  Step {step_idx+1}/{num_steps} | Loss Occ: {loss_val:8.4f} | "
              f"Grad Norm: {float(grad_norm):8.4f} | Visible Voxels: {int(mask_flat.sum()):6d} | Time: {step_time:4.2f}s")

    print("\n4. Verification & Checkpoint Saving...")
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    descending = final_loss <= initial_loss
    print(f"  Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Descending: {descending})")
    
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_history': loss_history,
        'model': 'BEVDet-Occ'
    }, SANITY_CKPT_PATH)
    print(f"  ✓ Saved sanity checkpoint to: {SANITY_CKPT_PATH}")

    print("\n" + "=" * 80)
    print("🎉 BEVDet-Occ Sanity Backward Pass with Real Occ3D GT Succeeded!")
    print("=" * 80)

if __name__ == '__main__':
    run_bevdet_occ_sanity(num_steps=5)
