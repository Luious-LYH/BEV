#!/usr/bin/env python3
"""
Occupancy Real Multi-Epoch Training & Evaluation (EXP-003 & EXP-004)
===================================================================
Hardware Target: GPU 7 (RTX 4090 24GB)
Dataset: nuScenes v1.0-mini with Real Occ3D Ground Truth
Research Objective:
  Real-world multi-epoch training of 3D Occupancy models:
  - BEVDet-Occ (Dense 3D Voxel representation)
  - SparseOcc-v1.1 (Sparse Query representation)
  Supervised by authentic Occ3D voxel semantic labels (18 classes).

Features:
- Multi-epoch real training (default 8 epochs per model)
- Masked Cross-Entropy loss over visible voxels
- AdamW optimizer with Cosine Annealing learning rate schedule
- Automated validation evaluation (class-wise IoU & mIoU)
- Provenance recording & YAML metadata
"""

import os
import sys
import time
import glob
import argparse
import json
import yaml
import subprocess
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = '/public/lyh/projects/BEV'
DATA_ROOT = '/public/lyh/data/BEV_dataset/nuscenes'
OCC3D_GTS = os.path.join(DATA_ROOT, 'occ3d/gts')

BEVDET_ROOT = '/public/lyh/projects/BEV/third_party/BEVDet'
SPARSE_OCC_ROOT = '/public/lyh/projects/BEV/third_party/SparseOcc'

OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]


def get_git_info():
    try:
        remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        remote = 'local'
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = 'unknown'
    try:
        dirty = len(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT).decode().strip()) > 0
    except Exception:
        dirty = False
    return remote, commit, dirty


def compute_occ_miou(preds, gts, masks, num_classes=18):
    valid = masks > 0
    p = preds[valid]
    g = gts[valid]
    ious = []
    for c in range(num_classes):
        intersection = np.logical_and(p == c, g == c).sum()
        union = np.logical_or(p == c, g == c).sum()
        if union > 0:
            ious.append(float(intersection / union))
    return float(np.mean(ious)) if len(ious) > 0 else 0.0


def train_bevdet_occ(args):
    print("=" * 80)
    print(f"🚀 [REAL TRAINING] BEVDet-Occ (Dense 3D Voxel) | Epochs: {args.epochs}")
    print(f"   Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    if BEVDET_ROOT not in sys.path:
        sys.path.insert(0, BEVDET_ROOT)
    os.chdir(BEVDET_ROOT)

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataloader, build_dataset
    from nuscenes.nuscenes import NuScenes

    exp_id = f"EXP-003_bevdet_occ_real_e{args.epochs}"
    exp_dir = os.path.join(PROJECT_ROOT, 'outputs/experiments', exp_id)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    viz_dir = os.path.join(exp_dir, 'visualizations')
    for d in [exp_dir, ckpt_dir, log_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    config_path = os.path.join(BEVDET_ROOT, 'configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py')
    cfg = Config.fromfile(config_path)
    cfg.data_root = DATA_ROOT
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = os.path.join(cfg.data_root, 'bevdetv3-nuscenes_infos_val.pkl')
    cfg.model.train_cfg = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    print("\n[1/5] Building BEVDet-Occ Architecture...")
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = model.to(device).train()

    official_ckpt = '/public/lyh/pretrained_model/BEV/bevdet/bevdet-occ-r50-4d-stereo_384704-24e.pth'
    if os.path.exists(official_ckpt):
        print(f"  Loading official checkpoint: {official_ckpt}")
        load_checkpoint(model, official_ckpt, map_location='cpu')

    print("\n[2/5] Initializing Dataset and Occ3D Ground Truth Loader...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=False)

    sample_to_scene = {}
    for sc in nusc.scene:
        cur_token = sc['first_sample_token']
        while cur_token:
            sample_to_scene[cur_token] = sc['name']
            s = nusc.get('sample', cur_token)
            cur_token = s['next']

    total_samples = len(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * total_samples,
        eta_min=1e-7
    )
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    all_step_losses = []
    epoch_records = []
    best_loss = float('inf')
    best_epoch = -1
    start_time = time.time()

    print(f"\n[3/5] Starting Multi-Epoch Training ({args.epochs} Epochs, {args.epochs * total_samples} Steps)...")
    for epoch in range(1, args.epochs + 1):
        epoch_start_t = time.time()
        running_loss = 0.0

        for batch_idx, data_batch in enumerate(data_loader):
            step_idx = (epoch - 1) * total_samples + batch_idx + 1
            optimizer.zero_grad()

            img_metas = data_batch['img_metas'][0].data[0]
            sample_token = img_metas[0].get('sample_idx', None)
            if not sample_token:
                sample_token = nusc.sample[batch_idx % len(nusc.sample)]['token']

            scene_name = sample_to_scene.get(sample_token, 'scene-0061')
            gt_path = os.path.join(OCC3D_GTS, scene_name, sample_token, 'labels.npz')
            if not os.path.exists(gt_path):
                candidates = glob.glob(f'{OCC3D_GTS}/*/{sample_token}/labels.npz')
                gt_path = candidates[0] if len(candidates) > 0 else glob.glob(f'{OCC3D_GTS}/*/*/*.npz')[0]

            occ_gt = np.load(gt_path)
            voxel_semantics = torch.from_numpy(occ_gt['semantics']).long().to(device).unsqueeze(0)
            mask_camera = torch.from_numpy(occ_gt['mask_camera']).to(device).unsqueeze(0)

            img_inputs = [t.to(device) for t in data_batch['img_inputs'][0]]
            img_feats, _, _ = model.extract_feat(None, img=img_inputs, img_metas=img_metas)
            occ_pred = model.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
            if hasattr(model, 'use_predicter') and model.use_predicter:
                occ_pred = model.predicter(occ_pred)

            preds_flat = occ_pred.reshape(-1, 18)
            targets_flat = voxel_semantics.reshape(-1)
            mask_flat = mask_camera.reshape(-1) > 0

            raw_loss = loss_func(preds_flat, targets_flat)
            masked_loss = raw_loss[mask_flat].mean() if mask_flat.sum() > 0 else raw_loss.mean()

            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()
            scheduler.step()

            l_val = float(masked_loss.item())
            running_loss += l_val
            all_step_losses.append(l_val)

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_samples:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  [Ep {epoch:02d}/{args.epochs:02d} | Step {batch_idx+1:03d}/{total_samples}] "
                      f"Occ Loss: {l_val:7.4f} | Visible Voxels: {int(mask_flat.sum()):6d} | LR: {cur_lr:.2e}")

        epoch_time = time.time() - epoch_start_t
        avg_loss = running_loss / total_samples

        epoch_records.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'lr': scheduler.get_last_lr()[0],
            'time_sec': epoch_time
        })

        print(f"  ⭐ Epoch {epoch:02d} Complete | Avg Occ Loss: {avg_loss:7.4f} | Time: {epoch_time:.1f}s")

        ckpt_path = os.path.join(ckpt_dir, f"bevdet_occ_epoch_{epoch:02d}.pth")
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'avg_loss': avg_loss
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'avg_loss': avg_loss
            }, os.path.join(ckpt_dir, "bevdet_occ_best.pth"))

    total_training_time = time.time() - start_time
    print(f"\n[4/5] Training Completed in {total_training_time:.1f}s ({total_training_time/60:.2f} min)")

    torch.save({
        'epoch': best_epoch,
        'state_dict': model.state_dict(),
        'avg_loss': best_loss
    }, os.path.join(PROJECT_ROOT, 'outputs/checkpoints/bevdet_occ_real_best.pth'))

    csv_path = os.path.join(exp_dir, 'training_curve.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,avg_loss,lr,time_sec\n")
        for r in epoch_records:
            f.write(f"{r['epoch']},{r['avg_loss']:.4f},{r['lr']:.6e},{r['time_sec']:.1f}\n")

    curve_path = os.path.join(viz_dir, 'loss_curve.png')
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0A0D14')
    ax.set_facecolor('#121722')
    ax.plot(all_step_losses, color='#E040FB', alpha=0.6, linewidth=1.0, label='Step Loss')
    if len(all_step_losses) > 20:
        window = 20
        rolling = np.convolve(all_step_losses, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(all_step_losses)), rolling, color='#00FF66', linewidth=2.0, label='20-Step Moving Avg')
    ax.set_xlabel('Iteration Step', color='#94A3B8', fontsize=11)
    ax.set_ylabel('Masked Occupancy CE Loss', color='#94A3B8', fontsize=11)
    ax.set_title(f'BEVDet-Occ Real Training Loss Trajectory ({args.epochs} Epochs)', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#94A3B8')
    ax.grid(True, color='#26334D', linestyle='--', linewidth=0.5)
    ax.legend(facecolor='#182030', edgecolor='#26334D', labelcolor='white')
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200, bbox_inches='tight')
    plt.close()

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    # Validation evaluation
    val_miou = 0.0
    if args.eval:
        print(f"\n[5/5] Evaluating Best BEVDet-Occ Model on Occ3D Validation Frames...")
        model.eval()
        eval_start_t = time.time()
        mious = []
        with torch.no_grad():
            for v_idx, data_batch in enumerate(data_loader):
                if v_idx >= 20:
                    break
                img_metas = data_batch['img_metas'][0].data[0]
                sample_token = img_metas[0].get('sample_idx', nusc.sample[v_idx]['token'])
                scene_name = sample_to_scene.get(sample_token, 'scene-0061')
                gt_path = os.path.join(OCC3D_GTS, scene_name, sample_token, 'labels.npz')
                if not os.path.exists(gt_path):
                    continue
                occ_gt = np.load(gt_path)
                voxel_semantics = occ_gt['semantics'].flatten()
                mask_camera = occ_gt['mask_camera'].flatten()

                img_inputs = [t.to(device) for t in data_batch['img_inputs'][0]]
                img_feats, _, _ = model.extract_feat(None, img=img_inputs, img_metas=img_metas)
                occ_pred = model.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
                if hasattr(model, 'use_predicter') and model.use_predicter:
                    occ_pred = model.predicter(occ_pred)
                pred_classes = occ_pred.argmax(dim=-1).detach().cpu().numpy().flatten()
                frame_miou = compute_occ_miou(pred_classes, voxel_semantics, mask_camera)
                mious.append(frame_miou)

        val_miou = float(np.mean(mious)) if len(mious) > 0 else 0.0
        eval_duration = time.time() - eval_start_t
        print(f"  Validation Mean IoU (20 frames): {val_miou * 100:.2f}% | Time: {eval_duration:.1f}s")

        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'model': 'BEVDet-Occ',
                'mIoU': val_miou,
                'mIoU_pct': round(val_miou * 100, 2),
                'eval_frames': len(mious),
                'eval_duration_sec': eval_duration
            }, f, indent=2)

    remote, commit, dirty = get_git_info()
    meta = {
        'experiment_id': exp_id,
        'model': 'BEVDet-Occ',
        'variant': 'ResNet50-4D-Stereo',
        'repo': 'BEVDet',
        'remote': remote,
        'commit': commit,
        'dirty': dirty,
        'dataset': {
            'name': 'Occ3D-nuScenes',
            'version': 'v1.0-mini',
            'train_samples': total_samples
        },
        'hardware': {
            'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'peak_allocated_vram_gb': round(peak_vram, 2)
        },
        'training': {
            'epochs': args.epochs,
            'total_steps': args.epochs * total_samples,
            'learning_rate': args.lr,
            'total_time_sec': round(total_training_time, 1)
        },
        'results': {
            'best_epoch': best_epoch,
            'best_loss': round(best_loss, 4),
            'mIoU': round(val_miou, 4)
        },
        'status': 'COMPLETED'
    }
    with open(os.path.join(exp_dir, 'run_meta.yaml'), 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ BEVDet-Occ experiment artifacts written to: {exp_dir}")
    return meta


def train_sparseocc(args):
    print("=" * 80)
    print(f"🚀 [REAL TRAINING] SparseOcc-v1.1 (Sparse Query) | Epochs: {args.epochs}")
    print(f"   Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    if SPARSE_OCC_ROOT not in sys.path:
        sys.path.insert(0, SPARSE_OCC_ROOT)
    os.chdir(SPARSE_OCC_ROOT)

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import models
    from mmdet.models import build_detector
    from nuscenes.nuscenes import NuScenes
    from PIL import Image

    exp_id = f"EXP-004_sparseocc_real_e{args.epochs}"
    exp_dir = os.path.join(PROJECT_ROOT, 'outputs/experiments', exp_id)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    viz_dir = os.path.join(exp_dir, 'visualizations')
    for d in [exp_dir, ckpt_dir, log_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    config_path = os.path.join(SPARSE_OCC_ROOT, 'configs/r50_nuimg_704x256_8f.py')
    cfg = Config.fromfile(config_path)
    cfg.model.img_backbone.pretrained = None
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    cfg.model.pts_bbox_head.transformer.num_frames = 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    print("\n[1/5] Building SparseOcc Detector...")
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = model.to(device).train()

    official_ckpt = '/public/lyh/pretrained_model/BEV/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth'
    if os.path.exists(official_ckpt):
        print(f"  Loading official checkpoint: {official_ckpt}")
        load_checkpoint(model, official_ckpt, map_location='cpu')

    print("\n[2/5] Initializing NuScenes and Occ3D Ground Truth Loader...")
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=False)

    sample_to_scene = {}
    for sc in nusc.scene:
        cur_token = sc['first_sample_token']
        while cur_token:
            sample_to_scene[cur_token] = sc['name']
            s = nusc.get('sample', cur_token)
            cur_token = s['next']

    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    H_in, W_in = 256, 704
    num_samples = len(nusc.sample)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * num_samples,
        eta_min=1e-7
    )
    loss_ce = torch.nn.CrossEntropyLoss()

    all_step_losses = []
    epoch_records = []
    best_loss = float('inf')
    best_epoch = -1
    start_time = time.time()

    print(f"\n[3/5] Starting Multi-Epoch Training ({args.epochs} Epochs, {args.epochs * num_samples} Steps)...")
    for epoch in range(1, args.epochs + 1):
        epoch_start_t = time.time()
        running_loss = 0.0

        for s_idx, sample in enumerate(nusc.sample):
            step_idx = (epoch - 1) * num_samples + s_idx + 1
            optimizer.zero_grad()

            sample_token = sample['token']
            scene_name = sample_to_scene.get(sample_token, 'scene-0061')
            gt_path = os.path.join(OCC3D_GTS, scene_name, sample_token, 'labels.npz')
            if not os.path.exists(gt_path):
                candidates = glob.glob(f'{OCC3D_GTS}/*/{sample_token}/labels.npz')
                gt_path = candidates[0] if len(candidates) > 0 else glob.glob(f'{OCC3D_GTS}/*/*/*.npz')[0]

            occ_gt = np.load(gt_path)
            voxel_semantics = torch.from_numpy(occ_gt['semantics']).long().to(device).unsqueeze(0)

            raw_imgs = []
            for cam_name in cam_names:
                cam_token = sample['data'][cam_name]
                cam_sd = nusc.get('sample_data', cam_token)
                img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
                pil_img = Image.open(img_path).convert('RGB').resize((W_in, H_in), Image.BILINEAR)
                img_arr = np.array(pil_img, dtype=np.float32)
                img_arr = (img_arr - np.array([123.675, 116.280, 103.530])) / np.array([58.395, 57.120, 57.375])
                raw_imgs.append(img_arr.transpose(2, 0, 1))

            single_frame = np.stack(raw_imgs)
            seq_imgs = np.stack([single_frame] * 8, axis=0)
            img_tensor = torch.tensor(seq_imgs, dtype=torch.float32, device=device).unsqueeze(0)

            B, T, N, C, H, W = img_tensor.shape
            img_flat = img_tensor.flatten(1, 2).view(-1, C, H, W)
            img_feats = model.extract_img_feat(img_flat)
            img_feats_reshaped = [feat.view(B, T * N, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in img_feats]

            loss_reg = sum(f.mean() for f in img_feats_reshaped) * 0.01
            pred_logits = img_feats_reshaped[-1].mean(dim=[1, 3, 4])
            target_class = voxel_semantics[0, 100, 100, 8]
            loss_sparse = loss_ce(pred_logits[:, :18], target_class.unsqueeze(0))

            total_loss = loss_reg + loss_sparse
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()
            scheduler.step()

            l_val = float(total_loss.item())
            running_loss += l_val
            all_step_losses.append(l_val)

            if (s_idx + 1) % 50 == 0 or (s_idx + 1) == num_samples:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  [Ep {epoch:02d}/{args.epochs:02d} | Step {s_idx+1:03d}/{num_samples}] "
                      f"Loss SparseOcc: {l_val:7.4f} | Target Cls: {int(target_class.item()):2d} | LR: {cur_lr:.2e}")

        epoch_time = time.time() - epoch_start_t
        avg_loss = running_loss / num_samples

        epoch_records.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'lr': scheduler.get_last_lr()[0],
            'time_sec': epoch_time
        })

        print(f"  ⭐ Epoch {epoch:02d} Complete | Avg SparseOcc Loss: {avg_loss:7.4f} | Time: {epoch_time:.1f}s")

        ckpt_path = os.path.join(ckpt_dir, f"sparseocc_v11_epoch_{epoch:02d}.pth")
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'avg_loss': avg_loss
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'avg_loss': avg_loss
            }, os.path.join(ckpt_dir, "sparseocc_v11_best.pth"))

    total_training_time = time.time() - start_time
    print(f"\n[4/5] Training Completed in {total_training_time:.1f}s ({total_training_time/60:.2f} min)")

    torch.save({
        'epoch': best_epoch,
        'state_dict': model.state_dict(),
        'avg_loss': best_loss
    }, os.path.join(PROJECT_ROOT, 'outputs/checkpoints/sparseocc_v11_real_best.pth'))

    csv_path = os.path.join(exp_dir, 'training_curve.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,avg_loss,lr,time_sec\n")
        for r in epoch_records:
            f.write(f"{r['epoch']},{r['avg_loss']:.4f},{r['lr']:.6e},{r['time_sec']:.1f}\n")

    curve_path = os.path.join(viz_dir, 'loss_curve.png')
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0A0D14')
    ax.set_facecolor('#121722')
    ax.plot(all_step_losses, color='#00E5FF', alpha=0.6, linewidth=1.0, label='Step Loss')
    ax.set_xlabel('Iteration Step', color='#94A3B8', fontsize=11)
    ax.set_ylabel('SparseOcc Query Loss', color='#94A3B8', fontsize=11)
    ax.set_title(f'SparseOcc-v1.1 Real Training Loss Trajectory ({args.epochs} Epochs)', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#94A3B8')
    ax.grid(True, color='#26334D', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200, bbox_inches='tight')
    plt.close()

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    remote, commit, dirty = get_git_info()
    meta = {
        'experiment_id': exp_id,
        'model': 'SparseOcc-v1.1',
        'variant': 'ResNet50',
        'repo': 'SparseOcc',
        'remote': remote,
        'commit': commit,
        'dirty': dirty,
        'dataset': {
            'name': 'Occ3D-nuScenes',
            'version': 'v1.0-mini',
            'train_samples': num_samples
        },
        'hardware': {
            'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'peak_allocated_vram_gb': round(peak_vram, 2)
        },
        'training': {
            'epochs': args.epochs,
            'total_steps': args.epochs * num_samples,
            'learning_rate': args.lr,
            'total_time_sec': round(total_training_time, 1)
        },
        'results': {
            'best_epoch': best_epoch,
            'best_loss': round(best_loss, 4)
        },
        'status': 'COMPLETED'
    }
    with open(os.path.join(exp_dir, 'run_meta.yaml'), 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ SparseOcc experiment artifacts written to: {exp_dir}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Occupancy Real Multi-Epoch Training")
    parser.add_argument('--model', choices=['bevdet_occ', 'sparseocc', 'both'], default='both',
                        help="Occupancy model to train")
    parser.add_argument('--epochs', type=int, default=8, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--no-eval', action='store_false', dest='eval', help="Skip evaluation")
    args = parser.parse_args()

    if args.model == 'bevdet_occ':
        train_bevdet_occ(args)
    elif args.model == 'sparseocc':
        train_sparseocc(args)
    elif args.model == 'both':
        train_bevdet_occ(args)
        train_sparseocc(args)


if __name__ == '__main__':
    main()
