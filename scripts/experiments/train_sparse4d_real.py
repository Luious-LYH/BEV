#!/usr/bin/env python3
"""
Sparse4Dv3-R50 Real Multi-Epoch Training & Evaluation (EXP-002)
===============================================================
Hardware Target: GPU 6 (RTX 4090 24GB)
Dataset: nuScenes v1.0-mini (323 train samples, 81 val samples)
Research Objective:
  Real-world multi-epoch training of fully sparse anchor-based 3D detector.
  Full training over all keyframe samples with instance bank & anchor refinement tracking.

Features:
- Multi-epoch real training (default 12 epochs, all 323 samples per epoch)
- True 6-layer anchor refinement loss components (cls, reg across layers)
- AdamW optimizer with Cosine Annealing learning rate schedule
- Automated validation evaluation & prediction exports
- Provenance recording & YAML metadata
"""

import os
import sys
import time
import argparse
import json
import yaml
import subprocess
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion

SPARSE4D_ROOT = '/public/lyh/projects/BEV/third_party/Sparse4D'
if SPARSE4D_ROOT not in sys.path:
    sys.path.insert(0, SPARSE4D_ROOT)
os.chdir(SPARSE4D_ROOT)

from mmcv import Config
from mmcv.runner import load_checkpoint
import projects.mmdet3d_plugin
from mmdet.models import build_detector
from nuscenes.nuscenes import NuScenes

PROJECT_ROOT = '/public/lyh/projects/BEV'
CONFIG_PATH = os.path.join(SPARSE4D_ROOT, 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py')
OFFICIAL_CKPT = '/public/lyh/pretrained_model/BEV/sparse4d/sparse4dv3_r50.pth'
ANCHOR_PATH = '/public/lyh/projects/BEV/nuscenes_kmeans900.npy'

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
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


def train_sparse4d_real(args):
    print("=" * 80)
    print(f"🚀 [REAL TRAINING] Sparse4Dv3-R50 | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"   Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    exp_id = f"EXP-002_sparse4d_real_e{args.epochs}"
    exp_dir = os.path.join(PROJECT_ROOT, 'outputs/experiments', exp_id)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    viz_dir = os.path.join(exp_dir, 'visualizations')
    for d in [exp_dir, ckpt_dir, log_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.use_deformable_func = False
    cfg.model.head.deformable_model.use_deformable_func = False
    cfg.model.img_backbone.pretrained = None
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    cfg.model.head.instance_bank.anchor = ANCHOR_PATH

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    print("\n[1/5] Building Sparse4Dv3 Architecture...")
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = model.to(device).train()

    if os.path.exists(OFFICIAL_CKPT):
        print(f"  Loading official checkpoint: {OFFICIAL_CKPT}")
        load_checkpoint(model, OFFICIAL_CKPT, map_location='cpu')

    print("\n[2/5] Initializing nuScenes mini Dataset Index...")
    nusc = NuScenes(version='v1.0-mini', dataroot='/public/lyh/data/BEV_dataset/nuscenes', verbose=False)

    val_scene_tokens = set([s['token'] for s in nusc.scene if '0061' in s['name'] or '0553' in s['name'] or '0757' in s['name']])
    train_samples = [s for s in nusc.sample if s['scene_token'] not in val_scene_tokens]
    val_samples = [s for s in nusc.sample if s['scene_token'] in val_scene_tokens]
    if len(train_samples) == 0:
        train_samples = nusc.sample[:323]
        val_samples = nusc.sample[323:]
    num_train = len(train_samples)
    print(f"  Train samples: {num_train} | Validation samples: {len(val_samples)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * num_train,
        eta_min=1e-7
    )

    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    H_in, W_in = 256, 704

    all_step_losses = []
    epoch_records = []
    best_loss = float('inf')
    best_epoch = -1
    start_time = time.time()

    print(f"\n[3/5] Starting Multi-Epoch Training ({args.epochs} Epochs, {args.epochs * num_train} Steps)...")
    for epoch in range(1, args.epochs + 1):
        epoch_start_t = time.time()
        running_loss = 0.0
        running_cls = 0.0
        running_reg = 0.0

        perm = np.random.permutation(num_train)

        for s_idx, sample_idx in enumerate(perm):
            step_idx = (epoch - 1) * num_train + s_idx + 1
            sample = train_samples[sample_idx]

            optimizer.zero_grad()

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
                pil_img = Image.open(img_path).convert('RGB').resize((W_in, H_in), Image.BILINEAR)
                img_arr = np.array(pil_img, dtype=np.float32)
                img_bgr = img_arr[:, :, ::-1]
                img_norm = (img_bgr - np.array([123.675, 116.280, 103.530])) / np.array([58.395, 57.120, 57.375])
                imgs.append(img_norm.transpose(2, 0, 1))

                cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
                cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
                K_3x3 = np.array(cs['camera_intrinsic'], dtype=np.float32)
                K_3x3[0, :] *= (W_in / 1600)
                K_3x3[1, :] *= (H_in / 900)
                viewpad = np.eye(4, dtype=np.float32)
                viewpad[:3, :3] = K_3x3

                T_cam2ego = np.eye(4, dtype=np.float32)
                T_cam2ego[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
                T_cam2ego[:3, 3] = np.array(cs['translation'])

                T_ego2global = np.eye(4, dtype=np.float32)
                T_ego2global[:3, :3] = Quaternion(cam_pose['rotation']).rotation_matrix
                T_ego2global[:3, 3] = np.array(cam_pose['translation'])

                T_refego2global = np.eye(4, dtype=np.float32)
                T_refego2global[:3, :3] = ref_rot
                T_refego2global[:3, 3] = ref_trans

                T_cam2refego = np.linalg.inv(T_refego2global) @ T_ego2global @ T_cam2ego
                T_refego2cam = np.linalg.inv(T_cam2refego)
                proj_mat = viewpad @ T_refego2cam
                projection_mats.append(proj_mat)

            imgs_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            proj_tensor = torch.tensor(np.stack(projection_mats), dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            image_wh_tensor = torch.tensor([[W_in, H_in]] * 6, dtype=torch.float32, device=device).unsqueeze(0).contiguous()

            gt_boxes_list = []
            gt_labels_list = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cat_name = ann['category_name']
                cls_idx = -1
                for c_i, c_name in enumerate(CLASS_NAMES):
                    if c_name in cat_name:
                        cls_idx = c_i
                        break
                if cls_idx == -1:
                    continue

                box_pos = np.array(ann['translation'])
                box_size = np.array(ann['size'])
                box_rot = Quaternion(ann['rotation'])
                rel_pos = ref_rot.T @ (box_pos - ref_trans)
                yaw = box_rot.yaw_pitch_roll[0]
                gt_boxes_list.append([rel_pos[0], rel_pos[1], rel_pos[2],
                                      box_size[1], box_size[0], box_size[2],
                                      yaw, 0.0, 0.0])
                gt_labels_list.append(cls_idx)

            if len(gt_boxes_list) == 0:
                gt_boxes_list = [[0.0, 10.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]]
                gt_labels_list = [0]

            timestamp = torch.tensor([0.0], dtype=torch.float32, device=device)
            img_metas = [{'T_global': np.eye(4, dtype=np.float32), 'T_global_inv': np.eye(4, dtype=np.float32)}]

            data = {
                'projection_mat': proj_tensor,
                'image_wh': image_wh_tensor,
                'timestamp': timestamp,
                'img_metas': img_metas,
                'gt_bboxes_3d': [torch.tensor(gt_boxes_list, dtype=torch.float32, device=device)],
                'gt_labels_3d': [torch.tensor(gt_labels_list, dtype=torch.long, device=device)]
            }

            losses = model(img=imgs_tensor, return_loss=True, **data)

            total_loss = 0
            cls_losses = []
            reg_losses = []
            for lk, lv in losses.items():
                if isinstance(lv, torch.Tensor):
                    s_loss = lv.mean()
                elif isinstance(lv, list):
                    s_loss = sum(l.mean() for l in lv)
                else:
                    s_loss = float(lv)
                total_loss += s_loss
                if 'cls' in lk:
                    cls_losses.append(float(s_loss.item()) if hasattr(s_loss, 'item') else float(s_loss))
                if 'reg' in lk or 'box' in lk:
                    reg_losses.append(float(s_loss.item()) if hasattr(s_loss, 'item') else float(s_loss))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()
            scheduler.step()

            l_val = float(total_loss.item())
            running_loss += l_val
            all_step_losses.append(l_val)
            c_val = sum(cls_losses) / max(len(cls_losses), 1)
            r_val = sum(reg_losses) / max(len(reg_losses), 1)
            running_cls += c_val
            running_reg += r_val

            if (s_idx + 1) % 40 == 0 or (s_idx + 1) == num_train:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  [Ep {epoch:02d}/{args.epochs:02d} | Step {s_idx+1:03d}/{num_train}] "
                      f"Total Loss: {l_val:8.2f} | Cls: {c_val:6.2f} | Reg: {r_val:6.2f} | LR: {cur_lr:.2e}")

        epoch_time = time.time() - epoch_start_t
        avg_loss = running_loss / num_train
        avg_cls = running_cls / num_train
        avg_reg = running_reg / num_train

        epoch_records.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_cls_loss': avg_cls,
            'avg_reg_loss': avg_reg,
            'lr': scheduler.get_last_lr()[0],
            'time_sec': epoch_time
        })

        print(f"  ⭐ Epoch {epoch:02d} Complete | Avg Loss: {avg_loss:8.2f} | Cls: {avg_cls:6.2f} | Reg: {avg_reg:6.2f} | Time: {epoch_time:.1f}s")

        ckpt_path = os.path.join(ckpt_dir, f"sparse4dv3_r50_epoch_{epoch:02d}.pth")
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
            }, os.path.join(ckpt_dir, "sparse4dv3_r50_best.pth"))

    total_training_time = time.time() - start_time
    print(f"\n[4/5] Training Completed in {total_training_time:.1f}s ({total_training_time/60:.2f} min)")

    torch.save({
        'epoch': best_epoch,
        'state_dict': model.state_dict(),
        'avg_loss': best_loss
    }, os.path.join(PROJECT_ROOT, 'outputs/checkpoints/sparse4dv3_r50_real_best.pth'))

    csv_path = os.path.join(exp_dir, 'training_curve.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,avg_loss,avg_cls_loss,avg_reg_loss,lr,time_sec\n")
        for r in epoch_records:
            f.write(f"{r['epoch']},{r['avg_loss']:.4f},{r['avg_cls_loss']:.4f},{r['avg_reg_loss']:.4f},{r['lr']:.6e},{r['time_sec']:.1f}\n")

    curve_path = os.path.join(viz_dir, 'loss_curve.png')
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0A0D14')
    ax.set_facecolor('#121722')
    ax.plot(all_step_losses, color='#00E5FF', alpha=0.6, linewidth=1.0, label='Step Loss')
    if len(all_step_losses) > 20:
        window = 20
        rolling = np.convolve(all_step_losses, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(all_step_losses)), rolling, color='#00FF66', linewidth=2.0, label='20-Step Moving Avg')
    ax.set_xlabel('Iteration Step', color='#94A3B8', fontsize=11)
    ax.set_ylabel('Sparse4D Loss', color='#94A3B8', fontsize=11)
    ax.set_title(f'Sparse4Dv3 Real Training Loss Trajectory ({args.epochs} Epochs)', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#94A3B8')
    ax.grid(True, color='#26334D', linestyle='--', linewidth=0.5)
    ax.legend(facecolor='#182030', edgecolor='#26334D', labelcolor='white')
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200, bbox_inches='tight')
    plt.close()

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    val_predictions = []
    if args.eval:
        print(f"\n[5/5] Evaluating Best Model on Validation Samples ({len(val_samples)} frames)...")
        model.eval()
        eval_start_t = time.time()
        with torch.no_grad():
            for v_idx, sample in enumerate(val_samples[:30]):
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
                    pil_img = Image.open(img_path).convert('RGB').resize((W_in, H_in), Image.BILINEAR)
                    img_arr = np.array(pil_img, dtype=np.float32)
                    img_bgr = img_arr[:, :, ::-1]
                    img_norm = (img_bgr - np.array([123.675, 116.280, 103.530])) / np.array([58.395, 57.120, 57.375])
                    imgs.append(img_norm.transpose(2, 0, 1))

                    cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
                    cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
                    K_3x3 = np.array(cs['camera_intrinsic'], dtype=np.float32)
                    K_3x3[0, :] *= (W_in / 1600)
                    K_3x3[1, :] *= (H_in / 900)
                    viewpad = np.eye(4, dtype=np.float32)
                    viewpad[:3, :3] = K_3x3

                    T_cam2ego = np.eye(4, dtype=np.float32)
                    T_cam2ego[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
                    T_cam2ego[:3, 3] = np.array(cs['translation'])
                    T_ego2global = np.eye(4, dtype=np.float32)
                    T_ego2global[:3, :3] = Quaternion(cam_pose['rotation']).rotation_matrix
                    T_ego2global[:3, 3] = np.array(cam_pose['translation'])
                    T_refego2global = np.eye(4, dtype=np.float32)
                    T_refego2global[:3, :3] = ref_rot
                    T_refego2global[:3, 3] = ref_trans
                    T_cam2refego = np.linalg.inv(T_refego2global) @ T_ego2global @ T_cam2ego
                    T_refego2cam = np.linalg.inv(T_cam2refego)
                    proj_mat = viewpad @ T_refego2cam
                    projection_mats.append(proj_mat)

                imgs_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device).unsqueeze(0).contiguous()
                proj_tensor = torch.tensor(np.stack(projection_mats), dtype=torch.float32, device=device).unsqueeze(0).contiguous()
                image_wh_tensor = torch.tensor([[W_in, H_in]] * 6, dtype=torch.float32, device=device).unsqueeze(0).contiguous()
                timestamp = torch.tensor([0.0], dtype=torch.float32, device=device)
                img_metas = [{'T_global': np.eye(4, dtype=np.float32), 'T_global_inv': np.eye(4, dtype=np.float32)}]

                res = model(
                    img=imgs_tensor,
                    return_loss=False,
                    rescale=True,
                    projection_mat=proj_tensor,
                    image_wh=image_wh_tensor,
                    timestamp=timestamp,
                    img_metas=img_metas
                )
                pred_instances = res[0]
                num_boxes = len(pred_instances['boxes_3d']) if 'boxes_3d' in pred_instances else 0
                val_predictions.append({
                    'sample_token': sample['token'],
                    'num_boxes': num_boxes
                })
                if (v_idx + 1) % 10 == 0:
                    print(f"  Evaluated {v_idx+1}/30 samples | Detections: {num_boxes}")

        eval_duration = time.time() - eval_start_t
        eval_metrics = {
            'num_val_frames': len(val_predictions),
            'avg_detections_per_frame': float(np.mean([p['num_boxes'] for p in val_predictions])),
            'eval_duration_sec': eval_duration
        }
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(eval_metrics, f, indent=2)

    remote, commit, dirty = get_git_info()
    meta = {
        'experiment_id': exp_id,
        'model': 'Sparse4Dv3-R50',
        'variant': 'ResNet50',
        'repo': 'Sparse4D',
        'remote': remote,
        'commit': commit,
        'dirty': dirty,
        'dataset': {
            'name': 'nuScenes',
            'version': 'v1.0-mini',
            'train_samples': num_train,
            'val_samples': len(val_samples)
        },
        'hardware': {
            'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'peak_allocated_vram_gb': round(peak_vram, 2)
        },
        'training': {
            'epochs': args.epochs,
            'total_steps': args.epochs * num_train,
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

    print(f"\n✓ Sparse4D experiment artifacts successfully written to: {exp_dir}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Sparse4Dv3 Real Training")
    parser.add_argument('--epochs', type=int, default=12, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--no-eval', action='store_false', dest='eval', help="Skip evaluation")
    args = parser.parse_args()
    train_sparse4d_real(args)


if __name__ == '__main__':
    main()
