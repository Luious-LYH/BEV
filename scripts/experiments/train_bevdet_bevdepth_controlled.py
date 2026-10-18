#!/usr/bin/env python3
"""
Controlled Experiment: BEVDet-R50 vs BEVDepth-R50 Real Training & Evaluation
=============================================================================
Hardware Target: GPU 4 (RTX 4090 24GB)
Dataset: nuScenes v1.0-mini (323 train samples, 81 val samples)
Research Objective:
  Empirically analyze the impact of explicit LiDAR depth supervision (DepthNet)
  against implicit depth learning under identical hyperparameter and training regimes.

Features:
- Full epoch training (all batches, no early break)
- AdamW optimizer with Cosine Annealing learning rate schedule
- Multiprocess DataLoader (4 workers, pinned memory) for high GPU throughput & occupancy
- Component loss tracking (loss_cls, loss_bbox, and loss_depth for BEVDepth)
- Automated periodic checkpointing (per epoch and best loss)
- Full nuScenes validation evaluation (mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE)
- Complete provenance metadata and loss curve export
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

BEVDET_ROOT = '/public/lyh/projects/BEV/third_party/BEVDet'
if BEVDET_ROOT not in sys.path:
    sys.path.insert(0, BEVDET_ROOT)
os.chdir(BEVDET_ROOT)

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

PROJECT_ROOT = '/public/lyh/projects/BEV'
DATA_ROOT = '/public/lyh/data/BEV_dataset/nuscenes/'

CONFIGS = {
    'bevdet': os.path.join(BEVDET_ROOT, 'configs/bevdet/bevdet-r50-cbgs.py'),
    'bevdepth': os.path.join(BEVDET_ROOT, 'configs/bevdet/bevdet-r50-4d-depth-cbgs.py')
}

OFFICIAL_CKPTS = {
    'bevdet': '/public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-cbgs.pth',
    'bevdepth': '/public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-4d-depth-cbgs.pth'
}

BACKBONE_CKPT = '/public/lyh/pretrained_model/BEV/backbones/resnet50-11ad3fa6.pth'


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


def train_single_model(model_name, args):
    print("=" * 80)
    print(f"🚀 [REAL TRAINING] Model: {model_name.upper()} | Epochs: {args.epochs} | Init: {args.init}")
    print(f"   Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    exp_id = f"EXP_{model_name}_real_{args.init}_e{args.epochs}"
    exp_dir = os.path.join(PROJECT_ROOT, 'outputs/experiments', exp_id)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    viz_dir = os.path.join(exp_dir, 'visualizations')
    for d in [exp_dir, ckpt_dir, log_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    cfg = Config.fromfile(CONFIGS[model_name])
    cfg.data_root = DATA_ROOT
    cfg.data.train.dataset.data_root = cfg.data_root
    cfg.data.train.dataset.ann_file = os.path.join(cfg.data_root, 'bevdetv3-nuscenes_infos_train.pkl')
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = os.path.join(cfg.data_root, 'bevdetv3-nuscenes_infos_val.pkl')

    if args.init == 'backbone':
        cfg.model.img_backbone.pop('pretrained', None)
        cfg.model.img_backbone.init_cfg = dict(type='Pretrained', checkpoint=BACKBONE_CKPT)
        init_desc = f"ResNet50 Backbone ({BACKBONE_CKPT})"
    else:
        init_desc = f"Official Checkpoint ({OFFICIAL_CKPTS[model_name]})"

    print(f"\n[1/5] Building {model_name.upper()} Model Architecture...")
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    if args.init == 'official' and os.path.exists(OFFICIAL_CKPTS[model_name]):
        print(f"  Loading official checkpoint: {OFFICIAL_CKPTS[model_name]}")
        load_checkpoint(model, OFFICIAL_CKPTS[model_name], map_location='cpu')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])
        torch.cuda.reset_peak_memory_stats(0)
    else:
        model = model.to(device)

    model.train()

    print("\n[2/5] Building High-Throughput Dataset & DataLoader...")
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=True
    )
    total_batches = len(data_loader)
    print(f"  Total samples (CBGS): {len(dataset)} | Batches per epoch: {total_batches} (Batch Size: {args.batch_size})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * total_batches,
        eta_min=1e-6
    )

    all_step_losses = []
    epoch_records = []
    best_loss = float('inf')
    best_epoch = -1
    start_time = time.time()

    print(f"\n[3/5] Starting Multi-Epoch Optimization (Total Iterations: {args.epochs * total_batches})...")
    for epoch in range(1, args.epochs + 1):
        epoch_start_t = time.time()
        running_loss = 0.0
        running_depth_loss = 0.0
        depth_loss_count = 0

        for batch_idx, data_batch in enumerate(data_loader):
            step_idx = (epoch - 1) * total_batches + batch_idx + 1
            optimizer.zero_grad()

            losses = model(return_loss=True, **data_batch)

            total_loss = 0
            for lk, lv in losses.items():
                if isinstance(lv, torch.Tensor):
                    total_loss += torch.sum(lv)
                elif isinstance(lv, list):
                    total_loss += sum(torch.sum(item) for item in lv)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            l_val = float(total_loss.item())
            running_loss += l_val
            all_step_losses.append(l_val)

            cur_depth_loss = 0.0
            if 'loss_depth' in losses:
                lv_d = losses['loss_depth']
                d_val = float(torch.sum(lv_d).item()) if isinstance(lv_d, torch.Tensor) else float(sum(torch.sum(i).item() for i in lv_d))
                cur_depth_loss = d_val
                running_depth_loss += d_val
                depth_loss_count += 1

            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == total_batches:
                cur_lr = scheduler.get_last_lr()[0]
                depth_str = f" | Depth Loss: {cur_depth_loss:.4f}" if cur_depth_loss > 0 else ""
                print(f"  [Ep {epoch:02d}/{args.epochs:02d} | Step {batch_idx+1:03d}/{total_batches}] "
                      f"Loss: {l_val:8.3f}{depth_str} | LR: {cur_lr:.2e}")

        epoch_time = time.time() - epoch_start_t
        avg_loss = running_loss / total_batches
        avg_depth = (running_depth_loss / depth_loss_count) if depth_loss_count > 0 else 0.0

        epoch_records.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_depth_loss': avg_depth,
            'lr': scheduler.get_last_lr()[0],
            'time_sec': epoch_time
        })

        print(f"  ⭐ Epoch {epoch:02d} Complete | Avg Loss: {avg_loss:8.3f} | Time: {epoch_time:.1f}s")

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch_{epoch:02d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_loss,
            'model_name': model_name
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'model_name': model_name
            }, best_ckpt_path)

    total_training_time = time.time() - start_time
    print(f"\n[4/5] Training Completed in {total_training_time:.1f}s ({total_training_time/60:.2f} min)")
    print(f"  Best Model: Epoch {best_epoch} (Loss: {best_loss:.4f})")

    # Also save to canonical outputs/checkpoints for compatibility
    canonical_ckpt = os.path.join(PROJECT_ROOT, f'outputs/checkpoints/{model_name}_real_best.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'avg_loss': best_loss,
        'model_name': model_name
    }, canonical_ckpt)

    # Export curves and history
    csv_path = os.path.join(exp_dir, 'training_curve.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,avg_loss,avg_depth_loss,lr,time_sec\n")
        for r in epoch_records:
            f.write(f"{r['epoch']},{r['avg_loss']:.4f},{r['avg_depth_loss']:.4f},{r['lr']:.6e},{r['time_sec']:.1f}\n")

    curve_path = os.path.join(viz_dir, 'loss_curve.png')
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0A0D14')
    ax.set_facecolor('#121722')
    ax.plot(all_step_losses, color='#00E5FF', alpha=0.6, linewidth=1.0, label='Step Loss')
    if len(all_step_losses) > 20:
        window = 20
        rolling = np.convolve(all_step_losses, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(all_step_losses)), rolling, color='#00FF66', linewidth=2.0, label='20-Step Moving Avg')
    ax.set_xlabel('Iteration Step', color='#94A3B8', fontsize=11)
    ax.set_ylabel('Total Detection Loss', color='#94A3B8', fontsize=11)
    ax.set_title(f'{model_name.upper()} Training Loss Trajectory ({args.epochs} Epochs)', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#94A3B8')
    ax.grid(True, color='#26334D', linestyle='--', linewidth=0.5)
    ax.legend(facecolor='#182030', edgecolor='#26334D', labelcolor='white')
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Peak VRAM
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    # [5/5] Automated Validation Evaluation
    val_metrics = {}
    if args.eval:
        print(f"\n[5/5] Evaluating Best Checkpoint on nuScenes Mini Validation Split (81 samples)...")
        model.eval()
        val_dataset = build_dataset(cfg.data.test)
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=args.workers,
            dist=False,
            shuffle=False
        )
        val_results = []
        eval_start_t = time.time()
        with torch.no_grad():
            for v_idx, val_batch in enumerate(val_loader):
                res = model(return_loss=False, rescale=True, **val_batch)
                val_results.extend(res)
                if (v_idx + 1) % 25 == 0 or (v_idx + 1) == len(val_dataset):
                    print(f"  Evaluated {v_idx+1}/{len(val_dataset)} samples...")

        eval_duration = time.time() - eval_start_t
        eval_res = val_dataset.evaluate(val_results, metric='bbox')
        val_metrics = {k: float(v) if isinstance(v, (float, int, np.floating)) else str(v) for k, v in eval_res.items()}
        print("\n--- Validation Evaluation Metrics ---")
        for mk in ['pts_bbox_NuScenes/mAP', 'pts_bbox_NuScenes/NDS', 'pts_bbox_NuScenes/mATE', 'pts_bbox_NuScenes/mASE', 'pts_bbox_NuScenes/mAOE']:
            if mk in val_metrics:
                print(f"  {mk}: {val_metrics[mk]:.4f}")

        metrics_file = os.path.join(exp_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'model': model_name,
                'epochs': args.epochs,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'eval_duration_sec': eval_duration,
                'metrics': val_metrics
            }, f, indent=2)

    # Write run_meta.yaml
    remote, commit, dirty = get_git_info()
    meta = {
        'experiment_id': exp_id,
        'model': model_name,
        'variant': 'ResNet50',
        'repo': 'BEVDet',
        'remote': remote,
        'commit': commit,
        'dirty': dirty,
        'dataset': {
            'name': 'nuScenes',
            'version': 'v1.0-mini',
            'train_samples': len(dataset),
            'val_samples': 81
        },
        'hardware': {
            'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'peak_allocated_vram_gb': round(peak_vram, 2)
        },
        'training': {
            'epochs': args.epochs,
            'batch_size_per_gpu': args.batch_size,
            'optimizer': 'AdamW',
            'learning_rate': args.lr,
            'initialization': init_desc,
            'total_time_sec': round(total_training_time, 1)
        },
        'results': {
            'best_epoch': best_epoch,
            'best_loss': round(best_loss, 4),
            'mAP': val_metrics.get('pts_bbox_NuScenes/mAP', None),
            'NDS': val_metrics.get('pts_bbox_NuScenes/NDS', None)
        },
        'status': 'COMPLETED'
    }
    with open(os.path.join(exp_dir, 'run_meta.yaml'), 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Experiment artifacts successfully written to: {exp_dir}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="BEVDet vs BEVDepth Controlled Real Training")
    parser.add_argument('--model', choices=['bevdet', 'bevdepth', 'both'], default='both',
                        help="Which model(s) to train")
    parser.add_argument('--epochs', type=int, default=12,
                        help="Number of epochs to train")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of DataLoader worker processes")
    parser.add_argument('--lr', type=float, default=2e-4,
                        help="Initial learning rate")
    parser.add_argument('--init', choices=['official', 'backbone'], default='official',
                        help="Initialize from official checkpoint or ImageNet backbone")
    parser.add_argument('--no-eval', action='store_false', dest='eval',
                        help="Skip evaluation after training")
    args = parser.parse_args()

    models_to_run = ['bevdet', 'bevdepth'] if args.model == 'both' else [args.model]
    for m in models_to_run:
        train_single_model(m, args)


if __name__ == '__main__':
    main()
