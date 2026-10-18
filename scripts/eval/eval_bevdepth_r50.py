#!/usr/bin/env python3
"""
BEVDepth-R50 Evaluation on nuScenes mini
========================================
Repo: BEVDet
Branch: dev3.0
Config: configs/bevdet/bevdet-r50-4d-depth-cbgs.py
Checkpoint: outputs/checkpoints/bevdepth_real_best.pth
Dataset: nuScenes v1.0-mini (val split, 81 keyframes)
Output: outputs/eval/bevdepth_r50_mini_eval.json

Column in Benchmark Table: [Our Model (12 Epochs)]
"""

import os
import sys

BEVDET_ROOT = '/public/lyh/projects/BEV/third_party/BEVDet'
sys.path.insert(0, BEVDET_ROOT)

import json
import time
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

CONFIG_PATH = os.path.join(BEVDET_ROOT, 'configs/bevdet/bevdet-r50-4d-depth-cbgs.py')
CHECKPOINT_PATH = '/public/lyh/projects/BEV/outputs/checkpoints/bevdepth_real_best.pth'
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = '/public/lyh/pretrained_model/BEV/bevdet/bevdet-r50-4d-depth-cbgs.pth'
OUTPUT_DIR = '/public/lyh/projects/BEV/outputs/eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)
EVAL_JSON_PATH = os.path.join(OUTPUT_DIR, 'bevdepth_r50_mini_eval.json')

def run_evaluation():
    print("=" * 70)
    print("BEVDEPTH-R50 EVALUATION ON NUSCENES MINI (VAL SPLIT)")
    print("=" * 70)
    
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.data_root = '/public/lyh/data/BEV_dataset/nuscenes/'
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = os.path.join(cfg.data_root, 'bevdetv3-nuscenes_infos_val.pkl')
    cfg.model.train_cfg = None
    
    print("\n1. Building Model & Loading Checkpoint...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    if os.path.exists(CHECKPOINT_PATH):
        raw_ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = raw_ckpt.get('model_state_dict', raw_ckpt.get('state_dict', raw_ckpt))
        cleaned_sd = {}
        for k, v in state_dict.items():
            k_clean = k.replace('module.', '') if k.startswith('module.') else k
            cleaned_sd[k_clean] = v
        model.load_state_dict(cleaned_sd, strict=False)
        print(f"  Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print("  Warning: Checkpoint not found, using initialized model.")
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        model = model.to(device)
    model.eval()
    
    print("\n2. Building Test Dataset (81 samples)...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    
    print(f"\n3. Running Inference over all {len(dataset)} validation samples...")
    results = []
    start_t = time.time()
    
    with torch.no_grad():
        for idx, data_batch in enumerate(data_loader):
            result = model(return_loss=False, rescale=True, **data_batch)
            results.extend(result)
            if (idx + 1) % 20 == 0 or (idx + 1) == len(dataset):
                print(f"  Processed {idx+1}/{len(dataset)} samples...")
                
    eval_time = time.time() - start_t
    print(f"  Inference finished in {eval_time:.1f}s ({len(dataset)/eval_time:.1f} FPS)")
    
    print("\n4. Evaluating with nuScenes Official Protocol...")
    metric_results = dataset.evaluate(results, metric='bbox')
    
    print("\n" + "=" * 70)
    print("BEVDEPTH-R50 VAL EVALUATION RESULTS")
    print("=" * 70)
    for k, v in metric_results.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
            
    with open(EVAL_JSON_PATH, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (float, int)) else str(v) for k, v in metric_results.items()}, f, indent=2)
    print(f"  Results saved to: {EVAL_JSON_PATH}")

if __name__ == '__main__':
    run_evaluation()
