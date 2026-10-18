#!/usr/bin/env python3
"""
generate_perception_videos.py
Generates video-level dynamic demonstrations for multi-task perception in nuScenes scene-0061:
1. Task 1: Multi-Camera Surround + BEV 3D Object Detection & Tracking Sequence (MP4)
2. Task 2: 4D Spatio-Temporal 3D Occupancy Grid Flow Sequence (MP4)
3. Task 3: Temporal Ego-Motion Alignment & Fusion Comparison (MP4)

Uses official nuScenes devkit, real sample frames, and encodes via ffmpeg (H.264 MP4).
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

DATAROOT = '/public/lyh/data/BEV_dataset/nuscenes'
OUTPUT_DIR = '/public/lyh/projects/BEV/artifacts/learning_report/videos'
FFMPEG_BIN = '/public/lyh/.conda/envs/CV/bin/ffmpeg'
TMP_FRAME_DIR = '/public/lyh/projects/BEV/cache/video_frames'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_FRAME_DIR, exist_ok=True)

CATEGORY_COLORS = {
    'car': '#00E5FF',
    'truck': '#2979FF',
    'bus': '#D500F9',
    'trailer': '#00B0FF',
    'construction_vehicle': '#FFAB00',
    'pedestrian': '#FF1744',
    'bicycle': '#FF6D00',
    'motorcycle': '#FF3D00',
    'barrier': '#FFA000',
    'traffic_cone': '#FF5252'
}

def get_color(cat):
    for k, c in CATEGORY_COLORS.items():
        if k in cat:
            return c
    return '#818CF8'

def save_video_and_webp(frame_dir, base_name, fps=4):
    import glob
    import cv2
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not frames:
        print(f"Warning: No frames found in {frame_dir}")
        return None, None
        
    sample_img = cv2.imread(frames[0])
    h, w, _ = sample_img.shape
    
    # 1. Save MP4 via cv2.VideoWriter
    mp4_out = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_out, fourcc, fps, (w, h))
    for f in frames:
        img_bgr = cv2.imread(f)
        writer.write(img_bgr)
    writer.release()
    
    # 2. Save high-efficiency Animated WebP via Pillow
    webp_out = os.path.join(OUTPUT_DIR, f"{base_name}.webp")
    target_w = min(1200, w)
    target_h = int(h * (target_w / float(w)))
    pil_imgs = [Image.open(f).resize((target_w, target_h), Image.LANCZOS) for f in frames]
    pil_imgs[0].save(webp_out, save_all=True, append_images=pil_imgs[1:], duration=int(1000 / fps), loop=0, quality=85)
    
    print(f"  ✓ MP4 Video: {mp4_out} ({os.path.getsize(mp4_out)/1024:.1f} KB)")
    print(f"  ✓ Animated WebP: {webp_out} ({os.path.getsize(webp_out)/1024:.1f} KB)")
    return mp4_out, webp_out

def load_scene_samples(nusc, scene_name='scene-0061'):
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    samples = []
    cur_token = scene['first_sample_token']
    while cur_token:
        s = nusc.get('sample', cur_token)
        samples.append(s)
        cur_token = s['next']
    return samples

def render_detection_tracking_video(nusc, samples, max_frames=30):
    print(f"\n[1/3] Generating Multi-Camera Surround + BEV 3D Tracking Video ({len(samples[:max_frames])} frames)...")
    frame_dir = os.path.join(TMP_FRAME_DIR, 'det_track')
    os.makedirs(frame_dir, exist_ok=True)
    
    cam_order = [
        ('CAM_FRONT_LEFT', 'Front-Left 65°'), 
        ('CAM_FRONT', 'Front Main 65°'), 
        ('CAM_FRONT_RIGHT', 'Front-Right 65°'),
        ('CAM_BACK_LEFT', 'Rear-Left BSD'), 
        ('CAM_BACK', 'Rear Main 66°'), 
        ('CAM_BACK_RIGHT', 'Rear-Right BSD')
    ]
    
    # Pre-collect trajectory in global frame
    ego_poses = []
    for s in samples[:max_frames]:
        lidar_data = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        ep = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_poses.append(ep['translation'][:2])
    ego_poses = np.array(ego_poses)
    
    rendered_frames = []
    for f_idx, sample in enumerate(samples[:max_frames]):
        fig = plt.figure(figsize=(18, 9), facecolor='#050914')
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.4], wspace=0.04, hspace=0.08,
                               left=0.02, right=0.98, top=0.92, bottom=0.04)
        
        # 1. Render 6 Surround Cameras (Left 3 columns, 2 rows)
        for i, (cam_ch, cam_label) in enumerate(cam_order):
            row = 0 if i < 3 else 1
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            
            cam_data = nusc.get('sample_data', sample['data'][cam_ch])
            cam_path = os.path.join(DATAROOT, cam_data['filename'])
            img = Image.open(cam_path)
            ax.imshow(img)
            
            # Project 3D boxes onto camera
            cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                # Move to ego
                box.translate(-np.array(cam_pose['translation']))
                box.rotate(Quaternion(cam_pose['rotation']).inverse)
                # Move to cam
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                if box_in_image(box, np.array(cs_record['camera_intrinsic']), (1600, 900), vis_level=BoxVisibility.ANY):
                    c = get_color(ann['category_name'])
                    corners = view_points(box.corners(), np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]
                    
                    def draw_rect(idxes):
                        for j in range(len(idxes)):
                            p1 = corners[:, idxes[j]]
                            p2 = corners[:, idxes[(j+1)%len(idxes)]]
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=1.5, alpha=0.9)
                            
                    draw_rect([0, 1, 2, 3]) # front
                    draw_rect([4, 5, 6, 7]) # back
                    for j in range(4):
                        p1 = corners[:, j]
                        p2 = corners[:, j+4]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=1.2, linestyle=':', alpha=0.7)
            
            ax.set_title(f"📷 {cam_ch} [{cam_label}]", color='#00E5FF', fontsize=9, pad=3, fontweight='bold')
            ax.axis('off')
            
        # 2. Render Bird's-Eye View (Right column, spanning both rows)
        ax_bev = fig.add_subplot(gs[:, 3])
        ax_bev.set_facecolor('#080D1A')
        
        # Load LiDAR
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc = LidarPointCloud.from_file(os.path.join(DATAROOT, lidar_data['filename']))
        lcs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pc.rotate(Quaternion(lcs['rotation']).rotation_matrix)
        pc.translate(np.array(lcs['translation']))
        
        # Subsample lidar for clean render
        sub_pts = pc.points[:, ::6]
        dist = np.hypot(sub_pts[0], sub_pts[1])
        mask = (dist < 45.0) & (sub_pts[2] > -2.0) & (sub_pts[2] < 2.5)
        ax_bev.scatter(sub_pts[1, mask], sub_pts[0, mask], c=sub_pts[2, mask], cmap='viridis', s=0.3, alpha=0.4)
        
        # Draw 3D Bounding Boxes on BEV
        cur_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            box.translate(-np.array(cur_ego_pose['translation']))
            box.rotate(Quaternion(cur_ego_pose['rotation']).inverse)
            
            c = get_color(ann['category_name'])
            bottom = box.bottom_corners()[:2, :] # [x, y] in ego
            # Map ego (x=right/fwd) to plot (X=left/right, Y=forward)
            bx = bottom[1]
            by = bottom[0]
            bx = np.append(bx, bx[0])
            by = np.append(by, by[0])
            ax_bev.plot(bx, by, color=c, linewidth=1.8, alpha=0.9)
            
            # Draw heading arrow
            center = box.center
            yaw = box.orientation.yaw_pitch_roll[0]
            vx = np.sin(yaw) * 2.2
            vy = np.cos(yaw) * 2.2
            ax_bev.arrow(center[1], center[0], vx, vy, head_width=0.8, head_length=1.0, fc=c, ec=c, alpha=0.9)
            
        # Draw Ego Vehicle (center)
        ax_bev.add_patch(patches.Rectangle((-1.0, -2.0), 2.0, 4.2, facecolor='#00E5FF', edgecolor='#FFFFFF', linewidth=1.5, zorder=10))
        ax_bev.plot(0, 0, marker='*', color='#FFFFFF', markersize=8, zorder=11)
        
        # Grid & Limits
        ax_bev.set_xlim(-28, 28)
        ax_bev.set_ylim(-15, 45)
        ax_bev.set_aspect('equal')
        ax_bev.grid(True, color=(1.0, 1.0, 1.0, 0.08), linestyle='--', linewidth=0.6)
        
        # Range Rings
        for r in [10, 20, 30, 40]:
            circle = plt.Circle((0, 0), r, fill=False, edgecolor=(0.0, 0.9, 1.0, 0.25), linestyle=':', linewidth=0.8)
            ax_bev.add_patch(circle)
            ax_bev.text(r * 0.7, r * 0.7, f"{r}m", color=(0.0, 0.9, 1.0, 0.5), fontsize=7, fontfamily='monospace')
            
        ax_bev.set_title(f"🛸 BEV 3D Tracking [Frame #{f_idx+1}/{max_frames}]", color='#00E5FF', fontsize=11, fontweight='bold', pad=4)
        ax_bev.set_xlabel("Lateral Y (m)", color='#94A3B8', fontsize=8)
        ax_bev.set_ylabel("Longitudinal X (m)", color='#94A3B8', fontsize=8)
        ax_bev.tick_params(colors='#64748B', labelsize=7)
        
        # Telemetry banner
        fig.suptitle(f"Multi-Camera BEV 3D Perception & Temporal Tracking Stream | Scene: scene-0061 (Boston) | Frame: {f_idx+1:02d} / {max_frames} | dt: 0.5s", 
                     color='#FFFFFF', fontsize=12, fontweight='bold', y=0.98)
        
        frame_path = os.path.join(frame_dir, f"frame_{f_idx:04d}.png")
        plt.savefig(frame_path, dpi=110, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        rendered_frames.append(frame_path)
        if (f_idx + 1) % 5 == 0:
            print(f"  ... rendered {f_idx + 1}/{max_frames} frames")
            
    # Encode with cv2 and Pillow WebP
    mp4_out, webp_out = save_video_and_webp(frame_dir, 'bev_tracking_scene0061', fps=4)
    print(f"✓ [1/3] Successfully generated Tracking Video Suite: {mp4_out}")
    return mp4_out

def render_occupancy_temporal_video(nusc, samples, max_frames=24):
    print(f"\n[2/3] Generating 4D Spatio-Temporal 3D Occupancy Flow Video ({max_frames} frames)...")
    frame_dir = os.path.join(TMP_FRAME_DIR, 'occ_flow')
    os.makedirs(frame_dir, exist_ok=True)
    
    occ_palette = {
        1: '#00E5FF',  # Car
        2: '#2979FF',  # Truck
        4: '#D500F9',  # Bus
        8: '#FF1744',  # Pedestrian
        10: '#FFA000', # Barrier
        11: '#475569', # Drivable surface
        13: '#94A3B8', # Sidewalk
        14: '#166534', # Terrain
        15: '#22C55E', # Vegetation
        16: '#A855F7'  # Building
    }
    
    for f_idx, sample in enumerate(samples[:max_frames]):
        s_token = sample['token']
        occ_file = f"/public/lyh/data/BEV_dataset/nuscenes/occ3d/gts/scene-0061/{s_token}/labels.npz"
        if not os.path.exists(occ_file):
            continue
            
        occ_data = np.load(occ_file)
        semantics = occ_data['semantics'] # [200, 200, 16]
        mask_lidar = occ_data.get('mask_lidar', (semantics != 0) & (semantics != 17))
        mask_camera = occ_data.get('mask_camera', (semantics != 0) & (semantics != 17))
        
        fig, (ax_cam, ax_occ) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='#050914', 
                                            gridspec_kw={'width_ratios': [1, 1.1], 'wspace': 0.08})
        
        # Left: CAM_FRONT
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        img = Image.open(os.path.join(DATAROOT, cam_data['filename']))
        ax_cam.imshow(img)
        ax_cam.set_title(f"📷 前视实车相机 CAM_FRONT [1600×900]", color='#00E5FF', fontsize=11, fontweight='bold', pad=6)
        ax_cam.axis('off')
        
        # Right: 3D Occupancy Voxel Slice & Volume in BEV
        # Project 3D occupied voxels: collapse Z axis with top class or surface
        ax_occ.set_facecolor('#070B16')
        
        # Downsample grid for crisp 2D BEV projection
        occupied = (semantics > 0) & (semantics < 17) & mask_camera
        occ_indices = np.argwhere(occupied) # [N, 3] (x, y, z)
        
        if len(occ_indices) > 0:
            # Map grid (0~200) to meters (-40m to +40m)
            vx = (occ_indices[:, 0] - 100) * 0.4
            vy = (occ_indices[:, 1] - 100) * 0.4
            vz = occ_indices[:, 2] * 0.4 - 1.0
            classes = semantics[occ_indices[:, 0], occ_indices[:, 1], occ_indices[:, 2]]
            
            # Filter road vs foreground obstacles
            is_road = (classes == 11) | (classes == 13)
            # Draw road points
            ax_occ.scatter(vy[is_road][::4], vx[is_road][::4], c='#334155', s=1.2, alpha=0.3, label='Drivable/Sidewalk')
            # Draw foreground dynamic objects with palette colors
            fg_mask = ~is_road
            for cid, col in occ_palette.items():
                if cid in [11, 13]: continue
                m = fg_mask & (classes == cid)
                if np.any(m):
                    ax_occ.scatter(vy[m], vx[m], c=col, s=4.5, alpha=0.85, label=f'Class {cid}')
                    
        # Vehicle Footprint
        ax_occ.add_patch(patches.Rectangle((-1.0, -2.0), 2.0, 4.2, facecolor='#00E5FF', edgecolor='#FFFFFF', linewidth=1.5, zorder=10))
        
        # Range markings
        for r in [10, 20, 30]:
            circle = plt.Circle((0, 0), r, fill=False, edgecolor=(0.0, 0.9, 1.0, 0.25), linestyle=':', linewidth=0.8)
            ax_occ.add_patch(circle)
            ax_occ.text(r * 0.7, r * 0.7, f"{r}m", color=(0.0, 0.9, 1.0, 0.5), fontsize=7, fontfamily='monospace')
            
        ax_occ.set_xlim(-25, 25)
        ax_occ.set_ylim(-15, 35)
        ax_occ.set_aspect('equal')
        ax_occ.grid(True, color=(1.0, 1.0, 1.0, 0.08), linestyle='--', linewidth=0.6)
        ax_occ.set_title(f"🧊 Occ3D 4D 时空占据体素投影 [Frame #{f_idx+1}/{max_frames}]", color='#00E676', fontsize=11, fontweight='bold', pad=6)
        ax_occ.set_xlabel("Lateral Y (m)", color='#94A3B8', fontsize=8)
        ax_occ.set_ylabel("Longitudinal X (m)", color='#94A3B8', fontsize=8)
        ax_occ.tick_params(colors='#64748B', labelsize=7)
        
        fig.suptitle(f"4D Spatio-Temporal Occupancy Stream | 200×200×16 Voxel Flow (0.4m Resolution) | Active Voxels: {np.sum(occupied):,}",
                     color='#FFFFFF', fontsize=12, fontweight='bold', y=0.97)
        
        frame_path = os.path.join(frame_dir, f"frame_{f_idx:04d}.png")
        plt.savefig(frame_path, dpi=110, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        if (f_idx + 1) % 5 == 0:
            print(f"  ... rendered {f_idx + 1}/{max_frames} frames")
            
    # Encode with cv2 and Pillow WebP
    mp4_out, webp_out = save_video_and_webp(frame_dir, 'occupancy_temporal_flow', fps=4)
    print(f"✓ [2/3] Successfully generated Occupancy Flow Suite: {mp4_out}")
    return mp4_out

def render_ego_alignment_video(nusc, samples, max_frames=20):
    print(f"\n[3/3] Generating Temporal Ego-Motion Alignment Comparison Video ({max_frames} frames)...")
    frame_dir = os.path.join(TMP_FRAME_DIR, 'ego_align')
    os.makedirs(frame_dir, exist_ok=True)
    
    for f_idx in range(1, max_frames):
        prev_s = samples[f_idx - 1]
        curr_s = samples[f_idx]
        
        # Load LiDAR points for both frames
        l_prev = nusc.get('sample_data', prev_s['data']['LIDAR_TOP'])
        l_curr = nusc.get('sample_data', curr_s['data']['LIDAR_TOP'])
        
        pc_prev = LidarPointCloud.from_file(os.path.join(DATAROOT, l_prev['filename']))
        pc_curr = LidarPointCloud.from_file(os.path.join(DATAROOT, l_curr['filename']))
        
        # Calibrations
        cs_prev = nusc.get('calibrated_sensor', l_prev['calibrated_sensor_token'])
        cs_curr = nusc.get('calibrated_sensor', l_curr['calibrated_sensor_token'])
        ep_prev = nusc.get('ego_pose', l_prev['ego_pose_token'])
        ep_curr = nusc.get('ego_pose', l_curr['ego_pose_token'])
        
        # Transform prev to prev ego
        pc_prev.rotate(Quaternion(cs_prev['rotation']).rotation_matrix)
        pc_prev.translate(np.array(cs_prev['translation']))
        
        # Transform curr to curr ego
        pc_curr.rotate(Quaternion(cs_curr['rotation']).rotation_matrix)
        pc_curr.translate(np.array(cs_curr['translation']))
        
        # Calculate Delta Ego Transform: T_prev2curr = T_curr_ego^-1 @ T_prev_ego
        R_prev = Quaternion(ep_prev['rotation']).rotation_matrix
        t_prev = np.array(ep_prev['translation'])
        R_curr = Quaternion(ep_curr['rotation']).rotation_matrix
        t_curr = np.array(ep_curr['translation'])
        
        # Ego motion displacement
        delta_t = t_curr - t_prev
        delta_yaw = (Quaternion(ep_curr['rotation']).yaw_pitch_roll[0] - Quaternion(ep_prev['rotation']).yaw_pitch_roll[0]) * 180 / np.pi
        
        # Align prev to curr ego: P_curr = R_curr.T @ (R_prev @ P_prev + t_prev - t_curr)
        pc_aligned = pc_prev.points[:3].copy()
        pc_aligned = R_prev @ pc_aligned + t_prev.reshape(3, 1) # to global
        pc_aligned = R_curr.T @ (pc_aligned - t_curr.reshape(3, 1)) # to curr ego
        
        fig, (ax_unalign, ax_align) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='#050914',
                                                   gridspec_kw={'wspace': 0.08})
        
        # Subsample for rendering
        p_c = pc_curr.points[:, ::5]
        p_p = pc_prev.points[:, ::5]
        p_a = pc_aligned[:, ::5]
        
        # Left: Unaligned
        ax_unalign.set_facecolor('#070B16')
        ax_unalign.scatter(p_p[1], p_p[0], c='#EF4444', s=0.4, alpha=0.4, label='t-1 历史帧 (未补偿)')
        ax_unalign.scatter(p_c[1], p_c[0], c='#00E5FF', s=0.4, alpha=0.5, label='t 当前帧')
        ax_unalign.set_title("❌ 未做自车运动补偿 (Unaligned)\n背景地物严重双重虚影，位移错位 Δx={:.2f}m, ΔYaw={:.2f}°".format(
            np.linalg.norm(delta_t), delta_yaw), color='#F87171', fontsize=10.5, fontweight='bold', pad=4)
        ax_unalign.set_xlim(-25, 25)
        ax_unalign.set_ylim(-10, 40)
        ax_unalign.set_aspect('equal')
        ax_unalign.grid(True, color=(1.0, 1.0, 1.0, 0.08), linestyle='--', linewidth=0.6)
        ax_unalign.legend(loc='upper right', facecolor='#0B132B', edgecolor='#EF4444', labelcolor='#FFFFFF', fontsize=8)
        
        # Right: Aligned
        ax_align.set_facecolor('#070B16')
        ax_align.scatter(p_a[1], p_a[0], c='#10B981', s=0.4, alpha=0.4, label='t-1 经过 ΔT_ego 对齐')
        ax_align.scatter(p_c[1], p_c[0], c='#00E5FF', s=0.4, alpha=0.5, label='t 当前帧')
        ax_align.set_title("✓ 自车运动对齐 (Ego-Motion Aligned)\n静态道路与建筑 100% 吻合，动态前车清晰显露真实位移向量", color='#34D399', fontsize=10.5, fontweight='bold', pad=4)
        ax_align.set_xlim(-25, 25)
        ax_align.set_ylim(-10, 40)
        ax_align.set_aspect('equal')
        ax_align.grid(True, color=(1.0, 1.0, 1.0, 0.08), linestyle='--', linewidth=0.6)
        ax_align.legend(loc='upper right', facecolor='#0B132B', edgecolor='#10B981', labelcolor='#FFFFFF', fontsize=8)
        
        fig.suptitle(f"Temporal Ego-Motion Compensation Dynamic Comparison | Frame {f_idx:02d} vs {f_idx+1:02d} | Δt = 500ms",
                     color='#FFFFFF', fontsize=12, fontweight='bold', y=0.97)
        
        frame_path = os.path.join(frame_dir, f"frame_{f_idx:04d}.png")
        plt.savefig(frame_path, dpi=110, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        
    mp4_out, webp_out = save_video_and_webp(frame_dir, 'temporal_ego_alignment', fps=3)
    print(f"✓ [3/3] Successfully generated Ego Alignment Suite: {mp4_out}")
    return mp4_out

def main():
    print("=" * 70)
    print("🎬 Generating Video-Level Multi-Task Perception Suite for BEV & Occ")
    print("=" * 70)
    t0 = time.time()
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    samples = load_scene_samples(nusc, 'scene-0061')
    print(f"Loaded scene-0061 with {len(samples)} sequential samples.")
    
    render_detection_tracking_video(nusc, samples, max_frames=25)
    render_occupancy_temporal_video(nusc, samples, max_frames=20)
    render_ego_alignment_video(nusc, samples, max_frames=18)
    
    print("\n" + "=" * 70)
    print(f"🎉 All perception demonstration videos completed in {time.time()-t0:.1f}s!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
