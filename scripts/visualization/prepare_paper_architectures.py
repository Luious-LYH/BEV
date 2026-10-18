#!/usr/bin/env python3
"""
prepare_paper_architectures.py — Prepare Paper Architectures & Core Module Diagrams
=============================================================================
Consolidates official paper framework diagrams and core module figures from
third_party repositories and generates high-resolution publication vector diagrams
for all key paradigms:
1. LSS (Lift, Splat, Shoot - ECCV 2020)
2. BEVDet & BEVDet4D (Huang et al., 2021/2022)
3. BEVDepth (Li et al., AAAI 2023 - LiDAR Supervised DepthNet & CADN)
4. BEVFormer (Li et al., ECCV 2022 - 3D Query Pillar & SCA/TSA)
5. Sparse4D & Sparse4Dv2 (Lin et al., 2023 - 4D Deformable Aggregation & 900 Anchors)
6. BEVDet-Occ (Huang et al., 2023 - Dense 3D Voxel CNN)
7. SparseOcc (Liu et al., 2023 - Sparse Surface 3D Query Tree & RayIoU)

Target Directory: artifacts/learning_report/images/paper_architectures/
=============================================================================
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = '/public/lyh/projects/BEV'
DEST_DIR = os.path.join(PROJECT_ROOT, 'artifacts/learning_report/images/paper_architectures')
os.makedirs(DEST_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Copy Existing Official Third-Party Figures
# -----------------------------------------------------------------------------
def copy_third_party_figures():
    copies = [
        # BEVFormer official architecture
        ('third_party/BEVFormer/figs/arch.png', 'bevformer_official_arch.png'),
        # Sparse4D official architecture & framework
        ('third_party/Sparse4D/resources/sparse4d_architecture.jpg', 'sparse4d_official_arch.jpg'),
        ('third_party/Sparse4D/resources/sparse4d_framework.png', 'sparse4d_official_framework.png'),
        ('third_party/Sparse4D/resources/sparse4dv2_framework.png', 'sparse4dv2_official_framework.png'),
        ('third_party/Sparse4D/resources/efficient_deformable_aggregation.jpg', 'sparse4d_deformable_aggregation.jpg'),
        # SparseOcc official architecture & RayIoU
        ('third_party/SparseOcc/asserts/arch.jpg', 'sparseocc_official_arch.jpg'),
        ('third_party/SparseOcc/asserts/rayiou.jpg', 'sparseocc_rayiou.jpg'),
        # BEVPoolv2 (BEVDet / BEVDepth)
        ('third_party/BEVDet/resources/bevpoolv2.png', 'bevdet_bevpoolv2.png'),
        ('third_party/BEVDet/resources/bevpoolv2_performance.png', 'bevdet_bevpoolv2_perf.png'),
    ]

    for src_rel, dst_name in copies:
        src = os.path.join(PROJECT_ROOT, src_rel)
        dst = os.path.join(DEST_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Copied {src_rel} -> {dst_name}")
        else:
            print(f"⚠️ Source missing: {src_rel}")

# -----------------------------------------------------------------------------
# 2. Render Publication-Grade Diagram for LSS (Lift-Splat-Shoot)
# -----------------------------------------------------------------------------
def render_lss_architecture():
    output_path = os.path.join(DEST_DIR, 'lss_architecture_diagram.png')
    print("🎨 Rendering LSS Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor='#080c14')
    ax.set_facecolor('#0d131f')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    # Color palette
    c_img = '#3b82f6'
    c_lift = '#00e5ff'
    c_frustum = '#10b981'
    c_splat = '#f59e0b'
    c_bev = '#8b5cf6'
    c_head = '#ec4899'

    def draw_box(x, y, w, h, title, sub, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      facecolor='#0f172a', edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, color=color, fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(x + w/2, y + h*0.28, sub, color='#94a3b8', fontsize=8.5, ha='center', va='center', fontfamily='monospace')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', width=2, headwidth=7, headlength=7))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2, label, color='#38bdf8', fontsize=8.5, ha='center', fontfamily='monospace')

    # Boxes
    draw_box(2, 14, 14, 14, "Multi-View Cams", "[6, 3, 900, 1600]", c_img)
    draw_arrow(16, 21, 20.5, 21, "Backbone")

    draw_box(20.5, 14, 15, 14, "2D Backbone / FPN", "Context C + Depth D\n[6, 512, 28, 50]", c_lift)
    draw_arrow(35.5, 21, 39.5, 21, "Outer Prod")

    draw_box(39.5, 10, 17.5, 22, "Lift: Frustum Grid", "Point p = (u,v,d)\nFeatures = C ⊗ α\n[6, 64, 118, 28, 50]", c_frustum)
    draw_arrow(57, 21, 61.5, 21, "Extrinsics")

    draw_box(61.5, 14, 15, 14, "Splat: Voxel Pool", "Sum into BEV Pillars\nCUDA Prefix Sum\n[1, 64, 128, 128]", c_splat)
    draw_arrow(76.5, 21, 80.5, 21, "BEV Conv")

    draw_box(80.5, 14, 16.5, 14, "Task Head (Shoot)", "Map Segmentation\nPlanning Costmap", c_head)

    # Core difference callout box at bottom
    diff_rect = patches.FancyBboxPatch((2, 1), 95, 6.5, boxstyle="round,pad=0.4",
                                       facecolor='#0f172a', edgecolor='#0284c7', linewidth=1.2)
    ax.add_patch(diff_rect)
    diff_text = (r"$\bf{Core\ Difference\ from\ Standard\ 2D\ CV:}$ Standard 2D CV assumes planar perspective geometry. "
                 r"LSS explicitly generates a discrete depth distribution $P(d)$ per pixel to cast 2D rays into 3D camera space, "
                 r"then projects multi-view camera frustums into a unified metric Ego frame via sensor extrinsics $(R, T)$.")
    ax.text(49.5, 4.25, diff_text, color='#e2e8f0', fontsize=8.4, ha='center', va='center')

    plt.suptitle("LSS (Lift, Splat, Shoot) Official Architecture & Tensor Transformation Pipeline", 
                 color='#ffffff', fontsize=14, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Rendered LSS diagram.")

# -----------------------------------------------------------------------------
# 3. Render Publication-Grade Diagram for BEVDet & BEVDet4D
# -----------------------------------------------------------------------------
def render_bevdet_architecture():
    output_path = os.path.join(DEST_DIR, 'bevdet_architecture_diagram.png')
    print("🎨 Rendering BEVDet Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor='#080c14')
    ax.set_facecolor('#0d131f')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    c_img = '#3b82f6'
    c_trans = '#00e5ff'
    c_enc = '#10b981'
    c_align = '#f59e0b'
    c_head = '#ec4899'

    def draw_box(x, y, w, h, title, sub, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      facecolor='#0f172a', edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, color=color, fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(x + w/2, y + h*0.28, sub, color='#94a3b8', fontsize=8.5, ha='center', va='center', fontfamily='monospace')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', width=2, headwidth=7, headlength=7))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2, label, color='#38bdf8', fontsize=8.5, ha='center', fontfamily='monospace')

    draw_box(2, 14, 15, 14, "Multi-Camera Images", "Frame t & Frame t-1\n[6, 3, 256, 704]", c_img)
    draw_arrow(17, 21, 22, 21, "Image Backbone")

    draw_box(22, 14, 16, 14, "View Transformer", "LSS Lift-Splat\nImage Augment Aware\n[B, C, 128, 128]", c_trans)
    draw_arrow(38, 21, 43, 21, "BEV Feature")

    draw_box(43, 14, 16, 14, "BEV Encoder", "ResNet / Second FPN\nBEV Data Augment\n[B, 256, 128, 128]", c_enc)
    draw_arrow(59, 21, 64, 21, "BEV Alignment")

    draw_box(64, 14, 16, 14, "BEVDet4D Ego Warping", "Warp F_{t-1} to F_t via\nΔEgo Pose Matrix", c_align)
    draw_arrow(80, 21, 85, 21, "Concat & Fuse")

    draw_box(85, 14, 13, 14, "CenterHead 3D", "Heatmap / Reg / Vel\n10 3D Classes", c_head)

    # Core difference callout box at bottom
    diff_rect = patches.FancyBboxPatch((2, 1), 95, 6.5, boxstyle="round,pad=0.4",
                                       facecolor='#0f172a', edgecolor='#0284c7', linewidth=1.2)
    ax.add_patch(diff_rect)
    diff_text = (r"$\bf{Core\ Difference\ from\ Standard\ 2D\ CV:}$ Decouples representation learning into 2D Image View space and "
                 r"3D BEV space. It introduces dual data augmentation (Image-view Aug + BEV-space Aug) and compensates for "
                 r"ego-vehicle motion using a differentiable affine warp operator ($\Delta Ego$), eliminating temporal ghosting.")
    ax.text(49.5, 4.25, diff_text, color='#e2e8f0', fontsize=8.4, ha='center', va='center')

    plt.suptitle("BEVDet & BEVDet4D Architecture: Dual Space Decoupling & Ego-Motion Compensation", 
                 color='#ffffff', fontsize=14, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Rendered BEVDet diagram.")

# -----------------------------------------------------------------------------
# 4. Render Publication-Grade Diagram for BEVDepth
# -----------------------------------------------------------------------------
def render_bevdepth_architecture():
    output_path = os.path.join(DEST_DIR, 'bevdepth_architecture_diagram.png')
    print("🎨 Rendering BEVDepth Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor='#080c14')
    ax.set_facecolor('#0d131f')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    c_cam = '#3b82f6'
    c_cadn = '#00e5ff'
    c_lidar = '#10b981'
    c_pool = '#f59e0b'
    c_head = '#ec4899'

    def draw_box(x, y, w, h, title, sub, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      facecolor='#0f172a', edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, color=color, fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(x + w/2, y + h*0.28, sub, color='#94a3b8', fontsize=8.5, ha='center', va='center', fontfamily='monospace')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', width=2, headwidth=7, headlength=7))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2, label, color='#38bdf8', fontsize=8.5, ha='center', fontfamily='monospace')

    draw_box(2, 14, 15, 14, "Multi-Camera Images", "6 Views + Intrinsics\n[6, 3, 256, 704]", c_cam)
    draw_arrow(17, 21, 22, 21, "Intrinsics PE")

    draw_box(22, 14, 18, 14, "Camera-Aware DepthNet", "DepthNet(F, K, R, T)\nExplicit Depth Distribution\n[6, D, H, W]", c_cadn)
    draw_arrow(40, 21, 45, 21, "Lift to Frustum")

    draw_box(45, 14, 17, 14, "Voxel Pooling v2", "CUDA Prefix Sum & Sort\n10x Faster than LSS\n[B, C, 128, 128]", c_pool)
    draw_arrow(62, 21, 67, 21, "BEV Features")

    draw_box(67, 14, 14, 14, "BEV Encoder", "Residual Convs\n[B, 256, 128, 128]", '#8b5cf6')
    draw_arrow(81, 21, 85, 21, "")

    draw_box(85, 14, 13, 14, "CenterHead 3D", "Class Heatmap\nBox Regression", c_head)

    # LiDAR Supervised Depth Branch at top
    rect_lidar = patches.FancyBboxPatch((22, 30), 18, 8, boxstyle="round,pad=0.5",
                                        facecolor='#0f172a', edgecolor=c_lidar, linewidth=1.8)
    ax.add_patch(rect_lidar)
    ax.text(31, 35, "LiDAR Projected Depth GT", color=c_lidar, fontsize=10, fontweight='bold', ha='center')
    ax.text(31, 32, "Supervised Loss L_depth (BCE)", color='#94a3b8', fontsize=8.5, ha='center', fontfamily='monospace')
    ax.annotate('', xy=(31, 28), xytext=(31, 30),
                arrowprops=dict(facecolor=c_lidar, edgecolor=c_lidar, width=1.5, headwidth=6, headlength=6))

    # Core difference callout box at bottom
    diff_rect = patches.FancyBboxPatch((2, 1), 95, 6.5, boxstyle="round,pad=0.4",
                                       facecolor='#0f172a', edgecolor='#0284c7', linewidth=1.2)
    ax.add_patch(diff_rect)
    diff_text = (r"$\bf{Core\ Difference\ from\ Standard\ 2D\ CV:}$ Standard CV lacks depth supervision and suffers from severe scale ambiguity. "
                 r"BEVDepth introduces Camera-Aware Depth Module (CADN) with intrinsic/extrinsic awareness and explicitly supervises "
                 r"the depth distribution $P(d)$ with projected sparse LiDAR point clouds, slashing depth error from 1.42m to 0.48m.")
    ax.text(49.5, 4.25, diff_text, color='#e2e8f0', fontsize=8.4, ha='center', va='center')

    plt.suptitle("BEVDepth Architecture: Explicit LiDAR Depth Supervision & Camera-Aware Depth Module", 
                 color='#ffffff', fontsize=14, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Rendered BEVDepth diagram.")

# -----------------------------------------------------------------------------
# 5. Render Publication-Grade Diagram for BEVDet-Occ
# -----------------------------------------------------------------------------
def render_bevdet_occ_architecture():
    output_path = os.path.join(DEST_DIR, 'bevdet_occ_architecture_diagram.png')
    print("🎨 Rendering BEVDet-Occ Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor='#080c14')
    ax.set_facecolor('#0d131f')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    c_cam = '#3b82f6'
    c_lift = '#00e5ff'
    c_3d = '#10b981'
    c_head = '#ec4899'

    def draw_box(x, y, w, h, title, sub, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      facecolor='#0f172a', edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, color=color, fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(x + w/2, y + h*0.28, sub, color='#94a3b8', fontsize=8.5, ha='center', va='center', fontfamily='monospace')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', width=2, headwidth=7, headlength=7))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2, label, color='#38bdf8', fontsize=8.5, ha='center', fontfamily='monospace')

    draw_box(2, 14, 16, 14, "Multi-Camera Images", "6 Views (1600x900)\nResNet-50 2D Backbone", c_cam)
    draw_arrow(18, 21, 23, 21, "2D to 3D Lift")

    draw_box(23, 14, 18, 14, "3D Voxel View Transformer", "Lift-Splat to 3D Volume\nNo Height Collapsing!\n[B, C, 16, 200, 200]", c_lift)
    draw_arrow(41, 21, 46, 21, "3D Convolutions")

    draw_box(46, 14, 20, 14, "3D ResNet UNet Encoder", "3D Conv (3x3x3 kernels)\nSkip Connections\n[B, 32, 16, 200, 200]", c_3d)
    draw_arrow(66, 21, 71, 21, "Voxel Classifier")

    draw_box(71, 14, 25, 14, "17-Class Occupancy Head", "Softmax over 17 Categories\nFree Space Masked out\n[B, 17, 16, 200, 200]", c_head)

    # Core difference callout box at bottom
    diff_rect = patches.FancyBboxPatch((2, 1), 95, 6.5, boxstyle="round,pad=0.4",
                                       facecolor='#0f172a', edgecolor='#0284c7', linewidth=1.2)
    ax.add_patch(diff_rect)
    diff_text = (r"$\bf{Core\ Difference\ from\ Standard\ 2D\ CV:}$ Standard 2D CV and 2D BEV collapse the vertical Z dimension ($Z \to 1$). "
                 r"BEVDet-Occ retains the full physical 3D volume ($200 \times 200 \times 16$, 0.4m resolution) and utilizes 3D convolutions "
                 r"to reconstruct geometric occupancy and free-form arbitrary non-bounding-box obstacles across the entire space.")
    ax.text(49.5, 4.25, diff_text, color='#e2e8f0', fontsize=8.4, ha='center', va='center')

    plt.suptitle("BEVDet-Occ Architecture: Preserving Full 3D Voxel Volume without Height Collapsing", 
                 color='#ffffff', fontsize=14, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Rendered BEVDet-Occ diagram.")

def main():
    copy_third_party_figures()
    render_lss_architecture()
    render_bevdet_architecture()
    render_bevdepth_architecture()
    render_bevdet_occ_architecture()
    print("🎉 All paper architecture and core module diagrams prepared successfully!")

if __name__ == '__main__':
    main()
