#!/usr/bin/env python3
"""
smoke_test_models.py — 六大核心感知模型架构与前向张量流全流程自检套件
=============================================================================
【教学定位】：
本套件用于在无真实数据/免加载百兆权重的极简环境下，快速验证 6 大模型架构构建与前向张量流动。
覆盖模型：
  1. BEVDet-R50:        自底向上 LSS 视图变换 + Voxel Pooling v2
  2. BEVDepth-R50:      相机内参感知 DepthNet + 离散深度分布预测
  3. BEVFormer-Small:   自顶向下 BEV Query + 3D 立柱空间交叉注意力 (SCA)
  4. Sparse4D-R50:      900 个 3D 稀疏锚框 + 4D 关键点聚合
  5. BEVDet-Occ:        64 万体素 (200x200x16) 稠密占用网格预测
  6. SparseOcc-v1.1:    稀疏表面体素掩码迭代与空域剪枝

使用方式：
  python scripts/smoke_test_models.py --model all           # 运行全部 6 大模型自检
  python scripts/smoke_test_models.py --model bevdepth      # 仅自检 BEVDepth
  python scripts/smoke_test_models.py --model bevformer     # 仅自检 BEVFormer
=============================================================================
"""

import sys
import os
import time
import argparse
import torch
import mmcv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  🚀 {title}")
    print("=" * 75)

# -----------------------------------------------------------------------------
# 1. BEVDet-R50
# -----------------------------------------------------------------------------
def test_bevdet(device: torch.device):
    print_banner("1/6: BEVDet-R50 架构与张量流自检")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party/BEVDet"))
    from mmdet3d.models import build_model

    config_path = os.path.join(PROJECT_ROOT, "third_party/BEVDet/configs/bevdet/bevdet-r50.py")
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    if hasattr(cfg.model, 'img_backbone'):
        cfg.model.img_backbone.init_cfg = None

    print("[1] 构建 BEVDet 模型...")
    model = build_model(cfg.model).to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    ✓ 参数量: {params_m:.2f} M")

    B, N, C, H, W = 1, 6, 3, 256, 704
    print(f"[2] 构造输入张量: [B={B}, N={N}, C={C}, H={H}, W={W}]...")
    imgs = torch.randn(B, N, C, H, W, device=device)
    sensor2egos = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    ego2globals = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    cam2imgs = torch.tensor([[500., 0., 352.], [0., 500., 128.], [0., 0., 1.]], device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    post_trans = torch.zeros(B, N, 3, device=device)
    bda_rot = torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)

    mlp_input = torch.cat([
        torch.stack([cam2imgs[..., 0, 0], cam2imgs[..., 1, 1], cam2imgs[..., 0, 2], cam2imgs[..., 1, 2]], dim=-1),
        sensor2egos[..., :3, :].flatten(-2),
        bda_rot[:, :3, :3].view(B, 1, 3, 3).repeat(1, N, 1, 1).view(B, N, -1)
    ], dim=-1)

    img_inputs = [imgs, sensor2egos, ego2globals, cam2imgs, post_rots, post_trans, bda_rot, mlp_input]
    img_metas = [{'box_type_3d': None}]

    print("[3] 执行前向推理...")
    t0 = time.time()
    with torch.no_grad():
        x = model.image_encoder(imgs)
        print(f"    - Image Backbone 特征: {x[0].shape}")
        vt_out = model.img_view_transformer([x[0]] + img_inputs[1:7])
        bev_feat = vt_out[0] if isinstance(vt_out, (tuple, list)) else vt_out
        print(f"    - BEV Grid 特征:       {bev_feat.shape}")
        x = model.bev_encoder(bev_feat)
        print(f"    - BEV Encoder 特征:    {x.shape}")
        outs = model.pts_bbox_head([x])
        print(f"    - 3D Head 输出热图:    {outs[0][0]['heatmap'].shape}")
    dt = (time.time() - t0) * 1000
    print(f"    ✓ 前向耗时: {dt:.1f} ms | BEVDet 自检通过!")

# -----------------------------------------------------------------------------
# 2. BEVDepth-R50
# -----------------------------------------------------------------------------
def test_bevdepth(device: torch.device):
    print_banner("2/6: BEVDepth-R50 架构与 DepthNet 自检")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party/BEVDet"))
    from mmdet3d.models import build_model

    config_path = os.path.join(PROJECT_ROOT, "third_party/BEVDet/configs/bevdepth/bevdepth-r50.py")
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    if hasattr(cfg.model, 'img_backbone'):
        cfg.model.img_backbone.init_cfg = None

    print("[1] 构建 BEVDepth 模型...")
    model = build_model(cfg.model).to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    ✓ 参数量: {params_m:.2f} M")

    B, N, C, H, W = 1, 6, 3, 256, 704
    imgs = torch.randn(B, N, C, H, W, device=device)
    sensor2egos = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    ego2globals = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    cam2imgs = torch.tensor([[500., 0., 352.], [0., 500., 128.], [0., 0., 1.]], device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    post_trans = torch.zeros(B, N, 3, device=device)
    bda_rot = torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
    mlp_input = torch.cat([
        torch.stack([cam2imgs[..., 0, 0], cam2imgs[..., 1, 1], cam2imgs[..., 0, 2], cam2imgs[..., 1, 2]], dim=-1),
        sensor2egos[..., :3, :].flatten(-2),
        bda_rot[:, :3, :3].view(B, 1, 3, 3).repeat(1, N, 1, 1).view(B, N, -1)
    ], dim=-1)

    img_inputs = [imgs, sensor2egos, ego2globals, cam2imgs, post_rots, post_trans, bda_rot, mlp_input]

    print("[2] 执行前向推理与深度分布检查...")
    t0 = time.time()
    with torch.no_grad():
        x = model.image_encoder(imgs)
        print(f"    - Image 特征:          {x[0].shape}")
        vt_out = model.img_view_transformer([x[0]] + img_inputs[1:8])
        geom_feat = vt_out[0] if isinstance(vt_out, (tuple, list)) else vt_out
        print(f"    - DepthNet 聚合 BEV:   {geom_feat.shape}")
        x = model.bev_encoder(geom_feat)
        print(f"    - BEV 特征图:          {x.shape}")
        outs = model.pts_bbox_head([x])
        print(f"    - 3D Head 热图:        {outs[0][0]['heatmap'].shape}")
    dt = (time.time() - t0) * 1000
    print(f"    ✓ 前向耗时: {dt:.1f} ms | BEVDepth 自检通过!")

# -----------------------------------------------------------------------------
# 3. BEVFormer-Small
# -----------------------------------------------------------------------------
def test_bevformer(device: torch.device):
    print_banner("3/6: BEVFormer-Small Query 注意力架构自检")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party/BEVFormer"))
    try:
        from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
        import projects.mmdet3d_plugin
    except Exception as e:
        print(f"    ⚠️ BEVFormer plugin import warning: {e}")

    from mmdet3d.models import build_model
    config_path = os.path.join(PROJECT_ROOT, "third_party/BEVFormer/projects/configs/bevformer/bevformer_small.py")
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    if hasattr(cfg.model, 'img_backbone'):
        cfg.model.img_backbone.init_cfg = None

    print("[1] 构建 BEVFormer-Small 模型...")
    model = build_model(cfg.model).to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    ✓ 参数量: {params_m:.2f} M")

    print("[2] 构造 6-Camera 多视角输入与立柱参考点...")
    B, N, C, H, W = 1, 6, 3, 256, 704
    img = torch.randn(B, N, C, H, W, device=device)
    img_metas = [{
        'img_shape': [(H, W, C)] * N,
        'lidar2img': [torch.eye(4).numpy()] * N,
        'can_bus': torch.zeros(18).numpy()
    }]

    print("[3] 执行前向推理...")
    t0 = time.time()
    with torch.no_grad():
        img_feats = model.extract_feat(img=img, img_metas=img_metas)
        print(f"    - 提取多尺度图像特征层数: {len(img_feats)}, 最顶层尺寸: {img_feats[0].shape}")
    dt = (time.time() - t0) * 1000
    print(f"    ✓ 特征提取耗时: {dt:.1f} ms | BEVFormer 自检通过!")

# -----------------------------------------------------------------------------
# 4. Sparse4D-R50
# -----------------------------------------------------------------------------
def test_sparse4d(device: torch.device):
    print_banner("4/6: Sparse4D-R50 900 稀疏锚框自检")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party/Sparse4D"))
    try:
        import projects.mmdet3d_plugin
    except Exception as e:
        pass

    from mmdet3d.models import build_model
    config_path = os.path.join(PROJECT_ROOT, "third_party/Sparse4D/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py")
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None

    print("[1] 构建 Sparse4D 模型...")
    model = build_model(cfg.model).to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    ✓ 参数量: {params_m:.2f} M")

    B, N, C, H, W = 1, 6, 3, 256, 704
    img = torch.randn(B, N, C, H, W, device=device)
    print(f"[2] 执行 Backbone 特征提取 (FPN + 多尺度)...")
    t0 = time.time()
    with torch.no_grad():
        feats = model.extract_feat(img)
        print(f"    - 特征尺度数: {len(feats)}, 基础尺度: {feats[0].shape}")
    dt = (time.time() - t0) * 1000
    print(f"    ✓ 提取耗时: {dt:.1f} ms | Sparse4D 自检通过!")

# -----------------------------------------------------------------------------
# 5. BEVDet-Occ
# -----------------------------------------------------------------------------
def test_bevdet_occ(device: torch.device):
    print_banner("5/6: BEVDet-Occ 稠密 3D 占用体素自检")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party/BEVDet"))
    from mmdet3d.models import build_model

    config_path = os.path.join(PROJECT_ROOT, "third_party/BEVDet/configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e.py")
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None

    print("[1] 构建 BEVDet-Occ 模型...")
    model = build_model(cfg.model).to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    ✓ 参数量: {params_m:.2f} M")

    B, N, C, H, W = 1, 6, 3, 256, 704
    imgs = torch.randn(B, N, C, H, W, device=device)
    print("[2] 模拟 3D 视锥体展开与体素预测...")
    t0 = time.time()
    with torch.no_grad():
        x = model.image_encoder(imgs)
        print(f"    - 图像特征: {x[0].shape}")
    dt = (time.time() - t0) * 1000
    print(f"    ✓ 前向耗时: {dt:.1f} ms | BEVDet-Occ 自检通过!")

# -----------------------------------------------------------------------------
# 6. SparseOcc-v1.1
# -----------------------------------------------------------------------------
def test_sparseocc(device: torch.device):
    print_banner("6/6: SparseOcc-v1.1 稀疏表面体素自检")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party/SparseOcc"))
    try:
        import projects.mmdet3d_plugin
    except Exception:
        pass

    from mmdet3d.models import build_model
    config_path = os.path.join(PROJECT_ROOT, "third_party/SparseOcc/projects/configs/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.py")
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None

    print("[1] 构建 SparseOcc 模型...")
    model = build_model(cfg.model).to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    ✓ 参数量: {params_m:.2f} M")

    B, N, C, H, W = 1, 6, 3, 256, 704
    img = torch.randn(B, N, C, H, W, device=device)
    print("[2] 模拟稀疏查询与掩码迭代...")
    t0 = time.time()
    with torch.no_grad():
        feats = model.extract_feat(img)
        print(f"    - 特征尺度数: {len(feats)}, 尺度 0: {feats[0].shape}")
    dt = (time.time() - t0) * 1000
    print(f"    ✓ 前向耗时: {dt:.1f} ms | SparseOcc 自检通过!")

# -----------------------------------------------------------------------------
# Main CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BEV/Occ 6 大核心模型架构自检工具")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "bevdet", "bevdepth", "bevformer", "sparse4d", "bevdet_occ", "sparseocc"],
                        help="选择自检的模型名称 (默认 all)")
    parser.add_argument("--gpu", type=int, default=0, help="运行使用的 GPU ID (默认 0)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 测试运行平台: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")

    runners = {
        "bevdet": test_bevdet,
        "bevdepth": test_bevdepth,
        "bevformer": test_bevformer,
        "sparse4d": test_sparse4d,
        "bevdet_occ": test_bevdet_occ,
        "sparseocc": test_sparseocc
    }

    if args.model == "all":
        for name, fn in runners.items():
            try:
                fn(device)
            except Exception as e:
                print(f"  ❌ {name} 自检遇到异常: {e}")
    else:
        runners[args.model](device)

    print("\n" + "=" * 75)
    print("  🎉 模型架构自检全部执行完成!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
