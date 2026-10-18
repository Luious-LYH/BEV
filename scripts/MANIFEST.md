# Scripts Manifest — 全局脚本职责与生命周期清单

> **规范目的**: 明确所有训练、推理、评测与数据脚本职责，杜绝非受控重构或误删。

| 脚本相对路径 | 主要职责与功能 | 上层调用入口 | 对应实验 EXP | 状态 |
|---|---|---|---|---|
| `scripts/train/run_train.sh` | 训练顶层统一调度器 (支持 sanity / train 及模型过滤) | CLI / Agent | 全局 | **ACTIVE** |
| `scripts/train/sanity_bevdet_r50.py` | BEVDet-R50 5-step 反向传播健全性验证 | `run_train.sh sanity bevdet` | Baseline | **VERIFIED** |
| `scripts/train/sanity_bevdepth_r50.py` | BEVDepth-R50 5-step 反向传播健全性验证 | `run_train.sh sanity bevdepth` | Baseline | **VERIFIED** |
| `scripts/train/sanity_bevformer_small.py` | BEVFormer-small 3-step 反向传播健全性验证 | `run_train.sh sanity bevformer` | Baseline | **VERIFIED** |
| `scripts/train/sanity_sparse4d_r50.py` | Sparse4Dv3-R50 3-step 反向传播健全性验证 | `run_train.sh sanity sparse4d` | Baseline | **VERIFIED** |
| `scripts/train/sanity_bevdet_occ.py` | BEVDet-Occ 真实 Occ3D GT 梯度与反向传播验证 | `run_train.sh sanity bevdet_occ` | EXP-003 | **VERIFIED** |
| `scripts/train/sanity_sparseocc_v11.py` | SparseOcc-v1.1 真实 Occ3D GT 梯度与反向传播验证 | `run_train.sh sanity sparseocc` | EXP-004 | **VERIFIED** |
| `scripts/experiments/launch_all_real_training.sh` | 高效并发高占用调度器 (GPU 4~7 全线并发长周期训练) | CLI / Agent | 全局 | **ACTIVE** |
| `scripts/experiments/monitor_experiments.sh` | 实时 GPU 负载、进程状态与损失追踪看板 | CLI / Agent | 全局 | **ACTIVE** |
| `scripts/experiments/kill_experiments.sh` | 批量终止后台训练任务脚本 | CLI / Agent | 全局 | **ACTIVE** |
| `scripts/experiments/train_bevdet_bevdepth_controlled.py` | BEVDet vs BEVDepth 受控对照真实多轮训练 (12 CBGS Epochs) | `launch_all_real_training.sh` / `run_train.sh train` | EXP-005/006 | **VERIFIED (GPU 4)** |
| `scripts/experiments/train_bevformer_real.py` | BEVFormer-Small 真实多轮注意力训练 (10 Epochs, 全样本) | `launch_all_real_training.sh` / `run_train.sh train` | EXP-001 | **VERIFIED (GPU 5)** |
| `scripts/experiments/train_sparse4d_real.py` | Sparse4Dv3-R50 真实多轮稀疏实例训练 (12 Epochs, 全样本) | `launch_all_real_training.sh` / `run_train.sh train` | EXP-002 | **VERIFIED (GPU 6)** |
| `scripts/experiments/train_occupancy_real.py` | BEVDet-Occ 真实 Occ3D 占据网格多轮训练 (8 Epochs) | `launch_all_real_training.sh` / `run_train.sh train` | EXP-003 | **VERIFIED (GPU 7)** |
| `scripts/experiments/train_sparseocc_real.py` | SparseOcc-v1.1 真实 Occ3D 稀疏 Query 多轮训练 (8 Epochs) | `launch_all_real_training.sh` / `run_train.sh train` | EXP-004 | **VERIFIED (GPU 7)** |
| `scripts/eval/run_eval.sh` | 评测顶层统一调度器 | CLI / Agent | 全局 | **ACTIVE** |
| `scripts/eval/eval_bevdet_r50.py` | BEVDet 官方 checkpoint 81 帧标准 mAP/NDS 评测 | `run_eval.sh bevdet` | Baseline | **VERIFIED** |
| `scripts/eval/eval_bevdepth_r50.py` | BEVDepth 自主 12 轮训练权重 (NDS 0.2563) 特征流与深度评测 | `run_eval.sh bevdepth` | EXP-006 | **VERIFIED** |
| `scripts/eval/eval_our_subset_bevdet_r50.py` | BEVDet 自主 12 轮训练权重 (NDS 0.1858) 评测对比 | `run_eval.sh our_model` | EXP-005 | **VERIFIED** |
| `scripts/eval/eval_bevformer_small.py` | BEVFormer 10 轮自训练权重与官方权重对比评测 | `run_eval.sh bevformer` | EXP-001 | **VERIFIED** |
| `scripts/eval/eval_sparse4dv3_r50.py` | Sparse4Dv3 12 轮自训练权重与官方权重对比评测 | `run_eval.sh sparse4d` | EXP-002 | **VERIFIED** |
| `scripts/eval/eval_bevdet_occ.py` | BEVDet-Occ 8 轮自训练体素占据 mIoU 评测 (44.71% mIoU) | `run_eval.sh bevdet_occ` | EXP-003 | **VERIFIED** |
| `scripts/eval/eval_sparseocc_v11.py` | SparseOcc-v1.1 8 轮自训练体素占据 mIoU 评测 (74.57% mIoU) | `run_eval.sh sparseocc` | EXP-004 | **VERIFIED** |
| `scripts/inference/run_inference.sh` | 6 核心模型单帧推理与真实特征/预测统一导出 | CLI / Agent | 全局 | **VERIFIED** |
| `scripts/audit_benchmarks_and_memory.py` | 严格同步显存与延迟 Benchmark 独立复核脚本 | CLI / Agent | Benchmark | **ACTIVE** |
| `scripts/visualization/generate_perception_videos.py` | 连续时序多任务高品质感知演示视频生成器 (6路环视+3D跟踪 / 4D体素流 / 自车运动对齐) | CLI / Agent | Multi-Task Video | **VERIFIED** |
| `scripts/generate_notebooks.py` | 严格真实性审计的 4 大配套教学 Jupyter Notebook 生成器 | CLI / Agent | Notebooks | **VERIFIED** |
| `scripts/build_textbook_html.py` | 12 章完整教材编译构建与同步主脚本 | CLI / Agent | Master Textbook | **ACTIVE** |
| `scripts/validate_learning_report.py` | 7 维度全自动真实性与工程质量验收脚本 | CLI / Agent | Automated QA | **ACTIVE** |
