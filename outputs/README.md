# Outputs Directory Placeholder

本目录用于存放模型推理、评估、Sanity 检查以及多轮训练的实时运行产物。

- **`outputs/checkpoints/`**: 存放本地训练保存的最佳模型权重 (`*_best.pth`)。
- **`outputs/experiments/`**: 存放多轮受控训练的收敛曲线 (`training_curve.csv`)、运行元数据 (`run_meta.yaml`) 与评测指标 (`metrics.json`)。
- **`outputs/eval/`**: 存放验证集评估指标输出。

> **说明**：所有重量级模型权重 (`*.pth`) 及本地运行生成的临时中间产物由 `.gitignore` 排除，不提交至 Git 代码仓。教学报告所需的所有可视化图像与张量流已汇总归档至 `artifacts/` 目录。
