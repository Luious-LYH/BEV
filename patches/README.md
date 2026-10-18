# 第三方库兼容性补丁说明 (Third-party Patches)

> **本项目直接在 `third_party/` 目录中预置了针对主流自动驾驶开源算法库的关键兼容性补丁。**  
> 本目录下的 `.patch` 文件作为补丁备份与审计记录，方便排查与复现。

---

## 1. 补丁清单与修复目标

| 补丁文件 | 目标上游仓库 | 解决的核心问题 | 触发错误现象 (未打补丁时) |
|---|---|---|---|
| **`bevdet.patch`** | `HuangJunJie2017/BEVDet` (dev3.0) | 1. 提高 MMCV 版本上限检查至 1.8.0<br>2. 修复 `HungarianAssigner3D` 重复注册冲突<br>3. 增加 Lyft 模块的可选安全引入 | `AssertionError: MMCV 1.7.2 > 1.7.0`<br>`KeyError: HungarianAssigner3D is already registered` |
| **`bevformer.patch`** | `fundamentalvision/BEVFormer` | 1. 修复与 BEVDet 在同一 Python 运行时的 `HungarianAssigner3D` 命名冲突<br>2. 针对非必要 `dd3d` / `lyft` 模块添加导入异常捕获 | `KeyError: HungarianAssigner3D is already registered in bbox_assigner`<br>`ModuleNotFoundError: dd3d` |
| **`sparseocc.patch`** | `MCG-NJU/SparseOcc` (v1.1) | 为 `_msmv_sampling_cuda` 算子添加纯 PyTorch 回退实现 (`msmv_sampling_pytorch`) | 在无编译 CUDA 扩展环境或 CPU 调试时抛出 `ImportError: cannot import name _msmv_sampling_cuda` |

---

## 2. 如何手动应用补丁 (若重新 Clone 上游官方代码)

若因二次开发需要重新克隆官方原版上游代码，可在项目根目录下通过以下命令一键打上补丁：

```bash
# 1. 对 BEVDet 应用补丁
git -C third_party/BEVDet apply ../../patches/bevdet.patch

# 2. 对 BEVFormer 应用补丁
git -C third_party/BEVFormer apply ../../patches/bevformer.patch

# 3. 对 SparseOcc 应用补丁
git -C third_party/SparseOcc apply ../../patches/sparseocc.patch
```

> **提示**：当前代码仓库中的 `third_party/` 已全部打好上述补丁，无需重复执行。
