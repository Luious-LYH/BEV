import os
import shutil

src = '/public/lyh/projects/BEV/third_party/Sparse4D_legacy_v2/projects/mmdet3d_plugin/ops/deformable_aggregation_ext.cpython-38-x86_64-linux-gnu.so'
dst = '/public/lyh/projects/BEV/third_party/Sparse4D/projects/mmdet3d_plugin/ops/deformable_aggregation_ext.cpython-38-x86_64-linux-gnu.so'

print(f"Copying {src} -> {dst}...")
shutil.copy2(src, dst)
print("File copied successfully! Size:", os.path.getsize(dst))

# Test import
import sys
import torch
sys.path.insert(0, '/public/lyh/projects/BEV/third_party/Sparse4D/projects/mmdet3d_plugin/ops')
import deformable_aggregation_ext
print("Successfully imported deformable_aggregation_ext with torch!")
print("Functions:", dir(deformable_aggregation_ext))
