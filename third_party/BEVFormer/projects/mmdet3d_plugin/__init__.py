import mmdet3d
from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.utils import *
from .models.opt.adamw import AdamW2
from .bevformer import *
try:
    from .dd3d import *
except ImportError:
    pass

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
BBOX_ASSIGNERS._module_dict['HungarianAssigner3D'] = HungarianAssigner3D
BBOX_ASSIGNERS._module_dict['HungarianAssigner3D_BEVFormer'] = HungarianAssigner3D

