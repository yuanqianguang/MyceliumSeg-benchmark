# 在 mmseg/datasets/pipelines 目录下新建 remap_labels.py
from mmcv.utils import Registry
from mmseg.datasets.builder import PIPELINES
import numpy as np

@PIPELINES.register_module()
class RemapLabels:
    def __init__(self, mapping: dict):
        # mapping: {old_value: new_value}
        self.mapping = mapping

    def __call__(self, results):
        gt = results['gt_semantic_seg']
        new_gt = np.zeros_like(gt, dtype=np.uint8)
        for old, new in self.mapping.items():
            new_gt[gt == old] = new
        results['gt_semantic_seg'] = new_gt
        return results