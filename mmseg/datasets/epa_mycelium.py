# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
import mmcv
from PIL import Image
import numpy as np


@DATASETS.register_module()
class EPA_Mycelium_Dataset(CustomDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    # CLASSES = ('background', 'mushroom', 'unkonwn1', 'unknown2')
    # CLASSES = ('mushroom', 'unkonwn1')
    CLASSES = ('background', 'mycelium')
    # CLASSES = ('background', 'mushroom', 'unkonwn1')
    # CLASSES = ('mushroom')

    # PALETTE = [[128, 128, 128]]
    PALETTE = [[0, 0, 0], [128, 0, 0]]
    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]
    # PALETTE = [[128, 0, 0], [0, 128, 0]]

    def __init__(self, **kwargs):
        super(EPA_Mycelium_Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)

  