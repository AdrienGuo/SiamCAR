from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import os

import cv2
import ipdb
import numpy as np
from pysot.datasets.crop_image import crop_origin
from pysot.datasets.process import resize, translate


class PCBCrop:
    """ This class is used for ORIGIN dataset.
    """
    def __init__(self, zf_size_min):
        """
        Args:
            zf_size_min: smallest z size after res50 backbone
        """
        # 公式: 從 zf 轉換回 z_img 最小的 size
        z_img_size_min = (((((zf_size_min - 1) * 2) + 2) * 2) * 2 + 7)
        self.z_min = z_img_size_min

    def _template_crop(self, img, box, bg, padding=(0, 0, 0)):
        crop_img = crop_origin(img, box, bg)

        crop_img_h, crop_img_w = crop_img.shape[:2]
        short_side = min(crop_img_w, crop_img_h)
        r = 1
        if short_side < self.z_min:
            # 處理小於 z_min 的情況
            r = self.z_min / short_side
            crop_img, _ = resize(crop_img, box, r)

        return crop_img, r

    def _search_crop(self, img, gt_boxes, z_box, r):
        _, z_box = resize(img, z_box, r)
        x_img, gt_boxes = resize(img, gt_boxes, r)
        return x_img, gt_boxes, z_box

    def get_template(self, img, box, bg):
        crop_img = self._template_crop(img, box.squeeze(), bg)
        return crop_img

    def get_search(self, img, gt_boxes, z_box, r):
        assert r >= 1, f"ERROR, r must greater than or equal 1 but got {r}"
        
        img, gt_boxes, z_box = self._search_crop(img, gt_boxes, z_box, r)

        return img, gt_boxes, z_box
