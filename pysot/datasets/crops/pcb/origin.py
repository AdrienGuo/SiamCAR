import math
import os

import cv2
import ipdb
import numpy as np

from ...utils.process import resize, translate_and_crop
from ..box import box_add_bg
from ..crop import crop, crop_origin


class PCBCropOrigin:
    """ This class is used for ORIGIN dataset.
    """

    def __init__(self, zf_size_min, background):
        """
        Args:
            zf_size_min: smallest z size after res50 backbone
        """
        # 公式: 從 zf 轉換回 z_img 最小的 size
        z_img_size_min = (((((zf_size_min - 1) * 2) + 2) * 2) * 2 + 7)
        self.z_min = z_img_size_min
        self.bg = background

    def _make_template(self, img, box, bg):
        if bg == "All":
            # origin method 不能有這種設定
            assert False, "bg can not be set to All"

        # 先做好包含 bg 的 box
        bg = float(bg)
        box_bg = box_add_bg(box, bg)
        # TODO: 怕會有 box 超出圖片的情況
        # 再去切出 crop_img
        crop_img = crop(img, box=box_bg)

        # crop_img = crop_origin(img, box, bg)

        crop_img_h, crop_img_w = crop_img.shape[:2]
        short_side = min(crop_img_w, crop_img_h)
        r = 1
        if short_side < self.z_min:
            # 處理小於 z_min 的情況
            r = self.z_min / short_side
            crop_img, _ = resize(crop_img, box, r)

        return crop_img, r

    def _make_search(self, img, gt_boxes, z_box, r):
        _, z_box = resize(img, z_box, r)
        x_img, gt_boxes = resize(img, gt_boxes, r)
        return x_img, gt_boxes, z_box

    def get_template(self, img, box, bg):
        crop_img = self._make_template(img, box.squeeze(), bg)
        return crop_img

    def get_search(self, img, gt_boxes, z_box, r):
        assert r >= 1, f"'r' must greater than or equal to 1 but got {r}"

        img, gt_boxes, z_box = self._make_search(img, gt_boxes, z_box, r)
        return img, gt_boxes, z_box

    def get_data(
        self,
        img,
        z_box,
        gt_boxes,
    ):
        # z_box: (1, [x1, y1, x2, y2])
        # gt_boxes: (num, [x1, y1, x2, y2])
        z_img, r = self.get_template(img, z_box, self.bg)
        # search image 會根據 template 算出來的 r 做縮放
        x_img, gt_boxes, z_box = self.get_search(img, gt_boxes, z_box, r)

        return z_img, x_img, z_box, gt_boxes
