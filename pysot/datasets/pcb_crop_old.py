from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import ipdb
import numpy as np
from pysot.core.config import cfg
from pysot.datasets.pcb_crop.crop import crop_like_SiamFC
from pysot.datasets.utils.process import resize, translate_and_crop
from pysot.utils.bbox import center2corner, corner2center, ratio2real


class PCBCrop:
    def __init__(self, template_size, search_size) -> None:
        self.z_size = template_size
        self.x_size = search_size

    def _template_crop(self, img, box, padding=(0, 0, 0)):
        img, scale = crop_like_SiamFC(img, box, padding=padding)
        assert abs(scale[0] - scale[1]) < 0.001, f"(scale[0]: {scale[0]} & scale[1]: {scale[1]}) should be the same"

        return img, scale

    def _search_crop(self, img, gt_boxes, z_box, scale, padding=(0, 0, 0)):
        img_h, img_w = img.shape[:2]
        r = scale[0]

        # 如果原圖縮放後大於 search size，要將他縮放成 search size
        if (img_w * r > self.x_size) or (img_h * r > self.x_size):
            # r: 改成根據長邊縮放
            if (img_w > img_h):
                r = self.x_size / img_w
            else:
                r = self.x_size / img_h
            x_img, gt_boxes = resize(img, gt_boxes, r)
            x_img, gt_boxes, spatium = translate_and_crop(x_img, gt_boxes, self.x_size, padding)
            z_img, z_box = resize(img, z_box, r)
            _, z_box, _ = translate_and_crop(z_img, z_box, self.x_size, padding)
        # 原圖縮放後小於 search size，則直接做 r 的縮放
        else:
            x_img, gt_boxes = resize(img, gt_boxes, r)
            x_img, gt_boxes, spatium = translate_and_crop(x_img, gt_boxes, self.x_size, padding)
            z_img, z_box = resize(img, z_box, r)
            _, z_box, _ = translate_and_crop(z_img, z_box, self.x_size, padding)

        return x_img, gt_boxes, z_box, r, spatium

    def get_template(self, img, box, padding=(0, 0, 0)):
        box = box.squeeze()
        img, z_scale = self._template_crop(img, box, padding=padding)
        self.z_scale = z_scale    # 拿來調整 search 的大小

        return img

    def get_search(self, img, gt_boxes, z_box, padding=(0, 0, 0)):
        img, gt_boxes, z_box, r, spatium = self._search_crop(
            img,
            gt_boxes,
            z_box,
            scale=self.z_scale,
            padding=padding
        )

        # z_box: k-means 也會用到
        # r, spatium 是為了將在 search image 上的 pred_boxes 還原回在原圖上
        return img, gt_boxes, z_box, r, spatium
