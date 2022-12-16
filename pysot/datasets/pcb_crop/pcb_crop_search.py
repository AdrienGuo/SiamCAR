from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import os

import cv2
import ipdb
import numpy as np

from pysot.datasets.pcb_crop.crop_image import crop_like_teacher
from pysot.datasets.utils.process import resize, translate_and_crop


class PCBCropSearch:
    def __init__(self, template_size, search_size, background) -> None:
        self.z_size = template_size
        self.x_size = search_size
        self.bg = background

    def _template_crop(self, img, box, bg, padding=(0, 0, 0)):
        box = box.squeeze()
        img = crop_like_teacher(
            img, box, bg, exemplar_size=self.z_size, padding=padding
        )
        return img

    def _search_crop(self, img, gt_boxes, z_box, padding=(0, 0, 0)):
        img_h, img_w = img.shape[:2]

        long_side = max(img_w, img_h)
        r = self.x_size / long_side

        x_img, gt_boxes = resize(img, gt_boxes, r)
        x_img_h, x_img_w = x_img.shape[:2]
        x = (self.x_size / 2) - (x_img_w / 2)
        y = (self.x_size / 2) - (x_img_h / 2)
        x_img, gt_boxes, spatium = translate_and_crop(
            x_img, gt_boxes, (x, y), self.x_size, padding)

        z_img, z_box = resize(img, z_box, r)
        _, z_box, _ = translate_and_crop(
            z_img, z_box, (x, y), self.x_size, padding)

        return x_img, gt_boxes, z_box, r, spatium

    def get_template(self, img, box, bg):
        """
        Args:
            box: (1, [x1, y1, x2, y2]) #real
        """
        img = self._template_crop(img, box, bg)
        return img

    def get_search(self, img, gt_boxes, z_box):
        img, gt_boxes, z_box, r, spatium = self._search_crop(
            img,
            gt_boxes,
            z_box
        )
        return img, gt_boxes, z_box, r, spatium

    def get_data(
        self,
        img,
        z_box,
        gt_boxes,
        padding
    ):
        # 先做好 search image (x_img)，再從上面裁切出 template image (z_img)。
        # z_box: (1, [x1, y1, x2, y2])
        # gt_boxes: (num, [x1, y1, x2, y2])
        x_img, gt_boxes, z_box, r, spatium = self.get_search(img, gt_boxes, z_box)
        z_img = self.get_template(x_img, z_box, self.bg)
        return z_img, x_img, z_box, gt_boxes
