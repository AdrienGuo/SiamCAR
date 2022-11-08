from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import os

import cv2
import ipdb
import numpy as np
from pysot.datasets.crop_image import crop_like_teacher
from pysot.datasets.process import resize, translate
from pysot.utils.bbox import (Center, Corner, center2corner, corner2center,
                              ratio2real)


class PCBCrop:
    def __init__(self, template_size, search_size) -> None:
        self.template_size = template_size
        self.search_size = search_size

    def _template_crop(self, img, box, bg, padding=(0, 0, 0)):
        img = crop_like_teacher(
            img, box, bg, exemplar_size=self.template_size, padding=padding
        )
        return img

    def _search_crop(self, img, gt_boxes, z_box, padding=(0, 0, 0)):
        img_h, img_w = img.shape[:2]

        long_side = max(img_w, img_h)
        r = self.search_size / long_side

        x_img, gt_boxes = resize(img, gt_boxes, r)
        x_img, gt_boxes, spatium = translate(x_img, gt_boxes, self.search_size, padding)
        z_img, z_box = resize(img, z_box, r)
        _, z_box, _ = translate(z_img, z_box, self.search_size, padding)

        return x_img, gt_boxes, z_box, r, spatium

    def get_template(self, img, box, bg):
        """
        Args:
            box: (1, [x1, y1, x2, y2]) #real
        """
        box = box.squeeze()
        img = self._template_crop(img, box, bg)
        return img

    def get_search(self, img, gt_boxes, z_box):
        # gt_boxes = center2corner(gt_boxes)
        # gt_boxes = ratio2real(img, gt_boxes)
        # z_box = center2corner(z_box)
        # z_box = ratio2real(img, z_box)

        img, gt_boxes, z_box, r, spatium = self._search_crop(
            img,
            gt_boxes,
            z_box
        )

        return img, gt_boxes, z_box, r, spatium
