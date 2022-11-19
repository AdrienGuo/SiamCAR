from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import ipdb
import numpy as np
from pysot.core.config import cfg
from pysot.datasets.crop_image import crop_tri
from pysot.datasets.process import resize, translate
from pysot.utils.bbox import center2corner, corner2center, ratio2real


class PCBCrop:
    def __init__(self, template_size, search_size) -> None:
        self.z_size = template_size
        self.x_size = search_size

    def _template_crop(self, img, box, r, padding=(0, 0, 0)):
        img, box = resize(img, box, r)
        img, box, _ = translate(img, box, self.z_size, padding)
        return img, box

    def _search_crop(self, img, boxes, padding=(0, 0, 0)):
        img_h, img_w = img.shape[:2]
        long_side = max(img_w, img_h)
        r = self.x_size / long_side

        img, boxes = resize(img, boxes, r)
        img, _, _ = translate(img, boxes, self.x_size, padding)

        return img, r

    def get_template(self, img, box, r):
        img, box = self._template_crop(img, box, r)
        return img, box

    def get_search(self, img, boxes):
        img, r = self._search_crop(img, boxes)
        return img, r
