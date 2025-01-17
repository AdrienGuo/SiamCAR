from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import ipdb
import numpy as np
from pysot.core.config import cfg
from pysot.datasets.pcb_crop.crop import crop_tri
from pysot.datasets.utils.process import resize, translate_and_crop
from pysot.utils.bbox import center2corner, corner2center, ratio2real


class PCBCrop:
    def __init__(self, template_size, search_size) -> None:
        self.z_size = template_size
        self.x_size = search_size

    def _template_crop(self, img, box, padding=(0, 0, 0)):
        img_h, img_w = img.shape[:2]
        long_side = max(img_h, img_w)
        r = self.z_size / long_side
        img, box = resize(img, box, r)
        img, box, _ = translate_and_crop(img, box, self.z_size, padding)
        return img, box, r

    def _search_crop(self, img, boxes, r, padding=(0, 0, 0)):
        img, boxes = resize(img, boxes, r)
        img, _, _ = translate_and_crop(img, boxes, self.x_size, padding)
        return img

    def get_template(self, img, box, padding=(0, 0, 0)):
        img, box, r = self._template_crop(img, box, padding)
        return img, box, r

    def get_search(self, img, boxes, r, padding):
        img = self._search_crop(img, boxes, r, padding)
        return img
