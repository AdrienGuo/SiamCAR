import cv2
import ipdb
import numpy as np

from ...utils.process import resize, translate_and_crop
from ..crop import crop_tri


class CropTriOrigin(object):
    """ This class is used for (tri with origin) dataset.
    """
    def __init__(self, zf_size_min):
        """
        Args:
            zf_size_min: smallest z size after res50 backbone
        """
        # 公式: 從 zf 轉換回 z_img 最小的 size
        z_img_size_min = (((((zf_size_min - 1) * 2) + 2) * 2) * 2 + 7)
        self.z_min = z_img_size_min

    def _template_crop(self, img, box, padding=(0, 0, 0)):
        img_h, img_w = img.shape[:2]
        short_side = min(img_w, img_h)
        r = 1
        if short_side < self.z_min:
            # 處理小於 z_min 的情況
            r = self.z_min / short_side
            img, box = resize(img, box, r)
        return img, box, r

    def _search_crop(self, img, boxes, r, padding=(0, 0, 0)):
        img, _ = resize(img, boxes, r)
        return img

    def get_template(self, img, box):
        img, box, r = self._template_crop(img, box.squeeze())
        return img, box, r

    def get_search(self, img, boxes, r):
        img = self._search_crop(img, boxes, r)
        return img

    def get_data(
        self,
        z_img,
        x_img,
        z_box,
        gt_boxes,
    ):
        # 確保 z_img 的最小邊不會小於 threshold，
        # 若是 z_img 有做縮放 -> x_img 一樣要做縮放
        z_img, z_box, r = self.get_template(z_img, z_box)
        x_img = self.get_search(x_img, gt_boxes, r)
        return z_img, x_img, z_box
