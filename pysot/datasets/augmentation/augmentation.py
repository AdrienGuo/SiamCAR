# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import numpy as np

from pysot.utils.bbox import Center, Corner, center2corner, corner2center


class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):
        """
        Args:
            z:
                image: (511, 511, 3)
                bbox: ground truth
                crop_bbox ([x1, y1, x2, y2]): [192, 192, 318, 318]
                size: 127
            x:
                image: (511, 511, 3)
                bbox: ground truth
                crop_bbox ([x1, y1, x2, y2]): [128, 128, 383, 383]
                size: 255
        """
        # im_h, im_w = 511, 511
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        # crop_bbox_center: [cx, cy, w, h]
        crop_bbox_center = corner2center(crop_bbox)
        if self.scale:
            # 這裡就單純對影像做縮放
            scale_x = (1.0 + Augmentation.random() * self.scale)
            scale_y = (1.0 + Augmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)
            # crop_bbox_center: [cx, cy, w', h']
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        # crop_bbox: [x1, y1, x2, y2]
        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            # self.shift (z: 4, x: 64)
            # shift 在做的事情是把 crop_bbox 做平移，範圍就是給定的 self.shift。
            # sx, sy: -64~64
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            # 這兩行其實沒啥用，可以不用寫...
            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box
        # 因為 crop_bbox 的範圍被調整了，所以 bbox (ground truth) 的座標也要修正。
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            # 一樣要調整 bbox 的大小。
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox

    def horizontal_flip(self, image, bbox):
        image = image.copy()
        bbox = bbox.copy()

        image = cv2.flip(image, 1)  # 水平翻轉
        width = image.shape[1]
        bbox_x1 = width - bbox[:, 2] - 1  # x1 = w - x2
        bbox_x2 = width - bbox[:, 0] - 1  # x2 = w - x1
        bbox[:, 0] = bbox_x1
        bbox[:, 2] = bbox_x2
        return image, bbox

    def vertical_flip(self, img, bbox):
        img = img.copy()
        bbox = bbox.copy()

        img = cv2.flip(img, 0)  # 垂直翻轉
        img_h = img.shape[0]
        bbox_y1 = img_h - bbox[:, 3] - 1  # y1 = h - y2
        bbox_y2 = img_h - bbox[:, 1] - 1  # y2 = h - y1
        bbox[:, 1] = bbox_y1
        bbox[:, 3] = bbox_y2
        return img, bbox

    def __call__(self, image, bbox, size=None, gray=False):
        shape = image.shape
        # crop_bbox: (0, 0, 126, 126)
        # crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
        #                                  size-1, size-1))
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        # image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        # if self.flip and self.flip > np.random.random():
        #     image, bbox = self._flip_aug(image, bbox)

        return image, bbox
