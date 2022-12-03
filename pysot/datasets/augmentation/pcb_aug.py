# Data augmentation for pcb dataset


import ipdb
import numpy as np

from .augmentation import Augmentation


class PCBAug(Augmentation):
    def z_aug(self, img, box):
        # template 做 flip 就好，search 不用
        if self.flip and self.flip > np.random.random():
            img, box = self.vertical_flip(img, box)
        if self.flip and self.flip > np.random.random():
            img, box = self.horizontal_flip(img, box)

        return img, box

    def x_aug(self, img, gt_boxes, z_box):
        return img, gt_boxes, z_box
