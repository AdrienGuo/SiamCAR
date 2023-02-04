# Data augmentation for pcb dataset

import cv2
import ipdb
import numpy as np

from .augmentation import Augmentation


class PCBAugmentation(Augmentation):
    def __init__(self, args) -> None:
        self.clahe = args.CLAHE
        self.flip = args.FLIP
        self.blur = args.BLUR
        self.color = args.COLOR
        self.gray = args.GRAY

    def __call__(self, img, box):
        # Flip
        if self.flip and self.flip > np.random.random():
            img, box = self.vertical_flip(img, box)
            img, box = self.horizontal_flip(img, box)

        # CLAHE 3.0
        if self.clahe:
            img = self.clahe_aug(img)

        # gray augmentation
        if self.gray:
            img = self._gray_aug(img)

        # color augmentation
        if self.color > np.random.random():
            img = self._color_aug(img)

        # blur augmentation
        if self.blur > np.random.random():
            img = self._blur_aug(img)

        return img, box
