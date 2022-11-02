# Author: Adiren Guo

import numpy as np


def overlap_ratio_one(rect1, rect2):
    """ Compute overlap ratio between two rects

    Args
        rect: 2d array of N x [x, y, w, h]
    Return:
        iou
    """
    left = np.maximum(rect1[0], rect2[0])
    right = np.minimum(rect1[0] + rect1[2], rect2[0] + rect2[2])
    top = np.maximum(rect1[1], rect2[1])
    bottom = np.minimum(rect1[1] + rect1[3], rect2[1] + rect2[3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou
