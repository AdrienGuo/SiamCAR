from typing import Tuple

import cv2
import ipdb
import numpy as np

__all__ = ['resize', 'translate_and_crop', 'z_score_norm']


def resize(img, boxes, scale):
    img_h, img_w = img.shape[:2]

    # 我不懂為啥在 2080ti 這台上面會回傳 float
    new_w = int(round(img_w * scale))
    new_h = int(round(img_h * scale))

    img = cv2.resize(img, (new_w, new_h))
    boxes = boxes * scale

    return img, boxes


def translate_and_crop(
    img,
    boxes,
    translate_px: Tuple[int, int],
    size,
    padding=(0, 0, 0)
):
    """ 對圖片做 translation，把圖片移動到中心。
    Args:
        translate_px (tuple of int): x, y 軸的位移。
        size (int): 要輸出的圖片大小
    """
    x = translate_px[0]
    y = translate_px[1]
    mapping = np.array([[1, 0, x],
                        [0, 1, y]])
    img = cv2.warpAffine(
        img, mapping,
        (size, size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding
    )
    boxes = boxes + [x, y, x, y]

    return img, boxes, (x, y)


def z_score_norm(img):
    assert isinstance(img, np.ndarray), "ERROR, img type should be numpy.ndarray!!"
    assert img.shape[-1] == 3, "ERROR, order of dimension is wrong!!"
    img = ((img - img.mean(axis=(0, 1))) / img.std(axis=(0, 1)))
    return img
