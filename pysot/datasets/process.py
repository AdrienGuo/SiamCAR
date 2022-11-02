import cv2
import ipdb
import numpy as np


def resize(img, boxes, scale):
    img_h, img_w = img.shape[:2]

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    img = cv2.resize(img, (new_w, new_h))
    boxes = boxes * scale

    return img, boxes


def translate(img, boxes, size, padding=(0, 0, 0)):
    img_h, img_w = img.shape[:2]

    # 把圖片移到中心
    x = (size - img_w) / 2
    y = (size - img_h) / 2
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
