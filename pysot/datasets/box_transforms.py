import numpy as np


def ratio2real(image, box):
    h, w = image.shape[:2]
    box[:, 0] = box[:, 0] * w
    box[:, 1] = box[:, 1] * h
    box[:, 2] = box[:, 2] * w
    box[:, 3] = box[:, 3] * h
    return box


def center2corner(center):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]

    Args:
        center (np.array): (N, 4=[cx, cy, w, h])
    Return:
        center (np.array): (N, 4=[x1, y1, x2, y2])
    """

    if isinstance(center, np.ndarray):
        center[:, 0] = center[:, 0] - center[:, 2] * 0.5  # x1
        center[:, 1] = center[:, 1] - center[:, 3] * 0.5  # y1
        center[:, 2] = center[:, 0] + center[:, 2]  # x2 = x1 + w
        center[:, 3] = center[:, 1] + center[:, 3]  # y2 = y1 + h
        return center
    else:
        assert False, "type error"


def corner2center(corner):
    """Convert [x1, y1, x2, y2] -> [cx, cy, w, h]"""

    center = np.zeros(corner.shape)
    if isinstance(corner, np.ndarray):
        center[:, 0] = (corner[:, 0] + corner[:, 2]) / 2  # cx = (x1 + x2) / 2
        center[:, 1] = (corner[:, 1] + corner[:, 3]) / 2  # cy = (y1 + y2) / 2
        center[:, 2] = corner[:, 2] - corner[:, 0]  # w = x2 - x1
        center[:, 3] = corner[:, 3] - corner[:, 1]  # h = y2 - y1
        return center
    else:
        assert False, "type error"


def x1y1x2y2tox1y1wh(boxes):
    """Convert [x1, y1, x2, y2] -> [x1, y1, w, h]"""

    boxes_transformed = np.zeros(boxes.shape)
    if isinstance(boxes, np.ndarray):
        if boxes.ndim == 1:
            boxes_transformed[0] = boxes[0]
            boxes_transformed[1] = boxes[1]
            boxes_transformed[2] = boxes[2] - boxes[0]  # w = x2 - x1
            boxes_transformed[3] = boxes[3] - boxes[1]  # h = y2 - y1
        elif boxes.ndim == 2:
            boxes_transformed[:, 0] = boxes[:, 0]
            boxes_transformed[:, 1] = boxes[:, 1]
            boxes_transformed[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
            boxes_transformed[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
        return boxes_transformed
    else:
        assert False, "type error"


def x1y1x2y2tocxcywh(boxes):
    """Convert [x1, y1, x2, y2] -> [cx, cy, w, h]"""
    boxes_transformed = np.zeros(boxes.shape)

    if isinstance(boxes, np.ndarray):
        if boxes.ndim == 1:
            boxes_transformed = np.array([
                (boxes[0] + boxes[2]) / 2,
                (boxes[1] + boxes[3]) / 2,
                (boxes[2] - boxes[0]) + 1,
                (boxes[3] - boxes[1]) + 1
            ])
        elif boxes.ndim == 2:
            boxes_transformed = np.stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2,
                (boxes[:, 2] - boxes[:, 0]) + 1,
                (boxes[:, 3] - boxes[:, 1]) + 1
            ], axis=1)
        return boxes_transformed
    else:
        assert False, "type error"


def cxcywhtox1y1x2y2(boxes):
    """Convert [cx, cy, w, h] -> [x1, y1, x2, y2]"""
    boxes_transformed = np.zeros(boxes.shape)

    if isinstance(boxes, np.ndarray):
        if boxes.ndim == 1:
            boxes_transformed = np.array([
                boxes[0] - ((boxes[2] - 1) / 2),  # cx - ((w-1) / 2)
                boxes[1] - ((boxes[3] - 1) / 2),  # cy - ((h-1) / 2)
                boxes[0] + ((boxes[2] - 1) / 2),  # cx + ((w-1) / 2)
                boxes[1] + ((boxes[3] - 1) / 2),  # cy + ((h-1) / 2)
            ])
        elif boxes.ndim == 2:
            boxes_transformed = np.stack([
                boxes[:, 0] - ((boxes[:, 2] - 1) / 2),  # cx - ((w-1) / 2)
                boxes[:, 1] - ((boxes[:, 3] - 1) / 2),  # cy - ((h-1) / 2)
                boxes[:, 0] + ((boxes[:, 2] - 1) / 2),  # cx + ((w-1) / 2)
                boxes[:, 1] + ((boxes[:, 3] - 1) / 2),  # cy + ((h-1) / 2)
            ], axis=1)
        return boxes_transformed
    else:
        assert False, "type error"
