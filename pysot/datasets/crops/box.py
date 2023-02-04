from ..box_transforms import cxcywhtox1y1x2y2, x1y1x2y2tocxcywh


def box_add_bg(box, bg):
    """
    Args:
        box: ([x1, y1, x2, y2])
    """
    # x1y1x2y2 -> cxcywh
    box = x1y1x2y2tocxcywh(box)
    # [w, h] * bg
    box[2:] *= bg
    # cxcywh -> x1y1x2y2
    box = cxcywhtox1y1x2y2(box)

    return box