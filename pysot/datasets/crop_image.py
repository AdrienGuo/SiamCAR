import cv2
import ipdb
import numpy as np


def pos_s_2_bbox(pos, s):
    return [pos[0] - s[0] / 2,
            pos[1] - s[1] / 2,
            pos[0] + s[0] / 2,
            pos[1] + s[1] / 2]


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    # 如果是 SiamFC 的話，bbox_w 其實等於 bbox_h, 因為 bbox 是一個 “正方形”
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    long_side = max(bbox_w, bbox_h)
    r = (out_sz - 1) / long_side

    # 做反向移動，把 bbox 的 (x1, y1) 移動到 (0, 0)
    c = -r * bbox[0]
    d = -r * bbox[1]

    mapping = np.array([[r, 0, c],
                        [0, r, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop, (r, r, c, d)


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0])/2., (bbox[3] + bbox[1])/2.]
    target_size = [(bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    s_z = (s_z, s_z)
    # scale_z = exemplar_size / s_z
    # d_search = (instance_size - exemplar_size) / 2
    # pad = d_search / scale_z
    # s_x = s_z + 2 * pad

    z, scale = crop_hwc(
        image,
        pos_s_2_bbox(target_pos, s_z),
        exemplar_size,
        padding=padding
    )
    # x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instance_size, padding)
    return z, scale


def crop_like_teacher(img, box, bg, exemplar_size=127, padding=(0, 0, 0)):
    img_h, img_w = img.shape[:2]
    box_pos = [(box[2] + box[0]) / 2., (box[3] + box[1]) / 2.]
    box_size = [(box[2] - box[0]), (box[3] - box[1])]

    # --- 全部背景 ---
    if bg == "All":
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        x = (exemplar_size / 2) - (cx)
        y = (exemplar_size / 2) - (cy)
        crop_img = img    # 要裁切的對象是整張圖
    # --- n 倍的背景 ---
    else:
        # 決定裁切的寬高
        bg = float(bg)
        assert bg >= 1, f"Error, (bg: {bg}) should not smaller than 1"
        crop_w = box_size[0] * bg
        crop_h = box_size[1] * bg
        box = [box_pos[0] - crop_w * 0.5,
               box_pos[1] - crop_h * 0.5,
               box_pos[0] + crop_w * 0.5,
               box_pos[1] + crop_h * 0.5]

        # 不要讓裁切範圍超出原圖
        box[0] = int(max(0, box[0]))
        box[1] = int(max(0, box[1]))
        box[2] = int(min(img_w, box[2]))
        box[3] = int(min(img_h, box[3]))
        box_size = [(box[2] - box[0]), (box[3] - box[1])]

        x = (exemplar_size / 2) - (box_size[0] / 2)
        y = (exemplar_size / 2) - (box_size[1] / 2)
        crop_img = img[box[1]: box[3], box[0]: box[2]]

    mapping = np.array([[1, 0, x],
                        [0, 1, y]]).astype(np.float)
    crop_img = cv2.warpAffine(
        crop_img,
        mapping,
        (exemplar_size, exemplar_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding
    )

    return crop_img


def crop_tri(img, r, exemplar_size, padding=(0, 0, 0)):
    img_h, img_w = img.shape[:2]

    x = (exemplar_size // 2) - (img_w // 2)
    y = (exemplar_size // 2) - (img_h // 2)
    mapping = np.array([[1, 0, x],
                        [0, 1, y]]).astype(np.float)

    crop_img = cv2.warpAffine(
        img,
        mapping,
        (exemplar_size, exemplar_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding
    )

    # virtual box for ratio_penalty
    box = np.array([[0, 0, img_w, img_h]])

    return crop_img, box
