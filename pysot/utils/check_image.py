# This file only used for checking the images.
# Nothing relates to the traing process.

import os
import re

import cv2
import numpy as np


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Create new dir: {dir_path}")


def save_image(image, save_path):
    # image_new = image -> will share the same memory address
    image_new = np.copy(image)
    cv2.imwrite(save_path, image_new)
    print(f"Save image to: {save_path}")


def draw_box(image, boxes, type=None, scores=None):
    """
    Args:
        image: type=array
        boxes: (box_num, [x1, y1, w, h])
        type: template / pred / gt
    """
    image_new = np.copy(image)
    image_new = np.ascontiguousarray(image_new)
    boxes = np.asarray(boxes, dtype=np.int32)

    if type == "template":
        color = (0, 0, 255)    # red
        thickness = 2
    elif type == "pred":
        color = (0, 255, 0)    # green
        thickness = 2
    elif type == "gt":
        color = (255, 0, 0)    # blue
        thickness = 4
    else:
        color = (0, 0, 0)    # white?
        thickness = 2

    # draw targets
    for idx, box in enumerate(boxes):
        # 畫框框
        cv2.rectangle(image_new, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=color, thickness=thickness)
        # 在框框上面打分數
        # if np.any(scores):
        #     fontFace = cv2.FONT_HERSHEY_COMPLEX
        #     fontScale = 0.5
        #     score = f"{scores[idx]:.3f}"
        #     labelSize = cv2.getTextSize(score, fontFace, fontScale, thickness=1)
        #     _x1 = box[0] # bottomleft x of text
        #     _y1 = box[1] # bottomleft y of text
        #     _x2 = box[0] + labelSize[0][0] # topright x of text
        #     _y2 = box[1] + labelSize[0][1] # topright y of text
        #     cv2.rectangle(image_new, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)   # text background
        #     cv2.putText(image_new, score, (_x1, _y2), fontFace, fontScale, color=(0, 0, 0), thickness=1)

    return image_new


def draw_preds(sub_dir, search_image, annotation_path, idx):
    imgs = []
    names = []
    preds = []
    pred_image = None

    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        template = lines[0]
        annos = lines[1:]

        # template
        template = template.split(',')
        template = list(map(float, template))

        # preds
        if not annos:    # 當沒有偵測到物件時
            print("--- There is no predicted item in this image. ---")
        else:
            for anno in annos:
                anno = anno.strip('\n')
                anno = re.sub("\[|\]", "", anno)
                anno = anno.split(',')
                anno = list(map(float, anno))
                preds.append(anno)

    pred_image = draw_box(search_image, [template], type="template")
    pred_image = draw_box(pred_image, preds, type="pred")

    return pred_image
