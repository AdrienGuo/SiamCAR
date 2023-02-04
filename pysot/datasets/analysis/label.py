# To find not match labels.
# Some ground truth were set for same label (same class),
# but actually their shapes have large variation.
# Hence, if their ratio of area are larger than the given threshold,
# find them and save images to ./wrong_labels

import argparse
import os
import random
from collections import defaultdict

import cv2
import ipdb
import numpy as np

from pysot.core.config import cfg
from pysot.utils.check_image import create_dir, draw_box, save_image


def cal_area(w, h):
    area = w * h
    return area


def area_ratio(area1, area2):
    ratio = max(area1 / area2, area2 / area1)
    return ratio


def cxcywh_to_x1y1wh(boxes):
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    return boxes


def find_not_match_labels_one_img(img_path: str, boxes_cls_map: dict, threshold: float):
    invalid_boxes_cls_map = defaultdict(list)
    for cls, boxes in boxes_cls_map.items():
        max_area = 0
        min_area = float("inf")
        for box in boxes:
            curr_area = cal_area(box[2], box[3])
            max_area = max(curr_area, max_area)
            min_area = min(curr_area, min_area)
        ratio = area_ratio(min_area, max_area)
        if ratio > threshold:
            invalid_boxes_cls_map[cls] = boxes_cls_map[cls]
    return invalid_boxes_cls_map


def find_not_match_labels(directory: str, threshold: float):
    imgs = list()
    invalid_labels = list()
    for root, _, files in sorted(os.walk(directory, followlinks=True)):
        for file in sorted(files):  # 排序
            box = []
            if file.endswith(('.jpg', '.png', 'bmp')):
                # 一張圖片
                img_path = os.path.join(root, file)
                anno_path = os.path.join(root, file[:-3] + "txt")
                if not os.path.isfile(anno_path):
                    assert False, f"ERROR, annotation doesn't exist: {anno_path}"
                f = open(anno_path, 'r')
                lines = f.readlines()
                cls = list()
                anno = []
                for line in lines:
                    line = line.strip('\n')
                    line = line.split(' ')
                    cls.append(str(line[0]))
                    anno.append(list(map(float, line[1:5])))

                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                boxes_cls_map = defaultdict(list)
                for i in range(len(cls)):
                    # cls = anno[i][0]
                    box = np.array(anno[i])
                    box[[0, 2]] *= img_w
                    box[[1, 3]] *= img_h
                    # boxes_cls_map[cls].append(box)
                    boxes_cls_map[cls[i]].append(box)

                invalid_boxes_cls_map = find_not_match_labels_one_img(img_path, boxes_cls_map, threshold)
                if invalid_boxes_cls_map:
                    imgs.append(img_path)
                    invalid_labels.append(invalid_boxes_cls_map)
    return imgs, invalid_labels


def draw_not_match_labels_one_img(img_path: str, invalid_boxes_cls_map: dict, save_dir: str):
    # 一張圖片
    img_name = img_path.split('/')[-1].rsplit('.', 1)[0]
    img_dir = os.path.join(save_dir, img_name)
    create_dir(img_dir)
    img = cv2.imread(img_path)
    # Save origin image
    origin_path = os.path.join(img_dir, f"{img_name}.jpg")
    save_image(img, origin_path)

    for cls, boxes in invalid_boxes_cls_map.items():
        img = cv2.imread(img_path)
        boxes = np.array(boxes)  # list -> array
        boxes = cxcywh_to_x1y1wh(boxes)
        # 每一個 cls 的 gt 都各自儲存成一個 {cls}.jpg
        img = draw_box(img, boxes, type="gt")
        save_path = os.path.join(img_dir, f"{cls}.jpg")
        save_image(img, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='find label they shonldn\'t be the same')
    parser.add_argument('--part', type=str, default='', help='train / test')
    parser.add_argument('--dataset_name', type=str, default='', help='dataset name')
    parser.add_argument('--dataset', type=str, default='', help='dataset')
    parser.add_argument('--cfg', type=str,
                                 default='./experiments/siamcar_r50/config.yaml', help='configuration of tracking')
    args = parser.parse_args()

    THRESHOLD = 0
    print(f"Threshold is: {THRESHOLD}")

    imgs, invalid_labels = find_not_match_labels(args.dataset, THRESHOLD)

    save_dir = os.path.join(
        "./datasets", args.part, args.dataset_name, "visualization")
    create_dir(save_dir)
    for img_path, invalid_boxes_cls_map in zip(imgs, invalid_labels):
        draw_not_match_labels_one_img(img_path, invalid_boxes_cls_map, save_dir)

    print('=' * 20, "Done", '=' * 20)
