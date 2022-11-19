# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, annotations, division, print_function,
                        unicode_literals)

import argparse
import os
import random
import time

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from pysot.core.config import cfg
from pysot.datasets.collate import collate_fn_new
# 記得要改
from pysot.datasets.pcbdataset_origin import PCBDataset
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
# from pysot.tracker.siamcar_tracker_amy import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.check_image import draw_preds, save_image
from pysot.utils.model_load import load_pretrain
from toolkit.statistics import overlap_ratio_one
# from toolkit.datasets import DatasetFactory
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.cuda.amp import autocast

torch.set_num_threads(1)


def calculate_metrics(pred_boxes, label_boxes):
    """
    Args:
        pred_boxes (list): (data_num, pred_num, 4)
        label_boxes (list): (data_num, label_num, 4)
    """
    assert len(pred_boxes) == len(label_boxes), "length of pred_boxes and label_boxes should be the same"

    tp = list()
    fp = list()
    boxes_num = list()

    for idx in range(len(pred_boxes)):
        # 一個 data
        tp_one = torch.zeros(len(pred_boxes[idx]))
        fp_one = torch.zeros(len(pred_boxes[idx]))
        boxes_one_num = len(label_boxes[idx])
        for pred_idx in range(len(pred_boxes[idx])):
            best_iou = 0
            for label_idx in range(len(label_boxes[idx])):
                iou = overlap_ratio_one(pred_boxes[idx][pred_idx], label_boxes[idx][label_idx])
                if iou > best_iou:
                    best_iou = iou
            # 所有預測出來的 pred_boxes，他們都已經是 positive 了；而且不是 true (TP) 就是 false (FP)
            if best_iou >= 0.5:
                tp_one[pred_idx] = 1
            else:
                fp_one[pred_idx] = 1

        tp_one_sum = sum(tp_one)
        fp_one_sum = sum(fp_one)

        tp.append(tp_one_sum)
        fp.append(fp_one_sum)
        boxes_num.append(boxes_one_num)

    if (sum(tp) + sum(fp) == 0):
        precision = 0
    else:
        precision = sum(tp) / (sum(tp) + sum(fp))
    recall = sum(tp) / sum(boxes_num)

    return precision, recall


def evaluate(test_loader, tracker):
    params = getattr(cfg.HP_SEARCH, "PCB")
    # [0.4, 0.2, 0.3]
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    all_pred_boxes = list()
    pred_classes = list()
    all_gt_boxes = list()
    label_classes = list()

    clocks = 0
    period = 0
    idx = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            """
                data['z_box']: array, (1, [x1, y1, x2, y2])
                data['gt_boxes']: tensor, (n, [0, x1, y1, x2, y2])
            """
            img_path = data['img_path'][0]
            z_box = data['z_box'][0]  # 不要 batch
            z_img = data['z_img'].cuda()
            x_img = data['x_img'].cuda()
            gt_boxes = data['gt_boxes']

            print(f"Load image from: {img_path}")
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ######################################
            # 調整 z_box, gt_boxes
            ######################################
            # (1, [x1, y1, x2, y2]) -> ([x1, y1, w, h])
            z_box = z_box.squeeze()
            z_box[2] = z_box[2] - z_box[0]
            z_box[3] = z_box[3] - z_box[1]
            # z_box = np.around(z_box, decimals=2)

            gt_boxes = gt_boxes.cpu().numpy()
            # gt_boxes: (n, [x1, y1, x2, y2])
            gt_boxes = gt_boxes[:, 1:]
            # gt_boxes: (n, [x1, y1, w, h])
            gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]

            ######################################
            # Init tracker
            ######################################
            # tic = cv2.getTickCount()
            start = time.time()

            # 用 template image 將 tracker 初始化
            with autocast():
                _ = tracker.init(z_img, z_box)
            # _ = tracker.init(img, z_box)

            ######################################
            # Do tracking (predict)
            ######################################
            # 用 search image 進行 "track" 的動作
            with autocast():
                outputs = tracker.track(x_img, hp)
            # outputs = tracker.track(img, hp)

            # toc = cv2.getTickCount()
            # clocks += toc - tic    # 總共有多少個 clocks (clock cycles)

            end = time.time()
            period += end - start

            all_pred_boxes.append(outputs['pred_boxes'])
            all_gt_boxes.append(gt_boxes.tolist())

            # precision, recall = calculate_metrics([outputs['pred_boxes']], [gt_boxes.tolist()])
            # ipdb.set_trace()
        precision, recall = calculate_metrics(all_pred_boxes, all_gt_boxes)
        precision = precision * 100
        recall = recall * 100

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # period = clocks / cv2.getTickFrequency()
        fps = (idx + 1) / period
        print(f"Speed: {fps} fps")

        return {
            "precision": precision,
            "recall": recall,
            "fps": fps
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='siamcar tracking')
    parser.add_argument('--model', type=str, default='', help='model to eval')
    parser.add_argument('--dataset_name', type=str, default='', help='dataset name')
    parser.add_argument('--dataset', type=str, default='', help='training dataset')
    parser.add_argument('--test_dataset', type=str, default='', help='testing dataset')
    parser.add_argument('--criteria', type=str, default='', help='criteria of dataset')
    parser.add_argument('--neg', type=float, default=0.0, help='negative pair')
    parser.add_argument('--bg', type=str, help='background of template')
    parser.add_argument('--cfg', type=str, default='./experiments/siamcar_r50/config.yaml', help='configuration of tracking')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)  # 不加 ModelBuilder() 會出問題ㄟ ??

    print("Building dataset...")
    test_dataset = PCBDataset(args, "test")
    print(f"Dataset size: {len(test_dataset)}")
    assert len(test_dataset) != 0, "ERROR, empty dataset"

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 只能設 1
        num_workers=0,
        collate_fn=collate_fn_new
    )

    # Create model
    model = ModelBuilder()
    # Load model
    model = load_pretrain(model, args.model).cuda().train()
    # Build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # evaluate(test_loader, tracker)
    print("Start evaluating...")
    metrics = evaluate(test_loader, tracker)

    print("=" * 20, "DONE!", "=" * 20, "\n")
