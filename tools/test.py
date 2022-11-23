# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import math
import os
import re
import shutil
import sys

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from eval import calculate_metrics
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pysot.core.config import cfg
from pysot.datasets.collate import collate_fn_new
# new / tri / origin
from pysot.datasets.pcbdataset_origin import PCBDataset
from pysot.models.model_builder import ModelBuilder
# tracker 可以改
from pysot.tracker.siamcar_tracker import SiamCARTracker
# from pysot.tracker.siamcar_tracker_amy import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.check_image import (create_dir, draw_box, draw_heatmap,
                                     draw_preds, save_image)
from pysot.utils.model_load import load_pretrain
from toolkit.datasets.testdata import ImageFolderWithSelect

sys.path.append('../')

torch.set_num_threads(1)


def create_img_dir(save_dir, img_name) -> dict:
    sub_dir = os.path.join(save_dir, img_name)
    create_dir(sub_dir)

    origin_dir = os.path.join(sub_dir, "origin")
    create_dir(origin_dir)
    z_dir = os.path.join(sub_dir, "template")
    create_dir(z_dir)
    x_dir = os.path.join(sub_dir, "search")
    create_dir(x_dir)
    anno_dir = os.path.join(sub_dir, "pred_annotation")
    create_dir(anno_dir)
    pred_dir = os.path.join(sub_dir, "pred")
    create_dir(pred_dir)
    heatmap_dir = os.path.join(sub_dir, "heatmap")
    create_dir(heatmap_dir)

    path_dir_map = dict()
    path_dir_map['origin'] = origin_dir
    path_dir_map['template'] = z_dir
    path_dir_map['search'] = x_dir
    path_dir_map['annotation'] = anno_dir
    path_dir_map['pred'] = pred_dir
    path_dir_map['heatmap'] = heatmap_dir

    return path_dir_map


def save_fail_img(img_name, img, z_img, x_img, pred_img, heatmap, idx):
    path_dir_map = create_img_dir(fail_dir, img_name)
    origin_path = os.path.join(path_dir_map['origin'], f"{img_name}.jpg")
    save_image(img, origin_path)
    z_path = os.path.join(path_dir_map['template'], f"{idx}.jpg")
    save_image(z_img, z_path)
    x_path = os.path.join(path_dir_map['search'], f"{idx}.jpg")
    save_image(x_img, x_path)
    pred_path = os.path.join(path_dir_map['pred'], f"{idx}.jpg")
    save_image(pred_img, pred_path)
    heatmap_path = os.path.join(path_dir_map['heatmap'], f"{idx}.jpg")
    heatmap.savefig(heatmap_path)
    print(f"Save heatmap to: {heatmap_path}")


def test_and_eval(tracker, test_loader):
    # hp_search
    params = getattr(cfg.HP_SEARCH, "PCB")
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    all_pred_boxes = list()
    all_gt_boxes = list()

    for idx, data in enumerate(test_loader):
        img_name = data['img_name'][0]
        img_path = data['img_path'][0]
        z_img = data['z_img'].cuda()
        x_img = data['x_img'].cuda()
        gt_boxes = data['gt_boxes']  # (N, [0, x1, y1, x2, y2])
        z_box = data['z_box'][0]  # ([x1, y1, x2, y2])
        # scale = data['scale'][0]

        z_box = z_box.squeeze()
        gt_boxes = gt_boxes[:, 1:]  # 不要 0 那項

        # 用 PatternMatch_test 資料集的時候不能加，
        # 因為 img_path 的路徑是 dir 不是 path
        img = cv2.imread(img_path)
        print(f"Load image from: {img_path}")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        path_dir_map = create_img_dir(save_dir, img_name)

        ##########################################
        # Predict
        ##########################################
        pred_boxes = []

        # 調整 z_box, gt_boxes 的框框，tracker.init() 的格式需要
        z_box[2] = z_box[2] - z_box[0]    # x2 -> w
        z_box[3] = z_box[3] - z_box[1]    # y2 -> h
        pred_boxes.append(z_box)

        gt_boxes = gt_boxes.cpu().numpy()
        gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]    # x2 -> w
        gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]    # y2 -> h

        ######################################
        # Init tracker
        ######################################
        with torch.no_grad():
            z_img = tracker.init(z_img, bbox=z_box)
            # z_img = tracker.init(img, bbox=z_box, z_img=z_img)

        ######################################
        # Do tracking (predict)
        ######################################
        with torch.no_grad():
            outputs = tracker.track(x_img, hp)
            # outputs = tracker.track(img, hp, x_img=x_img)

        # Save original image
        origin_path = os.path.join(path_dir_map['origin'], f"{img_name}.jpg")
        save_image(img, origin_path)

        # Save z_img
        z_img = z_img.cpu().numpy().squeeze()
        z_img = np.transpose(z_img, (1, 2, 0))  # (3, 127, 127) -> (127, 127, 3)
        z_path = os.path.join(path_dir_map['template'], f"{idx}.jpg")
        save_image(z_img, z_path)

        # Save x_img
        x_img = outputs['x_img']
        x_img = x_img.cpu().numpy().squeeze()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_path = os.path.join(path_dir_map['search'], f"{idx}.jpg")
        save_image(x_img, x_path)

        # Save heatmap
        cls = outputs['cls']
        heatmap = draw_heatmap(x_img, cls)
        heatmap_path = os.path.join(path_dir_map['heatmap'], f"{idx}.jpg")
        heatmap.savefig(heatmap_path)
        print(f"Save heatmap to: {heatmap_path}")

        # pred_scores on x_img
        scores = np.around(outputs['top_scores'], decimals=2)
        # pred_boxes on x_img
        for box in outputs['pred_boxes']:
            box = np.around(box, decimals=2)
            pred_boxes.append(box)

        # Save annotation on x_img
        anno_path = os.path.join(path_dir_map['annotation'], f"{idx}.txt")
        with open(anno_path, 'w') as f:
            # template
            f.write(', '.join(map(str, pred_boxes[0])) + '\n')
            # preds
            for i, x in enumerate(pred_boxes[1:]):
                # format: [x1, y1, w, h]
                f.write(', '.join(map(str, x)) + ', ' + str(scores[i]) + '\n')
        print(f"Save annotation result to: {anno_path}")

        ##########################################
        # Draw
        ##########################################
        # gt_boxes on x_img
        pred_img = draw_box(x_img, gt_boxes, type="gt")
        # pred_boxes on x_img
        pred_img = draw_preds(pred_img, scores, anno_path, idx)
        if pred_img is None:  # 如果沒偵測到物件，存 x_img
            pred_img = x_img
        pred_path = os.path.join(path_dir_map['pred'], f"{idx}.jpg")
        save_image(pred_img, pred_path)

        # ##########################################
        # # Save fail pred image
        # ##########################################
        precision, recall = calculate_metrics([outputs['pred_boxes']], [gt_boxes.tolist()])
        if precision != 1 or recall != 1:
            save_fail_img(img_name, img, z_img, x_img, pred_img, heatmap, idx)
            # save_fail_img(img_name, img, z_img, x_img, pred_img, idx)

        # For evaluating
        all_pred_boxes.append(outputs['pred_boxes'])
        all_gt_boxes.append(gt_boxes.tolist())

        # ipdb.set_trace()

    precision, recall = calculate_metrics(all_pred_boxes, all_gt_boxes)
    precision = precision * 100
    recall = recall * 100

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


def main():
    # load config
    cfg.merge_from_file(args.cfg)

    model = ModelBuilder()
    # load model
    # model = load_pretrain(model, args.model).cuda().eval()
    model = load_pretrain(model, args.model).cuda().train()
    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # Create dataset
    print("Building dataset...")
    # dataset = ImageFolderWithSelect(args.test_dataset)
    dataset = PCBDataset(args=args, mode="test")
    assert len(dataset) != 0, "ERROR, dataset is empty!!"
    print(f"Test dataset size: {len(dataset)}")

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn_new
    )

    test_and_eval(tracker, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='siamcar tracking')
    parser.add_argument('--model', type=str, default='', help='model to eval')
    parser.add_argument('--dataset_name', type=str, default='', help='dataset name')
    parser.add_argument('--part', type=str, default='', help='train / test')
    parser.add_argument('--test_dataset', type=str, default='', help='testing dataset')
    parser.add_argument('--criteria', type=str, default='', help='criteria of dataset')
    parser.add_argument('--neg', type=float, default=0.0, help='negative pair')
    parser.add_argument('--bg', type=str, help='background of template')
    parser.add_argument('--cfg', type=str, default='./experiments/siamcar_r50/config.yaml', help='configuration of tracking')
    args = parser.parse_args()

    # TODO: 應該要先 load model 才能創資料夾
    print(f"Load model from: {args.model}")
    model_dir = args.model.split('/')[-2]
    model_epoch = args.model.split('/')[-1].rsplit('.', 1)[0]
    model_name = model_dir + '_' + model_epoch
    model_name = re.sub("_model", "", model_name)
    print(f"Model name: {model_name}")

    save_dir = os.path.join("./results", args.part, args.dataset_name, args.criteria, model_name)
    create_dir(save_dir)
    print(f"Test results saved to: {save_dir}")

    fail_dir = os.path.join(save_dir, "FAILED")
    create_dir(fail_dir)
    print(f"Failed results saved to: {fail_dir}")

    main()

    print('=' * 20, "DONE!", '=' * 20)
