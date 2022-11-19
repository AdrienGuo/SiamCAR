# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import math
import os
import shutil
import sys

import cv2
import ipdb
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker_amy import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.datasets.testdata import ImageFolderWithSelect
from pysot.utils.model_load import load_pretrain

sys.path.append('../')
torch.set_num_threads(1)


def test_amy(snapshot, dataset_dir):
    """
    Args:
        snapshot: model path
        dataset_dir: dataset directory
    """
    print(f"Load model from: {snapshot}")
    model_name = snapshot.split('/')[-2]
    print("Model name:", model_name)
    dataset_name = dataset_dir.split('/')[-1]
    print(f"Dataset name: {dataset_name}")

    # hp_search
    params = getattr(cfg.HP_SEARCH, "PCB")
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset
    dataset = ImageFolderWithSelect(dataset_dir)

    for idx, (_, annotation, path, check) in enumerate(dataset):
        img_name = path.split('/')[-1]
        count = 0
        print("name:", img_name)
        for i in range(len(annotation)):
            annotation[i][1] = float(annotation[i][1])
            annotation[i][2] = float(annotation[i][2])
            annotation[i][3] = float(annotation[i][3])
            annotation[i][4] = float(annotation[i][4])
            tic = cv2.getTickCount()
            toc = 0

            pred_bboxes = []

            # init
            count += 1
            classid = annotation[i][0]

            if annotation[i][0] != 26:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if check == 0:
                    gt_bbox = [annotation[i][1]*img.shape[1], annotation[i][2]*img.shape[0],
                               annotation[i][3]*img.shape[1], annotation[i][4]*img.shape[0]]
                    gt_bbox = [gt_bbox[0]-gt_bbox[2]/2, gt_bbox[1] -
                               gt_bbox[3]/2, gt_bbox[2], gt_bbox[3]]
                else:
                    gt_bbox = [annotation[i][1], annotation[i][2], annotation[i]
                               [3]-annotation[i][1], annotation[i][4]-annotation[i][2]]
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                init_info = {'init_bbox': gt_bbox_}

                # Init tracker
                with torch.no_grad():
                    z_crop = tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)

                # track
                with torch.no_grad():
                    outputs = tracker.track(img, hp)
                pred_bbox = outputs['pred_boxes']
                pred_bboxes.append(pred_bbox)
                toc += cv2.getTickCount() - tic

                # === Save z_crop ===
                # z_crop = z_crop.cpu().numpy().squeeze()
                # z_crop = z_crop.transpose(1, 2, 0)
                # save_z_crop = f"./image_check/z_crop/{idx}__{classid}__{count}.jpg"
                # cv2.imwrite(save_z_crop, z_crop)
                # print(f"Save z_crop: {save_z_crop}")

                # # === Save x_crop ===
                # x_crop = outputs['x_crop'].cpu().numpy().squeeze()
                # x_crop = x_crop.transpose(1, 2, 0)
                # save_x_crop = f"./image_check/x_crop/{idx}__{classid}__{count}.jpg"
                # cv2.imwrite(save_x_crop, x_crop)
                # print(f"Save x_crop: {save_x_crop}")

                # ipdb.set_trace()

                anno_save_path = os.path.join(
                    'results_amy', dataset_name, model_name, "annotation")
                if not os.path.isdir(anno_save_path):
                    os.makedirs(anno_save_path)
                    print(f"Create new dir: {anno_save_path}")

                # save results
                img_save_path = img_name + "__" + str(classid) + "__" + str(count)
                result_path = os.path.join(
                    anno_save_path, '{}.txt'.format(img_save_path))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+"="+'\n')
                toc /= cv2.getTickFrequency()

                print(f"Save annotation to: {result_path} | time: {toc:3.1f}s")

                # print('Time: {:5.1f}s Speed: {:3.1f}fps'.format(toc, 1 / toc))

        # ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='siamcar tracking')
    parser.add_argument('--snapshot', type=str,
                        default='./snapshot/amy/checkpoint_e999.pth', help='snapshot of models to eval')
    parser.add_argument('--dataset', default='./datasets/train/allOld', type=str, help='dataset')
    parser.add_argument('--cfg', type=str,
                        default='./experiments/siamcar_r50/config_amy.yaml', help='config file')
    args = parser.parse_args()

    # load config
    cfg.merge_from_file(args.cfg)

    test_amy(args.snapshot, args.dataset)
