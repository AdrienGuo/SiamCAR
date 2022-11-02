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
from pysot.core.config import cfg
from pysot.datasets.collate import collate_fn_new
# new / tri
from pysot.datasets.pcbdataset_tri import PCBDataset
from pysot.models.model_builder import ModelBuilder
# tracker 可以改
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.check_image import (create_dir, draw_box, draw_preds,
                                     save_image)
from pysot.utils.model_load import load_pretrain
from toolkit.datasets.testdata import ImageFolderWithSelect
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append('../')

parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--model', type=str, default='', help='model to eval')
parser.add_argument('--dataset_name', type=str, default='', help='dataset name')
parser.add_argument('--dataset_path', type=str, default='', help='testing dataset')
parser.add_argument('--criteria', type=str, default='', help='criteria of dataset')
parser.add_argument('--neg', type=float, default=0.0, help='negative pair')
parser.add_argument('--bg', type=str, help='background of template')
parser.add_argument('--cfg', type=str, default='./experiments/siamcar_r50/config.yaml', help='configuration of tracking')
args = parser.parse_args()

torch.set_num_threads(1)


def main(save_dir):
    # load config
    cfg.merge_from_file(args.cfg)

    # hp_search
    params = getattr(cfg.HP_SEARCH, "PCB")
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.model).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # Create dataset
    # dataset = ImageFolderWithSelect(args.test_dataset)
    dataset = PCBDataset(args=args, mode="test")
    assert len(dataset) != 0, "Error, dataset is empty!!"

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn_new
    )

    model_name = args.model.split('/')[-2]
    print("Model:", model_name)

    # for idx, (_, annotation, path, check) in enumerate(dataset):
    #     img_name = path.split('/')[-1].rsplit('.', 1)[0]
    #     count = 0
    #     print("Img name:", img_name)

    #     ##########################################
    #     # Create directories
    #     ##########################################
    #     sub_dir = os.path.join(save_dir, img_name)
    #     create_dir(sub_dir)

    #     z_dir = os.path.join(sub_dir, "template")
    #     create_dir(z_dir)
    #     x_dir = os.path.join(sub_dir, "search")
    #     create_dir(x_dir)
    #     anno_dir = os.path.join(sub_dir, "pred_annotation")
    #     create_dir(anno_dir)
    #     pred_dir = os.path.join(sub_dir, "pred")
    #     create_dir(pred_dir)

    #     for i in range(len(annotation)):
    #         annotation[i][1] = float(annotation[i][1])
    #         annotation[i][2] = float(annotation[i][2])
    #         annotation[i][3] = float(annotation[i][3])
    #         annotation[i][4] = float(annotation[i][4])
    #         tic = cv2.getTickCount()
    #         toc = 0

    #         pred_bboxes = []

    #         # init
    #         count += 1
    #         classid = annotation[i][0]
            
    #         if annotation[i][0] != 26:
    #             img = cv2.imread(path)
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #             if check == 0:
    #                 gt_bbox = [annotation[i][1]*img.shape[1], annotation[i][2]*img.shape[0], annotation[i][3]*img.shape[1], annotation[i][4]*img.shape[0]]
    #                 gt_bbox = [gt_bbox[0]-gt_bbox[2]/2, gt_bbox[1]-gt_bbox[3]/2, gt_bbox[2], gt_bbox[3]]
    #             else:
    #                 gt_bbox = [annotation[i][1], annotation[i][2], annotation[i][3]-annotation[i][1], annotation[i][4]-annotation[i][2]]
    #             cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    #             gt_bbox_ = [cx-w/2, cy-h/2, w, h]
    #             init_info = {'init_bbox': gt_bbox_}

    #             with torch.no_grad():
    #                 z_crop = tracker.init(img, gt_bbox_)
    #             pred_bbox = gt_bbox_
    #             pred_bboxes.append(pred_bbox)

    #             # track
    #             with torch.no_grad():
    #                 outputs = tracker.track(img, hp)
    #             pred_bbox = outputs['pred_boxes']
    #             pred_bboxes.append(pred_bbox)

    #             toc += cv2.getTickCount() - tic

    #             # === Save z_crop ===
    #             z_crop = z_crop.cpu().numpy().squeeze()
    #             z_crop = z_crop.transpose(1, 2, 0)
    #             save_z_crop = os.path.join(z_dir, f"{idx}.jpg")
    #             cv2.imwrite(save_z_crop, z_crop)
    #             print(f"Save z_crop: {save_z_crop}")

    #             # === Save x_crop ===
    #             x_crop = outputs['x_crop'].cpu().numpy().squeeze()
    #             x_crop = x_crop.transpose(1, 2, 0)
    #             save_x_crop = os.path.join(x_dir, f"{idx}.jpg")
    #             cv2.imwrite(save_x_crop, x_crop)
    #             print(f"Save x_crop: {save_x_crop}")

    #             ipdb.set_trace()

    #             model_path = os.path.join('results', args.datasetimg_name, model_name)
    #             if not os.path.isdir(model_path):
    #                 os.makedirs(model_path)

    #             # save results
    #             temp = img_name + "__" + str(classid) + "__" + str(count)
    #             result_path = os.path.join(model_path, '{}.txt'.format(img_name + "__" + str(classid) + "__" + str(count)))
    #             with open(result_path, 'w') as f:
    #                 for x in pred_bboxes:
    #                     f.write(','.join([str(i) for i in x]) + "=" + '\n')
    #             print(f"save result to: {result_path}")
    #             toc /= cv2.getTickFrequency()

    #             print('Time: {:5.1f}s Speed: {:3.1f}fps'.format(toc, 1 / toc))
            
    #         ipdb.set_trace()

    # return

    for idx, data in enumerate(test_loader):
        img_path = data['img_path'][0]
        z_img = data['z_img'].cuda()
        x_img = data['x_img'].cuda()
        gt_boxes = data['gt_boxes']  # (N, [0, x1, y1, x2, y2])
        z_box = data['z_box'][0]  # ([x1, y1, x2, y2])
        scale = data['scale'][0]

        z_box = z_box.squeeze()
        gt_boxes = gt_boxes[:, 1:]  # 不要 0 那項

        # img = cv2.imread(img_path)
        # 應該不用轉成 rgb 吧？？
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ##########################################
        # Create directories
        ##########################################
        img_name = img_path.split('/')[-1].rsplit('.', 1)[0]
        sub_dir = os.path.join(save_dir, img_name)
        create_dir(sub_dir)

        z_dir = os.path.join(sub_dir, "template")
        create_dir(z_dir)
        x_dir = os.path.join(sub_dir, "search")
        create_dir(x_dir)
        anno_dir = os.path.join(sub_dir, "pred_annotation")
        create_dir(anno_dir)
        pred_dir = os.path.join(sub_dir, "pred")
        create_dir(pred_dir)

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

        ######################################
        # Do tracking (predict)
        ######################################
        with torch.no_grad():
            outputs = tracker.track(x_img, hp)

        # ipdb.set_trace()

        # === save z_img ===
        z_img = z_img.cpu().numpy().squeeze()
        z_img = np.transpose(z_img, (1, 2, 0))        # (3, 127, 127) -> (127, 127, 3)
        z_path = os.path.join(z_dir, f"{idx}.jpg")
        save_image(z_img, z_path)

        # === save x_img ===
        x_img = outputs['x_img']
        x_img = x_img.cpu().numpy().squeeze()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_path = os.path.join(x_dir, f"{idx}.jpg")
        save_image(x_img, x_path)

        # pred_scores on x_img
        scores = np.around(outputs['top_scores'], decimals=2)
        # === pred_boxes on x_img ===
        for box in outputs['pred_boxes']:
            box = np.around(box, decimals=2)
            pred_boxes.append(box)

        # === Save annotation on x_img ===
        anno_path = os.path.join(anno_dir, f"{idx}.txt")
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
        x_img = draw_box(x_img, gt_boxes, type="gt")
        # pred_boxes on x_img
        pred_path = os.path.join(pred_dir, f"{idx}.jpg")
        pred_image = draw_preds(sub_dir, x_img, scores, anno_path, idx)
        if pred_image is None:    # 如果沒偵測到物件，存 x_img
            save_image(x_img, pred_path)
        else:
            save_image(pred_image, pred_path)

        # ipdb.set_trace()


if __name__ == '__main__':
    model_name = args.model.split('/')[-2]
    print(f"Model: {model_name}")

    save_dir = os.path.join("./results", args.dataset_name, args.criteria, model_name)
    create_dir(save_dir)
    print(f"Test results saved to: {save_dir}")

    main(save_dir)
