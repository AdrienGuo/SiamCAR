# Copyright (c) SenseTime. All Rights Reserved.
# 一張影像依類別分開 有Text

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
import sys
from collections import namedtuple

import cv2
import ipdb
import numpy as np
import torch
from PIL import Image
from pysot.core.config import cfg
from pysot.datasets.image_crop import crop, resize
from pysot.datasets.pcb_crop_origin import PCBCrop
from pysot.utils.bbox import Center, center2corner, ratio2real
from pysot.utils.check_image import create_dir, draw_box, save_image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

# from re import template


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

    
class PCBDataset(Dataset):
    def __init__(self, args, mode: str, loader=default_loader):
        """ 代號
            z: template
            x: search
        Args:
            mode: train / test
        """
        self.args = args

        if mode == "test":
            self.dataset_dir = args.test_dataset
        else:
            self.dataset_dir = args.dataset
        # self.anno = args.dataset
        # self.loader = loader
        images, templates, searches = self._make_dataset(self.dataset_dir)
        images, templates, searches = self._filter_dataset(images, templates, searches, args.criteria)

        self.images = images
        self.templates = templates
        self.searches = searches

        # zf_size_min: smallest z size after res50 backbone
        zf_size_min = 5
        # PCBCrop: Crop template & search (preprocess)
        self.pcb_crop = PCBCrop(zf_size_min)

    # def _find_classes(self,directory: str):
    #     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    #     if not classes:
    #         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    #     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    #     return classes, class_to_idx
    
    def _make_dataset(self, directory: str):
        images = []
        templates = []
        searches = []

        # directory = os.path.expanduser(directory)
        for root, _, files in sorted(os.walk(directory, followlinks=True)):
            for file in sorted(files):    # 排序
                box = []
                if file.endswith(('.jpg', '.png', 'bmp')):
                    img_path = os.path.join(root, file)
                    anno_path = os.path.join(root, file[:-3] + "txt")
                    if os.path.isfile(anno_path):
                        f = open(anno_path, 'r')
                        lines = f.readlines()
                        anno = []
                        for line in lines:
                            line = line.strip('\n')
                            line = line.split(' ')
                            line = list(map(float, line))
                            anno.append(line)

                        for i in range(len(anno)):
                            # TODO: 現在新的資料集，所以這裡要改掉
                            if anno[i][0] != 26 or self.args.dataset_name == "all":
                                item = img_path, str(int(anno[i][0]))
                                images.append(item)
                                templates.append([anno[i][1], anno[i][2], anno[i][3], anno[i][4]])
                                box = []
                            if anno[i][0] != 26 or self.args.dataset_name == "all":
                                for j in range(len(anno)):
                                    if anno[j][0] == anno[i][0]:
                                        box.append([anno[j][1], anno[j][2], anno[j][3], anno[j][4]])
                                box = np.stack(box).astype(np.float32)
                                searches.append(box)
                    # text 類型
                    elif os.path.isfile(os.path.join(root, file[:-3] + "label")):
                        anno_path = os.path.join(root, file[:-3] + "label")
                        f = open(anno_path, 'r')
                        img = cv2.imread(img_path)
                        imh, imw = img.shape[:2]
                        lines = f.readlines()
                        anno = []
                        for line in lines:
                            line = line.strip('\n')
                            line = line.split(',')
                            line = list(line)
                            anno.append(line)
                        for i in range(len(anno)):
                            if (float(anno[i][1]) > 0) and (float(anno[i][2]) > 0):
                                item = img_path, anno[i][0]
                                images.append(item)
                                cx = float(anno[i][1]) + (float(anno[i][3]) - float(anno[i][1])) / 2
                                cy = float(anno[i][2]) + (float(anno[i][4]) - float(anno[i][2])) / 2
                                w = float(anno[i][3]) - float(anno[i][1])
                                h = float(anno[i][4]) - float(anno[i][2])
                                templates.append([cx/imw, cy/imh, w/imw, h/imh])
                                box = []
                                for j in range(len(anno)):
                                    if anno[j][0] == anno[i][0]:
                                        cx = float(anno[i][1]) + (float(anno[i][3]) - float(anno[i][1])) / 2
                                        cy = float(anno[i][2]) + (float(anno[i][4]) - float(anno[i][2])) / 2
                                        w = float(anno[i][3]) - float(anno[i][1])
                                        h = float(anno[i][4]) - float(anno[i][2])
                                        box.append([cx/imw, cy/imh, w/imw, h/imh])

                                box = np.stack(box).astype(np.float32)
                                searches.append(box)
                    else:
                        assert False, f"ERROR, no annotation for image: {img_path}"

        return images, templates, searches

    def _filter_dataset(self, images, templates, searches, criteria):
        # criteria == all
        if criteria == "all":
            return images, templates, searches
        # for
        inds_match = list()
        for idx, image in enumerate(images):
            # read image
            img = cv2.imread(image[0])
            # get w & h
            img_h, img_w = img.shape[:2]
            z_w = templates[idx][2] * img_w
            z_h = templates[idx][3] * img_h
            # calculate r by resize to 255
            long_side = max(img_w, img_h)
            r = cfg.TRAIN.SEARCH_SIZE / long_side
            # calculate template new w, h
            z_w = z_w * r
            z_h = z_h * r
            if criteria == "small":
                if max(z_w, z_h) <= 32:
                    inds_match.append(idx)
            elif criteria == "mid":
                if 32 < max(z_w, z_h) <= 64:
                    inds_match.append(idx)
            elif criteria == "big":
                if max(z_w, z_h) > 64:
                    inds_match.append(idx)
            else:
                assert False, "ERROR, chosen criteria is wrong!"
        images = [images[i] for i in inds_match]
        templates = [templates[i] for i in inds_match]
        searches = [searches[i] for i in inds_match]

        return images, templates, searches

    def _get_image_anno(self, idx, data):
        img_path, template_cls = self.images[idx]
        image_anno = data[idx]
        return img_path, image_anno

    def _get_positive_pair(self, idx):
        return self._get_image_anno(idx, self.templates), \
               self._get_image_anno(idx, self.searches)

    def _get_negative_pair(self, idx):
        while True:
            idx_neg = np.random.randint(0, len(self.images))
            if self.images[idx][0] != self.images[idx_neg][0]:
                # idx 和 idx_neg 不是對應到同一張圖
                break
        return self._get_image_anno(idx, self.templates), \
               self._get_image_anno(idx_neg, self.searches)
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()

        # 加入 neg 的原因要去看 [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
        neg = cfg.DATASET.NEG and self.args.neg > np.random.random()

        # Get dataset.
        if neg:
            template, search = self._get_negative_pair(idx)
        else:
            template, search = self._get_positive_pair(idx)

        # Get image.
        assert template[0] == search[0], f"ERROR, should be the same if neg is False"
        img_path = template[0]
        img_name = img_path.split('/')[-1].rsplit('.', 1)[0]
        # origin_img = cv2.imread(img_path)
        # template_image = cv2.imread(img_path)
        # search_image = cv2.imread(img_path)
        img = cv2.imread(img_path)
        assert img is not None, f"Error image: {template[0]}"
        z_img = img
        x_img = img

        # img_cls = template[2]
        # assert isinstance(img_cls, str), f"Error, class should be string"
        # if template_image is None:
        #     print('error image:',template[0])

        ##########################################
        # Crop the template & search image.
        ##########################################
        z_box = template[1].copy()
        gt_boxes = search[1].copy()  # gt_boxes: (num, [cx, cy, w, y]) #ratio

        z_box = np.asarray(z_box)
        z_box = z_box[np.newaxis, :]  # [cx, cy, w, h] -> (1, [cx, cy, w, h]) 轉成跟 gt_boxes 一樣是二維的
        gt_boxes = np.asarray(gt_boxes)
        # center -> corner & ratio -> real
        z_box = center2corner(z_box)
        z_box = ratio2real(img, z_box)
        gt_boxes = center2corner(gt_boxes)
        gt_boxes = ratio2real(img, gt_boxes)

        # z_box: (1, [x1, y1, x2, y2])
        # gt_boxes: (num, [x1, y1, x2, y2])
        z_img, r = self.pcb_crop.get_template(x_img, z_box, self.args.bg)
        x_img, gt_boxes, z_box = self.pcb_crop.get_search(x_img, gt_boxes, z_box, r)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ########################
        # 創 directory
        ########################
        # dir = f"./image_check/x{cfg.TRAIN.SEARCH_SIZE}_bg{self.args.bg}"
        # sub_dir = os.path.join(dir, img_name)
        # create_dir(sub_dir)

        # # sub_dir/origin，裡面存 origin image
        # origin_dir = os.path.join(sub_dir, "origin")
        # create_dir(origin_dir)
        # # sub_dir/template，裡面存 template image
        # template_dir = os.path.join(sub_dir, "template")
        # create_dir(template_dir)
        # # sub_dir/search，裡面存 search image
        # search_dir = os.path.join(sub_dir, "search")
        # create_dir(search_dir)

        # #########################
        # # 存圖片
        # #########################
        # origin_path = os.path.join(origin_dir, "origin.jpg")
        # save_image(img, origin_path)
        # template_path = os.path.join(template_dir, f"{idx}.jpg")
        # save_image(z_img, template_path)

        # # Draw gt_boxes on search image
        # tmp_gt_boxes = np.asarray(gt_boxes).astype(None).copy()
        # tmp_gt_boxes[:, 2] = tmp_gt_boxes[:, 2] - tmp_gt_boxes[:, 0]
        # tmp_gt_boxes[:, 3] = tmp_gt_boxes[:, 3] - tmp_gt_boxes[:, 1]
        # gt_image = draw_box(x_img, tmp_gt_boxes, type="gt")
        # # tmp_z_box = np.asarray(z_box).astype(None).copy()
        # # tmp_z_box[:, 2] = tmp_z_box[:, 2] - tmp_z_box[:, 0]
        # # tmp_z_box[:, 3] = tmp_z_box[:, 3] - tmp_z_box[:, 1]
        # # z_gt_image = draw_box(gt_image, tmp_z_box, type="template")
        # search_path = os.path.join(search_dir, f"{idx}.jpg")
        # save_image(gt_image, search_path)

        # ipdb.set_trace()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        col_num = len(gt_boxes)
        gt_boxes = np.c_[np.zeros(col_num), gt_boxes]  # Add a all zeros column to gt_boxes
        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.int64)

        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        cls = torch.as_tensor(cls, dtype=torch.int64)

        z_img = z_img.transpose((2, 0, 1)).astype(np.float32)
        x_img = x_img.transpose((2, 0, 1)).astype(np.float32)

        z_img = torch.as_tensor(z_img, dtype=torch.float32)
        x_img = torch.as_tensor(x_img, dtype=torch.float32)

        # 小於 15 的卷積後會變成 0
        assert z_img.size(1) > 15, f"ERROR, should > 15, but got {z_img.size(1)}"
        assert z_img.size(2) > 15, f"ERROR, should > 15, but got {z_img.size(2)}"

        # cls: (size, size) 都是 0
        # box: (n, 5: [0, x1, y1, x2, y2])
        # z_box: Corner(x1, y1, x2, y2)

        return img_name, img_path, z_img, x_img, cls, gt_boxes, z_box, r

        return {
            "img_path": img_path,
            "z_img": z_img,
            "x_img": x_img,
            "cls": cls,
            "box": box,
            "z_box": z_box,
            "r": r
        }
