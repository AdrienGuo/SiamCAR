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
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from pysot.core.config import cfg
from pysot.datasets.pcb_crop.pcb_crop_origin import PCBCropOrigin
from pysot.datasets.augmentation.pcb_aug import PCBAug
from pysot.datasets.utils.process import z_score_norm
from pysot.utils.bbox import Center, center2corner, ratio2real
from pysot.utils.check_image import create_dir, draw_box, save_image


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

    
class PCBDatasetOrigin(Dataset):
    def __init__(self, args, mode: str, loader = default_loader):
        """ 代號
            z: template
            x: search
        Args:
            mode: train / test
        """
        self.args = args

        self.mode = mode
        if mode == "test":
            self.dataset_dir = args.test_dataset
        else:
            self.dataset_dir = args.dataset
        images, templates, searches = self._make_dataset(self.dataset_dir, args.target)
        images, templates, searches = self._filter_dataset(
            images, templates, searches, args.criteria)
        assert len(images) != 0, "ERROR, dataset is empty"
        self.images = images
        self.templates = templates
        self.searches = searches

        # zf_size_min: smallest z size after res50 backbone
        zf_size_min = 4
        # PCBCropOrigin: Crop template & search (preprocess)
        self.pcb_crop = PCBCropOrigin(zf_size_min)

        # Augmentation
        self.pcb_aug = PCBAug(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR
        )

    def _make_dataset(self, directory: str, target: str):
        images = []
        templates = []
        searches = []

        # 標記錯誤的影像
        imgs_exclude = ["6_cae_cae_20200803_10.bmp"]
        mid_imgs_exclude = ['17_ic_ic_20200810_solder_40.bmp', '17_ic_Sot23_20200820_solder_81.bmp', '5_sod_sod (7).jpg']
        small_imgs_exclude = ['']
        if self.args.criteria == "mid":
            imgs_exclude += mid_imgs_exclude
        elif self.args.criteria == "small":
            # TODO
            pass

        # directory = os.path.expanduser(directory)
        for root, _, files in sorted(os.walk(directory, followlinks=True)):
            for file in sorted(files):  # 排序
                # box = []
                if file in imgs_exclude:
                    # These images cause OOM
                    continue
                if file.endswith(('.jpg', '.png', 'bmp')):
                    # one image
                    img_path = os.path.join(root, file)
                    anno_path = os.path.join(root, file[:-3] + "txt")
                    if os.path.isfile(anno_path):
                        # annotation matches the image
                        f = open(anno_path, 'r')
                        lines = f.readlines()
                        cls = list()
                        anno = list()
                        for line in lines:
                            line = line.strip('\n')
                            line = line.split(' ')
                            # line[0] = str(line[0])
                            # 因為這裡 line 裡面存了兩種 type (str, float) 所以報錯
                            # line[1:5] = map(float, line[1:5])
                            # anno.append(line)
                            cls.append(str(line[0]))
                            anno.append(list(map(float, line[1:5])))

                        for i in range(len(cls)):
                            item = img_path, cls[i]
                            images.append(item)
                            templates.append([anno[i][0], anno[i][1], anno[i][2], anno[i][3]])
                            box = list()
                            if target == "one":
                                # 單目標偵測
                                box.append([anno[i][0], anno[i][1], anno[i][2], anno[i][3]])
                            elif target == "multi":
                                # 多目標偵測
                                for j in range(len(cls)):
                                    if cls[j] == cls[i]:
                                        box.append([anno[j][0], anno[j][1], anno[j][2], anno[j][3]])
                            box = np.stack(box).astype(np.float32)
                            searches.append(box)
                    # text 類型
                    # elif os.path.isfile(os.path.join(root, file[:-3] + "label")):
                    #     anno_path = os.path.join(root, file[:-3] + "label")
                    #     f = open(anno_path, 'r')
                    #     img = cv2.imread(img_path)
                    #     imh, imw = img.shape[:2]
                    #     lines = f.readlines()
                    #     anno = []
                    #     for line in lines:
                    #         line = line.strip('\n')
                    #         line = line.split(',')
                    #         line = list(line)
                    #         anno.append(line)
                    #     for i in range(len(anno)):
                    #         if (float(anno[i][1]) > 0) and (float(anno[i][2]) > 0):
                    #             item = img_path, anno[i][0]
                    #             images.append(item)
                    #             cx = float(anno[i][1]) + (float(anno[i][3]) - float(anno[i][1])) / 2
                    #             cy = float(anno[i][2]) + (float(anno[i][4]) - float(anno[i][2])) / 2
                    #             w = float(anno[i][3]) - float(anno[i][1])
                    #             h = float(anno[i][4]) - float(anno[i][2])
                    #             templates.append([cx/imw, cy/imh, w/imw, h/imh])
                    #             box = []
                    #             for j in range(len(anno)):
                    #                 if anno[j][0] == anno[i][0]:
                    #                     cx = float(anno[i][1]) + (float(anno[i][3]) - float(anno[i][1])) / 2
                    #                     cy = float(anno[i][2]) + (float(anno[i][4]) - float(anno[i][2])) / 2
                    #                     w = float(anno[i][3]) - float(anno[i][1])
                    #                     h = float(anno[i][4]) - float(anno[i][2])
                    #                     box.append([cx/imw, cy/imh, w/imw, h/imh])

                    #             box = np.stack(box).astype(np.float32)
                    #             searches.append(box)
                    else:
                        # 影像對應的 annotation 不存在
                        assert False, f"ERROR, no annotation for image: {img_path}"
        return images, templates, searches

    def _filter_dataset(self, images, templates, searches, criteria):
        # criteria == all
        if criteria == "all":
            return images, templates, searches
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
        img_path, img_cls = self.images[idx]
        box_anno = data[idx]
        return img_path, box_anno, img_cls

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
        # center -> corner | ratio -> real
        z_box = center2corner(z_box)
        z_box = ratio2real(img, z_box)
        gt_boxes = center2corner(gt_boxes)
        gt_boxes = ratio2real(img, gt_boxes)

        # z_box: (1, [x1, y1, x2, y2])
        # gt_boxes: (num, [x1, y1, x2, y2])
        z_img, r = self.pcb_crop.get_template(x_img, z_box, self.args.bg)
        x_img, gt_boxes, z_box = self.pcb_crop.get_search(x_img, gt_boxes, z_box, r)

        # Augmentation
        if self.mode == "train":
            z_img, _ = self.pcb_aug.z_aug(z_img, z_box)
            x_img, gt_boxes, z_box = self.pcb_aug.x_aug(x_img, gt_boxes, z_box)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ########################
        # 創 directory
        ########################
        # dir = f"./image_check/origin/x{cfg.TRAIN.SEARCH_SIZE}_bg{self.args.bg}"
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
        # tmp_z_box = np.asarray(z_box).astype(None).copy()
        # tmp_z_box[:, 2] = tmp_z_box[:, 2] - tmp_z_box[:, 0]
        # tmp_z_box[:, 3] = tmp_z_box[:, 3] - tmp_z_box[:, 1]
        # z_gt_image = draw_box(gt_image, tmp_z_box, type="template")
        # search_path = os.path.join(search_dir, f"{idx}.jpg")
        # save_image(z_gt_image, search_path)

        # ipdb.set_trace()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        col_num = len(gt_boxes)
        gt_boxes = np.c_[np.zeros(col_num), gt_boxes]  # Add a all zeros column to gt_boxes
        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.int64)

        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        cls = torch.as_tensor(cls, dtype=torch.int64)

        # Z-score normalization on z_img, x_img
        # z_img = z_score_norm(z_img)
        # x_img = z_score_norm(x_img)

        z_img = z_img.transpose((2, 0, 1)).astype(np.float32)
        x_img = x_img.transpose((2, 0, 1)).astype(np.float32)
        z_img = torch.as_tensor(z_img, dtype=torch.float32)
        x_img = torch.as_tensor(x_img, dtype=torch.float32)

        # 小於 15 的卷積後會變成 0
        assert z_img.size(1) >= 15, f"ERROR, should > 15, but got {z_img.size(1)}"
        assert z_img.size(2) >= 15, f"ERROR, should > 15, but got {z_img.size(2)}"

        # cls: (size, size) 都是 0
        # box: (n, 5: [0, x1, y1, x2, y2])
        # z_box: Corner(x1, y1, x2, y2)
        return img_name, img_path, z_img, x_img, cls, gt_boxes, z_box, r
        return {
            'img_name': img_name,
            'img_path': img_path,
            'z_img': z_img,
            'z_box': z_box,
            'z_cls': template[2],
            'x_img': x_img,
            'gt_boxes': gt_boxes,
            'cls': cls,
        }
