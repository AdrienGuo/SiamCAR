# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import logging
import os
import re
import sys
import time

import cv2
import ipdb
import numpy as np
import torch
from pysot.core.config import cfg
from pysot.datasets.pcb_crop_tri_origin import PCBCrop
from pysot.utils.bbox import Center, Corner, center2corner
from pysot.utils.check_image import create_dir, draw_box, save_image
from torchvision import transforms

logger = logging.getLogger("global")


# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class PCBDataset():
    """ 定義代號
        x: template
        z: search
    """
    def __init__(self, args, mode) -> None:
        super(PCBDataset, self).__init__()

        self.root = args.test_dataset
        images, templates, searches = self._make_dataset(self.root)
        images, templates, searches = self._filter_dataset(
            images, templates, searches, args.criteria)
        assert len(images) != 0, "ERROR, dataset is empty"
        self.images = images
        self.templates = templates
        self.searches = searches

        # zf_size_min: smallest z size after res50 backbone
        zf_size_min = 4
        # PCBCrop: Crop template & search (preprocess)
        self.pcb_crop = PCBCrop(zf_size_min)


    def _make_dataset(self, dir_path):
        """
        Returns:
            images (list):
            templates (list):
            search (list):
        """

        images = list()
        templates = list()
        searches = list()

        for root, _, files in os.walk(dir_path):
            img_dir = root
            z_regex = re.compile(r"^(Template)")  # 開頭要有 Template
            x_regex = re.compile(r"^(?!Template)")  # 開頭不能有 Template
            zs = [file for file in files if z_regex.match(file)]
            xs = [file for file in files if x_regex.match(file)]
            for z in zs:
                z_path = os.path.join(root, z)
                for x in xs:
                    x_path = os.path.join(root, x)
                    images.append(img_dir)
                    templates.append(z_path)
                    searches.append(x_path)

        return images, templates, searches

    def _filter_dataset(self, images, templates, searches, criteria):
        # criteria == all
        if criteria == "all":
            return images, templates, searches
        # for
        inds_match = list()
        for idx, (z, x) in enumerate(zip(templates, searches)):
            # read image
            z_img = cv2.imread(z)
            x_img = cv2.imread(x)
            # get w & h
            z_h, z_w = z_img.shape[:2]
            x_h, x_w = x_img.shape[:2]
            # calculate r by resize to 255
            long_side = max(x_w, x_h)
            r = cfg.TRAIN.SEARCH_SIZE / long_side
            # calculate templates new w, h
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        logger.debug("__getitem__")

        img_path = self.images[idx]
        template = self.templates[idx]
        search = self.searches[idx]

        ##########################################
        # Step 1.
        # Get template and search images (raw data)
        ##########################################
        img_name = img_path.split('/')[-1]
        # print(f"Load image from: {img_path}")

        z_img = cv2.imread(template)
        x_img = cv2.imread(search)
        assert z_img is not None, f"Error image: {template}"
        assert x_img is not None, f"Error image: {search}"

        # Create virtual boxes
        z_box = np.array([[0, 0, z_img.shape[1], z_img.shape[0]]])
        gt_boxes = np.array([[0, 0, 0, 0]])  # useless

        ##########################################
        # Step 2.
        # Crop the template and search images
        ##########################################
        # 確保 z_img 的最小邊不會小於 threshold，
        # 若是 z_img 有做縮放 -> x_img 一樣要做縮放
        z_img, z_box, r = self.pcb_crop.get_template(z_img, z_box)
        x_img = self.pcb_crop.get_search(x_img, gt_boxes, r)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ########################
        # 創 directory
        ########################
        # dir = f"./image_check/test/x{cfg.TRACK.INSTANCE_SIZE}"
        # img_name = img_path.split('/')[-1]

        # # 以 “圖片名稱” 當作 sub_dir 的名稱
        # sub_dir = os.path.join(dir, img_name)
        # create_dir(sub_dir)

        # # 創 sub_dir/search，裡面存 search image
        # search_dir = os.path.join(sub_dir, "search")
        # create_dir(search_dir)
        # # 創 sub_dir/template，裡面存 template image
        # template_dir = os.path.join(sub_dir, "template")
        # create_dir(template_dir)

        # #########################
        # # 存圖片
        # #########################
        # template_path = os.path.join(template_dir, f"{idx}.jpg")
        # save_image(z_img, template_path)
        # search_path = os.path.join(search_dir, f"{idx}.jpg")
        # save_image(x_img, search_path)

        # ipdb.set_trace()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        ##########################################
        # Step 3.
        # (127, 127, 3) -> (3, 127, 127) for CNN using
        ##########################################
        z_img = z_img.transpose((2, 0, 1)).astype(np.float32)
        x_img = x_img.transpose((2, 0, 1)).astype(np.float32)
        z_img = torch.as_tensor(z_img, dtype=torch.float32)
        x_img = torch.as_tensor(x_img, dtype=torch.float32)

        box = []
        for i in range(len(gt_boxes)):
            box.append([0, gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2], gt_boxes[i][3]])
        box = np.stack(box).astype(np.float32)
        box = torch.as_tensor(box, dtype=torch.int64)

        # cls: 一個沒用的東西 (only used in training)
        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        cls = torch.as_tensor(cls, dtype=torch.int64)

        return img_name, img_path, z_img, x_img, cls, box, z_box, r

        return {
            'img_path': search,
            'img_name': img_name,
            'z_box': z_box,
            'z_img': z_img,
            'x_img': x_img,

            'gt_boxes': gt_boxes,    # useless
            'scale': r,    # useless
            'spatium': 0,    # useless
            # === 下面是檢查 anchor 的時候，要存檔用的 ===
            'idx': idx,
        }
