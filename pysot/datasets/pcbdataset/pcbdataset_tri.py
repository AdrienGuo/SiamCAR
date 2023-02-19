# 這是專門給 PatternMatch_test 資料集的。
# 只會用在 test，
# 因為 PatternMatch_test 沒有標籤


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
from pysot.datasets.augmentation.pcb_aug import PCBAugmentation
from pysot.datasets.pcb_crop import get_pcb_crop
from utils.file_organizer import create_dir, save_img

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class PCBDatasetTri(object):
    """ 代號
        z: template
        x: search
    """

    def __init__(
        self,
        args,
        mode: str,
        augmentation: PCBAugmentation
    ) -> None:
        super(PCBDatasetTri, self).__init__()

        self.args = args
        self.method = args['method']
        self.criteria = args['criteria']
        images, templates, searches = self._make_dataset(args['data_path'])
        images, templates, searches = self._filter_dataset(
            images, templates, searches, args['criteria'])
        assert len(images) != 0, "ERROR, dataset is empty"
        self.images = images
        self.templates = templates
        self.searches = searches

        pcb_crop = get_pcb_crop(args['method'])
        if args['method'] == "tri_origin":
            # zf_size_min: smallest z size after res50 backbone
            zf_size_min = 4
            self.pcb_crop = pcb_crop(zf_size_min)
        else:
            assert False, "Method is wrong"

        # Augmentation
        # TODO: Refactor
        self.z_aug = augmentation['template']
        self.x_aug = augmentation['search']

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
            r = 255 / long_side
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

    def _save_img(self, img_path, z_img, x_img, idx):
        # 創 directory
        dir = f"./image_check/{self.method}"
        img_name = img_path.split('/')[-1]
        sub_dir = os.path.join(dir, img_name)
        search_dir = os.path.join(sub_dir, "search")
        template_dir = os.path.join(sub_dir, "template")
        create_dir(sub_dir)
        create_dir(search_dir)
        create_dir(template_dir)

        # 存圖片
        template_path = os.path.join(template_dir, f"{idx}.jpg")
        search_path = os.path.join(search_dir, f"{idx}.jpg")
        save_img(z_img, template_path)
        save_img(x_img, search_path)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        template = self.templates[idx]
        search = self.searches[idx]

        ##########################################
        # Step 1.
        # Get template and search images (raw data)
        ##########################################
        img_name = img_path.split('/')[-1]

        z_img = cv2.imread(template)
        x_img = cv2.imread(search)
        assert z_img is not None, f"Error image: {template}"
        assert x_img is not None, f"Error image: {search}"

        # Create virtual boxes
        z_box = np.array([[0, 0, z_img.shape[1], z_img.shape[0]]])
        gt_boxes = np.array([[0, 0, 0, 0, 0]])  # useless

        ##########################################
        # Step 2.
        # Crop the template and search images.
        ##########################################
        z_img, x_img, z_box = self.pcb_crop.get_data(
            z_img, x_img, z_box, gt_boxes
        )

        # CLAHE 3.0
        # z_img, z_box = self.z_aug(z_img, z_box)
        # x_img, gt_boxes = self.x_aug(x_img, gt_boxes)

        # self._save_img(img_path, z_img, x_img, idx)
        # ipdb.set_trace()

        ##########################################
        # Step 3.
        # (127, 127, 3) -> (3, 127, 127) for CNN using
        ##########################################
        z_img = z_img.transpose((2, 0, 1)).astype(np.float32)
        x_img = x_img.transpose((2, 0, 1)).astype(np.float32)
        z_img = torch.as_tensor(z_img, dtype=torch.float32)
        x_img = torch.as_tensor(x_img, dtype=torch.float32)

        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.int64)

        # useless
        gt_cls = np.zeros(
            (cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        gt_cls = torch.as_tensor(gt_cls, dtype=torch.int64)

        return img_name, img_path, z_img, x_img, z_box, gt_boxes, gt_cls
        return {
            'img_name': img_name,
            'z_img': z_img,
            'x_img': x_img,
            'z_box': z_box,
            'gt_boxes': gt_boxes
        }
