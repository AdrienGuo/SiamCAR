# 這是用在除了 PatternMatch_test 以外的資料集。
# trian, test 都會用


import logging
import os
import sys
from collections import namedtuple

import cv2
import ipdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from pysot.core.config import cfg
from pysot.datasets.augmentation.augmentation_siamcar import Augmentation
from pysot.datasets.augmentation.pcb_aug import PCBAugmentation
from pysot.datasets.crops import get_pcb_crop
from pysot.utils.bbox import Center, center2corner, ratio2real
from pysot.utils.check_image import create_dir, draw_box, save_image
from utils.file_organizer import save_img
from utils.painter import draw_boxes

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class PCBDataset(Dataset):
    """
    代號
        z: template
        x: search

    Args:
        mode: train / test
    """

    def __init__(
        self,
        args: dict,
        mode: str,
        augmentation: PCBAugmentation,
    ):
        self.args = args
        self.method = args['method']
        self.criteria = args['criteria']
        self.mode = mode

        images, templates, searches = self._make_dataset(
            args['data_path'], args['target'])
        images, templates, searches = self._filter_dataset(
            images, templates, searches, args['criteria'])
        self.images = images
        self.templates = templates
        self.searches = searches

        # Several crop methods
        pcb_crop = get_pcb_crop(args['method'])
        if args['method'] == "official":
            self.pcb_crop = pcb_crop(
                template_size=127,
                search_size=255,
                shift=64
            )
        elif args['method'] == "origin":
            # zf_size_min: smallest z size after res50 backbone
            zf_size_min = 4
            self.pcb_crop = pcb_crop(
                zf_size_min,
                background=args['bg']
            )
        elif args['method'] == "official_origin":
            self.pcb_crop = pcb_crop(
                template_size=127,
                background=args['bg']
            )
        elif args['method'] == "siamcar":
            self.pcb_crop = pcb_crop(
                template_size=127,
                search_size=cfg.TRAIN.SEARCH_SIZE,
                template_shift=cfg.TRAIN.DATASET.TEMPLATE.SHIFT,
                search_shift=cfg.TRAIN.DATASET.SEARCH.SHIFT,
            )
        else:
            assert False, "method is wrong"

        # Augmentations
        # self.z_aug = augmentation['template']
        # self.x_aug = augmentation['search']

        # data augmentation
        self.template_aug = Augmentation(
            cfg.TRAIN.DATASET.TEMPLATE.SHIFT,
            cfg.TRAIN.DATASET.TEMPLATE.SCALE,
            cfg.TRAIN.DATASET.TEMPLATE.BLUR,
            cfg.TRAIN.DATASET.TEMPLATE.FLIP,
            cfg.TRAIN.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            cfg.TRAIN.DATASET.SEARCH.SHIFT,
            cfg.TRAIN.DATASET.SEARCH.SCALE,
            cfg.TRAIN.DATASET.SEARCH.BLUR,
            cfg.TRAIN.DATASET.SEARCH.FLIP,
            cfg.TRAIN.DATASET.SEARCH.COLOR
        )

    def _make_dataset(self, directory: str, target: str):
        images = []
        templates = []
        searches = []

        # 標記錯誤的影像 & 會造成OOM的影像 & 太大張的影像
        imgs_exclude = ['6_cae_cae_20200803_10.bmp',
                        '20200629_ok (42).jpg', '16_bga_BGA_20220106_uniform_1.bmp']
        mid_imgs_exclude = ['17_ic_ic_20200810_solder_40.bmp',
                            '17_ic_Sot23_20200820_solder_81.bmp', '5_sod_sod (7).jpg']
        small_imgs_exclude = ['']
        if self.criteria == "all":
            imgs_exclude = imgs_exclude + mid_imgs_exclude + small_imgs_exclude
        elif self.criteria == "mid":
            imgs_exclude += mid_imgs_exclude
        elif self.criteria == "small":
            imgs_exclude += small_imgs_exclude

        for root, _, files in sorted(os.walk(directory, followlinks=True)):
            for file in sorted(files):  # 排序
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
                            cls.append(str(line[0]))
                            anno.append(list(map(float, line[1:5])))

                        for i in range(len(cls)):
                            if cls[i] == "36":
                                # "36" 是極性物件，先不考慮。
                                continue
                            item = img_path, cls[i]
                            images.append(item)
                            templates.append(
                                [anno[i][0], anno[i][1], anno[i][2], anno[i][3]])
                            box = list()
                            if target == "one":
                                # 單目標偵測
                                box.append([anno[i][0], anno[i][1],
                                            anno[i][2], anno[i][3]])
                            elif target == "multi":
                                # 多目標偵測
                                for j in range(len(cls)):
                                    if cls[j] == cls[i]:
                                        box.append(
                                            [anno[j][0], anno[j][1], anno[j][2], anno[j][3]])
                            box = np.stack(box).astype(np.float32)
                            searches.append(box)
                    else:
                        # 影像對應的 annotation 不存在
                        assert False, f"ERROR, no annotation for image: {img_path}"
        return images, templates, searches

    def _filter_dataset(self, images, templates, searches, criteria):
        if criteria == "all":
            return images, templates, searches
        inds_match = list()
        for idx, image in enumerate(images):
            img = cv2.imread(image[0])
            # get w & h
            img_h, img_w = img.shape[:2]
            z_w = templates[idx][2] * img_w
            z_h = templates[idx][3] * img_h
            # calculate r by resize to 255
            long_side = max(img_w, img_h)
            r = 255 / long_side
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
        # TODO: Refactor
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

    def _save_img(self, img_name, img, z_img, x_img, z_box, gt_boxes, idx):
        # 創 directory
        dir = os.path.join("./image_check", f"{self.method}")
        sub_dir = os.path.join(dir, img_name)
        origin_dir = os.path.join(sub_dir, "origin")
        template_dir = os.path.join(sub_dir, "template")
        search_dir = os.path.join(sub_dir, "search")
        create_dir(sub_dir)
        create_dir(origin_dir)
        create_dir(template_dir)
        create_dir(search_dir)

        # 存圖片
        origin_path = os.path.join(origin_dir, "origin.jpg")
        template_path = os.path.join(template_dir, f"{idx}.jpg")
        save_img(img, origin_path)
        save_img(z_img, template_path)
        # Draw gt_boxes on search image
        gt_image = draw_boxes(x_img, gt_boxes, type="gt")
        z_gt_image = draw_boxes(gt_image, z_box, type="template")
        search_path = os.path.join(search_dir, f"{idx}.jpg")
        save_img(z_gt_image, search_path)

    def __getitem__(self, idx):
        # 加入 neg 的原因要去看 [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
        neg = None
        # Get dataset.
        if not neg:
            template, search = self._get_positive_pair(idx)
        else:
            template, search = self._get_negative_pair(idx)

        # Get image.
        assert template[0] == search[0], f"Should be the same if neg is False"
        img_path = template[0]
        img_name = img_path.split('/')[-1].rsplit('.', 1)[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img is not None, f"Error image: {template[0]}"

        # Crop the template & search image.
        z_box = template[1].copy()  # z_box: [cx, cy, w, h]
        gt_boxes = search[1].copy()  # gt_boxes: (N, [cx, cy, w, y]) #ratio

        z_box = np.asarray(z_box)
        gt_boxes = np.asarray(gt_boxes)
        # ([cx, cy, w, h]) -> (1, [cx, cy, w, h]) 轉成二維的
        z_box = z_box[np.newaxis, :]
        # center -> corner | ratio -> real
        z_box = center2corner(z_box)
        z_box = ratio2real(img, z_box)
        gt_boxes = center2corner(gt_boxes)
        gt_boxes = ratio2real(img, gt_boxes)

        # z_box: (1, [x1, y1, x2, y2])
        # gt_boxes: (N, [x1, y1, x2, y2])
        z_img, x_img, z_box, gt_boxes = self.pcb_crop.get_data(
            img, z_box, gt_boxes)

        # Augmentations
        # if self.mode == "train":
        #     z_img, _ = self.z_aug(z_img, z_box)
        #     x_img, gt_boxes = self.x_aug(x_img, gt_boxes)

        # augmentation
        if self.mode == "train" or self.mode == "test":
            z_box = z_box.squeeze()
            gt_boxes = gt_boxes.squeeze()

            gray = cfg.TRAIN.DATASET.GRAY and cfg.TRAIN.DATASET.GRAY > np.random.random()
            template, _ = self.template_aug(z_img,
                                            z_box,
                                            cfg.TRAIN.EXEMPLAR_SIZE,
                                            gray=gray)

            search, bbox = self.search_aug(x_img,
                                           gt_boxes,
                                           cfg.TRAIN.SEARCH_SIZE,
                                           gray=gray)
            z_img = template
            x_img = search
            gt_boxes = bbox[np.newaxis, :]
            z_box = gt_boxes

        # Save images to ./image_check
        # self._save_img(img_name, img, z_img, x_img, z_box, gt_boxes, idx)
        # ipdb.set_trace()

        # Add one all zeros column to gt_boxes
        col_num = len(gt_boxes)
        gt_boxes = np.c_[np.zeros(col_num), gt_boxes]
        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.int64)

        z_img = z_img.transpose((2, 0, 1)).astype(np.float32)
        x_img = x_img.transpose((2, 0, 1)).astype(np.float32)
        z_img = torch.as_tensor(z_img, dtype=torch.float32)
        x_img = torch.as_tensor(x_img, dtype=torch.float32)

        gt_cls = np.zeros(
            (cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        gt_cls = torch.as_tensor(gt_cls, dtype=torch.int64)

        # gt_boxes: (n, 4=[x1, y1, x2, y2])
        # z_box: (1, 4=[x1, y1, x2, y2])
        return img_name, img_path, z_img, x_img, z_box, gt_boxes, gt_cls
        return {
            'img_name': img_name,
            'img_path': img_path,
            'z_img': z_img,
            'x_img': x_img,
            'z_box': z_box,
            'gt_boxes': gt_boxes,
            'gt_cls': gt_cls
        }
