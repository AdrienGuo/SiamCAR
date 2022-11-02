# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.loss_car_multi import make_siamcar_loss_evaluator
from pysot.models.neck import get_neck
from pysot.utils.xcorr import xcorr_depthwise

from ..utils.location_grid import compute_locations


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)

        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], self.zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)

        return {
            'cls': cls,
            'loc': loc,
            'cen': cen
        }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        z_img = data['z_img'].cuda()
        x_img = data['x_img'].cuda()
        gt_cls = data['gt_cls'].cuda()
        gt_boxes = data['gt_boxes'].cuda()  # (?, [x1, y1, x2, y2])

        # Backbone (ResNet50)
        zf = self.backbone(z_img)
        xf = self.backbone(x_img)

        # Neck
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        # Depthwise Correlation
        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        # features: (b, c=256, h, w)
        features = self.down(features)

        # Classificaitn, Regression
        cls, loc, cen = self.car_head(features)

        # 做出 meshgrid
        # locations: (size * size, 2)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        # cls: (b, 2, h, w) -> (b, 1, h, w, 2)
        cls = self.log_softmax(cls)

        # 算 loss
        cen_loss, cls_loss, loc_loss = self.loss_evaluator(
            locations,
            cen,
            cls,
            loc,
            gt_cls,
            gt_boxes
        )

        # get loss
        outputs = {}
        outputs['cen'] = cen_loss
        outputs['cls'] = cls_loss
        outputs['loc'] = loc_loss
        outputs['total'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss

        return outputs
