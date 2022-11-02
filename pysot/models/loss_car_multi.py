"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

INF = 100000000


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    label = torch.tensor(label, dtype=torch.long).cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: (pos_inds)
            target: (pos_inds)
            weight: centerness，越中心的權重越大
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_aera = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) \
                      + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) \
                      + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:  # weight.sum() 不會小於 0 吧？
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class GIOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        
        iou = (area_intersect + 1.0) / (area_union + 1.0)
        
        # enclosure
        gw_intersect = torch.max(pred_left, target_left) + \
                      torch.max(pred_right, target_right)
        gh_intersect = torch.max(pred_bottom, target_bottom) + \
                      torch.max(pred_top, target_top)
        g_area = gw_intersect*gh_intersect+ 1e-7
        giou = iou - (g_area - area_union)/(g_area)
        losses = 1. - giou
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamCARLossComputation(object):
    """ This class computes the SiamCAR losses.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.box_reg_loss_func = IOULoss()
        # self.box_reg_loss_func = GIOULoss()

        # BCEWithLogitsLoss 就是把 Sigmoid 和 BCELoss 合成一步
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, gt_cls, gt_bbox):
        gt_cls, reg_targets = self.compute_targets_for_locations(points, gt_cls, gt_bbox)
        return gt_cls, reg_targets

    def compute_targets_for_locations(
        self,
        locations,
        gt_cls,
        gt_bbox  # (?, [idx, x1, y1, x2, y2])
    ):
        """
        Returns:
            gt_cls_all (list[array])
            reg_targets (list[array])
        """
        xs, ys = locations[:, 0], locations[:, 1]  # torch.Size([3025])
        
        gt_cls_all = []
        reg_targets = []
        bboxes = gt_bbox
        # TODO: 其實這行沒用啊
        gt_cls = gt_cls.view(-1, self.cfg.TRAIN.OUTPUT_SIZE**2)  # gt_cls: (b, 625)

        for i in range(gt_cls.shape[0]):  # 每次處裡一張，有 batch 張
            # curr_bbox = []  # 一張影像的 bounding box

            # 選出和這張對應到的那些 gt_bbox
            curr_bbox = bboxes[(bboxes[:, 0] == i).nonzero().squeeze(dim=1)]
            # for j in range(len(bboxes)):
            #     if bboxes[j][0] == i:
            #         # 代表是對應到的 boxes
            #         curr_bbox.append(bboxes[j])
            #     else:
            #         break
            # bboxes = bboxes[j:, :]
            # curr_bbox = torch.stack(curr_bbox)

            l = xs[:, None] - curr_bbox[:, 1][None].float()
            t = ys[:, None] - curr_bbox[:, 2][None].float()
            r = curr_bbox[:, 3][None].float() - xs[:, None]
            b = curr_bbox[:, 4][None].float() - ys[:, None]
            # N 為所有採樣點的數，M 為 gt_boxes 數量
            # reg_targets_per_im: (N, M, 4)
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            # is_in_boxes: (N, M)
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            # is_in_boxes: (M, N)
            is_in_boxes = is_in_boxes.permute(1, 0).contiguous()
            # gt_cls_per_im: (M, size * size)
            gt_cls_per_im = torch.zeros(is_in_boxes.shape[0], self.cfg.TRAIN.OUTPUT_SIZE**2)
            for n in range(is_in_boxes.shape[0]):
                pos = np.where(is_in_boxes[n].cpu() == 1)
                gt_cls_per_im[n][pos] = 1
            gt_cls_all.append(gt_cls_per_im)

            # reg_targets_per_im: (M, N, 4)
            reg_targets_per_im = reg_targets_per_im.permute(1, 0, 2).contiguous()
            reg_targets.append(reg_targets_per_im)

        return gt_cls_all, reg_targets

    def compute_centerness_targets(self, reg_targets):
        # left_right: (size * size, 2)
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        # centerness = 1 代表最中心，越遠離中心數值越接近 0
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) \
                      * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(
        self,
        locations: torch.Tensor,  # (size * size, 2), meshgrid
        pred_cen,  # (b, 1, size, size)
        pred_cls,  # (b, 1, size, size, 2)
        pred_boxes,  # (b, 4, size, size)
        gt_cls,  # (b, size, size)
        gt_boxes  # (?, [idx, x1, y1, x2, y2])
    ):
        """
        Args:
            gt_cls: 其實都是 0，到 compute_targets_for_locations 才會算

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        # TODO: 這裡的 N 算錯的吧
        N = pred_cls.size(0)  # 影像張數，其實就是 batch 大小
        # gt_cls (list[array]), gt_boxes (list[array])
        gt_cls, gt_boxes = self.prepare_targets(locations, gt_cls, gt_boxes)
        pred_boxes_flatten = []
        pred_cen_flatten = []
        gt_cls_flatten = []
        gt_boxes_flatten = []

        cen_loss_all = []
        cls_loss_all = []
        reg_loss_all = []

        for n in range(N):  # 有 N 張影像，batch 大小
            # 一張影像 (1 batch)
            reg_loss = 0
            cen_loss = 0
            cls_loss = 0
            K = len(gt_cls[n])  # 這張影像裡面有 K 個 gt_boxes
            # TODO: 亭儀是一個 gt 一個 gt 去算然後加起來，
            # 但我的 SiamRPN++ 是全部的 gt 都一起算，
            # 這兩種做法會等價嗎？
            # (全部一起算的話要處理每一個 grid point 要去對哪一個 gt 的問題)
            for k in range(K):
                # 這張影像對應到的其中一個 gt_box

                # pred 的邊界框回歸
                # pred_boxes_flatten: (4, size, size) -> (size, size, 4) -> (size * size, 4)
                pred_boxes_flatten = (pred_boxes[n].permute(1, 2, 0).contiguous().view(-1, 4))
                # gt 分類
                # gt_cls_flatten: (size * size)
                gt_cls_flatten = (gt_cls[n][k].view(-1))
                # gt 邊界框回歸
                # gt_boxes_flatten: (size * size, 4)
                gt_boxes_flatten = (gt_boxes[n][k].view(-1, 4))
                # pred 的中心度
                # pred_cen_flatten: (size * size)
                pred_cen_flatten = (pred_cen[n].view(-1))

                # pos_inds: (size * size)
                # 這裡的 torch.nonzero() 是取 "不是 False" 的 index
                pos_inds = torch.nonzero(gt_cls_flatten > 0).squeeze(1)
                # TODO
                # assert pos_inds.numel() > 0, "ERROR, No positive cls in this image."

                # 有在 gt 框框裡面的才要去算 loss
                pred_boxes_flatten = pred_boxes_flatten[pos_inds]
                gt_boxes_flatten = gt_boxes_flatten[pos_inds]
                pred_cen_flatten = pred_cen_flatten[pos_inds]

                ##################################
                # 算 loss
                ##################################
                # TODO: SiamCAR 的 cls 好像不是 1 就是 0，不像 SiamRPN++ 會有 -1
                cls_loss += select_cross_entropy_loss(pred_cls[n], gt_cls_flatten)

                if pos_inds.numel() > 0:
                    # centerness_targets: 距離中心點的分數 0~1，1 代表最接近中心點
                    centerness_targets = self.compute_centerness_targets(gt_boxes_flatten)
                    reg_loss += self.box_reg_loss_func(
                        pred_boxes_flatten,
                        gt_boxes_flatten,
                        centerness_targets
                    )
                    cen_loss += self.centerness_loss_func(
                        pred_cen_flatten,
                        centerness_targets
                    )
                else:
                    # 根本不該進來這裡面吧 (但還是會進來...)
                    # assert False, "ERROR, You should not come in."
                    reg_loss += pred_boxes_flatten.sum()
                    cen_loss += pred_cen_flatten.sum()

            cen_loss = cen_loss / K
            cls_loss = cls_loss / K
            reg_loss = reg_loss / K

            cen_loss_all.append(cen_loss)
            cls_loss_all.append(cls_loss)
            reg_loss_all.append(reg_loss)

        cen_loss = sum(cen_loss_all) / N
        cls_loss = sum(cls_loss_all) / N
        reg_loss = sum(reg_loss_all) / N
        return cen_loss, cls_loss, reg_loss


def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
