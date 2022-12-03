"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pysot.core.config import cfg

INF = 100000000


# 阿這個怎麼根本沒用到？？
def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class CLSLoss(object):
    def __init__(self, method) -> None:
        """ 會根據 method 使用相對應的 loss function
            目前有 bce / focal 兩種
        """
        self.method = method

    def get_cls_loss(self, pred, label, select):
        if len(select.size()) == 0 or \
                select.size() == torch.Size([0]):
            # 完全沒有正樣本或負樣本
            return 0
        pred = torch.index_select(pred, dim=0, index=select)
        label = torch.index_select(label, dim=0, index=select)
        return F.nll_loss(pred, label)

    # 看了這麼久才發現，
    # 原來他這裡分類是用 BCE 而不是用 focal loss？
    # 可是他在 car_head 那裡卻又有提到 focal loss (沒有，是我看錯)
    def select_cross_entropy_loss(self, pred, label):
        pred = pred.view(-1, 2)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()
        # label = label.to(dtype=torch.long).cuda()
        label = torch.tensor(label, dtype=torch.long).cuda()
        # 他這樣 pos, neg 分開算，就已經有 "權重" 的概念了，
        # 這樣搞不好比 focal loss 好
        loss_pos = self.get_cls_loss(pred, label, pos)
        loss_neg = self.get_cls_loss(pred, label, neg)
        loss = loss_pos * 0.5 + loss_neg * 0.5
        return loss, loss_pos * 0.5, loss_neg * 0.5

    def focal_loss(self, logpt, label) -> torch.Tensor:
        """
        Ref: https://leimao.github.io/blog/Focal-Loss-Explained/
        Ref: https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py

        Args:
            logpt (1, H, W, 2): 已經有經過 log_softmax
            label (H*W)
        """
        logpt = logpt.view(-1, 2)
        label = label.to(dtype=torch.long).cuda()
        label = label.view(-1, 1)
        logpt = logpt.gather(dim=1, index=label)  # 抓出屬於 label 的那個 logpt 機率值
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)  # 把 log 的機率值轉回不是 log 的狀態

        alpha = cfg.TRAIN.LOSS_ALPHA
        gamma = cfg.TRAIN.LOSS_GAMMA
        if isinstance(alpha, (float, int)): alpha = torch.Tensor([alpha, 1-alpha])

        # TODO: 把 pos, neg 的 loss 分開算
        if alpha is not None:
            if not isinstance(type(alpha), type(logpt)):
                alpha = alpha.type_as(logpt.data)
            at = alpha.gather(0, label.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**gamma * logpt
        return loss.mean(), loss.mean(), loss.mean()

    def calculate(self, pred, label):
        # TODO: 其實這樣處理也不好，
        # 因為 bce 和 focal 是兩種完全不相干的算法，不應該寫在同一個 class 裡面。
        loss = tuple()
        if self.method == "bce":
            loss = self.select_cross_entropy_loss(pred, label)
        elif self.method == "focal":
            loss = self.focal_loss(pred, label)
        return loss


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: (pos_inds)
            target: (pos_inds)
            weight: centerness，越中心的權重越大 (0 ~ 1)
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) \
                      + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) \
                      + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        # TODO: Check no underflow or overflow.
        # Cause I use mixed precision in forward process.
        # pred_area, target_area, area_intersect, area_union

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

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) \
            + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) \
            + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

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

        self.score_h: 要預測的高度
        self.score_w: 要預測的寬度
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.box_reg_loss_func = IOULoss()
        # self.box_reg_loss_func = GIOULoss()

        # BCEWithLogitsLoss 就是把 Sigmoid 和 BCELoss 合成一步
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

        # TODO: 把 cls_loss 包成一個 object
        self.cls_loss_func = CLSLoss(cfg.TRAIN.CLS_LOSS_METHOD)

    def prepare_targets(self, points, gt_boxes, B):
        gt_cls, gt_regs = self.compute_targets_for_locations(
            points, gt_boxes, B
        )
        return gt_cls, gt_regs

    def compute_targets_for_locations(
        self,
        locations,
        gt_boxes,  # (?, [idx, x1, y1, x2, y2])
        B  # batch_size
    ):
        """
        Returns:
            gt_cls_all (list[array])
            gt_regs_all (list[array])
        """
        xs, ys = locations[:, 0], locations[:, 1]

        gt_cls_all = []
        gt_regs_all = []
        bboxes = gt_boxes

        for i in range(B):  # 每次處裡一張，有 batch 張
            # 選出和這張對應到的那些 gt_boxes
            match_boxes = bboxes[(bboxes[:, 0] == i).nonzero(as_tuple=False).squeeze(dim=1)]
            # for j in range(len(bboxes)):
            #     if bboxes[j][0] == i:
            #         # 代表是對應到的 boxes
            #         match_boxes.append(bboxes[j])
            #     else:
            #         break
            # bboxes = bboxes[j:, :]
            # match_boxes = torch.stack(match_boxes)

            l = xs[:, None] - match_boxes[:, 1][None].float()
            t = ys[:, None] - match_boxes[:, 2][None].float()
            r = match_boxes[:, 3][None].float() - xs[:, None]
            b = match_boxes[:, 4][None].float() - ys[:, None]
            # N 為所有採樣點的數，G 為 gt_boxes 數量
            # reg_targets_per_im: (N, G, 4)
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            # 整個 match_boxes 的區域都是 in_boxes
            # is_in_boxes: (N, G)
            # is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            # 不要整個 match_boxes 的區域都是 in_boxes
            # 只留下最中心的 0.3 x 0.3 的區塊
            # 可是要小心會不會變成完全沒有 in_boxes，因為我有些物件很小
            # RuntimeError: Integer division of tensors using div or / is no longer supported,
            # and in a future release div will perform true division as in Python 3.
            # Use true_divide or floor_divide (// in Python) instead
            s1 = reg_targets_per_im[:, :, 0] > 0.6 * ((match_boxes[:, 3] - match_boxes[:, 1]) / 2.).float()
            s2 = reg_targets_per_im[:, :, 2] > 0.6 * ((match_boxes[:, 3] - match_boxes[:, 1]) / 2.).float()
            s3 = reg_targets_per_im[:, :, 1] > 0.6 * ((match_boxes[:, 4] - match_boxes[:, 2]) / 2.).float()
            s4 = reg_targets_per_im[:, :, 3] > 0.6 * ((match_boxes[:, 4] - match_boxes[:, 2]) / 2.).float()
            # is_in_boxes: (N, G)
            is_in_boxes = s1 * s2 * s3 * s4

            # is_in_boxes: (G, N)
            is_in_boxes = is_in_boxes.permute(1, 0).contiguous()
            # gt_cls_per_im: (G, size * size)
            gt_cls_per_im = torch.zeros(is_in_boxes.shape[0], self.score_h * self.score_w)
            for g in range(is_in_boxes.shape[0]):
                pos = np.where(is_in_boxes[g].cpu() == 1)
                gt_cls_per_im[g][pos] = 1
            gt_cls_all.append(gt_cls_per_im)

            # reg_targets_per_im: (G, N, 4=[l,t,r,b])
            reg_targets_per_im = reg_targets_per_im.permute(1, 0, 2).contiguous()
            gt_regs_all.append(reg_targets_per_im)

        return gt_cls_all, gt_regs_all

    def compute_centerness_targets(self, reg_targets):
        # reg_targets: (pos_inds, 4=[l,t,r,b])
        # left_right: (size * size, 2=[l,r])
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        # centerness = 1 代表最中心，越遠離中心數值越接近 0
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) \
                      * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(
        self,
        locations: torch.Tensor,  # (size * size, 2), meshgrid
        pred_cen,  # (B, 1, score_h, score_w)
        pred_cls,  # (B, 1, score_h, score_w, 2)
        pred_boxes,  # (B, 4=[l,t,r,b], score_h, score_w)
        gt_cls,  # (B, size, size)
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
        B = pred_cls.size(0)  # 影像張數，其實就是 batch 大小

        # 取得要預測特徵圖的 寬 & 高
        self.score_w = pred_boxes.size(3)
        self.score_h = pred_boxes.size(2)

        # gt_cls: list[ (G, N) ], gt_boxes: list[ (G, N, 4=[l,t,r,b]) ]
        # length of list = B
        gt_cls, gt_boxes = self.prepare_targets(locations, gt_boxes, B)

        # ipdb.set_trace()

        pred_boxes_flatten = []
        pred_cen_flatten = []
        gt_cls_flatten = []
        gt_boxes_flatten = []

        cen_loss_all = []
        cls_loss_all = []
        reg_loss_all = []

        cls_pos_losses = list()
        cls_neg_losses = list()

        for b in range(B):  # 有 B 張影像，batch 大小
            # 一張影像 (1 batch)
            reg_loss = 0
            cen_loss = 0
            cls_loss = 0

            cls_pos_loss = 0
            cls_neg_loss = 0

            G = len(gt_cls[b])  # 這張影像裡面有 G 個 gt_boxes
            # TODO: 亭儀是一個 gt 一個 gt 去算然後加起來，
            # 但我的 SiamRPN++ 是全部的 gt 都一起算，
            # 這兩種做法會等價嗎？
            # (全部一起算的話要處理每一個 grid point 要去對哪一個 gt 的問題)
            for g in range(G):
                # 這張影像對應到的其中一個 gt_box

                # pred 的邊界框回歸
                # pred_boxes_flatten: (4, size, size) -> (size, size, 4) -> (size * size, 4)
                pred_boxes_flatten = (pred_boxes[b].permute(1, 2, 0).contiguous().view(-1, 4))
                # gt 分類
                # gt_cls_flatten: (size * size)
                gt_cls_flatten = (gt_cls[b][g].view(-1))
                # gt 邊界框回歸
                # gt_boxes_flatten: (size * size, 4)
                gt_boxes_flatten = (gt_boxes[b][g].view(-1, 4))
                # pred 的中心度
                # pred_cen_flatten: (size * size)
                pred_cen_flatten = (pred_cen[b].view(-1))

                # pos_inds: (size * size)
                # 這裡的 torch.nonzero() 是取 "不是 False" 的 index
                pos_inds = torch.nonzero(gt_cls_flatten > 0, as_tuple=False).squeeze(1)
                # TODO:
                # assert pos_inds.numel() > 0, "ERROR, No positive cls in this image."

                # 有在 gt 框框裡面的才要去算 loss
                pred_boxes_flatten = pred_boxes_flatten[pos_inds]
                gt_boxes_flatten = gt_boxes_flatten[pos_inds]
                pred_cen_flatten = pred_cen_flatten[pos_inds]

                ##################################
                # 算 loss
                ##################################
                # TODO: SiamCAR 的 cls 好像不是 1 就是 0，不像 SiamRPN++ 會有 -1
                # 使用 Weighted BCE Loss
                # cls_loss_tmp, cls_pos_loss_tmp, cls_neg_loss_tmp \
                #     = select_cross_entropy_loss(pred_cls[b], gt_cls_flatten)

                # 使用 Focal Loss
                # cls_loss += focal_loss(pred_cls[b], gt_cls_flatten)

                cls_loss_tmp, cls_pos_loss_tmp, cls_neg_loss_tmp = \
                    self.cls_loss_func.calculate(pred_cls[b], gt_cls_flatten)
                cls_loss += cls_loss_tmp
                cls_pos_loss += cls_pos_loss_tmp
                cls_neg_loss += cls_neg_loss_tmp

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
                    # 但他都已經是空的了，進來不就也只是 +0 而已嗎
                    # assert False, "ERROR, You should not come in."
                    reg_loss += pred_boxes_flatten.sum()
                    cen_loss += pred_cen_flatten.sum()

            cen_loss = cen_loss / G
            cls_loss = cls_loss / G
            reg_loss = reg_loss / G
            cls_pos_loss = cls_pos_loss / G
            cls_neg_loss = cls_neg_loss / G

            cen_loss_all.append(cen_loss)
            cls_loss_all.append(cls_loss)
            reg_loss_all.append(reg_loss)
            cls_pos_losses.append(cls_pos_loss)
            cls_neg_losses.append(cls_neg_loss)

        cen_loss = sum(cen_loss_all) / B
        cls_loss = sum(cls_loss_all) / B
        reg_loss = sum(reg_loss_all) / B
        cls_pos_loss = sum(cls_pos_losses) / B
        cls_neg_loss = sum(cls_neg_losses) / B

        return cen_loss, cls_loss, reg_loss, cls_pos_loss, cls_neg_loss


def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
