"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


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

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
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
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self,cfg):
        self.box_reg_loss_func = IOULoss()
        #self.box_reg_loss_func = GIOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(points, labels, gt_bbox)

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        xs, ys = locations[:, 0], locations[:, 1]# torch.Size([3025])
        
        labels_all = []
        reg_targets = []
        bboxes = gt_bbox
        labels = labels.view(-1,self.cfg.TRAIN.OUTPUT_SIZE**2)#torch.Size([batch,625])
        for i in range(labels.shape[0]): #每次處裡一張，有i張(batch大小)
            bbox=[] #一張影像的bounding box
            
            for j in range(len(bboxes)):
                if bboxes[j][0]==i:
                    bbox.append(bboxes[j])
                    
                else:
                    break
            bboxes = bboxes[j:,:]
            
            bbox = torch.stack(bbox)
            l = xs[:, None] - bbox[:, 1][None].float()
            t = ys[:, None] - bbox[:, 2][None].float()
            r = bbox[:, 3][None].float() - xs[:, None]
            b = bbox[:, 4][None].float() - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            #reg_targets_per_im变量（形状为(N, M, 4)，N为所有采样点的个数，M为bbox数量）
            
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0  
            is_in_boxes = is_in_boxes.permute(1,0).contiguous() 
            reg_targets_per_im = reg_targets_per_im.permute(1,0,2).contiguous() 
            reg_targets.append(reg_targets_per_im)
            labels_per_im = torch.zeros(is_in_boxes.shape[0],self.cfg.TRAIN.OUTPUT_SIZE**2) 
            
            for n in range(is_in_boxes.shape[0]):
                pos = np.where(is_in_boxes[n].cpu() == 1)
                labels_per_im[n][pos] = 1
            labels_all.append(labels_per_im)
        
        return labels_all, reg_targets


    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor]) #[batch,4,25,25]
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        
        N = box_cls[0].size(0) #影像張數
        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        
        reg_loss_all=[]
        centerness_loss_all=[]
        cls_loss_all=[]
        
        
        for l in range(N):#有N個影像
            reg_loss =0
            centerness_loss=0
            cls_loss=0
            for k in range(len(label_cls[l])): #有k個box
                #預測的邊界框回歸
                box_regression_flatten= (box_regression[l].permute(1, 2, 0).contiguous().view(-1, 4))
                #邊界框分類label
                labels_flatten = (label_cls[l][k].view(-1))
                
                #邊界框回歸label
                reg_targets_flatten = (reg_targets[l][k].view(-1, 4))
                #預測的中心度
                centerness_flatten = (centerness[l].view(-1))
                
                pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

                box_regression_flatten = box_regression_flatten[pos_inds]
                reg_targets_flatten = reg_targets_flatten[pos_inds]
                centerness_flatten = centerness_flatten[pos_inds]
                cls_loss += select_cross_entropy_loss(box_cls, labels_flatten)
                
                if pos_inds.numel() > 0:
                    centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
                    reg_loss += self.box_reg_loss_func(
                        box_regression_flatten,
                        reg_targets_flatten,
                        centerness_targets
                    )
                    centerness_loss += self.centerness_loss_func(
                        centerness_flatten,
                        centerness_targets
                    )
                else:
                    reg_loss += box_regression_flatten.sum()
                    centerness_loss += centerness_flatten.sum()
           
            cls_loss = cls_loss.sum()/len(label_cls[l])
            reg_loss = reg_loss.sum()/len(label_cls[l])
            centerness_loss = centerness_loss.sum()/len(label_cls[l])
            
            cls_loss_all.append(cls_loss)
            reg_loss_all.append(reg_loss)
            centerness_loss_all.append(centerness_loss)
            
            

        cls_loss = sum(cls_loss_all)/N
        reg_loss = sum(reg_loss_all)/N
        centerness_loss = sum(centerness_loss_all)/N

        return cls_loss, reg_loss, centerness_loss


def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
