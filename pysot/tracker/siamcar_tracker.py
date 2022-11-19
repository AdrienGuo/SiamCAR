# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip
from torchvision import transforms


class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        # self.window 下面有再重新自己做一次
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        # 一定要特別標起來紀念一下 (11/08/2022)
        # self.model.eval()

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:, :, :, :], dim=1).data[:, 1, :, :].cpu().numpy()
        return cls

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        # Debug
        if pad.min() < 0:
            print(f"invalid pad value: {pad.min()}")
        if w.min() < 0:
            print(f"ERROR, invalid w value: {w.min()}")
        if h.min() < 0:
            print(f"ERROR, invalid w value: {h.min()}")
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, ltrbs, penalty_lk):
        bboxes_w = ltrbs[0, :, :] + ltrbs[2, :, :]
        bboxes_h = ltrbs[1, :, :] + ltrbs[3, :, :]
        # s_c: size change
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        # TODO: 如果有旋轉的怎麼辦勒？
        # r_c: ratio change
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location_up(self, max_r_up, max_c_up):
        dist_h = int((self.x_img_h - (self.score_h - 1) * cfg.TRACK.STRIDE) / 2)  # 31
        dist_w = int((self.x_img_w - (self.score_w - 1) * cfg.TRACK.STRIDE) / 2)  # 31
        max_r_up += dist_h  # r: row
        max_c_up += dist_w  # c: col
        p_cool_s = np.array([max_r_up, max_c_up])
        # 把框框的座標改成以 (instance_size, instance_size) 當作 (0, 0)
        disp = p_cool_s - (np.array([self.x_img_h, self.x_img_w]) - 1.) / 2.
        return disp

    def accurate_location(self, max_r, max_c):
        # 因為是縮小的位置，所以要先放大 8 倍
        max_r = max_r * cfg.TRACK.STRIDE
        max_c = max_c * cfg.TRACK.STRIDE
        dist_h = int((self.x_img_h - (self.score_h - 1) * cfg.TRACK.STRIDE) / 2)  # 31
        dist_w = int((self.x_img_w - (self.score_w - 1) * cfg.TRACK.STRIDE) / 2)  # 31
        max_r += dist_h  # r: row
        max_c += dist_w  # c: col
        p_cool_s = np.array([max_r, max_c])
        # 把框框的座標改成以 (instance_size, instance_size) 當作 (0, 0)
        disp = p_cool_s - (np.array([self.x_img_h, self.x_img_w]) - 1.) / 2.
        return disp

    def coarse_location(
        self,
        hp_score_up,
        p_score_up,  # (upsize, upsize)
        scale_score,
        ltrbs: np.array  # (size, size, 4)
    ):
        """
        看起來亭儀這裡就不管了，直接回傳和原本輸入一模一樣的 p_score_up，
        簡單來說，就是一個沒用的 method。

        這個 method 我沒有看得很懂，不過我猜原本的用意，
        是要限制預測的範圍，會以 max_r_up_hp, max_c_up_hp 為中心，
        框出一個框框，超出這個框框的 score 全部都設為 0，
        但這是追蹤才能這樣做，因為他們只有一個 target，
        我們有很多個 targets 所以整張影像都要看。

        Returns:
            p_score_up: Same as the input p_score_up
        """
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        # 找 hp_score_up 裡面的 maximum 的 (i, j)
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        # 從 (upsize, upsize) 還原到 (size, size) 的位置上
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        # 將超出 (size, size) 範圍的移到範圍內
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE - 1)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE - 1)
        bbox_region = ltrbs[max_r, max_c, :]  # bbox_region: (4,)

        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)
        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)

        mask = np.zeros_like(p_score_up)
        mask = np.ones_like(p_score_up)  # 原本的 code 沒有，亭儀加的
        mask[max_r_up_hp - t_region: max_r_up_hp + b_region + 1, max_c_up_hp - l_region: max_c_up_hp + r_region + 1] = 1
        # mask[:, :] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    # 原版的 code
    def get_center_up(
        self,
        hp_score_up,  # useless
        p_score_up,
        scale_score,  # useless
        ltrbs,  # useless, (size, size, 4)
        ltrbs_up: np.array,  # (upsize, upsize, 4)
        hp: dict,
        cls_up  # (upsize, upsize)
    ):
        # score_up: (upsize, upsize)
        # coarse_location 完全沒用，score_up 就等於 p_score_up
        # score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, ltrbs)
        score_up = p_score_up

        boxes = []
        scores = []
        score = score_up
        # TODO: 這裡可能會錯，應該要全部都找才對吧？
        # 但全部都找的話超慢...
        for i in range(max(score_up.shape[0], score_up.shape[1])):  # 所以只會找 193 個
            # 找 score 裡面的 maximum 的 (i, j)
            max_r_up, max_c_up = np.unravel_index(score.argmax(), score.shape)
            scores.append(score[max_r_up][max_c_up])
            # disp: 將座標軸改以 (instance_size / 2, instance_size / 2) 當作 (0, 0)
            disp = self.accurate_location_up(max_r_up, max_c_up)
            # self.scale_z = 1，因為我直接放在 search image 上面
            disp_ori = disp / self.scale_z
            # 將座標軸 "還原" 回以 左上角為 (0, 0)
            new_cx = disp_ori[1] + self.center_pos[0]
            new_cy = disp_ori[0] + self.center_pos[1]

            ave_w = (ltrbs_up[max_r_up, max_c_up, 0] + ltrbs_up[max_r_up, max_c_up, 2]) / self.scale_z
            ave_h = (ltrbs_up[max_r_up, max_c_up, 1] + ltrbs_up[max_r_up, max_c_up, 3]) / self.scale_z
            # s_c: size change
            s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
            # TODO: 如果有旋轉的怎麼辦勒？
            # r_c: ratio change
            r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
            # penalty: <= 1，越小代表懲罰越重
            penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
            # 再去調整原本預測出來的 ave_w, ave_h，
            # 如果 cls_up 的分數越高 --> lr 就越高，
            # lr 越高 --> 新的 w, h 會越近原本預測的 ave_w, ave_h
            lr = cls_up[max_r_up, max_c_up] * hp['lr'] * penalty
            new_width = lr * ave_w + (1 - lr) * self.size[0]
            new_height = lr * ave_h + (1 - lr) * self.size[1]

            box = [new_cx, new_cy, new_width, new_height]
            boxes.append(box)

            score[max_r_up][max_c_up] = -1

        return boxes, scores

    # 我做的 code。
    # 和原版就差在，我不是用經過 cv2.INTER_CUBIC 變成 up，
    # 而是直接用最原始由 outputs 出來的。
    def get_center(
        self,
        p_score,
        ltrbs: np.array,  # (score_h, score_w, 4)
        hp: dict,
        cls  # (score_h, score_w)
    ):
        boxes = []
        scores = []
        # TODO: 這裡可能會錯，應該要全部都找才對吧？
        # 但全部都找的話超慢...
        for i in range(max(p_score.shape[0], p_score.shape[1])):
            # 找 p_score 裡面的 maximum 的 (i, j)
            # 要注意這裡傳回來的是 “縮小 8 倍“ 的 index (因為不像上面有做 up)
            max_r, max_c = np.unravel_index(p_score.argmax(), p_score.shape)
            scores.append(p_score[max_r][max_c])
            # disp: 將座標軸改以 (x_img_h / 2, x_img_w / 2) 當作 (0, 0)
            disp = self.accurate_location(max_r, max_c)
            # self.scale_z = 1，因為我直接放在 search image 上面
            disp_ori = disp / self.scale_z
            # 將座標軸 "還原" 回以 左上角為 (0, 0)
            new_cx = disp_ori[1] + self.center_pos[0]
            new_cy = disp_ori[0] + self.center_pos[1]

            ave_w = (ltrbs[max_r, max_c, 0] + ltrbs[max_r, max_c, 2]) / self.scale_z
            ave_h = (ltrbs[max_r, max_c, 1] + ltrbs[max_r, max_c, 3]) / self.scale_z
            # s_c: size change
            s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
            # TODO: 如果有旋轉的怎麼辦勒？
            # r_c: ratio change
            r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
            # penalty: <= 1，越小代表懲罰越重
            penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
            # 再去調整原本預測出來的 ave_w, ave_h，
            # 如果 cls 的分數越高 --> lr 就越高，
            # lr 越高 --> 新的 w, h 會越近原本預測的 ave_w, ave_h
            lr = cls[max_r, max_c] * hp['lr'] * penalty
            new_width = lr * ave_w + (1 - lr) * self.size[0]
            new_height = lr * ave_h + (1 - lr) * self.size[1]

            box = [new_cx, new_cy, new_width, new_height]
            boxes.append(box)

            p_score[max_r][max_c] = -1

        return boxes, scores

    def nms(self, bbox, scores, iou_threshold):
        cx = bbox[:, 0]
        cy = bbox[:, 1]
        width = bbox[:, 2]
        height = bbox[:, 3]

        x1 = cx - width / np.array(2)
        y1 = cy - height / np.array(2)
        x2 = x1 + width
        y2 = y1 + height
        areas = (x2 - x1) * (y2 - y1)

        # 结果列表
        result = []
        index = scores.argsort()[::-1]  # 由高到低排序信心值取得 index
        while index.size > 0:
            i = index[0]  # 目前 score 最大的那個 index
            result.append(i)
            # 計算現在這個框框 與其他所有框框的 IOU
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            # 只留下小於 iou_threshold 的框框 (因為大於就代表他們重疊了)
            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx + 1]  # 剩下的框

        return result

    def init(self, z_img: torch.tensor, bbox: np.array):
        """
        Args:
            img(np.ndarray): BGR image
            bbox: (x1, y1, w, h): bbox
        """

        self.box = bbox
        # self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
        #                             bbox[1]+(bbox[3]-1)/2])

        # self.size: template 的原始寬, 高
        self.size = np.array([bbox[2], bbox[3]])

        # # calculate z crop size
        # w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # s_z = round(np.sqrt(w_z * h_z))

        # # calculate channle average
        # self.channel_average = np.mean(img, axis=(0, 1))

        # # get crop
        # z_crop = self.get_subwindow(img, self.center_pos,
        #                             cfg.TRACK.EXEMPLAR_SIZE,
        #                             s_z, self.channel_average)
        # if cfg.CUDA:
        #     z_crop = z_crop.cuda()

        self.model.template(z_img)

        return z_img

    def track(self, x_img: torch.tensor, hp: dict):
        """
        Args:
            x_img (tensor): (b, c, h, w) BGR image
        Return:
            bbox (list): [x, y, width, height]
        """
        # === 把框框移動到 "原圖" or "search image" ===
        # - 把框框移動到 “原圖” 上時，就加上 “原圖” 的中心點位置 -
        # self.center_pos = np.array([img.shape[1]/2, img.shape[0]/2])
        # - 把框框移動到 "search image" 上時，就加上 "search image" 的中心點位置 -
        # TODO: 要動態調整
        # 調整成 x_img 的大小，記得寬高要分開
        self.x_img_h = x_img.size()[2]
        self.x_img_w = x_img.size()[3]
        # x_img 的中心點
        self.center_pos = np.array([self.x_img_w / 2, self.x_img_h / 2])

        # - 若上面的框框是放在 原圖 上面，要計算縮放比例 s_z -
        # w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # s_z = np.sqrt(w_z * h_z)
        # self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        # - 若上面的框框是放在 search image 上面，就不用縮放 -
        self.scale_z = 1

        # s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        # x_crop = self.get_subwindow(img, self.center_pos,
        #                             cfg.TRACK.INSTANCE_SIZE,
        #                             round(s_x), self.channel_average)
        # if cfg.CUDA:
        #     x_crop = x_crop.cuda()
        # outputs = self.model.track(x_crop)

        # Get pred outputs.
        outputs = self.model.track(x_img)

        # ipdb.set_trace()

        cls = self._convert_cls(outputs['cls']).squeeze(axis=0)

        cen = outputs['cen'].data.cpu().numpy()
        if outputs['cen'].shape[-1] != 1:
            cen = (cen - cen.min()) / cen.ptp()  # ptp: peek-to-peek
        cen = cen.squeeze(axis=(0, 1))  # cen: (size, size)

        # ltrbs: (4, score_h, score_w)
        ltrbs = outputs['loc'].data.cpu().numpy().squeeze(axis=0)

        # TODO: 高、寬要分開動態調整
        self.score_h = ltrbs.shape[1]
        self.score_w = ltrbs.shape[2]
        upsize_h = (self.score_h - 1) * cfg.TRACK.STRIDE + 1
        upsize_w = (self.score_w - 1) * cfg.TRACK.STRIDE + 1

        # ipdb.set_trace()

        # 計算 (大小 & 長寬比) 的懲罰量
        penalty = self.cal_penalty(ltrbs, hp['penalty_k'])
        # 加入 penalty
        # p_score = cls * cen * penalty
        p_score = cls * cen

        hanning_h = np.hanning(self.score_h)
        hanning_w = np.hanning(self.score_w)
        self.window = np.outer(hanning_h, hanning_w)
        # hp_score: (score_h, score_w)
        if cfg.TRACK.hanming:
            # TODO: 這應該是不能加吧，變成越四周的分數越低，和我們的 task 不合
            hp_score = p_score * (1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score

        # TODO: upsize_h, upsize_w 順序不能顛倒！！
        # upsize = (size - 1) * 8 + 1
        hp_score_up = cv2.resize(hp_score, (upsize_h, upsize_w), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize_h, upsize_w), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize_h, upsize_w), interpolation=cv2.INTER_CUBIC)
        ltrbs = np.transpose(ltrbs, (1, 2, 0))  # ltrbs: (score_h, score_w, 4)
        ltrbs_up = cv2.resize(ltrbs, (upsize_h, upsize_w), interpolation=cv2.INTER_CUBIC)

        # ipdb.set_trace()

        scale_score = upsize_h / cfg.TRACK.SCORE_SIZE  # useless

        # 兩種模式:
        # 使用 upsize 的模式
        # bbox, rescore = self.get_center_up(
        #     hp_score_up,  # useless
        #     p_score_up,
        #     scale_score,  # useless
        #     ltrbs,  # useless
        #     ltrbs_up,
        #     hp,
        #     cls_up
        # )
        # 使用原本 size 的模式
        bbox, rescore = self.get_center(
            p_score,
            ltrbs,
            hp,
            cls
        )

        bbox = np.array(bbox)  # bbox: (N, [cx, cy, w, h])
        rescore = np.array(rescore)  # rescore: (N)

        # 加 nms，計算所有框
        boxes = []
        top_scores = []
        iou_threshold = 0.1
        results = self.nms(bbox, rescore, iou_threshold)

        # 有用圖片的寬高限制預測的範圍
        for i in range(len(results)):
            box = bbox[results[i], :]
            score = rescore[results[i]]
            if score >= 0.5:
                top_scores.append(score)
                cx = box[0]
                cy = box[1]
                width = box[2]  # self.size[0] * (1 - lr) + bbox[2] * lr
                height = box[3]  # self.size[1] * (1 - lr) + bbox[3] * lr

                # clip boundary
                cx = bbox_clip(cx, 0, self.x_img_w)
                cy = bbox_clip(cy, 0, self.x_img_h)
                width = bbox_clip(width, 0, self.x_img_w)
                height = bbox_clip(height, 0, self.x_img_h)

                box = [cx - width / 2,
                       cy - height / 2,
                       width,
                       height]
                boxes.append(box)
            else:
                # 分數 < 0.5 的忽略
                continue

        return {
            'top_scores': top_scores,
            'pred_boxes': boxes,
            'x_img': x_img,
            'cls': cls  # for heamap
        }
