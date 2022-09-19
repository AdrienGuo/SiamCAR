# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2
import time
import onnxruntime
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip
import torch
from openvino.inference_engine import IECore, IENetwork

class SiamCARTracker(SiameseTracker):
    def __init__(self, model, device,types,cfg):
        super(SiamCARTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.device = device
        self.type = types
        if self.type=='openvino':
             # Get Input Layer Information
            Input = iter(self.model.inputs)
            self.InputLayer_search = next(Input)
            self.InputLayer_template = next(Input)

            # Get Output Layer Information
            Output = iter(self.model.outputs)
            self.OutputLayer_cen = next(Output)
            self.OutputLayer_cls = next(Output)
            self.OutputLayer_loc = next(Output)


    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls
    
    
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
     
        self.box = bbox
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])

        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        self.z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        
        if self.device !='cpu' and self.device!='CPU':
            self.z_crop = self.z_crop.cuda()
        
    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):
       
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)#31
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE-1)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE-1)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask = np.ones_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        #mask[:, :] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self,hp_score_up, p_score_up, scale_score,lrtbs,lrtbs_up,hp,cls_up):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        
        # accurate location
        result=[]
        score_result =[]
        score = score_up
        
        for i in range(len(score_up)):
            
            max_r_up, max_c_up = np.unravel_index(score.argmax(), score.shape) #給x,y
            score_result.append(score[max_r_up][max_c_up])
            disp = self.accurate_location(max_r_up,max_c_up)
            disp_ori = disp / self.scale_z
            new_cx = disp_ori[1] + self.center_pos[0]
            new_cy = disp_ori[0] + self.center_pos[1]
            
            ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
            ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z
            
            s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
            r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
            penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
            lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
            new_width = lr*ave_w + (1-lr)*self.size[0]
            new_height = lr*ave_h + (1-lr)*self.size[1]
            
            
            ans = [new_cx, new_cy,new_width,new_height]
            
            score[max_r_up][max_c_up]=-1
            result.append(ans)
            
        
        return result,score_result

    
    def nms(self, bbox,scores,image,iou_threshold ):
        cx = bbox[:,0]
        cy = bbox[:,1]
        width = bbox[:,2]
        height = bbox[:,3]
        
        x1 = cx - width/np.array(2)
        y1 = cy - height/np.array(2)
        x2 = x1 + width
        y2 = y1 + height
        areas = (x2 - x1)*(y2 - y1)
        
        # 结果列表
        result = []
        index = scores.argsort()[::-1]  # 由高到低排序信心值取得index
        while index.size > 0:
            i = index[0]
            result.append(i)  

            # 計算其他框與該框的IOU
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            # 只保留满足IOU閥值的index
            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx + 1]  #剩下的框

        return result
    
    
    def track(self, img, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        timemean=[]
        self.center_pos = np.array([img.shape[1]/2,img.shape[0]/2])
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        
        if self.device !='cpu' and self.device!='CPU':
            x_crop = x_crop.cuda()
        
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        # Start Inference
        start = time.time()

        if self.type=='openvino':
            results = self.model.infer(inputs={self.InputLayer_template:self.z_crop,self.InputLayer_search: x_crop})
            cls = torch.from_numpy(results[self.OutputLayer_cls])
            loc = torch.from_numpy(results[self.OutputLayer_loc])
            cen = torch.from_numpy(results[self.OutputLayer_cen])
            
        elif self.type=='onnx':
            inputs = {self.model.get_inputs()[0].name: to_numpy(self.z_crop),
                      self.model.get_inputs()[1].name: to_numpy(x_crop)}
            results = self.model.run(None, inputs)
            cls = torch.from_numpy(results[0])
            loc = torch.from_numpy(results[1])
            cen = torch.from_numpy(results[2])
        
        outputs={'cls':cls,'loc':loc,'cen':cen}

        end = time.time()
        inf_time = end - start
        timemean.append(inf_time)
        print('Inference Time: {} Seconds Single Image'.format(inf_time))

        fps = 1./(end-start)
        print('Estimated FPS: {} FPS Single Image'.format(fps))
        
        
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        
        p_score = penalty * cls * cen
        hp_score = p_score #68x98
        
        if cfg.TRACK.hanming:
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score
        
        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC) #變193x193
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs,(1,2,0))#(88, 98, 4)
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE

        bbox,rescore = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs,lrtbs_up,hp,cls_up)
        bbox=np.array(bbox)
        rescore=np.array(rescore)
        
        #加nms，計算所有框
        boxes =[]
        best_score =[]
        iou_threshold = 0.1
        result=self.nms(bbox,rescore,img,iou_threshold)
        for i in range(len(result)):
            box = bbox[result[i],:]
            scores = rescore[result[i]]
            best_score.append(scores)
            if scores >=0.5:
                cx = box[0]
                cy = box[1]
                width = box[2] #self.size[0] * (1 - lr) + bbox[2] * lr
                height = box[3] #self.size[1] * (1 - lr) + bbox[3] * lr


                # clip boundary
                cx = bbox_clip(cx,0,img.shape[1])
                cy = bbox_clip(cy,0,img.shape[0])
                width = bbox_clip(width,0,img.shape[1])
                height = bbox_clip(height,0,img.shape[0])
                
                box = [cx - width / 2,
                       cy - height / 2,
                       width,
                       height]
                boxes.append(box)
                
            else:
                continue
        bbox = boxes
        
        
        return {
                'bbox': bbox,
               }
