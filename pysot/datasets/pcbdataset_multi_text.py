# Copyright (c) SenseTime. All Rights Reserved.
# 一張影像依類別分開 有Text

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple

import cv2
import ipdb
import numpy as np
import torch
from PIL import Image
from pysot.core.config import cfg
from pysot.datasets.augmentation.augmentation import Augmentation
from pysot.datasets.image_crop import crop, resize
from pysot.utils.bbox import Center, center2corner
from pysot.utils.check_image import create_dir, draw_box, save_image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

    
class PCBDataset(Dataset): #先讀所有圖片，再以類別去讀anno
    def __init__(self, args, mode, loader=default_loader) :
        for name in cfg.DATASET.NAMES:
            data_cfg = getattr(cfg.DATASET, name)

        if mode == "train":
            self.root = args.dataset
            self.anno = args.dataset
        elif mode == "test":
            self.root = args.test_dataset
            self.anno = args.test_dataset
        else:
            assert False, "ERROR, mode is wrong!!"

        self.loader = loader
        imgs,frame,temp = self._make_dataset(self.root)
        self.imgs = imgs
        self.frame = frame
        self.temp = temp
        
        self.loder = default_loader
        
        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
    def _find_classes(self,directory: str):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def _make_dataset(self,directory):
        imgs=[]
        frame=[]
        temp =[]
       
        directory = os.path.expanduser(directory)
        
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):#已排序
                box=[]
                if fname.endswith(('.jpg','.png','bmp')):
                    path = os.path.join(root, fname)
                    
                    anno_path = os.path.join(self.anno,fname[:-3]+"txt")
                    if os.path.isfile(anno_path):
                        
                        f = open(anno_path,'r')
                        lines = f.readlines()
                        anno=[]
                        for line in lines:
                            line=line.strip('\n')
                            line=line.split(' ')
                            line = list(map(float, line))
                            anno.append(line)

                        for i in range(len(anno)):
                            if anno[i][0]!=26:
                                item = path, str(int(anno[i][0]))
                                imgs.append(item)
                                temp.append([anno[i][1],anno[i][2],anno[i][3],anno[i][4]])
                                box=[]
                           
                            if anno[i][0]!=26:
                                for j in range(len(anno)):
                                    if anno[j][0] == anno[i][0]:
                                        box.append([anno[j][1],anno[j][2],anno[j][3],anno[j][4]])
                                box = np.stack(box).astype(np.float32)
                                frame.append(box)
                    elif os.path.isfile(os.path.join(self.anno,fname[:-3]+"label")):
                        
                        f = open(os.path.join(self.anno,fname[:-3]+"label"),'r')
                        img =  cv2.imread(path)
                        imh,imw = img.shape[:2]
                        lines = f.readlines()
                        anno=[]
                        for line in lines:
                            line=line.strip('\n')
                            line=line.split(',')
                            line = list(line)
                            anno.append(line)
                        for i in range(len(anno)):
                            if (float(anno[i][1])>0) and (float(anno[i][2]) > 0):
                                item = path,anno[i][0]
                                imgs.append(item)
                                cx = float(anno[i][1])+(float(anno[i][3])-float(anno[i][1]))/2
                                cy = float(anno[i][2])+(float(anno[i][4])-float(anno[i][2]))/2
                                w = float(anno[i][3])-float(anno[i][1])
                                h = float(anno[i][4])-float(anno[i][2])
                                temp.append([cx/imw,cy/imh,w/imw,h/imh])
                                box=[]
                                #print(anno[i])

                                for j in range(len(anno)):
                                    if anno[j][0] == anno[i][0]:
                                        cx = float(anno[i][1])+(float(anno[i][3])-float(anno[i][1]))/2
                                        cy = float(anno[i][2])+(float(anno[i][4])-float(anno[i][2]))/2
                                        w = float(anno[i][3])-float(anno[i][1])
                                        h = float(anno[i][4])-float(anno[i][2])

                                        box.append([cx/imw,cy/imh,w/imw,h/imh])
                                box = np.stack(box).astype(np.float32)
                                frame.append(box)
                        
                        
                        #else:
                        #    box.append([anno[i][1],anno[i][2],anno[i][3],anno[i][4]])
                        #box = np.stack(box).astype(np.float32)
                        #frame.append(box)

                    #cx,cy,w,h
        return imgs,frame,temp

    #追蹤的
    def _get_image_anno(self,index,typeName):
        
        image_path,target = self.imgs[index]
        if typeName=="template":
            bbox = self.temp[index]
        elif typeName=="search":
            bbox = self.frame[index]
        bbox = np.stack(bbox).astype(np.float32)    
        return image_path, bbox, target

    def _get_positive_pair(self, index):#一對一pair 以隨機挑選
        
        #傳template和search的index
        return self._get_image_anno(index,'template'), self._get_image_anno(index,'search')
        
        
       
        

    def _get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)    
    
    def _template_crop(self,image,box):
        imh, imw = image.shape[:2]
        cx = box[0]*imw
        cy = box[1]*imh
        w = box[2]*imw
        h = box[3]*imh
        
        minx = int(cx - w/2)
        miny = int(cy - h/2)
        maxx = int(cx + w/2)
        maxy = int(cy + h/2)
        
        crop_img = image[miny:maxy, minx:maxx]
        
        return crop_img
    
   
    #使用transform resize、centercrop    
    def _get_bbox(self, shape,direction,origin_size,temp):#原本cx,cy,w,h
        bbox=[]
        length = len(shape)
        
        shape[:,0]=shape[:,0]*origin_size[0]
        shape[:,1]=shape[:,1]*origin_size[1]
        shape[:,2]=shape[:,2]*origin_size[0]
        shape[:,3]=shape[:,3]*origin_size[1]

        for i in range (len(shape)):
            if direction=='x':
                x1 = (shape[i][0]-shape[i][2]/2)+temp
                y1 = (shape[i][1]-shape[i][3]/2)
                x2 = (shape[i][0]+shape[i][2]/2)+temp
                y2 = (shape[i][1]+shape[i][3]/2)
            else:
                x1 = (shape[i][0]-shape[i][2]/2)
                y1 = (shape[i][1]-shape[i][3]/2)+temp
                x2 = (shape[i][0]+shape[i][2]/2)
                y2 = (shape[i][1]+shape[i][3]/2)+temp
            bbox.append(center2corner(Center((x1+(x2-x1)/2),(y1+(y2-y1)/2), (x2-x1), (y2-y1))))   
        return bbox
    

    def _get_bbox_template(self, image, shape):
        imh, imw = image.shape[:2]
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        bbox=[]

        shape[2] = shape[2]*imw
        shape[3] = shape[3]*imh
        w, h = shape[2],shape[3]
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        
        bbox=(center2corner(Center(cx, cy, w, h)))
       
        return bbox    
    
    #使用image_crop 511
    def _search_gt_box(self, image, shape,scale): #整張都會縮小，不確定效果好嗎
        imh, imw = image.shape[:2]
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        bbox=[]
        cx=shape[:,0]*imw
        cy=shape[:,1]*imh
        w=shape[:,2]*imw
        h=shape[:,3]*imh
            
        for i in range (len(shape)):
            
            w1 ,h1 = w[i],h[i]
           
            cx1 = cx[i]*scale[0]+scale[2]
            cy1 = cy[i]*scale[1]+scale[3]
   
            wc_z = w1 + context_amount * (w1+h1)
            hc_z = h1 + context_amount * (w1+h1)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w1 = w1*scale_z
            h1 = h1*scale_z
            
            bbox.append(center2corner(Center(cx1, cy1, w1, h1)))
        return bbox 
    
    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = self._get_positive_pair(index)
        
       
        # get image
        assert template[0] == search[0], f"ERROR, should be the same if neg=False"
        img_path = template[0]
        img_name = img_path.split('/')[-1].rsplit('.', 1)[-2]
        origin_img = cv2.imread(img_path)
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        assert template_image is not None, f"Error image: {template[0]}"
        if template_image is None:
            print('error image:',template[0])
        
        img_cls = template[2]
        assert isinstance(img_cls, str), f"Error, class should be string"
        
        template_box = template[1]

        #crop to 511 (like SiamFC)
        imh,imw = template_image.shape[:2]
        if cfg.DATASET.Background:
            template_crop = self._template_crop(template_image,template[1])
            template,scale = crop(template_crop,template[1],cfg.DATASET.Background)
        else:
            template,scale = crop(template_image,template[1],cfg.DATASET.Background)
        template_box = self._get_bbox_template(template_image, template_box)
        
        
        #search crop (like SiamFC) 但影像都放在左上角
        if (imw*scale[0] > cfg.TRAIN.SEARCH_SIZE ) or (imh*scale[1] > cfg.TRAIN.SEARCH_SIZE ):
            center_crop = transforms.CenterCrop(cfg.TRAIN.SEARCH_SIZE)
            search_image = Image.fromarray((cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)))
            s_resize = resize(search_image,cfg.TRAIN.SEARCH_SIZE)
            origin_size = s_resize.size
            search_image = center_crop(s_resize)
        
            if origin_size[0] < cfg.TRAIN.SEARCH_SIZE: #x
                temp = (cfg.TRAIN.SEARCH_SIZE-origin_size[0])/2
                direction = 'x' 
            elif origin_size[1] < cfg.TRAIN.SEARCH_SIZE: #y
                temp = (cfg.TRAIN.SEARCH_SIZE-origin_size[1])/2
                direction ='y'
            else:
                temp=0
                direction ='x'

            # get bounding box
            search_box = self._get_bbox(search[1],direction,origin_size,temp)
            
            search = cv2.cvtColor(np.asarray(search_image), cv2.COLOR_RGB2BGR)
            bbox = search_box
        else:
            cx = (imw/2)*scale[0]+scale[2]
            cy = (imh/2)*scale[1]+scale[3]
            if (imw*scale[0] < (cfg.TRAIN.SEARCH_SIZE / 2) ) and (imh*scale[1] < (cfg.TRAIN.SEARCH_SIZE / 2) ):
                mapping = np.array([[scale[0], 0, (cfg.TRAIN.SEARCH_SIZE / 2)-cx],
                            [0, scale[1], (cfg.TRAIN.SEARCH_SIZE / 2)-cy]]).astype(np.float)
                scale[2] = (cfg.TRAIN.SEARCH_SIZE / 2)-cx
                scale[3] = (cfg.TRAIN.SEARCH_SIZE / 2)-cy
               
            else:
                mapping = np.array([[scale[0], 0, 0],
                            [0, scale[1], 0]]).astype(np.float)
              
                scale[2] = 0
                scale[3] = 0
            search_image2 = cv2.warpAffine(
                search_image, mapping,
                (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0)
            )
        
            search_box = self._search_gt_box(search_image,search[1],scale)
            bbox = search_box
            search = search_image2

        # augmentation
        
        template, _ = self.template_aug(template,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)
        
        search, bbox = self.search_aug(search,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ########################
        # 創 directory
        ########################
        # dir = f"./image_check/x{cfg.TRAIN.SEARCH_SIZE}"
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
        # save_name = f"{img_name}__{img_cls}__{index}.jpg"

        # origin_path = os.path.join(origin_dir, save_name)
        # save_image(origin_img, origin_path)
        # template_path = os.path.join(template_dir, save_name)
        # save_image(template, template_path)

        # # Draw gt_boxes on search image
        # tmp_gt_boxes = np.asarray(bbox).astype(None).copy()
        # tmp_gt_boxes[:, 2] = tmp_gt_boxes[:, 2] - tmp_gt_boxes[:, 0]
        # tmp_gt_boxes[:, 3] = tmp_gt_boxes[:, 3] - tmp_gt_boxes[:, 1]
        # gt_image = draw_box(search, tmp_gt_boxes, type="gt")
        # search_path = os.path.join(search_dir, save_name)
        # save_image(gt_image, search_path)

        # ipdb.set_trace()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        box=[]
        for i in range (len(bbox)):
            box.append([0,bbox[i].x1,bbox[i].y1,bbox[i].x2,bbox[i].y2])
        box = np.stack(box).astype(np.float32)
        box=torch.as_tensor(box, dtype=torch.int64)
        
        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        cls = torch.as_tensor(cls, dtype=torch.int64)
        
        
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        
        template = torch.as_tensor(template, dtype=torch.float32)
        search = torch.as_tensor(search, dtype=torch.float32)
        

        return template,search,cls,box
