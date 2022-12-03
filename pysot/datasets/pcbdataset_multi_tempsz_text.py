# Copyright (c) SenseTime. All Rights Reserved.
# 一張影像依類別分開 Text

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging
import sys
import os
from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation.augmentation import Augmentation
from pysot.core.config import cfg
import xml.etree.ElementTree as ET

#from torchvision import datasets
from torchvision.datasets.folder import default_loader
from pysot.datasets.image_crop_sz import resize,crop
from torchvision import transforms
from PIL import Image
#from pysot.datasets.transforms import get_transforms
import torch
 

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

    
class PCBDataset(Dataset): #先讀所有圖片，再以類別去讀anno
    def __init__(self,loader=default_loader) :
        for name in cfg.DATASET.NAMES:
            data_cfg = getattr(cfg.DATASET, name)
            
        self.root = data_cfg.ROOT
        self.anno = data_cfg.ANNO
        self.loader = loader
        #classes, class_to_idx = self._find_classes(self.root)
        #self.class_to_idx = class_to_idx
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
        '''
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            count = 0
            if not os.path.isdir(target_dir):
                continue
        '''
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
                                item = path,anno[i][0]
                                imgs.append(item)
                                temp.append([anno[i][1],anno[i][2],anno[i][3],anno[i][4]])
                                box=[]
                            #print(anno[i])
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
        
        #x1, y1, x2, y2 = bbox
        #print("target:",target)
        if typeName=="template":
            bbox = self.temp[index]
        elif typeName=="search":
            bbox = self.frame[index]
            
        
        '''
        #讀xml
        xml = ET.parse(anno_path)
        root = xml.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            bnd = obj.find('bndbox')
            if name == target and typeName=="search":
                bbox.append([int(bnd.find(tag).text)  for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            if name == target and typeName=="template":
                xmin = int(bnd.find('xmin').text)
                ymin = int (bnd.find('ymin').text)
                xmax = int (bnd.find('xmax').text)
                ymax = int (bnd.find('ymax').text)
                bbox =[xmin,ymin,xmax,ymax]
                break
        ''' 
        
        bbox = np.stack(bbox).astype(np.float32)    
        return image_path, bbox

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
    
    
    
   
    #使用transform resize、centercrop    
    def _get_bbox_255(self, shape,origin_size,tempx,tempy):#原本cx,cy,w,h
        bbox=[]
        #context_amount = 0.5
        #exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        #cx, cy = imw//2, imh//2
        length = len(shape)
        
        shape[:,0]=shape[:,0]*origin_size[0]
        shape[:,1]=shape[:,1]*origin_size[1]
        shape[:,2]=shape[:,2]*origin_size[0]
        shape[:,3]=shape[:,3]*origin_size[1]

        for i in range (len(shape)):
              
            x1 = (shape[i][0]-shape[i][2]/2)+tempx
            y1 = (shape[i][1]-shape[i][3]/2)+tempy
            x2 = (shape[i][0]+shape[i][2]/2)+tempx
            y2 = (shape[i][1]+shape[i][3]/2)+tempy
            
            bbox.append(center2corner(Center((x1+(x2-x1)/2),(y1+(y2-y1)/2), (x2-x1), (y2-y1))))
            
        return bbox
    
    def _get_bbox(self, shape,direction,origin_size,temp):#原本cx,cy,w,h
        bbox=[]
        #context_amount = 0.5
        #exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        #cx, cy = imw//2, imh//2
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

    def _get_bbox2(self, image, shape):
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
            cx1 = scale[0]*(cx[i])+scale[2]
            cy1 = scale[1]*(cy[i])+scale[3]
            wc_z = w1 + context_amount * (w1+h1)
            hc_z = h1 + context_amount * (w1+h1)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w1 = w1*scale_z
            h1 = h1*scale_z
            
            bbox.append(center2corner(Center(cx1, cy1, w1, h1)))
        return bbox
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
    
    def _template_crop_pil(self,image,box):
        imw, imh = image.size
        cx = box[0]*imw
        cy = box[1]*imh
        w = box[2]*imw
        h = box[3]*imh
        
        crop_img = image.crop((cx-w /2,cy-h/2,cx+w/2 ,cy+h/2))
        
        return crop_img
    
    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        #index = self.pick[index]
        #dataset, index = self._find_dataset(index)
        
        
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = self._get_positive_pair(index)
        #print("template:",template[0])
        #print("search:",search[0])
       
        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        if template_image is None:
            print('error image:',template[0])
        
        #print("template_box:",template[1])
        #print("search_box:",search[1])
        
        #crop to 511 (like SiamFC)
        
        imh,imw = template_image.shape[:2]
        template_crop = self._template_crop(template_image,template[1])
        template,scale = crop(template_crop,template[1])
        #template_image = crop(template_image,template[1])
        #template_box = self._get_bbox2(template_image, template[1])
        
        
        #search crop (like SiamFC) 但影像都放在左上角
        if (imw*scale[0] > 600 ) or (imh*scale[1] > 600 ):
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
            if (imw*scale[0] < 300 ) and (imh*scale[1] < 300 ):
                mapping = np.array([[scale[0], 0, 300-cx],
                            [0, scale[1], 300-cy]]).astype(np.float)
                scale[2] = 300-cx
                scale[3] = 300-cy
                check=1
            else:
                mapping = np.array([[scale[0], 0, 0],
                            [0, scale[1], 0]]).astype(np.float)
                check=0
                scale[2] = 0
                scale[3] = 0
            search_image2 = cv2.warpAffine(search_image, mapping, (600, 600), 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
            search_box = self._search_gt_box(search_image,search[1],scale)
            bbox = search_box
            search = search_image2

        
        #crop to 255 (transform)
        #法一
        '''
        t_center_crop = transforms.CenterCrop(cfg.TRAIN.EXEMPLAR_SIZE)
        s_center_crop = transforms.CenterCrop(cfg.TRAIN.SEARCH_SIZE)
        search_image = Image.fromarray((cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)))
        s_resize = resize(search_image,cfg.TRAIN.EXEMPLAR_SIZE)
        origin_size = s_resize.size
        search_image = s_center_crop(s_resize)
        
        tempx = (cfg.TRAIN.SEARCH_SIZE-origin_size[0])/2
        tempy = (cfg.TRAIN.SEARCH_SIZE-origin_size[1])/2
        # get bounding box
        search_box = self._get_bbox(search[1],origin_size,tempx,tempy)
        search = cv2.cvtColor(np.asarray(search_image), cv2.COLOR_RGB2BGR)
        bbox = search_box
        
        #template
        template_image = self._template_crop_pil(s_resize,template[1])
        template_image = t_center_crop(template_image)
        template = cv2.cvtColor(np.asarray(template_image), cv2.COLOR_RGB2BGR)
        
        '''
        '''
        if len(search[1]) >= 1 :
            #crop to 255 (transform)
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
            search_box = self._get_bbox(search[1],"search",direction,origin_size,temp)
            search = cv2.cvtColor(np.asarray(search_image), cv2.COLOR_RGB2BGR)
            bbox = search_box
        
        else:
            search_image_crop = crop(search_image,search[1][0])
            search_box = self._get_bbox2(search_image_crop, search[1][0])
            search, box = self.search_aug(search_image_crop,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
            bbox=[]
            bbox.append(box)
        '''
        # augmentation
        '''
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)
        
        search, bbox = self.search_aug(search_image_crop,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
        '''
        
        
        box=[]
        for i in range (len(bbox)):
            box.append([0,bbox[i].x1,bbox[i].y1,bbox[i].x2,bbox[i].y2])
        box = np.stack(box).astype(np.float32)
        box=torch.as_tensor(box, dtype=torch.float32)
        
        #print("box:",box)
        
        #template = torch.as_tensor(template, dtype=torch.float32)
        #search = torch.as_tensor(search, dtype=torch.float32)
        
        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        cls = torch.as_tensor(cls, dtype=torch.int64)
        
        
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        
        template = torch.as_tensor(template, dtype=torch.float32)
        search = torch.as_tensor(search, dtype=torch.float32)
        
        
        #print("box:",box.shape)
        return template,search,cls,box
        '''
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                #'bbox':bbox
                #'bbox': np.array([bbox.x1,bbox.y1,bbox.x2,bbox.y2])
                'bbox': np.array([bbox[0].x1,bbox[0].y1,bbox[0].x2,bbox[0].y2])
            
                }
         '''
         

        
    
    
    
    
    
'''   
class PCBDataset(Dataset):
    def __init__(self,):
        super(PCBDataset, self).__init__()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

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
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        if template_image is None:
            print('error image:',template[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])


        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)


        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'bbox': np.array([bbox.x1,bbox.y1,bbox.x2,bbox.y2])
                }

'''