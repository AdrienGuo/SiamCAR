# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import torchvision.transforms.functional as F
from pysot.utils.bbox import corner2center, \
        Center, center2corner, Corner
def resize(image,size):
    img_height = image.height
    img_width = image.width
    
    if (img_width / img_height) > 1:
            rate = size / img_width
    else:
            rate = size/ img_height
       
    width=int(img_width*rate)
    height=int(img_height*rate)
   
    return F.resize(image, (height,width))

def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    scale=[a,b,c,d]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop,scale


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z,scale = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    #x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z,scale


def crop(img, box,background,instanc_size=511):
    
    avg_chans = np.mean(img, axis=(0, 1))
    imh , imw = img.shape[:2]
    
    cx=box[0]*imw
    cy=box[1]*imh
    w=box[2]*imw
    h=box[3]*imh
    
    #xmin,ymin,xmax,ymax
    if background :
        bbox =[0,0,imw,imh]
    else:
        bbox=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
    
    
    x,scale = crop_like_SiamFC(img, bbox, instanc_size=instanc_size, padding=(0,0,0))
    return x,scale
