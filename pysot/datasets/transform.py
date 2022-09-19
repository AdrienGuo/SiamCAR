import torch
import torchvision.transforms.functional as F
from torchvision import transforms

import numpy as np

import cv2
from PIL import Image as Img

#CLAHE 3.0

class CLAHE(object):
    def __init__(self, clipLimit=3.0):#, tileGridSize=(8, 8)
        self.clipLimit = clipLimit
        #self.tileGridSize = tileGridSize

    def __call__(self, image):
        open_cv_image = np.array(image)
    
        bgr = open_cv_image[:,:,::-1].copy()
        
        lab = cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
        
        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clipLimit)

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        bgr = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
        
        im_pil = Img.fromarray(bgr)
        return im_pil
    
class Resize(transforms.Resize):

    def __init__(self, size):
        self.size=size  
    @staticmethod
    def target_size(w, h, size,ratio):
        w=int(w*ratio)
        h=int(h*ratio)
        size = (h,w)
        return size

    def __call__(self, img):
        size = self.size
        img_height = img.height
        img_width = img.width

        if (img_width / img_height) > 1:
            rate = size / img_width
        else:
            rate = size / img_height

        target_size = self.target_size(img_width, img_height, size, rate)
        return F.resize(img , target_size )





def get_transforms():

    transform_train = transforms.Compose([
                #CLAHE(),
                #Resize(img_size),#model.train_imsize config['size']
                #transforms.CenterCrop(img_size),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.216, 0.217, 0.198], std=[0.229, 0.216, 0.214])
            ])
    transform_val =  transforms.Compose([
                 #CLAHE(),
                 #Resize(img_size),# model.test_imsize
                 #transforms.CenterCrop(img_size),
                 transforms.ToTensor(),
                 #transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
                    
                 #transforms.Normalize(mean=[0.216, 0.217, 0.198], std=[0.229, 0.216, 0.214])
            ])
    return transform_train,transform_val
