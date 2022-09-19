#畫bbox
from PIL import Image , ImageDraw , ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse
import torch
import torchvision.transforms.functional as F
from torchvision import transforms


parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--dataset', type=str, default='/tf/SiamCAR/testing_dataset/DATA/',help='dataset path')
parser.add_argument('--annotation', type=str, default='/tf/SiamCAR/results/UAV123/SiamCAR_epoch1000_batch16_searchSize255_output25_pretrained200_searchTrans__tempsCrop_iou_checkpoint999_1200_text_0.4_0.2_0.3/',help='annotation path')

parser.add_argument('--save', type=str, default='/tf/SiamCAR/tools/result/',help='save path')

args = parser.parse_args()

torch.set_num_threads(1)

 
def pil_draw():
    imgs=[]
    names=[]
    annotation_file=[]
    #讀圖片檔
    for root, _, fnames in sorted(os.walk(args.dataset, followlinks=True)):
        for fname in sorted(fnames):
            if fname.endswith(('.jpg','.png','bmp')):
                filePath = os.path.join(args.dataset, fname)
                imgs.append(filePath)
                names.append(fname)
    imgs=sorted(imgs)
    names=sorted(names)
    
    #讀標註檔
    for fname in os.listdir(args.annotation):
        #if fname.endswith(('.txt')):
        filePath = os.path.join(args.annotation, fname)
        item = filePath,fname
        annotation_file.append(item)
    annotation_file=sorted(annotation_file)
   
    
    #畫圖
    img_name=''
    for i in range(len(annotation_file)):
        ann_path , fname = annotation_file[i]
        f = open(ann_path,'r')
        annotation=[]
        lines = f.readlines()[1:]
        for line in lines:
            line=line.strip('\n')
            line=line.strip('=')
            line = re.sub("\[|\]","",line)
            line=line.split(',')
            if line[0] == "":
                line=[0,0,0,0]
                annotation.append(line)
            else:
                line = list(map(float, line));
                annotation.append(line)
        
        fname1 = fname.split('__')
        im = Image.open(args.dataset+fname1[0])
        img_name = fname1[0]
        draw = ImageDraw.Draw(im)
        
        length = int(len(annotation[0])/4)
        for j in range (length):
            bbox = [annotation[0][0+j*4],annotation[0][1+j*4],annotation[0][2+j*4],annotation[0][3+j*4]]       
            draw.rectangle([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]],outline='red',width=4)

        del draw
        if not os.path.exists(args.save+img_name[:-4]):
            os.makedirs(args.save+img_name[:-4])

        im.save(args.save+img_name[:-4]+"/"+fname[:-4]+".jpg")
        #im2 = Image.open(args.save+img_name[:-4]+"/"+fname[:-4]+".jpg")
        #plt.imshow(im2)
        #plt.show()

if __name__ == '__main__':
    pil_draw()
