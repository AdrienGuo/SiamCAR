import torch
import os
import torchvision.transforms.functional as F
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import numpy as np
from torch.utils.data import DataLoader, Dataset
import re


class ImageFolderWithSelect(Dataset):#datasets.ImageFolder
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """ 
    
    def __init__(self, root,transform=None, target_transform=None,loader=default_loader) : 
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        #classes, class_to_idx = self._find_classes(root)
        #self.class_to_idx = class_to_idx
        imgs = self._make_dataset(root)
        self.imgs = imgs
        self.loder = default_loader
        
        
    
    def _find_classes(self,directory: str):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def _make_dataset(self,directory):
        imgs=[]
        train_name =[]
        
        directory = os.path.expanduser(directory)
        
        
        '''
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            
            if not os.path.isdir(target_dir):
                continue
       '''
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):#已排序
                if fname.endswith(('.jpg','.png','bmp')):
                    path = os.path.join(root, fname)
                    
                    #讀annotation
                    annotation=[]
                    anno_path = root+fname[:-4]+'.txt'
                    anno_path2 = root+fname[:-4]+'.label'
                    if os.path.isfile(anno_path):
                        check=0
                        f = open(anno_path,'r')
                        lines = f.readlines()
                        for line in lines:
                            line=line.strip('\n')
                            #line = re.sub("\[|\]","",line)
                            line=line.split(' ')
                            line = list(map(float, line))
                        
                            annotation.append(line)
                    elif os.path.isfile(anno_path2):
                        check=1
                        f = open(anno_path2,'r')
                        lines = f.readlines()
                        for line in lines:
                            line=line.strip('\n')
                            line=line.split(',')
                            line = list(line)
                            #if (float(line[1])>0) and (float(line[2])>0):
                            annotation.append(line)
                    
                        # cx,cy,w,h
                    
                    item = path, annotation,check
                    imgs.append(item)

        return imgs
    
        
    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        path, annotation,check = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img) 
             
        if self.target_transform is not None:
            annotation = self.target_transform(annotation)

        return img,annotation,path,check

