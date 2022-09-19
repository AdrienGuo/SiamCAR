# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.append('../')

import ipdb
import torchvision.transforms.functional as F
from PIL import Image
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets.testdata import ImageFolderWithSelect
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--datasetName', type=str, default='PCB',help='dataset name')#OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--snapshot', type=str, default='./snapshot/checkpoint_e999.pth',help='snapshot of models to eval')
parser.add_argument('--cfg', type=str, default='./experiments/siamcar_r50/config_test.yaml',help='config file')
parser.add_argument('--dataset', default='./dataset/testing_dataset/DATA/', type=str, help='dataset')
args = parser.parse_args()

torch.set_num_threads(1)


def main():
     # load config
    cfg.merge_from_file(args.cfg)

    # hp_search
    params = getattr(cfg.HP_SEARCH,args.datasetName)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset
    dataset = ImageFolderWithSelect(args.dataset)
    assert len(dataset) != 0, "dataset is empty!!"

    model_name = args.snapshot.split('/')[-2] + '_'+str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])
    print("model:",model_name)

    for idx, (_,annotation,path,check) in enumerate(dataset):
        path1 = path.split('/')
        name = path1[-1]
        count = 0
        print("name:",name)
        for i in range (len(annotation)):
            annotation[i][1]= float(annotation[i][1])
            annotation[i][2]= float(annotation[i][2])
            annotation[i][3]= float(annotation[i][3])
            annotation[i][4]= float(annotation[i][4])
            tic = cv2.getTickCount()
            toc = 0

            pred_bboxes = []

            #init
            count +=1
            classid = annotation[i][0]
            
            if annotation[i][0]!=26:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if check==0:
                    gt_bbox = [annotation[i][1]*img.shape[1],annotation[i][2]*img.shape[0],annotation[i][3]*img.shape[1],annotation[i][4]*img.shape[0]]
                    gt_bbox = [gt_bbox[0]-gt_bbox[2]/2,gt_bbox[1]-gt_bbox[3]/2,gt_bbox[2],gt_bbox[3]]
                else:
                    gt_bbox = [annotation[i][1],annotation[i][2],annotation[i][3]-annotation[i][1],annotation[i][4]-annotation[i][2]]
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-w/2, cy-h/2, w, h]
                init_info = {'init_bbox':gt_bbox_}
                

                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)


                #track
                outputs = tracker.track(img,hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                toc += cv2.getTickCount() - tic


                model_path = os.path.join('results', args.datasetName, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)

                # save results
                temp = name+"__"+str(classid)+"__"+str(count)
                result_path = os.path.join(model_path, '{}.txt'.format(name+"__"+str(classid)+"__"+str(count)))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+"="+'\n')
                toc /= cv2.getTickFrequency()

                print('Time: {:5.1f}s Speed: {:3.1f}fps'.format(toc, 1 / toc))


if __name__ == '__main__':
    main()
