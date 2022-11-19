# 算 recall precision

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from pysot.utils.check_image import create_dir


def overlap_ratio_one(rect1, rect2):
    ''' Compute overlap ratio between two rects
    Args
        rect: 2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    # if rect1.ndim==1:
    #     rect1 = rect1[np.newaxis, :]
    # if rect2.ndim==1:
    #     rect2 = rect2[np.newaxis, :]
    left = np.maximum(rect1[0], rect2[0])
    right = np.minimum(rect1[0]+rect1[2], rect2[0]+rect2[2])
    top = np.maximum(rect1[1], rect2[1])
    bottom = np.minimum(rect1[1]+rect1[3], rect2[1]+rect2[3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[2]*rect1[3] + rect2[2]*rect2[3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou


def evaluate_amy(snapshot, dataset_dir):
    model_name = snapshot.split('/')[-2]
    print("Model name:", model_name)
    dataset_name = dataset_dir.split('/')[-1]
    sub_dir = os.path.join("results_amy", dataset_name, model_name)
    anno_dir= os.path.join(sub_dir, "annotation")  # 要拿來 eval 的標記檔 (由 test 產生)
    metric_dir = os.path.join(sub_dir, "metric")
    create_dir(metric_dir)

    target_all = []
    annotation_file = []

    TP_all = []
    FP_all = []
    TPtotal = []
    FPtotal = []
    sumtotal = 0

    # 讀標註檔
    for fname in os.listdir(anno_dir):
        if fname.endswith(('.txt')):
            filePath = os.path.join(anno_dir, fname)
            item = filePath, fname
            annotation_file.append(item)
    annotation_file = sorted(annotation_file)

    img_name = ''
    metric_save_path = os.path.join(metric_dir, "metric.txt")
    f_r = open(metric_save_path, 'w')
    for i in range(len(annotation_file)):
        pre_box = []
        anno_box = []
        ann_path, fname = annotation_file[i]
        f = open(ann_path, 'r')
        annotation = []
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            line = line.strip('=')
            line = re.sub("\[|\]", "", line)
            line = line.split(',')

            if line[0] != '':
                line = list(map(float, line))
                annotation.append(line)

        fname1 = fname.rsplit('__', 2)
        target = fname1[1]
        if target not in target_all:
            target_all.append(target)

        img_path = os.path.join(dataset_dir, fname1[0])
        im = Image.open(img_path)
        imw, imh = im.size
        img_name = fname1[0]

        if annotation:
            length = int(len(annotation[0])/4)
            for j in range(length):
                bbox = [annotation[0][0+j*4], annotation[0][1+j*4],
                        annotation[0][2+j*4], annotation[0][3+j*4]]
                pre_box.append(bbox)

            # 算anno box
            anno_path = os.path.join(dataset_dir, fname1[0][:-4] + '.txt')
            anno_path2 = os.path.join(dataset_dir, fname1[0][:-4] + '.label')
            if os.path.isfile(anno_path):
                f2 = open(anno_path, 'r')
                anno_lines = f2.readlines()
                for line in anno_lines:
                    line = line.strip('\n')
                    line = line.split(' ')
                    line = list(map(float, line))
                    if line[0] == float(target):
                        cx = line[1] * imw
                        cy = line[2] * imh
                        w = line[3] * imw
                        h = line[4] * imh
                        xmin = cx - w/2
                        ymin = cy - h/2
                        anno_box.append([xmin, ymin, w, h])
            elif os.path.isfile(anno_path2):
                f2 = open(anno_path2, 'r')
                anno_lines = f2.readlines()
                for line in anno_lines:
                    line = line.strip('\n')
                    line = line.split(',')
                    line = list(line)
                    if line[0] == target:
                        xmin = float(line[1])
                        ymin = float(line[2])
                        w = float(line[3]) - float(line[1])
                        h = float(line[4]) - float(line[2])
                        anno_box.append([xmin, ymin, w, h])

            pre_box = np.stack(pre_box).astype(np.float32)
            anno_box = np.stack(anno_box).astype(np.float32)
            # 算recall
            TP = torch.zeros(len(pre_box))
            FP = torch.zeros(len(pre_box))
            total_true_bboxes = len(anno_box)

            for d_idx in range(len(pre_box)):
                best_iou = 0
                for g_idx in range(len(anno_box)):
                    iou = overlap_ratio_one(pre_box[d_idx], anno_box[g_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                if best_iou >= 0.5:
                    TP[d_idx] = 1
                else:
                    FP[d_idx] = 1

            TP_sum = sum(TP)
            FP_sum = sum(FP)
            TPtotal.append(TP_sum)
            FPtotal.append(FP_sum)
            sumtotal = sumtotal + total_true_bboxes

            recall = TP_sum / total_true_bboxes
            precision = TP_sum / (TP_sum + FP_sum)

            TP_all.append([target, TP_sum, total_true_bboxes])
            FP_all.append([target, FP_sum])

    print("total recall:", str(sum(TPtotal) / sumtotal))
    print("total precision:", str(sum(TPtotal) / (sum(TPtotal) + sum(FPtotal))))
    print("====================================")
    f_r.write("total recall:" + str(sum(TPtotal) / sumtotal)+'\n')
    f_r.write("total precision:" + str(sum(TPtotal) /
              (sum(TPtotal) + sum(FPtotal)))+'\n')
    f_r.write("====================================" + '\n')

    metrics = dict()
    # tensor -> float
    metrics['precision'] = (sum(TPtotal) / (sum(TPtotal) + sum(FPtotal))).item() * 100
    metrics['recall'] = (sum(TPtotal) / sumtotal).item() * 100

    target_all = sorted(target_all)
    for t in range(len(target_all)):
        tp = []
        fp = []
        count = 0
        for r in range(len(TP_all)):
            if TP_all[r][0] == target_all[t]:
                tp.append(TP_all[r][1])
                count = count + TP_all[r][2]
            if FP_all[r][0] == target_all[t]:
                fp.append(FP_all[r][1])
        if len(tp):
            print("TP:", sum(tp))
            print("FP:", sum(fp))
            print("total recall_" +
                  str(target_all[t]) + ":" + str(sum(tp) / count))
            print("total precision_" +
                  str(target_all[t])+":"+str(sum(tp)/(sum(tp)+sum(fp))))
            print("count:", count)
            print("====================================")
            f_r.write("total recall_" +
                      str(target_all[t])+":"+str(sum(tp)/count)+'\n')
            f_r.write("total precision_" +
                      str(target_all[t])+":"+str(sum(tp)/(sum(tp)+sum(fp)))+'\n')
            f_r.write("count:"+str(count)+'\n')
            f_r.write("===================================="+'\n')
        else:
            print("total recall_"+str(target_all[t])+":", 0)
            print("total precision_"+str(target_all[t])+":", 0)
            print("====================================")
            f_r.write("total recall_"+str(target_all[t])+":", str(0)+'\n')
            f_r.write("total precision_"+str(target_all[t])+":", str(0)+'\n')
            f_r.write("===================================="+'\n')

    print(f"Save metrics to: {metric_save_path}")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking evaluation')
    parser.add_argument('--snapshot', type=str,
                        default='./snapshot/amy/checkpoint_e999.pth', help='snapshot of models to eval')
    parser.add_argument('--dataset', default='./datasets/train/allOld', type=str, help='dataset')
    args = parser.parse_args()

    evaluate_amy(args.snapshot, args.dataset)
