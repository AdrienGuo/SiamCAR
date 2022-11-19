# 畫bbox
import argparse
import os
import re

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from pysot.utils.check_image import create_dir


torch.set_num_threads(1)


def pil_draw(snapshot, dataset_dir):
    model_name = snapshot.split('/')[-2]
    print("Model name:", model_name)
    dataset_name = dataset_dir.split('/')[-1]
    anno_dir = os.path.join("results_amy", dataset_name, model_name, "annotation")
    save_dir = f"./results_amy/{dataset_name}/{model_name}/image/"
    create_dir(save_dir)

    imgs = []
    names = []
    annotation_file = []
    # 讀圖片檔
    for root, _, fnames in sorted(os.walk(args.dataset, followlinks=True)):
        for fname in sorted(fnames):
            if fname.endswith(('.jpg', '.png', 'bmp')):
                filePath = os.path.join(args.dataset, fname)
                imgs.append(filePath)
                names.append(fname)
    imgs = sorted(imgs)
    names = sorted(names)

    # 讀標註檔
    for fname in os.listdir(anno_dir):
        # if fname.endswith(('.txt')):
        filePath = os.path.join(anno_dir, fname)
        item = filePath, fname
        annotation_file.append(item)
    annotation_file = sorted(annotation_file)

    # 畫圖
    img_name = ''
    for i in range(len(annotation_file)):
        ann_path, fname = annotation_file[i]
        f = open(ann_path, 'r')
        annotation = []
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            line = line.strip('=')
            line = re.sub("\[|\]", "", line)
            line = line.split(',')
            if line[0] == "":
                line = [0, 0, 0, 0]
                annotation.append(line)
            else:
                line = list(map(float, line))
                annotation.append(line)

        fname_split = fname.rsplit('__', 2)
        img_path = os.path.join(args.dataset, fname_split[0])
        im = Image.open(img_path)
        img_name = fname_split[0]
        draw = ImageDraw.Draw(im)

        length = int(len(annotation[0])/4)
        for j in range(length):
            bbox = [annotation[0][0+j*4], annotation[0][1+j*4],
                    annotation[0][2+j*4], annotation[0][3+j*4]]
            draw.rectangle([bbox[0], bbox[1], bbox[2]+bbox[0],
                           bbox[3]+bbox[1]], outline=(0, 255, 0), width=4)  # type: ignore

        del draw
        if not os.path.exists(save_dir + img_name[:-4]):
            os.makedirs(save_dir + img_name[:-4])
            print(f"Create new dir: {save_dir + img_name[:-4]}")

        save_path = save_dir + img_name[:-4] + "/" + fname[:-4] + ".jpg"
        im.save(save_path)
        print(f"Save result image to: {save_path}")

        # ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='siamcar tracking')
    parser.add_argument('--dataset', default='./datasets/train/allOld', type=str, help='dataset')
    parser.add_argument('--snapshot', type=str,
                        default='./snapshot/amy/checkpoint_e999.pth', help='snapshot of models to eval')
    args = parser.parse_args()

    pil_draw(args.snapshot, args.dataset)
