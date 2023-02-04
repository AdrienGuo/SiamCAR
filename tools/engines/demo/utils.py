import os

import numpy as np

from utils.file_organizer import create_dir, get_path, save_fig, save_img
from utils.painter import draw_heatmap


def tensor_to_numpy(tensor):
    # (B=1, C=3, H, W) -> (C=3, H, W)
    numpy = tensor.cpu().numpy().squeeze()
    # (3, H, W) -> (H, W, 3)
    numpy = np.transpose(numpy, (1, 2, 0))
    return numpy


def save_tensor_img(tensor_img, dir, idx):
    img = tensor_to_numpy(tensor_img)
    path = os.path.join(dir, f"{idx}.jpg")
    save_img(img, path)


def create_img_dir(save_dir, img_name) -> dict:
    sub_dir = os.path.join(save_dir, img_name)
    create_dir(sub_dir)

    origin_dir = create_dir(sub_dir, "origin")
    z_dir = create_dir(sub_dir, "template")
    x_dir = create_dir(sub_dir, "search")
    anno_dir = create_dir(sub_dir, "pred_annotation")
    pred_dir = create_dir(sub_dir, "pred")
    heatmap_cen_dir = create_dir(sub_dir, ['heatmap', 'cen'])
    heatmap_cls_dir = create_dir(sub_dir, ['heatmap', 'cls'])
    heatmap_score_dir = create_dir(sub_dir, ['heatmap', 'score'])

    dir_path_map = dict()
    dir_path_map['origin'] = origin_dir
    dir_path_map['template'] = z_dir
    dir_path_map['search'] = x_dir
    dir_path_map['annotation'] = anno_dir
    dir_path_map['pred'] = pred_dir
    dir_path_map['heatmap_cen'] = heatmap_cen_dir
    dir_path_map['heatmap_cls'] = heatmap_cls_dir
    dir_path_map['heatmap_score'] = heatmap_score_dir
    return dir_path_map


def save_heatmap(heatmap: np.ndarray, img: np.ndarray, dir: str, idx: int) -> np.ndarray:
    heatmap = draw_heatmap(img, heatmap)
    heatmap_path = os.path.join(dir, f"{idx}.jpg")
    save_fig(heatmap, heatmap_path)
    return heatmap


def save_fail_img(fail_dir, img_name, img, z_img, x_img, pred_img, cen_heatmap, cls_heatmap, score_heatmap, idx):
    path_dir_map = create_img_dir(fail_dir, img_name)

    origin_path = get_path(path_dir_map['origin'], f"{img_name}.jpg")
    z_path = get_path(path_dir_map['template'], f"{idx}.jpg")
    x_path = get_path(path_dir_map['search'], f"{idx}.jpg")
    pred_path = get_path(path_dir_map['pred'], f"{idx}.jpg")
    cen_heatmap_path = get_path(path_dir_map['heatmap_cen'], f"{idx}.jpg")
    cls_heatmap_path = get_path(path_dir_map['heatmap_cls'], f"{idx}.jpg")
    score_heatmap_path = get_path(path_dir_map['heatmap_score'], f"{idx}.jpg")
    save_img(img, origin_path)
    save_img(z_img, z_path)
    save_img(x_img, x_path)
    save_img(pred_img, pred_path)
    save_fig(cen_heatmap, cen_heatmap_path)
    save_fig(cls_heatmap, cls_heatmap_path)
    save_fig(score_heatmap, score_heatmap_path)
