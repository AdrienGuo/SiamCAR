import ipdb
import torch


def collate_fn(batch):
    # 其實就跟寫 for loop 用的那個 zip 原理一樣，超強的。
    img_name, img_path, z_img, x_img, gt_cls, gt_boxes, z_box, r \
        = zip(*batch)

    for i, l in enumerate(gt_boxes):
        l[:, 0] = i  # add target image index for build_targets()

    return {
        'img_name': img_name,
        'img_path': img_path,
        'z_img': torch.stack(z_img, 0),
        'x_img': torch.stack(x_img, 0),
        'gt_cls': torch.stack(gt_cls, 0),
        'gt_boxes': torch.cat(gt_boxes, 0),
        'z_box': z_box,
        'scale': r
    }
