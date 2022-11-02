import ipdb
import torch


def collate_fn_new(batch):
    img_path, z_img, x_img, cls, box, z_box, r = zip(*batch)

    for i, l in enumerate(box):
        l[:, 0] = i  # add target image index for build_targets()

    return {
        'img_path': img_path,
        'z_img': torch.stack(z_img, 0),
        'x_img': torch.stack(x_img, 0),
        'gt_cls': torch.stack(cls, 0),
        'gt_boxes': torch.cat(box, 0),
        'z_box': z_box,
        'scale': r
    }
