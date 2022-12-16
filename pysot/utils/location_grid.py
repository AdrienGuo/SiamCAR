import ipdb
import torch


def compute_locations(cls_, stride, x_img):
    h, w = cls_.size()[-2:]
    x_img_h, x_img_w = x_img.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride, cls_.device, x_img_h, x_img_w
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, device, x_img_h, x_img_w):
    grids_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    grids_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    grids_y, grids_x = torch.meshgrid((grids_y, grids_x))
    # width & height has their own shift value
    # 位移 = (原圖大小 - (score大小-1)*8) // 2
    shift_x = (x_img_w - (w - 1) * 8) // 2  # x 軸的起始點
    shift_y = (x_img_h - (h - 1) * 8) // 2  # y 軸的起始點
    grids_x = grids_x.reshape(-1) + shift_x
    grids_y = grids_y.reshape(-1) + shift_y
    # locations = torch.stack((grids_x, grids_y), dim=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = torch.stack((grids_x, grids_y), dim=1) + stride
    locations = torch.stack((grids_x, grids_y), dim=1)    # alex:48 // 32
    return locations
