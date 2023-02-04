import ipdb
import numpy as np

from ...utils.process import resize, translate_and_crop


class CropTri127Origin:
    """
    PatternMatch 資料集，
        z: 127
        x: 原圖
    """

    def __init__(self) -> None:
        pass

    def get_template(self, img, box, out_size, exemplar_size=127, context_amount=0.5, padding=(0, 0, 0)):
        box = box.squeeze()
        box_size = [(box[2] - box[0]), (box[3] - box[1])]
        wc_z = box_size[1] + context_amount * sum(box_size)
        hc_z = box_size[0] + context_amount * sum(box_size)
        crop_side = np.sqrt(wc_z * hc_z)
        # scale: 縮放比例
        scale = exemplar_size / crop_side
        img, box = resize(img, box, scale)

        # 計算 x, y 軸的位移距離
        img_size = [img.shape[1], img.shape[0]]
        x = out_size / 2 - img_size[0] / 2
        y = out_size / 2 - img_size[1] / 2
        avg_chans = np.mean(img, axis=(0, 1))
        img, box, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=out_size, padding=avg_chans)

        return img, box, scale

    def get_search(self, img, boxes, r):
        img, boxes = resize(img, boxes, r)
        return img

    def get_data(
        self,
        z_img,
        x_img,
        z_box,
        gt_boxes,
    ):
        z_img, z_box, r = self.get_template(
            z_img, z_box, out_size=127)
        x_img = self.get_search(
            x_img, gt_boxes, r)
        return z_img, x_img, z_box
