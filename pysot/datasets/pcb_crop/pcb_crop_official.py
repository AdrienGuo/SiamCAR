import ipdb
import numpy as np

from pysot.datasets.utils.process import resize, translate_and_crop


class PCBCropOfficial:
    """ 這是搭配 原paper (official) 的方法使用的。
    """
    def __init__(self, template_size, search_size, shift) -> None:
        self.z_size = template_size
        self.x_size = search_size
        self.shift = shift  # spatial aware sampling

    def make_img511(self, img, box, out_size, exemplar_size=127, context_amount=0.5, padding=(0, 0, 0)):
        """ 因為論文的作法是先做出一張 (511, 511, 3) 的影像，
            再去從這張影像上面切出 template, search。
        作法：
            先用 box 根據一套公式轉換，算出 crop_side，
            這個 crop_side 要變成 exemplar_size，所以得到 scale，
            最後就是對原圖做 scale 倍的縮放和裁切後回傳。
        """
        box = box.squeeze()
        # 裁切的公式算法，原始作法要去看 [SiamFC](https://arxiv.org/pdf/1606.09549.pdf)
        # 但其實 SiamCAR 這裡和原論文的作法不太一樣，不過最後結果應該是一樣的...吧？
        # 公式: crop_side = ((w + p) × (h + p)) ^ 1/2
        #                   p = (w + h) / 2
        gt_size = [(box[2] - box[0]), (box[3] - box[1])]
        wc_z = gt_size[1] + context_amount * sum(gt_size)
        hc_z = gt_size[0] + context_amount * sum(gt_size)
        crop_side = np.sqrt(wc_z * hc_z)
        # scale: 縮放比例
        scale = exemplar_size / crop_side
        img, box = resize(img, box, scale)

        # x, y 軸的位移距離
        x = out_size / 2 - (box[0] + box[2]) / 2
        y = out_size / 2 - (box[1] + box[3]) / 2
        img, box, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=out_size, padding=padding)
        return img, box

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def get_template(self, img, box, padding=(0, 0, 0)):
        # x, y 軸的位移距離
        x = self.z_size / 2 - (box[0] + box[2]) / 2
        y = self.z_size / 2 - (box[1] + box[3]) / 2
        img, _, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=self.z_size, padding=padding)
        return img

    def get_search(self, img, box, padding=(0, 0, 0)):
        # x, y 軸的位移距離
        x = self.x_size / 2 - (box[0] + box[2]) / 2
        y = self.x_size / 2 - (box[1] + box[3]) / 2
        # spatial aware sampling
        # 要去看 [SiamRPN++](https://arxiv.org/pdf/1812.11703.pdf)
        x = x + PCBCropOfficial.random() * self.shift
        y = y + PCBCropOfficial.random() * self.shift
        img, box, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=self.x_size, padding=padding)
        return img, box

    def get_data(
        self,
        img,
        z_box,
        gt_boxes,
        padding
    ):
        # img511: (511, 511, 3)
        img511, z_box511 = self.make_img511(
            img, z_box, out_size=511, padding=padding)

        # z_img: (127, 127, 3)
        z_img = self.get_template(
            img511, z_box511, padding=padding)
        # search image 要使用 spatial aware sampling
        # x_img: (255, 255, 3)
        x_img, z_box = self.get_search(
            img511, z_box511, padding=padding)

        z_box = z_box[np.newaxis, :]
        gt_boxes = z_box  # 官方的作法只有單物件，所以兩個一樣
        return z_img, x_img, z_box, gt_boxes
