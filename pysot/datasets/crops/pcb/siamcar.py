import ipdb
import numpy as np

from ...utils.process import resize, translate_and_crop


class PCBCropCAR:
    """這是 SiamCAR 使用的方法"""

    def __init__(self, template_size, search_size, template_shift=0, search_shift=0) -> None:
        self.z_size = template_size
        self.x_size = search_size
        self.z_shift = template_shift
        self.x_shift = search_shift  # spatial aware sampling

    def make_img511(self, img, box, out_size, exemplar_size=127, context_amount=0.5, padding=(0, 0, 0)):
        """
        因為論文的作法是先做出一張 (511, 511, 3) 的影像，
        再去從這張影像上面切出 template, search。
        作法：
            先用 box 根據一套公式轉換，算出 crop_side，
            這個 crop_side 要變成 exemplar_size，所以得到 rate，
            最後就是對原圖做 rate 倍的縮放和裁切後回傳。
        """

        # 裁切的公式算法，原始作法要去看 [SiamFC](https://arxiv.org/pdf/1606.09549.pdf)
        # 但其實 SiamCAR 這裡和原論文的作法不太一樣，不過最後結果應該是一樣的...吧？
        # 公式: crop_side = ((w + p) × (h + p)) ^ 1/2
        #                   p = (w + h) / 2
        gt_size = [(box[2] - box[0]), (box[3] - box[1])]
        wc_z = gt_size[1] + context_amount * sum(gt_size)
        hc_z = gt_size[0] + context_amount * sum(gt_size)
        crop_side = np.sqrt(wc_z * hc_z)
        # rate: 縮放比例
        rate = exemplar_size / crop_side
        img, box = resize(img, box, rate)

        # x, y 軸的位移距離
        x = out_size / 2 - (box[0] + box[2]) / 2
        y = out_size / 2 - (box[1] + box[3]) / 2
        img, box, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=out_size, padding=padding)
        return img, box

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    @staticmethod
    def _spatial_aware_sampling(x, y, shift):
        # spatial aware sampling
        # 要去看 [SiamRPN++](https://arxiv.org/pdf/1812.11703.pdf)
        x = x + PCBCropCAR.random() * shift
        y = y + PCBCropCAR.random() * shift
        return x, y

    def _make_template(self, img, box):
        # x, y 軸的位移距離
        x = self.z_size / 2 - (box[0] + box[2]) / 2
        y = self.z_size / 2 - (box[1] + box[3]) / 2
        x, y = self._spatial_aware_sampling(x, y, self.z_shift)
        avg_chans = np.mean(img, axis=(0, 1))

        img, _, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=self.z_size, padding=avg_chans)
        return img

    def _make_search(self, img, box):
        # x, y 軸的位移距離
        x = self.x_size / 2 - (box[0] + box[2]) / 2
        y = self.x_size / 2 - (box[1] + box[3]) / 2
        x, y = self._spatial_aware_sampling(x, y, self.x_shift)
        avg_chans = np.mean(img, axis=(0, 1))

        img, box, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=self.x_size, padding=avg_chans)
        return img, box

    def get_template(self, img, box):
        img = self._make_template(img, box)
        return img

    def get_search(self, img, box):
        img, box = self._make_search(img, box)
        return img, box

    def get_data(
        self,
        img,
        z_box,
        gt_boxes,
    ):
        z_box = z_box.squeeze()

        # img511: (511, 511, 3)
        avg_chans = np.mean(img, axis=(0, 1))
        img511, z_box511 = self.make_img511(
            img, z_box, out_size=511, padding=avg_chans)

        # 這個 avg_chans 會導致兩種補的 padding 顏色不一樣
        avg_chans = np.mean(img511, axis=(0, 1))
        # z_img: (127, 127, 3)
        # x_img: (255, 255, 3)
        # search image 要使用 spatial aware sampling
        z_img = self.get_template(img511, z_box511)
        x_img, z_box = self.get_search(img511, z_box511)

        # These 3 lines for using the SiamCAR official crop method.
        z_img = img511
        x_img = img511
        z_box = z_box511

        z_box = z_box[np.newaxis, :]
        gt_boxes = z_box  # 官方的作法只有單物件，所以兩個一樣

        return z_img, x_img, z_box, gt_boxes
