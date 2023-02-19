import os

import colorama
import cv2
import ipdb
import numpy as np
import torch
from colorama import Fore
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from pysot.core.config import cfg
from pysot.datasets.augmentation.pcb_aug import PCBAugmentation
from pysot.datasets.collate import collate_fn
from pysot.datasets.pcbdataset import get_pcbdataset
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.check_image import draw_box, draw_preds
from pysot.utils.model_load import load_pretrain
from tools.evaluate import calculate_metrics
from utils.file_organizer import create_dir, save_img
from utils.painter import draw_boxes

from .utils import create_img_dir, save_fail_img, save_heatmap, tensor_to_numpy

colorama.init(autoreset=True)


class Demoer(object):
    def __init__(self, args) -> None:
        self.args = args

        # Config
        siamcar_cfg = "./experiments/siamcar_r50/config.yaml"
        cfg.merge_from_file(siamcar_cfg)
        cfg.merge_from_file(args.cfg)

        self.tracker: SiamCARTracker = None
        self.demo_loader: DataLoader = None
        self.save_dir: str = None
        self.fail_dir: str = None

    def build_tracker(self, model_path):
        print(f"Loading model from: {model_path}")
        model = ModelBuilder(self.args.method)
        model = load_pretrain(model, model_path).cuda()
        model.eval()
        self.tracker = SiamCARTracker(model, cfg.TRACK)

    def build_dataloader(self):
        # Datasets arguments
        data_args = {
            'data_path': self.args.data,
            'method': self.args.method,
            'criteria': self.args.criteria,
            'bg': self.args.bg,
            'target': self.args.target,
        }

        # Data augmentations
        data_augmentation = {
            'template': PCBAugmentation(cfg.TEST.DATASET.TEMPLATE),
            'search': PCBAugmentation(cfg.TEST.DATASET.SEARCH),
        }

        print("Building dataset...")
        pcbdataset = get_pcbdataset(self.args.dataset)
        dataset = pcbdataset(data_args, mode="evaluate",
                             augmentation=data_augmentation)
        assert len(dataset) != 0, "ERROR, dataset is empty!!"
        print(f"Demo dataset size: {len(dataset)}")
        self.demo_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn)

    def create_save_dirs(self):
        model_dir = self.args.model.split('/')[-2]
        model_ckpt = self.args.model.split('/')[-1].rsplit('.', 1)[0]
        save_dir = os.path.join(
            "./results", self.args.dataset,
            self.args.train_date, self.args.cfg_name,
            # "official_model",
            self.args.part, self.args.criteria, self.args.target, self.args.method,
            model_dir, model_ckpt)
        create_dir(save_dir)
        print(f"Demo results saved to: {save_dir}")

        # PatternMatch_test 因為沒有標籤，沒有辦法判斷是否 fail
        fail_dir = None
        if self.args.dataset != "PatternMatch_test" and self.args.dataset != "tmp":
            fail_dir = os.path.join(save_dir, "FAILED")
            create_dir(fail_dir)
            print(f"Failed results saved to: {fail_dir}")

        self.save_dir = save_dir
        self.fail_dir = fail_dir

    def demo(self) -> list:
        # hp_search
        params = getattr(cfg.HP_SEARCH, "PCB")
        hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

        all_pred_boxes = list()
        all_gt_boxes = list()

        with torch.no_grad():
            for idx, data in enumerate(self.demo_loader):
                img_name = data['img_name'][0]
                img_path = data['img_path'][0]
                z_img = data['z_img'].cuda()
                x_img = data['x_img'].cuda()
                gt_boxes = data['gt_boxes']  # (N, [0, x1, y1, x2, y2])
                z_box = data['z_box'][0]  # ([x1, y1, x2, y2])

                print(f"Load image from: {Fore.GREEN}{img_path}")
                path_dir_map = create_img_dir(self.save_dir, img_name)

                # PatternMatch_test 資料集的時候不能加，因為 img_path 的路徑是 dir 不是 path
                img = None
                if self.fail_dir:
                    img = cv2.imread(img_path)
                    # Save original image
                    origin_path = os.path.join(
                        path_dir_map['origin'], f"{img_name}.jpg")
                    save_img(img, origin_path)

                pred_boxes = []
                # TODO
                # 調整 z_box, gt_boxes 的框框，tracker.init() 的格式需要
                z_box = z_box.squeeze()
                z_box[2] = z_box[2] - z_box[0]  # x2 -> w
                z_box[3] = z_box[3] - z_box[1]  # y2 -> h
                pred_boxes.append(z_box)

                gt_boxes = gt_boxes[:, 1:]  # 不要 0 那項
                gt_boxes = gt_boxes.cpu().numpy()
                gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]  # x2 -> w
                gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]  # y2 -> h

                # TODO: with autocast():
                z_img = self.tracker.init(z_img, bbox=z_box)
                outputs = self.tracker.track(x_img, hp)

                # Save z_img
                z_img = tensor_to_numpy(z_img)
                save_img(z_img,
                         os.path.join(path_dir_map['template'], f"{idx}.jpg"))

                # Save x_img
                x_img = outputs['x_img']
                x_img = tensor_to_numpy(x_img)
                save_img(x_img,
                         os.path.join(path_dir_map['search'], f"{idx}.jpg"))

                # Save cen, cls, score heatmaps
                heatmap_cen = save_heatmap(
                    outputs['cen'], x_img, path_dir_map['heatmap_cen'], idx)
                heatmap_cls = save_heatmap(
                    outputs['cls'], x_img, path_dir_map['heatmap_cls'], idx)
                heatmap_score = save_heatmap(
                    outputs['score'], x_img, path_dir_map['heatmap_score'], idx)

                # pred_boxes on x_img
                for box in outputs['pred_boxes']:
                    box = np.around(box, decimals=2)
                    pred_boxes.append(box)

                # Save annotation on x_img
                scores = np.around(outputs['top_scores'], decimals=2)
                anno_path = os.path.join(
                    path_dir_map['annotation'], f"{idx}.txt")
                with open(anno_path, 'w') as f:
                    # template
                    f.write(', '.join(map(str, pred_boxes[0])) + '\n')
                    # preds
                    for i, x in enumerate(pred_boxes[1:]):
                        # format: [x1, y1, w, h]
                        f.write(', '.join(map(str, x)) +
                                ', ' + str(scores[i]) + '\n')
                print(f"Save annotation result to: {anno_path}")

                # gt_boxes on x_img
                pred_img = draw_box(x_img, gt_boxes, type="gt")
                # pred_boxes on x_img
                pred_img = draw_preds(pred_img, scores, anno_path, idx)
                if pred_img is None:  # 如果沒偵測到物件，存 x_img
                    pred_img = x_img
                pred_path = os.path.join(path_dir_map['pred'], f"{idx}.jpg")
                save_img(pred_img, pred_path)

                # 因為 PatternMatch_test 資料集沒有標籤，不能去算 precision, recall
                if self.fail_dir:
                    precision, recall = calculate_metrics(
                        [outputs['pred_boxes']], [gt_boxes.tolist()])
                    if precision != 1 or recall != 1:
                        save_fail_img(
                            self.fail_dir, img_name, img, z_img, x_img, pred_img, heatmap_cen, heatmap_cls, heatmap_score, idx
                        )
                    # For evaluating
                    all_pred_boxes.append(outputs['pred_boxes'])
                    all_gt_boxes.append(gt_boxes.tolist())

    def evaluate(self, all_pred_boxes, all_gt_boxes):
        if self.fail_dir:
            precision, recall = calculate_metrics(all_pred_boxes, all_gt_boxes)
            precision = precision * 100
            recall = recall * 100
            print(f"Recall: {recall}")
            print(f"Precision: {precision}")
        else:
            print("Can't evaluate on PatternMatch_test dataset")
