# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import json
import logging
import math
import os
import random
import time

import ipdb
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import wandb
from pysot.core.config import cfg
from pysot.datasets.collate import collate_fn
# 選擇 new / origin 的裁切方式
# from pysot.datasets.pcbdataset.pcbdataset import PCBDataset
from pysot.datasets.pcbdataset import get_pcbdataset
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.average_meter import AverageMeter
from pysot.utils.check_image import create_dir
from pysot.utils.distributed import (average_reduce, dist_init, get_rank,
                                     get_world_size, reduce_gradients)
from pysot.utils.log_helper import add_file_handler, init_log, print_speed
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.misc import commit, describe
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.SetCallBack import SetCallback
from tools.engine import save_checkpoint, train_one_epoch, validate
from tools.engines.trainer import Trainer
from tools.evaluate import evaluate
from tools.eval_amy import evaluate_amy
from tools.test_amy import test_amy

# from convertOnnx import convert_onnx
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger('global')
# parse arguments
parser = argparse.ArgumentParser(description='SiamFC training arguments')
parser.add_argument('--date', type=str, help='Date')
parser.add_argument('--dataset', type=str, help='all / tmp')
parser.add_argument('--data', type=str, default='', help='path to train data')
parser.add_argument('--criteria', type=str, default='',
                    help='all / big / mid / small')
parser.add_argument('--method', type=str, default='', help='official / origin')
parser.add_argument('--bg', type=str, default='', help='background')
parser.add_argument('--test_data', type=str, default='',
                    help='path to test data')
parser.add_argument('--eval_criteria', type=str, default='',
                    help='all / big / mid / small')
parser.add_argument('--eval_method', type=str,
                    default='', help='official_origin')
parser.add_argument('--eval_bg', type=str, default='', help='background')
parser.add_argument('--target', type=str, default='', help='one / multi')
parser.add_argument('--cfg_name', type=str, default='',
                    help='File name of config file')
parser.add_argument('--cfg', type=str,
                    default='./experiments/siamcar_r50/config.yaml', help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


class SiamCARTrainer(object):
    def __init__(self):
        self._seed_torch(args.seed)

        rank, world_size = dist_init()

        if rank == 0:
            if not os.path.exists(cfg.TRAIN.LOG_DIR):
                os.makedirs(cfg.TRAIN.LOG_DIR)
            init_log('global', logging.INFO)
            # PermissionError
            # if cfg.TRAIN.LOG_DIR:
            #     add_file_handler('global',
            #                      os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
            #                      logging.INFO)

        # create tensorboard writer
        if rank == 0 and cfg.TRAIN.LOG_DIR:
            pass
            # PermissionError
            # self.tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
        else:
            self.tb_writer = None

        self.stop_training = False

    def _seed_torch(self, seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # 蔡杰他們在討論，跟加速相關的東東
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def CreateDatasets(self, validation_split: float = 0.0, random_seed: int = 42):
        logger.info("Building dataset...")
        pcb_dataset = get_pcbdataset(args.method)
        # 現在 neg = 0，所以 train_dataset 可以沿用
        dataset = pcb_dataset(args, "train")
        test_dataset = pcb_dataset(args, "test")

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.seed(random_seed)
        random.shuffle(indices)
        split = dataset_size - int(np.floor(validation_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        assert len(train_dataset) != 0, "ERROR, dataset is empty!!"

        def create_loader(dataset, batch_size, num_workers):
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False,
                collate_fn=collate_fn
            )
            return data_loader

        self.train_loader = create_loader(
            train_dataset, args.batch_size, cfg.TRAIN.NUM_WORKERS)
        self.train_eval_loader = create_loader(
            train_dataset, batch_size=1, num_workers=8)
        # self.val_loader = create_loader(val_dataset, args.batch_size, cfg.TRAIN.NUM_WORKERS)
        # self.val_eval_loader = create_loader(val_dataset, batch_size=1, num_workers=8)
        self.test_loader = create_loader(
            test_dataset, args.batch_size, cfg.TRAIN.NUM_WORKERS)
        self.test_eval_loader = create_loader(
            test_dataset, batch_size=1, num_workers=8)

    def SetTrainingCallback(self):
        nTrainSample = self.getTrainSampleNumber()
        TargetTrainBatchSize = self.getTrainBatch()
        TargetEpoch = self.getTrainEpoch()
        # Training Progress
        train_stepPerEpoch = math.ceil(nTrainSample / TargetTrainBatchSize)
        self.cb = SetCallback()

    def build_opt_lr(self, current_epoch=0):
        if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
            print("Train backbone")
            for layer in cfg.BACKBONE.TRAIN_LAYERS:
                for param in getattr(self.model.backbone, layer).parameters():
                    param.requires_grad = True
                for m in getattr(self.model.backbone, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        else:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for m in self.model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        trainable_params = []
        trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                               self.model.backbone.parameters()),
                              'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
        if cfg.ADJUST.ADJUST:
            trainable_params += [{'params': self.model.neck.parameters(),
                                  'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': self.model.car_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': self.model.down.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

        self.optimizer = torch.optim.SGD(trainable_params,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer, epochs=args.epoch)
        # Warning:
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        # self.lr_scheduler.step(cfg.TRAIN.START_EPOCH)

        # Prevent underflowing gradients
        self.scaler = GradScaler()

    def CompilModel(self):
        # create model
        self.model = ModelBuilder().cuda()

        # load pretrained backbone weights
        if cfg.BACKBONE.PRETRAINED:
            cur_path = os.path.dirname(os.path.realpath(__file__))
            backbone_path = os.path.join(
                cur_path, '../', cfg.BACKBONE.PRETRAINED)
            load_pretrain(self.model.backbone, backbone_path)

        # build optimizer and lr_scheduler
        self.build_opt_lr(cfg.TRAIN.START_EPOCH)

        # resume training
        print("cfg.TRAIN.RESUME:", cfg.TRAIN.RESUME)
        if cfg.TRAIN.RESUME:
            print("Resume")
            logger.info("resume from {}".format(cfg.TRAIN.RESUME))
            assert os.path.isfile(cfg.TRAIN.RESUME), \
                '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
            self.model, self.optimizer, cfg.TRAIN.START_EPOCH = \
                restore_from(self.model, self.optimizer,
                             ckpt_path=cfg.TRAIN.RESUME)
        # load pretrain
        elif cfg.TRAIN.PRETRAINED:
            print("pretrained")
            load_pretrain(self.model, cfg.TRAIN.PRETRAINED)

    def StartFit(self):
        model_name = \
            f"CLAHE_noPre_{cfg.TRAIN.CLS_LOSS_METHOD}_{args.dataset_name}_{args.criteria}" \
            f"_{args.target}_{args.method}" \
            f"_{cfg.TRAIN.CEN_WEIGHT}_{cfg.TRAIN.CLS_WEIGHT}_{cfg.TRAIN.LOC_WEIGHT}" \
            f"_neg{args.neg}_x{cfg.TRAIN.SEARCH_SIZE}_bg{args.bg}_e{args.epoch}_b{args.batch_size}"

        constants = {
            'Dataset': args.dataset_name,
            'Criteria': args.criteria,
            'Train Dataset Size': len(self.train_loader.dataset),
            'Test Dataset Size': len(self.test_loader.dataset),
            'Validation Ratio': cfg.DATASET.VALIDATION_SPLIT,
            'Method': args.method,
            'Negative Ratio': args.neg,
            'Background': args.bg,
            'Epochs': args.epoch,
            'Batch': args.batch_size,
            'Accumulate iter': args.accum_iters,
            'lr': cfg.TRAIN.LR.KWARGS.start_lr,
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
            'Model Pretrained': cfg.TRAIN.PRETRAINED,
            'Backbone Pretrained': cfg.BACKBONE.PRETRAINED,
            'Backbone Train Epoch': cfg.BACKBONE.TRAIN_EPOCH,
            'cen weight': cfg.TRAIN.CEN_WEIGHT,
            'cls weight': cfg.TRAIN.CLS_WEIGHT,
            'loc weight': cfg.TRAIN.LOC_WEIGHT,
            'cls Loss Method': cfg.TRAIN.CLS_LOSS_METHOD,
            'Alpha (Focal Loss)': cfg.TRAIN.LOSS_ALPHA,
            'Gamma (Focal Loss)': cfg.TRAIN.LOSS_GAMMA,
            'Template Flip Ratio': cfg.DATASET.TEMPLATE.FLIP,
        }
        # wandb.init(
        #     project="SiamCAR",
        #     entity="adrien88",
        #     name=model_name,
        #     config=constants
        # )

        print("Start Training!")
        for epoch in range(cfg.TRAIN.START_EPOCH, args.epoch):
            # one epoch
            logger.info(f"Epoch: {epoch + 1}")

            if (epoch + 1) == cfg.BACKBONE.TRAIN_EPOCH:
                self.build_opt_lr(current_epoch=epoch+1)
                print('Start training backbone')

            ######################################
            # Training
            ######################################
            # Switch to train mode.
            self.model.train()
            # train for one epoch
            train_losses = train_one_epoch(
                train_loader=self.train_loader,
                model=self.model,
                scaler=self.scaler,
                optimizer=self.optimizer,
                epoch=epoch,
                args=args
            )
            # Warning:
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            self.lr_scheduler.step()

            ######################################
            # Validating
            # 就直接用 test_loader 當作 validation
            ######################################
            # 注意，這裡的 mode 要根據 method 來調整
            # 這裡調整後，evaluating 也同樣是這個 mode
            # self.model.eval()
            test_losses = validate(
                test_loader=self.test_loader,
                model=self.model,
                epoch=epoch
            )

            ######################################
            # Evaluating
            ######################################
            if (epoch + 1) == 1 or (epoch + 1) % cfg.TRAIN.EVAL_FREQ == 0:
                dummy_model_dir = "./save_models/dummy_model_titan"
                create_dir(dummy_model_dir)
                dummy_model_name = "dummy_model"
                dummy_model_path = os.path.join(
                    dummy_model_dir, f"{dummy_model_name}.pth")
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, dummy_model_path)

                tracker = SiamCARTracker(self.model, cfg.TRACK)
                logger.info("Train evaluating...")
                train_metrics = evaluate(self.train_eval_loader, tracker)
                logger.info("Test evaluating...")
                test_metrics = evaluate(self.test_eval_loader, tracker)

                wandb.log({
                    "train_metrics": {
                        "Precision": train_metrics['precision'],
                        "Recall": train_metrics['recall']
                    },
                    "test_metrics": {
                        "Precision": test_metrics['precision'],
                        "Recall": test_metrics['recall']
                    }
                }, commit=False)

            wandb.log({
                "train": {
                    "cen_loss": train_losses.avg['cen'],
                    "cls_loss": train_losses.avg['cls'],
                    "loc_loss": train_losses.avg['loc'],
                    "total_loss": train_losses.avg['total'],
                    "cls_pos_loss": train_losses.avg['cls_pos'],
                    "cls_neg_loss": train_losses.avg['cls_neg']
                },
                "test": {
                    "cen_loss": test_losses.avg['cen'],
                    "cls_loss": test_losses.avg['cls'],
                    "loc_loss": test_losses.avg['loc'],
                    "total_loss": test_losses.avg['total'],
                    "cls_pos_loss": test_losses.avg['cls_pos'],
                    "cls_neg_loss": test_losses.avg['cls_neg']
                },
                "epoch": epoch + 1
            }, commit=True)

            ######################################
            # Save model
            ######################################
            if (epoch + 1) == 1 or ((epoch + 1) % cfg.TRAIN.SAVE_MODEL_FREQ) == 0:
                model_dir = os.path.join(
                    cfg.TRAIN.MODEL_DIR, args.dataset_name, args.criteria, args.target, args.method, model_name
                )
                create_dir(model_dir)
                model_path = os.path.join(model_dir, f"ckpt{epoch + 1}.pth")
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, model_path)

    def getTrainSampleNumber(self):
        return len(self.train_loader.dataset)

    def getTrainEpoch(self):
        return args.epoch

    def getTrainBatch(self):
        return args.batch_size

    def modelPredict(self):
        pass

    def stopTraining(self):
        self.stop_training = True
        print("--- The training has been stopped ---")

    def ConvertOnnx(self):
        convert_onnx()


if __name__ == '__main__':
    # load cfg
    # cfg.merge_from_file(args.cfg)

    # PermissionError
    os.environ['WANDB_DIR'] = './wandb_titan'

    trainer = Trainer(args)
    trainer.CompilModel()
    trainer.CreateDatasets(cfg.DATASET.VALIDATION_SPLIT, random_seed=42)
    trainer.StartFit()
    # trainer.ConvertOnnx()
