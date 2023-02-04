import argparse
import json
import logging
import math
import os
import random
import time

import colorama
import ipdb
import numpy as np
import torch
import torch.nn as nn
from colorama import Fore
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from pysot.core.config import cfg
from pysot.datasets.augmentation.pcb_aug import PCBAugmentation
from pysot.datasets.collate import collate_fn
from pysot.datasets.pcbdataset import get_pcbdataset
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.average_meter import AverageMeter
from pysot.utils.check_image import create_dir
from pysot.utils.distributed import (average_reduce, dist_init, get_rank,
                                     get_world_size, reduce_gradients)
from pysot.utils.log_helper import add_file_handler, init_log, print_speed
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.model_load import load_pretrain, restore_from
from tools.engine import save_checkpoint, train_one_epoch, validate
from tools.evaluate import evaluate
from utils.wandb import WandB

logger = logging.getLogger('global')
colorama.init(autoreset=True)


class Trainer(object):
    def __init__(self, args):
        rank, world_size = dist_init()

        self._seed_torch(args.seed)
        self.stop_training = False
        self.args = args

        # Config
        siamcar_cfg = "./experiments/siamcar_r50/config.yaml"
        cfg.merge_from_file(siamcar_cfg)
        cfg.merge_from_file(args.cfg)
        cfg.update({'Args': vars(args)})

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
        # Datasets arguments
        data_args = {
            'train': {
                'data_path': self.args.data,
                'method': self.args.method,
                'criteria': self.args.criteria,
                'bg': self.args.bg,
                'target': self.args.target,
            },
            'test': {
                'data_path': self.args.test_data,
                'method': self.args.method,
                'criteria': self.args.criteria,
                'bg': self.args.bg,
                'target': self.args.target
            },
            'eval_train': {
                'data_path': self.args.data,
                'method': self.args.eval_method,
                'criteria': self.args.eval_criteria,
                'bg': self.args.eval_bg,
                'target': self.args.target
            },
            'eval_test': {
                'data_path': self.args.test_data,
                'method': self.args.eval_method,
                'criteria': self.args.eval_criteria,
                'bg': self.args.eval_bg,
                'target': self.args.target
            }
        }

        # Data augmentations
        data_augmentations = {
            'train': {
                'template': PCBAugmentation(cfg.TRAIN.DATASET.TEMPLATE),
                'search': PCBAugmentation(cfg.TRAIN.DATASET.SEARCH),
            },
            'test': {
                'template': PCBAugmentation(cfg.TEST.DATASET.TEMPLATE),
                'search': PCBAugmentation(cfg.TEST.DATASET.SEARCH),
            }
        }

        logger.info("Building dataset...")
        pcb_dataset = get_pcbdataset(self.args.method)
        # 現在 neg = 0，所以 train_dataset 可以沿用
        dataset = pcb_dataset(
            data_args['train'], mode="train", augmentation=data_augmentations['train'])
        assert len(dataset) != 0, "Data is empty"
        test_dataset = pcb_dataset(
            data_args['test'], mode="test", augmentation=data_augmentations['test'])
        eval_dataset = pcb_dataset(
            data_args['eval_train'], mode="evaluate", augmentation=data_augmentations['test'])
        test_eval_dataset = pcb_dataset(
            data_args['eval_test'], mode="evaluate", augmentation=data_augmentations['test'])

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
        dataset_size = {
            'Train': len(dataset),
            'Test': len(test_dataset),
            'Train Eval': len(eval_dataset),
            'Test Eval': len(test_eval_dataset)
        }
        cfg.update({'Dataset Size': dataset_size})

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
            train_dataset, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_WORKERS)
        self.train_eval_loader = create_loader(
            eval_dataset, batch_size=1, num_workers=0)
        self.test_loader = create_loader(
            test_dataset, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_WORKERS)
        self.test_eval_loader = create_loader(
            test_eval_dataset, batch_size=1, num_workers=0)

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
            self.optimizer, epochs=cfg.TRAIN.EPOCH)
        # Warning:
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        # self.lr_scheduler.step(cfg.TRAIN.START_EPOCH)

        # Prevent underflowing gradients
        self.scaler = GradScaler()

    def CompilModel(self):
        # create model
        self.model = ModelBuilder(method=self.args.method).cuda()

        # load pretrained backbone weights
        if cfg.BACKBONE.PRETRAINED:
            print(f"{Fore.GREEN}Use pretrained backbone")
            load_pretrain(self.model.backbone, cfg.BACKBONE.PRETRAINED)

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
            print(f"{Fore.GREEN}Use pretrained model")
            load_pretrain(self.model, cfg.TRAIN.PRETRAINED)

    def StartFit(self):
        model_name = \
            f"{self.args.dataset}_{self.args.criteria}_{self.args.target}_{self.args.method}"
        model_dir = os.path.join(cfg.TRAIN.MODEL_DIR, self.args.date, self.args.cfg_name,
                                 self.args.dataset, self.args.criteria, self.args.target,
                                 self.args.method, model_name)

        # Initialize WandB
        wandb = WandB(name=model_name, config=cfg, init=True)

        # Metrics before training.
        init_info = {'Train': {}, 'Test': {}}
        train_metrics, test_metrics = self.evaluate()
        init_info['Train'].update(train_metrics)
        init_info['Test'].update(test_metrics)
        wandb.update(info=init_info, epoch=0)
        wandb.upload(commit=True)

        print("Start Training!")
        for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCH):
            logger.info(f"Epoch: {epoch + 1}")
            epoch_info = {'Train': {}, 'Test': {}}

            self.model.method = self.args.method

            if (epoch + 1) == cfg.BACKBONE.TRAIN_EPOCH:
                self.build_opt_lr(current_epoch=epoch+1)
                print('Start training backbone')

            # Train for one epoch
            train_losses = train_one_epoch(
                train_loader=self.train_loader,
                model=self.model,
                scaler=self.scaler,
                optimizer=self.optimizer,
                epoch=epoch,
                accum_iters=cfg.TRAIN.ACCUM_ITER)

            # Validation
            test_losses = validate(
                test_loader=self.test_loader,
                model=self.model,
                epoch=epoch)

            epoch_info['Train'].update(train_losses)
            epoch_info['Test'].update(test_losses)

            self.lr_scheduler.step()

            # Evaluate
            if (epoch+1) == 1 or (epoch+1) % cfg.TRAIN.EVAL_FREQ == 0:
                train_metrics, test_metrics = self.evaluate()
                epoch_info['Train'].update(train_metrics)
                epoch_info['Test'].update(test_metrics)

            wandb.update(info=epoch_info, epoch=epoch+1)
            wandb.upload(commit=True)

            # Save model
            if (epoch + 1) == 1 or ((epoch + 1) % cfg.TRAIN.SAVE_MODEL_FREQ) == 0:
                create_dir(model_dir)
                model_path = os.path.join(model_dir, f"ckpt{epoch+1}.pth")
                model_info = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}

                save_checkpoint(model_info, model_path)

    def evaluate(self):
        self.model.eval()
        self.model.method = self.args.eval_method
        tracker = SiamCARTracker(self.model, cfg.TRACK)
        train_metrics = evaluate(self.train_eval_loader, tracker)
        for key, value in train_metrics.items():
            print(f"{key}: {value}")
        test_metrics = evaluate(self.test_eval_loader, tracker)
        for key, value in test_metrics.items():
            print(f"{key}: {value}")
        return train_metrics, test_metrics

    def getTrainSampleNumber(self):
        return len(self.train_loader.dataset)

    def getTrainEpoch(self):
        return self.args.epoch

    def getTrainBatch(self):
        return cfg.TRAIN.BATCH_SIZE

    def modelPredict(self):
        pass

    def stopTraining(self):
        self.stop_training = True
        print("--- The training has been stopped ---")

    def ConvertOnnx(self):
        convert_onnx()
