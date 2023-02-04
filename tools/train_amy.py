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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import wandb
from pysot.core.config import cfg
from pysot.datasets.collate_amy import collate_fn_new
from pysot.datasets.pcbdataset_multi_text import PCBDataset
from pysot.models.model_builder_amy import ModelBuilder
from pysot.utils.average_meter_official import AverageMeter
from pysot.utils.check_image import create_dir
from pysot.utils.distributed import (average_reduce, dist_init, get_rank,
                                     get_world_size, reduce_gradients)
from pysot.utils.log_helper import add_file_handler, init_log, print_speed
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.misc import commit, describe
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.SetCallBack import SetCallback
from tools.eval_amy import evaluate_amy
from tools.test_amy import test_amy

# from convertOnnx import convert_onnx

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--dataset', type=str, default="./datasets/train/allOld/",
                    help="training dataset")
parser.add_argument('--test_dataset', type=str, default="./datasets/test/allOld/",
                    help="testing dataset")
parser.add_argument('--cfg', type=str, default='./experiments/siamcar_r50/config_amy.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
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
            # if cfg.TRAIN.LOG_DIR:
            #     add_file_handler('global',
            #                      os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
            #                      logging.INFO)

            #logger.info("Version Information: \n{}\n".format(commit()))
            #logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

        #dist_model = nn.DataParallel(model).cuda()

        # create tensorboard writer
        # if rank == 0 and cfg.TRAIN.LOG_DIR:
        #     self.tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
        # else:
        #     self.tb_writer = None

        self.stop_training = False

    def _seed_torch(self, seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def CreateDataset(self, validation_split, random_seed):
        train_dataset = PCBDataset(args, "train")
        val_dataset = PCBDataset(args, "val")

        train_sampler = None
        if get_world_size() > 1:
            train_sampler = DistributedSampler(self.train_dataset)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn_new,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn_new
        )

        print(f"Train dataset size: {len(self.train_loader.dataset)}")
        print(f"Val dataset size: {len(self.val_loader.dataset)}")
        assert len(self.train_loader.dataset) != 0, "ERROR, dataset is empty !!"

        return len(self.train_loader.dataset)

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
            self.optimizer, epochs=cfg.TRAIN.EPOCH)
        self.lr_scheduler.step(cfg.TRAIN.START_EPOCH)

    def CompilModel(self):
        # create model
        self.model = ModelBuilder().train().cuda()

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
            print("resume")
            logger.info("resume from {}".format(cfg.TRAIN.RESUME))
            assert os.path.isfile(cfg.TRAIN.RESUME), \
                '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
            self.model, self.optimizer, cfg.TRAIN.START_EPOCH = \
                restore_from(self.model, self.optimizer, cfg.TRAIN.RESUME)

        # load pretrain
        elif cfg.TRAIN.PRETRAINED:
            print("pretrained")
            load_pretrain(self.model, cfg.TRAIN.PRETRAINED)

    # start training
    def StartFit(self):
        cur_lr = self.lr_scheduler.get_cur_lr()
        rank = get_rank()
        print("train rank:", rank)

        print("start tarin!")
        average_meter = AverageMeter()

        def is_valid_number(x):
            return not (math.isnan(x) or math.isinf(x) or x > 1e4)

        world_size = get_world_size()
        num_per_epoch = len(self.train_loader)
        start_epoch = cfg.TRAIN.START_EPOCH

        epoch_record = {'epoch': 0, 'loss': 0.0}
        batch_record = {'batch': 0, 'loss': 0.0}
        self.cb.on_train_begin()

        if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and get_rank() == 0:
            os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

        for epoch in range(cfg.TRAIN.EPOCH):
            train_cen_loss = 0.0
            train_cls_loss = 0.0
            train_loc_loss = 0.0
            train_total_loss = 0.0
            val_total_loss, val_cen_loss, val_cls_loss, val_loc_loss = 0, 0, 0, 0

            self.lr_scheduler.step(epoch)
            cur_lr = self.lr_scheduler.get_cur_lr()
            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                self.build_opt_lr(epoch)
                print('start training backbone')

            end = time.time()
            for idx, data in enumerate(self.train_loader):
                if epoch == cfg.TRAIN.EPOCH:
                    return

                data_time = average_reduce(time.time() - end)

                outputs = self.model(data)

                loss = outputs['total_loss'].mean()

                train_total_loss += outputs['total_loss'].mean().item()
                train_cen_loss += outputs['cen_loss'].mean().item()
                train_cls_loss += outputs['cls_loss'].mean().item()
                train_loc_loss += outputs['loc_loss'].mean().item()

                if is_valid_number(loss.data.item()):
                    self.optimizer.zero_grad()
                    loss.backward()
                    reduce_gradients(self.model)

                    # clip gradient
                    clip_grad_norm_(self.model.parameters(),
                                    cfg.TRAIN.GRAD_CLIP)
                    self.optimizer.step()

                batch_time = time.time() - end
                batch_info = {}
                batch_info['batch_time'] = average_reduce(batch_time)
                batch_info['data_time'] = average_reduce(data_time)
                batch_record['batch'] = idx
                batch_record['loss'] = train_total_loss/(idx+1)
                self.cb.on_batch_end(batch_record)

                for k, v in sorted(outputs.items()):
                    batch_info[k] = average_reduce(v.mean().data.item())

                average_meter.update(**batch_info)  # 對所有batch_size的損失取平均

                if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                    info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                        epoch+1, (idx+1) % num_per_epoch,
                        num_per_epoch, cur_lr)
                    for cc, (k, v) in enumerate(batch_info.items()):
                        if cc % 2 == 0:
                            info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                        else:
                            info += ("{:s}\n").format(
                                getattr(average_meter, k))

                    logger.info(info)
                    print_speed(idx+1*num_per_epoch,
                                average_meter.batch_time.avg,
                                cfg.TRAIN.EPOCH * num_per_epoch)

                if self.stop_training:
                    return

            epoch_record['epoch'] = epoch
            epoch_record['loss'] = train_total_loss / len(self.train_loader)
            self.cb.on_epoch_end(epoch_record)

            elapsed = time.time() - end
            train_cls_loss = train_cls_loss / len(self.train_loader)
            train_loc_loss = train_loc_loss / len(self.train_loader)
            train_cen_loss = train_cen_loss / len(self.train_loader)
            train_total_loss = train_total_loss / len(self.train_loader)

            ######################################
            # Validating
            ######################################
            self.model.eval()

            for idx, data in enumerate(self.val_loader):
                with torch.no_grad():
                    outputs = self.model(data)
                val_total_loss += outputs['total_loss'].mean().item()
                val_cen_loss += outputs['cen_loss'].mean().item()
                val_cls_loss += outputs['cls_loss'].mean().item()
                val_loc_loss += outputs['loc_loss'].mean().item()

            val_cls_loss = val_cls_loss / len(self.val_loader)
            val_loc_loss = val_loc_loss / len(self.val_loader)
            val_cen_loss = val_cen_loss / len(self.val_loader)
            val_total_loss = val_total_loss / len(self.val_loader)

            ######################################
            # Evaluating
            ######################################
            # if (epoch + 1) % cfg.TRAIN.EVAL_FREQ == 0:
            #     dummy_model_dir = os.path.join("./snapshot", "dummy")
            #     create_dir(dummy_model_dir)
            #     dummy_model_name = "dummy_model"
            #     dummy_model_path = os.path.join(dummy_model_dir, f"{dummy_model_name}.pth")
            #     torch.save({
            #         'epoch': epoch+start_epoch,
            #         'state_dict': self.model.state_dict(),
            #         'optimizer': self.optimizer.state_dict()
            #     }, dummy_model_path)
            #     print(f"Save dummy model to: {dummy_model_path}")

            #     # 先做 test 產生 annotation file
            #     test_amy(snapshot=dummy_model_path, dataset_dir=args.test_dataset)
            #     # 再從 annotation 去算 precision, recall
            #     metrics = evaluate_amy(snapshot=dummy_model_path, dataset_dir=args.test_dataset)

            #     wandb.log({
            #         "test_metrics": {
            #             "Precision": metrics['precision'],
            #             "Recall": metrics['recall']
            #         }
            #     }, commit=False)

            wandb.log({
                "train": {
                    "cen_loss": train_cen_loss,
                    "cls_loss": train_cls_loss,
                    "loc_loss": train_loc_loss,
                    "total_loss": train_total_loss
                },
                "val": {
                    "cen_loss": val_cen_loss,
                    "cls_loss": val_cls_loss,
                    "loc_loss": val_loc_loss,
                    "total_loss": val_total_loss
                }
            }, commit=True)

            # self.tb_writer.add_scalar('total_loss',total_loss , epoch)
            # self.tb_writer.add_scalar('cls_loss',cls_loss , epoch)
            # self.tb_writer.add_scalar('loc_loss',loc_loss , epoch)
            # self.tb_writer.add_scalar('cen_loss',cen_loss , epoch)
            print("epoch:", epoch + 1)
            print(
                f"({elapsed:.3f}s,{elapsed/len(self.train_loader.dataset):.3}s/img)")

            print(f"\r(cls_loss:{train_cls_loss:6.4f}, loc_loss:{train_loc_loss:6.4f}, cen_loss{train_cen_loss:6.4f})"
                  f"\tTotal_loss:{train_total_loss:6.4f}")

            # save checkpoint
            # if ((epoch + 1) % cfg.TRAIN.SAVE_MODEL_FREQ) == 0:
            #     model_dir = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "my")
            #     create_dir(model_dir)
            #     model_path = os.path.join(model_dir, f'checkpoint_e{epoch}.pth')
            #     torch.save({
            #         'epoch': epoch + start_epoch,
            #         'state_dict': self.model.state_dict(),
            #         'optimizer': self.optimizer.state_dict()
            #     }, model_path)
            #     print(f"Save model to: {model_path}")

        self.cb.on_train_end()

    def getTrainSampleNumber(self):
        return len(self.train_loader.dataset)

    def getTrainEpoch(self):
        return cfg.TRAIN.EPOCH

    def getTrainBatch(self):
        return cfg.TRAIN.BATCH_SIZE

    def modelPredict(self):
        pass

    def stopTraining(self):
        self.stop_training = True
        print("-" * 20, "The training has been stopped", "-" * 20)

    def ConvertOnnx(self):
        convert_onnx()


if __name__ == '__main__':
    # load cfg
    cfg.merge_from_file(args.cfg)

    os.environ['WANDB_DIR'] = './wandb_titan'
    # constants = {
    #     "search_size": cfg.TRAIN.SEARCH_SIZE,
    #     "crop_method": "old",
    #     "score_size": cfg.TRAIN.OUTPUT_SIZE,
    #     "epochs": cfg.TRAIN.EPOCH,
    #     "batch_size": cfg.TRAIN.BATCH_SIZE,
    #     "start_lr": cfg.TRAIN.LR.KWARGS.start_lr,
    #     "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
    #     "train backbone epoch": cfg.BACKBONE.TRAIN_EPOCH
    # }
    # wandb.init(
    #     project="SiamCAR_origin",
    #     entity="adrien88",
    #     name=f"x{cfg.TRAIN.SEARCH_SIZE}_old_e{cfg.TRAIN.EPOCH}_b{cfg.TRAIN.BATCH_SIZE}",
    #     config=constants
    # )

    # Create save model directory
    create_dir(cfg.TRAIN.SNAPSHOT_DIR)

    Trainer = SiamCARTrainer()
    Trainer.CreateDataset(cfg.DATASET.VALIDATION_SPLIT, random_seed=42)
    Trainer.SetTrainingCallback()
    Trainer.CompilModel()
    Trainer.StartFit()
    # Trainer.ConvertOnnx()
