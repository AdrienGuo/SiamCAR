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
import wandb
from pysot.core.config import cfg
from pysot.datasets.collate import collate_fn_new
# === 選擇 new / origin 的裁切方式 ===
# from pysot.datasets.pcbdataset_multi_text import PCBDataset
from pysot.datasets.pcbdataset_new import PCBDataset
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
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from tools.eval import evaluate

# from convertOnnx import convert_onnx
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--dataset_name', type=str, default='', help='dataset name')
parser.add_argument('--dataset', type=str, default='', help='training dataset')
parser.add_argument('--test_dataset', type=str, default='', help='testing dataset')
parser.add_argument('--criteria', type=str, default='', help='criteria of dataset')
parser.add_argument('--neg', type=float, default=0.0, help='negative pair')
parser.add_argument('--bg', type=str, help='background of template')
parser.add_argument('--epoch', type=int, help='epoch')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--accum_iter', type=int, help='accumulate gradient iteration')
parser.add_argument('--cfg', type=str, default='./experiments/siamcar_r50/config.yaml', help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
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

            #logger.info("Version Information: \n{}\n".format(commit()))
            #logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

        
        #dist_model = nn.DataParallel(model).cuda()
        
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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def CreateDataset(self, validation_split: float = 0.0, random_seed: int = 42):
        logger.info("Building dataset...")
        # 現在 neg = 0，所以 train_dataset 可以沿用
        dataset = PCBDataset(args, "train")
        test_dataset = PCBDataset(args, "test")

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

        # train_sampler = None
        # if get_world_size() > 1:
        #     train_sampler = DistributedSampler(train_dataset)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn_new
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn_new
        )

        self.train_eval_loader = DataLoader(
            train_dataset,
            batch_size=1,  # 只能設為 1
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn_new
        )

        self.val_eval_loader = DataLoader(
            val_dataset,
            batch_size=1,  # 只能設為 1
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn_new
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # 只能設為 1
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn_new
        )

        # return len(self.train_loader.dataset)

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

        self.lr_scheduler = build_lr_scheduler(self.optimizer, epochs=args.epoch)
        # Warning:
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        # self.lr_scheduler.step(cfg.TRAIN.START_EPOCH)

    def CompilModel(self):
        # create model
        self.model = ModelBuilder().train().cuda()

        # load pretrained backbone weights
        if cfg.BACKBONE.PRETRAINED:
            cur_path = os.path.dirname(os.path.realpath(__file__))
            backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
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
                restore_from(self.model, self.optimizer, ckpt_path=cfg.TRAIN.RESUME)

        # load pretrain
        elif cfg.TRAIN.PRETRAINED:
            print("pretrained")
            load_pretrain(self.model, cfg.TRAIN.PRETRAINED)

    # Training
    def StartFit(self):
        cur_lr = self.lr_scheduler.get_cur_lr()
        rank = get_rank()
        print("Train rank:", rank)

        average_meter = AverageMeter()

        def is_valid_number(x):
            return not(math.isnan(x) or math.isinf(x) or x > 1e4)

        world_size = get_world_size()
        num_per_epoch = len(self.train_loader)
        start_epoch = cfg.TRAIN.START_EPOCH

        epoch_record = {'epoch': 0, 'loss': 0.0}
        batch_record = {'batch': 0, 'loss': 0.0}
        self.cb.on_train_begin()

        if get_rank() == 0:
            create_dir(cfg.TRAIN.MODEL_DIR)

        print("Start tarining!")
        for epoch in range(cfg.TRAIN.START_EPOCH, args.epoch):
            # one epoch
            logger.info(f"Epoch: {epoch + 1}")

            if (epoch + 1) == cfg.BACKBONE.TRAIN_EPOCH:
                self.build_opt_lr(current_epoch=epoch + 1)
                print('Start training backbone.')

            train_loss = dict(cen=0.0, cls=0.0, loc=0.0, total=0.0)
            val_loss = dict(cen=0.0, cls=0.0, loc=0.0, total=0.0)

            # self.lr_scheduler.step(epoch)
            # cur_lr = self.lr_scheduler.get_cur_lr()

            end = time.time()

            ######################################
            # Training
            ######################################
            self.model.train()
            for idx, data in enumerate(self.train_loader):
                # one batch

                data_time = average_reduce(time.time() - end)

                outputs = self.model(data)

                # ipdb.set_trace()

                # train_loss 會個別對應 outputs 的 key 後疊加
                train_loss = {key: value + outputs[key] for key, value in train_loss.items()}

                loss = outputs['total'].mean()
                loss.backward()
                # accumulate gradient
                if ((idx + 1) % args.accum_iter) == 0 or ((idx + 1) == len(self.train_loader)):
                    if is_valid_number(loss.data.item()):
                        reduce_gradients(self.model)
                        # clip gradient
                        clip_grad_norm_(self.model.parameters(), cfg.TRAIN.GRAD_CLIP)
                        self.optimizer.step()

                        self.optimizer.zero_grad()

                    print(
                        f"cen_loss: {outputs['cen']:<6.3f}"
                        f" | cls_loss: {outputs['cls']:<6.3f}"
                        f" | loc_loss: {outputs['loc']:<6.3f}"
                        f" | total_loss: {outputs['total']:<6.3f}"
                    )

                batch_time = time.time() - end
                # batch_info = {}
                # batch_info['batch_time'] = average_reduce(batch_time)
                # batch_info['data_time'] = average_reduce(data_time)
                # batch_record['batch'] = idx
                # batch_record['loss'] = train_loss['total'] / (idx + 1)
                # self.cb.on_batch_end(batch_record)

                # for k, v in sorted(outputs.items()):
                #     batch_info[k] = average_reduce(v.mean().data.item())

                # average_meter.update(**batch_info)    # 對所有batch_size的損失取平均

                # if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                #     info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                #         epoch, (idx + 1) % num_per_epoch,
                #         num_per_epoch, cur_lr
                #     )
                #     for cc, (k, v) in enumerate(batch_info.items()):
                #         if cc % 2 == 0:
                #             info += ("\t{:s}\t").format(
                #                     getattr(average_meter, k))
                #         else:
                #             info += ("{:s}\n").format(
                #                     getattr(average_meter, k))

                #     logger.info(info)
                #     print_speed(idx+1*num_per_epoch,
                #                 average_meter.batch_time.avg,
                #                 args.epoch * num_per_epoch)

            # Warning:
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            self.lr_scheduler.step()
            cur_lr = self.lr_scheduler.get_cur_lr()
            print(f"Curr lr: {cur_lr}")

            train_loss = {key: value / len(self.train_loader) for key, value in train_loss.items()}

            epoch_record['epoch'] = epoch + 1
            epoch_record['loss'] = train_loss['total']
            self.cb.on_epoch_end(epoch_record)

            elapsed = time.time() - end
            print(f"({elapsed:.3f}s, {elapsed / len(self.train_loader.dataset):.3}s/img)")

            print("--- Train ---")
            print(
                f"cen_loss: {train_loss['cen']:<6.3f}"
                f" | cls_loss: {train_loss['cls']:<6.3f}"
                f" | loc_loss: {train_loss['loc']:<6.3f}"
                f" | total_loss: {train_loss['total']:<6.3f}"
            )

            ######################################
            # Validating
            ######################################
            self.model.eval()

            for idx, data in enumerate(self.val_loader):
                with torch.no_grad():
                    outputs = self.model(data)
                val_loss = {key: value + outputs[key] for key, value in val_loss.items()}
            val_loss = {key: value / len(self.val_loader) for key, value in val_loss.items()}
            print("--- Validation ---")
            print(
                f"cen_loss: {val_loss['cen']:<6.3f}"
                f" | cls_loss: {val_loss['cls']:<6.3f}"
                f" | loc_loss: {val_loss['loc']:<6.3f}"
                f" | total_loss: {val_loss['total']:<6.3f}"
            )

            ######################################
            # Evaluating
            ######################################
            if (epoch + 1) % cfg.TRAIN.EVAL_FREQ == 0:
                dummy_model_dir = "./save_models/dummy_model"
                create_dir(dummy_model_dir)
                dummy_model_name = "dummy_model_1"
                dummy_model_path = os.path.join(dummy_model_dir, f"{dummy_model_name}.pth")
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, dummy_model_path)
                print(f"Save dummy_model: {dummy_model_path}")

                # Create model
                eval_model = ModelBuilder()
                # Load model
                eval_model = load_pretrain(eval_model, dummy_model_path).cuda().eval()
                # Build tracker
                tracker = SiamCARTracker(eval_model, cfg.TRACK)

                logger.info("Train evaluating...")
                train_metrics = evaluate(self.train_eval_loader, tracker)
                logger.info("Validation evaluating...")
                val_metrics = evaluate(self.val_eval_loader, tracker)
                logger.info("Test evaluating...")
                test_metrics = evaluate(self.test_loader, tracker)

                wandb.log({
                    "train_metrics": {
                        "Precision": train_metrics['precision'],
                        "Recall": train_metrics['recall']
                    },
                    "val_metrics": {
                        "Precision": val_metrics['precision'],
                        "Recall": val_metrics['recall']
                    },
                    "test_metrics": {
                        "Precision": test_metrics['precision'],
                        "Recall": test_metrics['recall']
                    }
                }, commit=False)

            wandb.log({
                "train": {
                    "cen_loss": train_loss['cen'],
                    "cls_loss": train_loss['cls'],
                    "loc_loss": train_loss['loc'],
                    "total_loss": train_loss['total']
                },
                "val": {
                    "cen_loss": val_loss['cen'],
                    "cls_loss": val_loss['cls'],
                    "loc_loss": val_loss['loc'],
                    "total_loss": val_loss['total']
                },
                "epoch": epoch + 1
            }, commit=True)

            ######################################
            # Save model
            ######################################
            if ((epoch + 1) % cfg.TRAIN.SAVE_MODEL_FREQ) == 0:
                model_dir = os.path.join(
                    cfg.TRAIN.MODEL_DIR, args.dataset_name, args.criteria,
                    f"{args.dataset_name}_{args.criteria}_neg{args.neg}_x{cfg.TRAIN.SEARCH_SIZE}_bg{args.bg}_e{args.epoch}_b{args.batch_size}")
                create_dir(model_dir)
                model_path = os.path.join(model_dir + f"/model_e{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + start_epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, model_path)
                print(f"Save model to: {model_path}")

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
    cfg.merge_from_file(args.cfg)

    # PermissionError
    os.environ['WANDB_DIR'] = './wandb_titan'
    # constants = {
    #     "Dataset": args.dataset_name,
    #     "Criteria": args.criteria,
    #     "Validation Ratio": cfg.DATASET.VALIDATION_SPLIT,
    #     "Negative Ratio": args.neg,
    #     "Background": args.bg,
    #     "Epochs": args.epoch,
    #     "Batch": args.batch_size,
    #     "Accumulate iter": args.accum_iter,
    #     "lr": cfg.TRAIN.LR.KWARGS.start_lr,
    #     "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
    #     "Model Pretrained": cfg.TRAIN.PRETRAINED,
    #     "Backbone Pretrained": cfg.BACKBONE.PRETRAINED,
    #     "Backbone Train Epoch": cfg.BACKBONE.TRAIN_EPOCH,
    #     "cen weight": cfg.TRAIN.CEN_WEIGHT,
    #     "cls weight": cfg.TRAIN.CLS_WEIGHT,
    #     "loc weight": cfg.TRAIN.LOC_WEIGHT,
    # }
    # wandb.init(
    #     project="SiamCAR",
    #     entity="adrien88",
    #     name=f"{args.dataset_name}_{args.criteria}_neg{args.neg}_x{cfg.TRAIN.SEARCH_SIZE}_bg{args.bg}_e{args.epoch}_b{args.batch_size}",
    #     config=constants
    # )

    Trainer = SiamCARTrainer()
    Trainer.CreateDataset(cfg.DATASET.VALIDATION_SPLIT, random_seed=42)
    Trainer.SetTrainingCallback()
    Trainer.CompilModel()
    Trainer.StartFit()
    # Trainer.ConvertOnnx()
