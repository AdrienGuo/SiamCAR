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

import numpy as np
import torch
import torch.nn as nn
from pysot.core.config import cfg
from pysot.datasets.collate import collate_fn_new
from pysot.datasets.pcbdataset_multi_text import PCBDataset
from pysot.models.model_builder import ModelBuilder
from pysot.utils.average_meter import AverageMeter
from pysot.utils.distributed import (average_reduce, dist_init, get_rank,
                                     get_world_size, reduce_gradients)
from pysot.utils.log_helper import add_file_handler, init_log, print_speed
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.misc import commit, describe
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.SetCallBack import SetCallback
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# from convertOnnx import convert_onnx
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--cfg', type=str, default='./experiments/siamcar_r50/config.yaml',
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

         # load cfg
        cfg.merge_from_file(args.cfg)
        if rank == 0:
            if not os.path.exists(cfg.TRAIN.LOG_DIR):
                os.makedirs(cfg.TRAIN.LOG_DIR)
            init_log('global', logging.INFO)
            if cfg.TRAIN.LOG_DIR:
                add_file_handler('global',
                                 os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                                 logging.INFO)

            #logger.info("Version Information: \n{}\n".format(commit()))
            #logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

        
        #dist_model = nn.DataParallel(model).cuda()
        
        # create tensorboard writer
        if rank == 0 and cfg.TRAIN.LOG_DIR:
            self.tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
        else:
            self.tb_writer = None
         
        self.stop_training = False
        
    def _seed_torch(self,seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True 
       
        
    def CreateDataset(self):
        self.train_dataset = PCBDataset()
        train_sampler = None
        if get_world_size() > 1:
            train_sampler = DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=True,
                                  collate_fn=collate_fn_new,
                                  shuffle = True,
                                  )
        
        return len(self.train_loader.dataset)
       
    def SetTrainingCallback(self):
        nTrainSample = self.getTrainSampleNumber()
        TargetTrainBatchSize = self.getTrainBatch()
        TargetEpoch = self.getTrainEpoch()
        # Training Progress
        train_stepPerEpoch = math.ceil(nTrainSample / TargetTrainBatchSize)
        self.cb = SetCallback()
    
    
    def build_opt_lr(self,current_epoch=0):
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

        self.lr_scheduler = build_lr_scheduler(self.optimizer, epochs=cfg.TRAIN.EPOCH)
        self.lr_scheduler.step(cfg.TRAIN.START_EPOCH)
        

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
        print("cfg.TRAIN.RESUME:",cfg.TRAIN.RESUME)
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
        print("train rank:",rank)

        print("start tarin!")
        average_meter = AverageMeter()

        def is_valid_number(x):
            return not(math.isnan(x) or math.isinf(x) or x > 1e4)

        world_size = get_world_size()
        num_per_epoch = len(self.train_loader)
        start_epoch = cfg.TRAIN.START_EPOCH
        
        epoch_record = {'epoch': 0, 'loss': 0}
        batch_record = {'batch':0,'loss':0}
        self.cb.on_train_begin()
       

        if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
                get_rank() == 0:
            os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

        for epoch in range(cfg.TRAIN.EPOCH):
            cen_loss = 0
            cls_loss = 0
            loc_loss = 0
            total_loss = 0
            self.lr_scheduler.step(epoch)
            cur_lr = self.lr_scheduler.get_cur_lr()
            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                print('start training backbone.')

            end = time.time()
            for idx, data in enumerate(self.train_loader):

                if epoch == cfg.TRAIN.EPOCH:
                    return

                data_time = average_reduce(time.time() - end)

                outputs = self.model(data)

                loss = outputs['total_loss'].mean()

                total_loss += outputs['total_loss'].mean().item()
                cen_loss += outputs['cen_loss'].mean().item()
                cls_loss += outputs['cls_loss'].mean().item()
                loc_loss += outputs['loc_loss'].mean().item()

                if is_valid_number(loss.data.item()):
                    self.optimizer.zero_grad()
                    loss.backward()
                    reduce_gradients(self.model)
                    
                    # clip gradient
                    clip_grad_norm_(self.model.parameters(), cfg.TRAIN.GRAD_CLIP)
                    self.optimizer.step()

                batch_time = time.time() - end
                batch_info = {}
                batch_info['batch_time'] = average_reduce(batch_time)
                batch_info['data_time'] = average_reduce(data_time)
                batch_record['batch'] = idx
                batch_record['loss'] = total_loss/(idx+1)
                self.cb.on_batch_end(batch_record)
                
                for k, v in sorted(outputs.items()):
                    batch_info[k] = average_reduce(v.mean().data.item())

                average_meter.update(**batch_info) #對所有batch_size的損失取平均


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
            epoch_record['loss'] = total_loss/len(self.train_loader)
            self.cb.on_epoch_end(epoch_record)
            
            elapsed = time.time() - end
            cls_loss = cls_loss/len(self.train_loader)
            loc_loss = loc_loss/len(self.train_loader)
            cen_loss = cen_loss/len(self.train_loader)
            total_loss = total_loss/len(self.train_loader)
            self.tb_writer.add_scalar('total_loss',total_loss , epoch)
            self.tb_writer.add_scalar('cls_loss',cls_loss , epoch)
            self.tb_writer.add_scalar('loc_loss',loc_loss , epoch)
            self.tb_writer.add_scalar('cen_loss',cen_loss , epoch)
            print("epoch:",epoch)
            print(f"({elapsed:.3f}s,{elapsed/len(self.train_loader.dataset):.3}s/img)")

            print (f"\r(cls_loss:{cls_loss:6.4f}, loc_loss:{loc_loss:6.4f},cen_loss{cen_loss:6.4f})"
                   f"\tTotal_loss:{total_loss:6.4f}")
            #save checkpoint
            if (epoch+1) % 1 == 0 and epoch != 0:
                torch.save({'epoch': epoch+start_epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                            cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))
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
        print('------------------------The training has been stopped-----------------------------')
   
    def ConvertOnnx(self):
        convert_onnx()



if __name__ == '__main__':
    
    Trainer = SiamCARTrainer()
    Trainer.CreateDataset()
    Trainer.SetTrainingCallback()
    Trainer.CompilModel()
    Trainer.StartFit()
    #Trainer.ConvertOnnx()
