# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamcar_r50"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
__C.TRAIN.EXEMPLAR_SIZE = 127
__C.TRAIN.SEARCH_SIZE = 255
__C.TRAIN.OUTPUT_SIZE = 25


__C.TRAIN.RESUME = ''


# Whole model pretrained
__C.TRAIN.PRETRAINED = ''
# __C.TRAIN.PRETRAINED = './tools/snapshot/model_general.pth'


__C.TRAIN.LOG_DIR = './logs/test'


__C.TRAIN.MODEL_DIR = './save_models/'


# __C.TRAIN.EPOCH = 200
__C.TRAIN.START_EPOCH = 0
# __C.TRAIN.BATCH_SIZE = 1
__C.TRAIN.NUM_WORKERS = 0

__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CEN_WEIGHT = 0.0
__C.TRAIN.CLS_WEIGHT = 1.0
__C.TRAIN.LOC_WEIGHT = 0.0

__C.TRAIN.EVAL_FREQ = 20
__C.TRAIN.SAVE_MODEL_FREQ = 20
__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.CLS_LOSS_METHOD = 'bce'  # bce / focal
# 當使用 Focal Loss 才會用到的 Hyperparameters
__C.TRAIN.LOSS_ALPHA = 0.25
__C.TRAIN.LOSS_GAMMA = 5.0


# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

__C.DATASET.CROP_METHOD = "new"

# validation_split ratio
__C.DATASET.VALIDATION_SPLIT = 0.0

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 0.0  # 4

__C.DATASET.TEMPLATE.SCALE = 0.0  # 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 0.0  # 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 0.0  # 64

__C.DATASET.SEARCH.SCALE =0.0  # 0.18
# __C.DATASET.SEARCH.SCALE = 0

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 0.0  # 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.Background = False

__C.DATASET.NAMES = ('PCB',)

__C.DATASET.PCB = CN()
# __C.DATASET.PCB.ROOT = './datasets/train/'  #'train_dataset/pcb/train'         # COCO dataset path
# __C.DATASET.PCB.ANNO = './datasets/train/'  #'train_dataset/pcb/annotation'
__C.DATASET.PCB.FRAME_RANGE = 1
__C.DATASET.PCB.NUM_USE = -1

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''
# __C.BACKBONE.PRETRAINED = './pretrained_models/resnet50.model'

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 1

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127
# Instance size
__C.TRACK.INSTANCE_SIZE = 255
__C.TRACK.SCORE_SIZE = 25
# 143 #143 #118 #143 #118 #93 #80 #73  #67 #55 #43 #25

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44



# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB100 = [0.35, 0.2, 0.45]

__C.HP_SEARCH.GOT10K = [0.7, 0.06, 0.1]

__C.HP_SEARCH.UAV123 = [0.4, 0.2, 0.3]

__C.HP_SEARCH.LaSOT = [0.33, 0.04, 0.3]

__C.HP_SEARCH.PCB = [0.4, 0.2, 0.3]
