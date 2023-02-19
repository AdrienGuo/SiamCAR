import math
import time

import ipdb
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

from pysot.core.config import cfg
from pysot.utils.average_meter import AverageMeter
from pysot.utils.distributed import reduce_gradients


def freeze_backbone_batchnorm(model):
    # 在 cfg.BACKBONE.TRAIN_EPOCH 之前要把 Backbone 的 Batch Norm 的層鎖住
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


def train_one_epoch(
    train_loader,
    model,
    scaler,
    optimizer,
    epoch,
    accum_iters,
) -> AverageMeter:
    # Switch model to train mode.
    model.train()

    losses_meter = AverageMeter("Loss")
    no_pos_num = 0

    if epoch + 1 < cfg.BACKBONE.TRAIN_EPOCH:
        freeze_backbone_batchnorm(model)

    end = time.time()
    for batch_idx, data in enumerate(train_loader):
        # one batch
        # Runs in mixed precision
        with autocast():
            losses = model(data)

        if losses['cls_pos'].item() == 0:
            # 當完全沒有正樣本的情況
            # TODO: 這應該要想更好的方法處理
            no_pos_num += 1
            continue

        loss = losses['total'] / accum_iters
        scaler.scale(loss).backward()

        losses = {key: value.item() for key, value in losses.items()}
        losses_meter.update(val=losses, num=1)

        # accumulate gradient
        if (batch_idx + 1) % accum_iters == 0 or (batch_idx + 1) == len(train_loader):
            if is_valid_number(loss.data.item()):
                reduce_gradients(model)

                # Ref: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
                # Clip gradient
                # Ref: https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()

                # TODO: 不知道 zero_grad 能不能放這裡
                optimizer.zero_grad()
            losses_meter.display(type="val")

    print(f"No cls_pos numbers in train: {no_pos_num}")
    losses_meter.display(type="avg")
    elapsed = time.time() - end
    print(f"({elapsed:.3f}s, {elapsed / len(train_loader.dataset):.3}s/img)")
    return losses_meter.avg


def validate(
    test_loader,
    model,
    epoch
):
    model.eval()

    losses_meter = AverageMeter("Loss")
    no_pos_num = 0

    if epoch + 1 < cfg.BACKBONE.TRAIN_EPOCH:
        freeze_backbone_batchnorm(model)

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Runs in mixed precision
            with autocast():
                losses = model(data)
            if losses['cls_pos'] == 0:
                no_pos_num += 1
                continue
            losses = {key: value.item() for key, value in losses.items()}
            losses_meter.update(val=losses, num=1)

    print(f"NO cls_pos numbers in validation: {no_pos_num}")
    losses_meter.display(type="avg")
    return losses_meter.avg


def save_checkpoint(
    stats,
    filename
):
    torch.save(stats, filename)
    print(f"Save model to: {filename}")
