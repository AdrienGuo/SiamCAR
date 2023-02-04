# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ipdb
import torch.nn as nn


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.BatchNorm2d(out_channels, track_running_stats=False),
        )

    def forward(self, x, method):
        """
        if method == "origin": 不要做裁切。
        else: 會對 template 做裁切，方法如下：
            原文 (SiamRPN++)
            Thus we crop the center 7 × 7 regions [41] as the template
            feature where each feature cell can still capture the entire
            target region.
        """

        x = self.downsample(x)

        if method == "origin":
            pass
        else:
            if x.size(3) < 20:
                l = 4
                r = l + 7
                x = x[:, :, l:r, l:r]

        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features, method):
        if self.num == 1:
            return self.downsample(features, method)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i], method).contiguous())
            return out
