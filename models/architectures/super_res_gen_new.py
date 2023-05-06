import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.architectures.super_res_gen import RRDBNet
from models.architectures.our_gen_unet_only_sr import UNetModel

class SRGen(nn.Module):
    def __init__(self, in_chans, scale):
        super(SRGen, self).__init__()
        self.scale = scale
        self.rrdb = RRDBNet(in_chans+3)
        self.unet = UNetModel(in_chans, 3, 4)

    def forward(self, x, noise):
        out = self.rrdb(torch.cat([x, noise], dim=1))

        # up_lr = F.interpolate(lr, scale_factor=4, mode='bicubic')
        out = self.unet(out, x)

        return out
