import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.architectures.super_res_gen import RRDBNet, UNet

class SRGen(nn.Module):
    def __init__(self, in_chans, scale):
        super(SRGen, self).__init__()
        self.rrdb = RRDBNet(in_chans)
        self.unet = UNet(in_channels=in_chans+1)

    def forward(self, x, noise):
        out = self.rrdb(x)
        cat_tensor = torch.cat([out, noise], dim=1)
        print(cat_tensor.shape)
        out = self.unet(cat_tensor)
        exit()

        return out