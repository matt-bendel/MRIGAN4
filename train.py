import os
import torch
import random

import numpy as np
import pytorch_lightning as pl

from data_loaders.prepare_data import create_data_loaders
from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from models.GAN import train_dataloader

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args.in_chans = 16
    args.out_chans = 16

    args.checkpoint_dir = "/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/trained_models"

    # train_loader, dev_loader = create_data_loaders(args, big_test=False)

    # init model
    model = rcGAN(args)
    dm = MRIDataModule(args)
    # fit trainer on 128 GPUs
    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", default_root_dir=args.checkpoint_dir)
    trainer.fit(model, dm)
