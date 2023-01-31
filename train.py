import os
import torch
import random

import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from pytorch_lightning import seed_everything

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1)

    args.in_chans = 16
    args.out_chans = 16

    args.checkpoint_dir = "/storage/matt_models/"

    model = rcGAN(args)

    dm = MRIDataModule(args)
    # fit trainer on 128 GPUs
    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", default_root_dir=args.checkpoint_dir,
                         max_epochs=args.num_epochs, auto_select_gpus=False)

    if args.resume:
        trainer.fit(model, dm, ckpt_path=args.checkpoint_dir + 'checkpoint.ckpt')
    else:
        trainer.fit(model, dm)
