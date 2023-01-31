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

    args.checkpoint_dir = "/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/trained_models/"

    # train_loader, dev_loader = create_data_loaders(args, big_test=False)

    # init model
    checkpoint_callback_best = ModelCheckpoint(
        monitor='final_val_psnr',
        mode='max',
        dirpath=args.checkpoint_dir,
        filename='best_model',
        save_top_k=1
    )

    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=args.checkpoint_dir,
        filename='checkpoint',
        save_top_k=1
    )

    model = rcGAN(args)

    dm = MRIDataModule(args)
    # fit trainer on 128 GPUs
    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", default_root_dir=args.checkpoint_dir,
                         max_epochs=args.num_epochs, callbacks=[checkpoint_callback_best, checkpoint_callback_epoch],
                         auto_select_gpus=False)

    if args.resume:
        trainer.fit(model, dm, ckpt_path=args.checkpoint_dir + 'checkpoint.ckpt')
    else:
        trainer.fit(model, dm)
