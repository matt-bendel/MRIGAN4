import torch
import os
import yaml
import types
import json

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModuleDiffusion import MRIDataModuleDiffusion
from utils.parse_args import create_arg_parser
from models.DDPM import DDPM
from models.l1_ssim_module import L1SSIMMRI
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")
    print(f"Number of GPUs: {args.num_gpus}")

    with open('configs/mri/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='best_model',
        save_top_k=1
    )

    if args.dp:
        cfg.batch_size = cfg.batch_size * args.num_gpus

    dm = MRIDataModuleDiffusion()
    # dm = FastMRIDataModule(base_path=cfg.data_path, batch_size=cfg.batch_size, num_data_loader_workers=2 if not args.dp else 20)

    model = DDPM()

    wandb_logger = WandbLogger(
        project="neurips",
        name=args.exp_name,
        log_model="all",
        save_dir=cfg.checkpoint_dir + 'wandb'
    )
    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='checkpoint-{epoch}',
        save_top_k=1
    )

    trainer = pl.Trainer(accelerator="gpu", devices=4, strategy='ddp',
                         max_epochs=50, callbacks=[checkpoint_callback_epoch],
                         num_sanity_val_steps=2, profiler="simple", logger=wandb_logger, benchmark=False, log_every_n_steps=10)

    trainer.fit(model, dm)

