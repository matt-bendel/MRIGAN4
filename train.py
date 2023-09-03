import torch
import os
import yaml
import types
import json

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModule import MRIDataModule
from datasets.fastmri_multicoil_general import FastMRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from data_loaders.BSD400DataModule import BSD400DataModule
from data_loaders.SRDataModule import SRDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from models.super_res_rcgan import SRrcGAN
from models.adler import Adler
from models.l1_ssim_module import L1SSIMMRI
from models.rcGAN_no_dc import rcGANNoDC
from models.ohayon import Ohayon
from models.mri_unet import MRIUnet
from models.CoModGAN import InpaintUNet
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

# TODO: REFACTOR GAN PL MODULES:
# TODO: SETUP BASE GAN
# TODO: SUBSEQUENT METHODS SHOULD INHERIT FROM BASEGAN
def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")
    print(f"Number of Noise Realizations: {args.num_noise}")
    print(f"Number of GPUs: {args.num_gpus}")

    if args.default_model_descriptor:
        print("USING DEFAULT MODEL DESCRIPTOR...")
        args.num_noise = 1

    if args.mri:
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

        dm = MRIDataModule(cfg, args.mask_type)
        # dm = FastMRIDataModule(base_path=cfg.data_path, batch_size=cfg.batch_size, num_data_loader_workers=2 if not args.dp else 20)

        if args.rcgan:
            noise_structure = {"AWGN": args.awgn, "structure": args.noise_structure}
            if args.nodc:
                model = rcGANNoDC(cfg, args.num_noise, args.default_model_descriptor, args.exp_name, noise_structure, args.num_gpus)
            else:
                model = rcGAN(cfg, args.num_noise, args.default_model_descriptor, args.exp_name, noise_structure, args.num_gpus)
        else:
            noise_structure = {"AWGN": args.awgn, "structure": args.noise_structure}
            model = L1SSIMMRI(cfg, args.num_noise, args.default_model_descriptor, args.exp_name, noise_structure, args.num_gpus)
    elif args.inpaint:
        with open('configs/inpaint/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_ssim',
            mode='max',
            dirpath=cfg.checkpoint_dir + args.exp_name + '/',
            filename='best_model',
            save_top_k=1
        )

        if args.dp:
            cfg.batch_size = cfg.batch_size * args.num_gpus

        dm = CelebAHQDataModule(cfg, args.mask_type)
        model = InpaintUNet(cfg, args.num_noise, args.default_model_descriptor, args.exp_name)
    elif args.sr:
        with open('configs/super_resolution/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        if args.dp:
            cfg.batch_size = cfg.batch_size * args.num_gpus

        dm = SRDataModule(cfg, args.sr_scale)

        noise_structure = {"AWGN": args.awgn, "structure": args.noise_structure}
        model = SRrcGAN(cfg, args.num_noise, args.default_model_descriptor, args.exp_name, noise_structure, args.num_gpus, args.sr_scale)
    elif args.cs:
        with open('configs/cs/config.yml', 'r') as f:
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

        dm = BSD400DataModule(cfg, args.mask_type)
        # TODO: CS Lighnting Module
        pass
    else:
        print("No valid application selected. Please include one of the following args: --mri, --inpaint, --cs.")
        exit()

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
        save_top_k=50
    )
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_psnr',
    #     mode='max',
    #     dirpath=cfg.checkpoint_dir + args.exp_name + '/',
    #     filename='best_model',
    #     save_top_k=1
    # )

    trainer = pl.Trainer(accelerator="gpu", devices=args.num_gpus, strategy='ddp' if not args.dp else 'dp',
                         max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback_epoch],
                         num_sanity_val_steps=2, profiler="simple", logger=wandb_logger, benchmark=False, log_every_n_steps=10)

    if args.resume:
        trainer.fit(model, dm,
                    ckpt_path=cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={args.resume_epoch}.ckpt')
    else:
        trainer.fit(model, dm)

    # trainer.fit(model, dm)
