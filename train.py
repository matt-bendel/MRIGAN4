import torch
import os
import yaml
import types
import json

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModule import MRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from data_loaders.BSD400DataModule import BSD400DataModule
from utils.parse_args import create_arg_parser
from models.rcGAN_no_d import rcGAN
from models.mri_unet import MRIUnet
from models.CoModGAN import InpaintUNet
from pytorch_lightning import seed_everything

# TODO: REFACTOR UNET INTO BASE PL MODULE
def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

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

        dm = MRIDataModule(cfg, args.mask_type)
        if args.rcgan:
            noise_structure = {"AWGN": args.awgn, "structure": args.noise_structure}
            model = rcGAN(cfg, args.num_noise, args.default_model_descriptor, args.exp_name, noise_structure)
        else:
            model = MRIUnet(cfg, args.num_noise, args.default_model_descriptor, args.exp_name)
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

        dm = CelebAHQDataModule(cfg, args.mask_type)
        model = InpaintUNet(cfg, args.num_noise, args.default_model_descriptor, args.exp_name)
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

        dm = BSD400DataModule(cfg, args.mask_type)
        # TODO: CS Lighnting Module
        pass
    else:
        print("No valid application selected. Please include one of the following args: --mri, --inpaint, --cs.")
        exit()

    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='checkpoint-{epoch}',
        save_top_k=50
    )

    trainer = pl.Trainer(accelerator="gpu", devices=args.num_gpus, strategy='ddp',
                         max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback, checkpoint_callback_epoch],
                         num_sanity_val_steps=0, profiler="simple")

    if args.resume:
        trainer.fit(model, dm,
                    ckpt_path=cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={args.resume_epoch}.ckpt')
    else:
        trainer.fit(model, dm)

    # trainer.fit(model, dm)
