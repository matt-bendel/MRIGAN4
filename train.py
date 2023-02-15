import torch
import os
import yaml

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModule import MRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from utils.parse_args import create_arg_parser
# from models.rcGAN import rcGAN
from models.mri_unet import MRIUnet
from models.inpaint_unet import InpaintUNet
from pytorch_lightning import seed_everything

# TODO: REFACTOR UNET INTO BASE PL MODULE

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    if args.default_model_descriptor:
        args.num_noise = 1

    if args.mri:
        with open(os.path.join('configs/mri', 'config.yml'), 'r') as f:
            cfg = yaml.load(f)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_psnr',
            mode='max',
            dirpath=cfg.checkpoint_dir + args.exp_name + '/',
            filename='best_model',
            save_top_k=1
        )

        dm = MRIDataModule(cfg)
        model = MRIUnet(cfg, args.num_noise, args.default_model_descriptor)
    elif args.inpaint:
        with open(os.path.join('configs/inpaint', 'config.yml'), 'r') as f:
            cfg = yaml.load(f)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_ssim',
            mode='max',
            dirpath=cfg.checkpoint_dir + args.exp_name + '/',
            filename='best_model',
            save_top_k=1
        )

        dm = CelebAHQDataModule(cfg)
        model = InpaintUNet(cfg, args.num_noise, args.default_model_descriptor)
    elif args.cs:
        args.checkpoint_dir = "/storage/matt_models/cs_baseline/"
        # TODO: LSUN?? data module
        # TODO: CS Lighnting Module
        pass
    else:
        print("No valid application selected. Please include one of the following args: --mri, --inpaint, --cs.")
        exit()

    trainer = pl.Trainer(accelerator="gpu", devices=args.num_gpus, strategy='ddp',
                         max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback],
                         num_sanity_val_steps=0, profiler="simple")

    if args.resume:
        trainer.fit(model, dm, ckpt_path=cfg.checkpoint_dir + args.exp_name + f'/checkpoint-{args.resume_epoch}.ckpt')
    else:
        trainer.fit(model, dm)

    # checkpoint_callback_gan = ModelCheckpoint(
    #     monitor='epoch',
    #     mode='max',
    #     dirpath='/storage/matt_models/',
    #     filename='checkpoint-{epoch}',
    #     save_top_k=50
    # )
    # model = rcGAN(args)

    # trainer = pl.Trainer(accelerator="gpu", devices=2, strategy='ddp',
    #                      max_epochs=args.num_epochs, callbacks=[checkpoint_callback],
    #                      num_sanity_val_steps=0, profiler="simple")
    #
    if args.resume:
        trainer.fit(model, dm, ckpt_path=args.checkpoint_dir + 'checkpoint.ckpt')
    else:
        trainer.fit(model, dm)
