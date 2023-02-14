import torch

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from models.mri_unet import MRIUnet
from pytorch_lightning import seed_everything

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.in_chans = 16
    args.out_chans = 16

    args.checkpoint_dir = "/storage/matt_models/random_baseline/"

    # TODO: MODULARITY ARGS
    checkpoint_callback_unet = ModelCheckpoint(
        monitor='val_psnr',
        mode='max',
        dirpath=args.checkpoint_dir,
        filename='checkpoint-{epoch}',
        save_top_k=1
    )

    dm = MRIDataModule(args)
    model = MRIUnet(args, 0)
    trainer = pl.Trainer(accelerator="gpu", devices=1, strategy='ddp',
                         max_epochs=args.num_epochs, callbacks=[checkpoint_callback_unet],
                         num_sanity_val_steps=0, profiler="simple")
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
    # if args.resume:
    #     trainer.fit(model, dm, ckpt_path=args.checkpoint_dir + 'checkpoint.ckpt')
    # else:
    #     trainer.fit(model, dm)
