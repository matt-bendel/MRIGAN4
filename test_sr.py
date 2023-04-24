import torch
import yaml
import os
import types
import json
import pathlib

import numpy as np

from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.super_res_rcgan import SRrcGAN
from data_loaders.SRDataModule import SRDataModule
import torchvision.transforms as T
from evaluation_scripts.metrics import psnr, ssim
import matplotlib.pyplot as plt
from PIL import Image
import PIL

# M_1: 2.15
# C_1: 3.50
# CFID_1: 5.65

# M_2: 1.76
# C_2: 1.18
# CFID_2: 2.94

# M_3: 1.69
# C_3: 0.77
# CFID_3: 2.46

# FID: 7.51

def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False, disc_num=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        ssim_val = ssim(target, image)
        if not kspace:
            pred = disc_num
            ax.set_title(
                f'PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f}, Pred: {pred * 100:.2f}% True') if disc_num else ax.set_title(
                f'PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f}')

    if method == 'Std. Dev':
        im = ax.imshow(image, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if kspace:
            image = image ** 0.4
            target = target ** 0.4
        im = ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(method)

    return im, ax


def generate_error_map(fig, target, recon, method, image_ind, rows, cols, relative=False, k=1, kspace=False):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    if kspace:
        recon = recon ** 0.4
        target = target ** 0.4

    error = (target - recon) if relative else np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    if relative:
        im = ax.imshow(k * error, cmap='bwr', origin='lower', vmin=-0.0001, vmax=0.0001)  # Plot image
        plt.gca().invert_yaxis()
    else:
        im = ax.imshow(k * error, cmap='jet', vmax=1) if kspace else ax.imshow(k * error, cmap='jet', vmax=0.0001)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.mask_type = 1

    if args.default_model_descriptor:
        args.num_noise = 1

    with open('configs/super_resolution/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    if args.dp:
        cfg.batch_size = cfg.batch_size * args.num_gpus

    dm = SRDataModule(cfg, args.sr_scale)
    dm.setup('')
    test_loader = dm.test_dataloader()
    transform = T.ToPILImage()

    with torch.no_grad():
        model = SRrcGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + '/checkpoint-epoch=800.ckpt')
        model.cuda()
        model.eval()
        total = 901

        for i, data in enumerate(test_loader):
            y, x, mean, std = data
            y = y.cuda()
            x = x.cuda()
            mean = mean.cuda()
            std = std.cuda()

            num_code = 32

            gens = torch.zeros(size=(y.size(0), 10, model.args.in_chans, x.shape[-1], x.shape[-1])).cuda()
            for z in range(num_code):
                gens[:, z, :, :, :] = model.forward(y) * std[:, :, None, None] + mean[:, :, None, None]

            avg = torch.mean(gens, dim=1)
            x = x * std[:, :, None, None] + mean[:, :, None, None]
            y = y * std[:, :, None, None] + mean[:, :, None, None]

            for j in range(y.size(0)):
                x = transform(x[j])
                x.save(cfg.checkpoint_dir + f'/gt/gt_{total}.png')

                for l in range(4):
                    temp = transform(avg[j])
                    temp.save(cfg.checkpoint_dir + f'/avg/{total:06d}_sample{l:05d}.png')

                # for z in range(num_code):
                #     temp = transform(gens[j, z])
                #     temp.save(cfg.checkpoint_dir + f'/samps/{total:06d}_sample{z:05d}.png')

                total += 1

