import torch
import yaml
import os
import types
import json
import pathlib
import lpips

import numpy as np
from matplotlib import gridspec

from data_loaders.MRIDataModule import MRIDataModule
from datasets.fastmri_multicoil_general import FastMRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.mri_unet import MRIUnet
from models.rcGAN import rcGAN
from models.rcGAN_no_dc import rcGANNoDC
from models.adler import Adler
from models.ohayon import Ohayon
from models.CoModGAN import InpaintUNet
from models.l1_ssim_module import L1SSIMMRI
from utils.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.metrics import psnr, ssim
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from evaluation_scripts.fid.fid_metric import FIDMetric
import matplotlib.pyplot as plt
from utils.fftc import ifft2c_new, fft2c_new
import sigpy as sp
import sigpy.mri as mr
from data.transforms import to_tensor
from models.architectures.old_gen import GeneratorModel
from DISTS_pytorch import DISTS
import matplotlib.patches as patches


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

# GRO TEST
# python test.py --mri --rcgan --exp-name neurips/rcgan_big --mask-type 1 && python test.py --mri --rcgan --exp-name neurips/random_blind --mask-type 1 && python test.py --mri --rcgan --exp-name neurips/random_proposed --mask-type 1 && python test.py --mri --rcgan --exp-name neurips/rcgan_statis --mask-type 1

# DYNAMIC RANDOM MASK TEST
# python test.py --mri --rcgan --exp-name neurips/rcgan_big --mask-type 2 && python test.py --mri --rcgan --exp-name neurips/random_blind --mask-type 2 && python test.py --mri --rcgan --exp-name neurips/random_proposed --mask-type 2 && python test.py --mri --rcgan --exp-name neurips/rcgan_statis --mask-type 2

# STATIC RANDOM MASK TEST
# python test.py --mri --rcgan --exp-name neurips/rcgan_big --mask-type 3 && python test.py --mri --rcgan --exp-name neurips/random_blind --mask-type 3 && python test.py --mri --rcgan --exp-name neurips/random_proposed --mask-type 3 && python test.py --mri --rcgan --exp-name neurips/rcgan_statis --mask-type 3

# python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 4 && python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 4


# python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 1 --R 8 && python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 1 --R 7 && python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 1 --R 6 && python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 1 --R 5 && python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 1 --R 4 && python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 1 --R 3 && python test.py --mri --rcgan --exp-name neurips/rcgan_proposed_many_R --mask-type 1 --R 2
# python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 1 --R 8 && python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 1 --R 7 && python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 1 --R 6 && python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 1 --R 5 && python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 1 --R 4 && python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 1 --R 3 && python test.py --mri --rcgan --exp-name neurips/rcgan_blind_many_R --mask-type 1 --R 2


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


def rgb(im, unit_norm=False):
    embed_ims = torch.zeros(size=(3, 384, 384)).cuda()
    tens_im = torch.tensor(im).cuda()

    if unit_norm:
        tens_im = (tens_im - torch.min(tens_im)) / (torch.max(tens_im) - torch.min(tens_im))
    else:
        tens_im = 2 * (tens_im - torch.min(tens_im)) / (torch.max(tens_im) - torch.min(tens_im)) - 1

    embed_ims[0, :, :] = tens_im
    embed_ims[1, :, :] = tens_im
    embed_ims[2, :, :] = tens_im

    return embed_ims.unsqueeze(0)


def get_com_fig(np_gt, np_avg, np_med, n, fig_num):
    zoom_startx = 80  # np.random.randint(120, 250)
    zoom_starty1 = 180  # np.random.randint(30, 80)
    zoom_starty2 = 180  # np.random.randint(260, 300)

    p = np.random.rand()
    zoom_starty = zoom_starty1
    if p <= 0.5:
        zoom_starty = zoom_starty2

    zoom_length = 80

    x_coord = zoom_startx + zoom_length
    y_coords = [zoom_starty, zoom_starty + zoom_length]

    nrow = 2
    ncol = 4

    fig = plt.figure(figsize=(ncol + 1, nrow + 1))

    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    ax = plt.subplot(gs[0, 0])
    ax.imshow(np_gt, cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Truth')

    ax1 = ax

    rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                             edgecolor='r',
                             facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    ax = plt.subplot(gs[0, 1])
    ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
              cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Truth')

    connection_path_1 = patches.ConnectionPatch([x_coord, y_coords[0]], [0, 0], coordsA=ax1.transData,
                                                coordsB=ax.transData, color='r')
    fig.add_artist(connection_path_1)
    connection_path_2 = patches.ConnectionPatch([x_coord, y_coords[1]], [0, zoom_length],
                                                coordsA=ax1.transData,
                                                coordsB=ax.transData, color='r')
    fig.add_artist(connection_path_2)

    ax = plt.subplot(gs[0, 2])
    ax.imshow(np_avg[zoom_starty:zoom_starty + zoom_length,
              zoom_startx:zoom_startx + zoom_length], cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Avg')

    ax = plt.subplot(gs[0, 3])
    ax.imshow(np_med[zoom_starty:zoom_starty + zoom_length,
              zoom_startx:zoom_startx + zoom_length], cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Median')

    ax = plt.subplot(gs[1, 2])
    ax.imshow(2 * np.abs(np_avg - gt_np)[zoom_starty:zoom_starty + zoom_length,
                  zoom_startx:zoom_startx + zoom_length], cmap='jet', vmin=0,
              vmax=np.max(np.abs(np_avg - gt_np)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(gs[1, 3])
    ax.imshow(2 * np.abs(med_np - gt_np)[zoom_starty:zoom_starty + zoom_length,
                  zoom_startx:zoom_startx + zoom_length], cmap='jet', vmin=0,
              vmax=np.max(np.abs(np_avg - gt_np)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(f'med_tests_{n}_samps_{fig_num}.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    seed_everything(1, workers=True)

    with open('configs/mri/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    cfg.batch_size = cfg.batch_size * 4

    dm = MRIDataModule(cfg, 1, big_test=True)

    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            y, x, mask, mean, std, maps, _, _ = data
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()
            maps = maps.cuda()

            reformatted_tensor = torch.zeros(size=(8, 384, 384, 2),
                                             device=x.device)
            reformatted_tensor[:, :, :, 0] = x[0, 0:8, :, :] * std[0] + mean[0]
            reformatted_tensor[:, :, :, 1] = x[0, 8:16, :, :] * std[0] + mean[0]

            x_hat = torch.view_as_complex(reformatted_tensor)
            maps_complex_conj = torch.view_as_complex(maps[0])

            new_im = torch.einsum('bij, bjk -> bik ', maps_complex_conj, x_hat)
            print(new_im.shape)
            exit()

            x_hat_mag = x_hat.abs().cpu().numpy()

            plt.imshow(x_hat_mag, cmap='gray')
            plt.savefig('coiltest.png')

            exit()
