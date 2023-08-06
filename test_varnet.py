import torch
import yaml
import os
import types
import json
import pathlib
import lpips

import numpy as np
from matplotlib import gridspec

from data_loaders.MRIDataModuleVarnet import MRIDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from fastmri.pl_modules.varnet_module import VarNetModule
from models.mri_unet import MRIUnet
from models.rcGAN import rcGAN
from models.l1_ssim_module import L1SSIMMRI
from utils.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.metrics import psnr, ssim
import matplotlib.pyplot as plt
from utils.fftc import ifft2c_new, fft2c_new
import sigpy as sp
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
    embed_ims = torch.zeros(size=(3, 384, 384))
    tens_im = torch.tensor(im)

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
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    with open('configs/mri/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    cfg.batch_size = cfg.batch_size * 4
    dm = MRIDataModule(cfg, args.mask_type, big_test=True)

    dm.setup()
    test_loader = dm.test_dataloader()
    model_alias = VarNetModule

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    lpips_met = lpips.LPIPS(net='alex')
    dists_met = DISTS()

    with torch.no_grad():
        model = model_alias.load_from_checkpoint(
            checkpoint_path='/storage/matt_models/mri/e2e_varnet/varnet/varnet_demo/checkpoints/epoch=14-step=137880.ckpt')
        model.cuda()
        model.eval()

        n_samps = [1]

        n_psnrs = []
        n_ssims = []
        n_lpipss = []
        n_distss = []

        for n in n_samps:
            trial_distss = []

            print(f"{n} SAMPLES")
            psnrs = []
            ssims = []
            apsds = []
            lpipss = []
            distss = []

            med_psnrs = []
            med_ssims = []
            med_lpipss = []
            med_distss = []

            for i, data in enumerate(test_loader):
                y, x, mask, maps, num_low_freqs = data
                y = y.cuda()
                x = x.cuda()
                mask = mask.cuda()
                num_low_freqs = num_low_freqs.cuda()

                recon = model(y, mask, num_low_freqs)
                gt = x

                batch_psnrs = []
                batchs_ssims = []
                batch_apsds = []

                for j in range(y.size(0)):
                    S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))
                    gt_ksp, avg_ksp = tensor_to_complex_np((gt[j]).cpu()), tensor_to_complex_np(
                        (recon[j]).cpu())

                    avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                    gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                    psnrs.append(psnr(gt_np, avg_gen_np))
                    ssims.append(ssim(gt_np, avg_gen_np))
                    lpipss.append(lpips_met(rgb(gt_np), rgb(avg_gen_np)).numpy())
                    distss.append(dists_met(rgb(gt_np, unit_norm=True), rgb(avg_gen_np, unit_norm=True)).numpy())


            n_psnrs.append(np.mean(psnrs))
            n_ssims.append(np.mean(ssims))
            n_lpipss.append(np.mean(lpipss))
            n_distss.append(np.mean(distss))

        psnr_str = ''
        ssim_str = ''
        lpips_str = ''
        dists_str = ''

        for i in range(len(n_psnrs)):
            psnr_str = f'{psnr_str} {n_psnrs[i]:.2f} \pm'
            ssim_str = f'{ssim_str} {n_ssims[i]:.4f} \pm'
            lpips_str = f'{lpips_str} {n_lpipss[i]:.4f} \pm'
            dists_str = f'{dists_str} {n_distss[i]:.4f} \pm'

        print(f'PSNR:\n{psnr_str}')
        print(f'SSIM:\n{ssim_str}')
        print(f'LPIPS:\n{lpips_str}')
        print(f'DISTS:\n{dists_str}')

            # print(f'APSD: {np.mean(apsds)}')
    exit()
    # TODO: CFID, FID for varnet
    cfids = []
    m_comps = []
    c_comps = []

    inception_embedding = VGG16Embedding(parallel=True)
    # CFID_1
    cfid_metric = CFIDMetric(gan=model,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=32)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)
    #
    # inception_embedding = VGG16Embedding(parallel=True)
    # # CFID_2
    # cfid_metric = CFIDMetric(gan=model,
    #                          loader=val_dataloader,
    #                          image_embedding=inception_embedding,
    #                          condition_embedding=inception_embedding,
    #                          cuda=True,
    #                          args=cfg,
    #                          ref_loader=False,
    #                          num_samps=8)
    #
    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)
    #
    # inception_embedding = VGG16Embedding(parallel=True)
    # # CFID_3
    # cfid_metric = CFIDMetric(gan=model,
    #                          loader=val_dataloader,
    #                          image_embedding=inception_embedding,
    #                          condition_embedding=inception_embedding,
    #                          cuda=True,
    #                          args=cfg,
    #                          ref_loader=train_dataloader,
    #                          num_samps=1)
    #
    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)
    #

    # n_samps = [1, 2, 4, 8, 16, 32]
    # for n in n_samps:
    # #     print(f"{n} SAMPLES")
    # inception_embedding = VGG16Embedding()
    # fid_metric = FIDMetric(gan=model,
    #                        ref_loader=train_dataloader,
    #                        loader=val_dataloader,
    #                        image_embedding=inception_embedding,
    #                        condition_embedding=inception_embedding,
    #                        cuda=True,
    #                        args=cfg)
    # fid, _ = fid_metric.get_fid()

    # print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
    # print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
    # print(f'APSD: {np.mean(apsds)}')
    for l in range(1):
        print(f'CFID_{l+1}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')
