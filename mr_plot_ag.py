import torch
import yaml
import os
import types
import json
import pathlib

import numpy as np
import matplotlib.patches as patches
from utils.fftc import ifft2c_new, fft2c_new

from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.rcGAN import rcGAN
from models.adler import Adler
from models.ohayon import Ohayon
from models.rcGAN_no_dc import rcGANNoDC
from models.l1_ssim_module import L1SSIMMRI
from utils.math import complex_abs, tensor_to_complex_np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sigpy as sp
from evaluation_scripts.metrics import psnr, ssim
from scipy import ndimage
from fastmri.pl_modules.varnet_module import VarNetModule

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

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.mask_type = 1

    if args.default_model_descriptor:
        args.num_noise = 1

    with open('configs/mri/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    cfg.batch_size = cfg.batch_size
    dm = MRIDataModule(cfg, args.mask_type, big_test=True)
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        rcGAN_model_wo_gr_w_dc = rcGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + '/neurips/rcgan_agnostic_blind/checkpoint_best.ckpt')
        rcGAN_model_w_gr_wo_dc = rcGANNoDC.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + '/neurips/rcgan_agnostic_proposed_no_dc/checkpoint_best.ckpt')
        rcGAN_model_w_gr_w_dc = rcGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + '/neurips/rcgan_agnostic_proposed/checkpoint_best.ckpt')
        l1_ssim_model = L1SSIMMRI.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + '/neurips/l1_ssim_agnostic/checkpoint_best.ckpt')
        varnet_model = VarNetModule.load_from_checkpoint(
            checkpoint_path='/storage/matt_models/mri/neurips/e2e_varnet_agnostic/varnet/varnet_demo/checkpoints/checkpoint_best.ckpt')

        rcGAN_model_wo_gr_w_dc.cuda()
        rcGAN_model_w_gr_wo_dc.cuda()
        rcGAN_model_w_gr_w_dc.cuda()
        l1_ssim_model.cuda()
        varnet_model.cuda()

        rcGAN_model_wo_gr_w_dc.eval()
        rcGAN_model_w_gr_wo_dc.eval()
        rcGAN_model_w_gr_w_dc.eval()
        l1_ssim_model.eval()
        varnet_model.eval()

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std, maps, fname, slice, num_low_freqs = data
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()
            num_low_freqs = num_low_freqs.cuda()
            varnet_y = fft2c_new(rcGAN_model_wo_gr_w_dc.reformat(y) * std[:] + mean[:])

            gens_rcgan_wo_gr_w_dc = torch.zeros(
                size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            gens_rcgan_w_gr_wo_dc = torch.zeros(
                size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            gens_rcgan_w_gr_w_dc = torch.zeros(
                size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            gens_l1_ssim = torch.zeros(
                size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            gens_varnet = torch.zeros(
                size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size)).cuda()

            for z in range(cfg.num_z_test):
                gens_rcgan_wo_gr_w_dc[:, z, :, :, :, :] = rcGAN_model_wo_gr_w_dc.reformat(rcGAN_model_wo_gr_w_dc.forward(y, mask))
                gens_rcgan_w_gr_wo_dc[:, z, :, :, :, :] = rcGAN_model_w_gr_wo_dc.reformat(rcGAN_model_w_gr_wo_dc.forward(y, mask))
                gens_rcgan_w_gr_w_dc[:, z, :, :, :, :] = rcGAN_model_w_gr_w_dc.reformat(rcGAN_model_w_gr_w_dc.forward(y, mask))
                gens_l1_ssim[:, z, :, :, :, :] = l1_ssim_model.reformat(l1_ssim_model.forward(y, mask))
                gens_varnet[:, z, :, :] = varnet_model(varnet_y.float(), mask == 1, num_low_freqs)

            avg_rcgan_wo_gr_w_dc = torch.mean(gens_rcgan_wo_gr_w_dc, dim=1)
            avg_rcgan_w_gr_wo_dc = torch.mean(gens_rcgan_w_gr_wo_dc, dim=1)
            avg_rcgan_w_gr_w_dc = torch.mean(gens_rcgan_w_gr_w_dc, dim=1)
            avg_l1_ssim = torch.mean(gens_l1_ssim, dim=1)
            avg_varnet = torch.mean(gens_varnet, dim=1)

            print("Got Recons")

            gt = rcGAN_model_wo_gr_w_dc.reformat(x)
            zfr = rcGAN_model_wo_gr_w_dc.reformat(y)

            # TODO: Add Langevin, L1+SSIM model

            for j in range(y.size(0)):
                np_avgs = {
                    'l1_ssim': None,
                    'varnet': None,
                    'rcgan_wo_gr_w_dc': None,
                    'rcgan_w_gr_wo_dc': None,
                    'rcgan_w_gr_w_dc': None
                }

                np_samps = {
                    'rcgan_wo_gr_w_dc': [],
                    'rcgan_w_gr_wo_dc': [],
                    'rcgan_w_gr_w_dc': []
                }

                np_stds = {
                    'rcgan_wo_gr_w_dc': None,
                    'rcgan_w_gr_wo_dc': None,
                    'rcgan_w_gr_w_dc': None
                }

                np_gt = None

                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))

                np_gt = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)
                np_zfr = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((zfr[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)

                np_avgs['rcgan_wo_gr_w_dc'] = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((avg_rcgan_wo_gr_w_dc[j] * std[j] + mean[j]).cpu())).abs().numpy(),
                    180)
                np_avgs['rcgan_w_gr_wo_dc'] = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((avg_rcgan_w_gr_wo_dc[j] * std[j] + mean[j]).cpu())).abs().numpy(),
                    180)
                np_avgs['rcgan_w_gr_w_dc'] = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((avg_rcgan_w_gr_w_dc[j] * std[j] + mean[j]).cpu())).abs().numpy(),
                    180)
                np_avgs['l1_ssim'] = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((avg_l1_ssim[j] * std[j] + mean[j]).cpu())).abs().numpy(),
                    180)
                np_avgs['varnet'] = ndimage.rotate(
                    torch.tensor(avg_varnet[j].cpu().numpy()).abs().numpy(),
                    180)

                print("Got Means")

                for z in range(cfg.num_z_test):
                    np_samps['rcgan_wo_gr_w_dc'].append(ndimage.rotate(torch.tensor(
                        S.H * tensor_to_complex_np((gens_rcgan_wo_gr_w_dc[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))
                    np_samps['rcgan_w_gr_wo_dc'].append(ndimage.rotate(torch.tensor(
                        S.H * tensor_to_complex_np((gens_rcgan_w_gr_wo_dc[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))
                    np_samps['rcgan_w_gr_w_dc'].append(ndimage.rotate(torch.tensor(
                        S.H * tensor_to_complex_np((gens_rcgan_w_gr_w_dc[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))

                print("Got Samps")

                np_stds['rcgan_wo_gr_w_dc'] = np.std(np.stack(np_samps['rcgan_wo_gr_w_dc']), axis=0)
                np_stds['rcgan_w_gr_wo_dc'] = np.std(np.stack(np_samps['rcgan_w_gr_wo_dc']), axis=0)
                np_stds['rcgan_w_gr_w_dc'] = np.std(np.stack(np_samps['rcgan_w_gr_w_dc']), axis=0)

                recon_directory = f'/storage/fastMRI_brain/Langevin_Recons_R={args.R}/'
                langevin_recons = np.zeros((32, 384, 384))
                recon_object = None
                exceptions = False

                for l in range(cfg.num_z_test):
                    try:
                        new_filename = recon_directory + fname[
                            j] + f'|langevin|slide_idx_{slice[j]}_R={args.R}_sample={l}_outputs.pt'
                        recon_object = torch.load(new_filename)
                    except Exception as e:
                        print(e)
                        exceptions = True
                        break
                    # temp_recon = unnormalize(recon_object['mvue'], recon_object['zfr'])

                    langevin_recons[l] = ndimage.rotate(
                        complex_abs(recon_object['mvue'][0].permute(1, 2, 0)).cpu().numpy(), 180)

                if exceptions:
                    exceptions = False
                    continue

                langevin_gt = ndimage.rotate(recon_object['gt'][0][0].abs().cpu().numpy(), 180)
                langevin_avg = np.mean(langevin_recons, axis=0)
                langevin_std = np.std(langevin_recons, axis=0)

                print("Got Langevin")

                recon_directory = f'/storage/matt_models/mri/ddrm_R={args.R}/'
                ddrm_recons = np.zeros((32, 384, 384))
                recon_object = None
                exceptions = False

                for l in range(cfg.num_z_test):
                    try:
                        new_filename = recon_directory + f'{fname[j]}_{slice[j]}_sample_{l}.pt'
                        recon_object = torch.load(new_filename)
                    except Exception as e:
                        print(e)
                        exceptions = True
                        continue
                    # temp_recon = unnormalize(recon_object['mvue'], recon_object['zfr'])

                    ddrm_recons[l, :, :] = recon_object[0, :, :].cpu().numpy()

                if exceptions:
                    exceptions = False
                    continue

                new_filename = recon_directory + f'{fname[j]}_{slice[j]}_gt.pt'
                ddrm_gt = torch.load(new_filename)[0, :, :].cpu().numpy()
                ddrm_avg = np.mean(ddrm_recons, axis=0)
                ddrm_std = np.std(ddrm_recons, axis=0)
                print("Got DDRM")
                print("Plot time baby")

                keys = ['l1_ssim', 'varnet', 'rcgan_wo_gr_w_dc', 'rcgan_w_gr_wo_dc', 'rcgan_w_gr_w_dc']
                if j == 0:
                    zoom_startx = 180
                    zoom_starty = 40
                    zoom_length = 80
                else:
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

                # TODO: OG fig plot
                # TODO: metrics
                # OG FIG
                nrow = 6
                ncol = 6

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

                count = 1
                for method in keys:
                    ax = plt.subplot(gs[0, count])
                    ax.imshow(np_avgs[method], cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    psnr_val = psnr(np_gt, np_avgs[method])
                    ssim_val = ssim(np_gt, np_avgs[method])

                    # ax.text(0.46, 0.04, f'PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}',
                    #         horizontalalignment='center', verticalalignment='center', fontsize=3.5, color='yellow',
                    #         transform=ax.transAxes)
                    count += 1

                ax = plt.subplot(gs[0, count])
                ax.imshow(langevin_avg, cmap='gray', vmin=0, vmax=0.7 * np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = plt.subplot(gs[0, count+1])
                ax.imshow(ddrm_avg, cmap='gray', vmin=0, vmax=0.7 * np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                psnr_val = psnr(langevin_gt, langevin_avg)
                ssim_val = ssim(langevin_gt, langevin_avg)

                # ax.text(0.46, 0.04, f'PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}',
                #         horizontalalignment='center', verticalalignment='center', fontsize=3.5, color='yellow',
                #         transform=ax.transAxes)

                count = 1
                for method in keys:
                    ax = plt.subplot(gs[1, count])
                    im = ax.imshow(2 * np.abs(np_avgs[method] - np_gt), cmap='jet', vmin=0,
                                   vmax=np.max(np.abs(np_avgs['rcgan_wo_gr_w_dc'] - np_gt)))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if count == 1:
                        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                        # Appropriately rescale final axis so that colorbar does not effect formatting
                        pad = 0.01
                        width = 0.02
                        cbar_ax = fig.add_axes([x10 - 2 * pad, y10, width, y11 - y10])
                        cbar = fig.colorbar(im, cax=cbar_ax, format='%.0e',
                                            orientation='vertical')  # Generate colorbar
                        cbar.ax.locator_params(nbins=3)
                        cbar.ax.yaxis.set_ticks_position("left")
                        cbar.ax.tick_params(labelsize=6)
                        cbar.ax.tick_params(rotation=0)
                        tl = cbar.ax.get_yticklabels()

                        # set the alignment for the first and the last
                        tl[0].set_verticalalignment('bottom')
                        tl[-1].set_verticalalignment('top')

                    count += 1

                ax = plt.subplot(gs[1, count])
                ax.imshow(3 * np.abs(langevin_avg - langevin_gt), cmap='jet', vmin=0,
                          vmax=np.max(np.abs(np_avgs['rcgan_wo_gr_w_dc'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = plt.subplot(gs[1, count+1])
                ax.imshow(3 * np.abs(ddrm_avg - ddrm_gt), cmap='jet', vmin=0,
                          vmax=np.max(np.abs(np_avgs['rcgan_wo_gr_w_dc'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 1
                for method in keys:
                    if method != 'l1_ssim' and method != 'varnet':
                        ax = plt.subplot(gs[2, count])
                        ax.imshow(np_stds[method], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax = plt.subplot(gs[2, count])
                        im = ax.imshow(np.zeros((384, 384)), cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
                        if method == 'l1_ssim':
                            [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                            # Appropriately rescale final axis so that colorbar does not effect formatting
                            pad = 0.01
                            width = 0.02
                            cbar_ax = fig.add_axes([x10 - 2 * pad, y10, width, y11 - y10])
                            cbar = fig.colorbar(im, cax=cbar_ax, format='%.0e',
                                                orientation='vertical')  # Generate colorbar
                            cbar.ax.locator_params(nbins=3)
                            cbar.ax.yaxis.set_ticks_position("left")
                            cbar.ax.tick_params(labelsize=6)
                            cbar.ax.tick_params(rotation=0)
                            tl = cbar.ax.get_yticklabels()

                            # set the alignment for the first and the last
                            tl[0].set_verticalalignment('bottom')
                            tl[-1].set_verticalalignment('top')

                    count += 1

                ax = plt.subplot(gs[2, count])
                ax.imshow(langevin_std, cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = plt.subplot(gs[2, count+1])
                ax.imshow(ddrm_std, cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = plt.subplot(gs[3, 0])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 1
                for method in keys:
                    ax = plt.subplot(gs[3, count])
                    ax.imshow(np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    psnr_val = psnr(np_gt, np_avgs[method])
                    ssim_val = ssim(np_gt, np_avgs[method])

                    # ax.text(0.46, 0.04, f'PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}',
                    #         horizontalalignment='center', verticalalignment='center', fontsize=3.5, color='yellow',
                    #         transform=ax.transAxes)
                    count += 1

                ax = plt.subplot(gs[3, count])
                ax.imshow(langevin_avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='gray', vmin=0, vmax=0.7 * np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = plt.subplot(gs[3, count + 1])
                ax.imshow(ddrm_avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='gray', vmin=0, vmax=0.7 * np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                # ax.text(0.46, 0.04, f'PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}',
                #         horizontalalignment='center', verticalalignment='center', fontsize=3.5, color='yellow',
                #         transform=ax.transAxes)

                count = 1
                for method in keys:
                    ax = plt.subplot(gs[4, count])
                    im = ax.imshow(2 * np.abs(np_avgs[method] - np_gt)[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='jet', vmin=0,
                                   vmax=np.max(np.abs(np_avgs['rcgan_wo_gr_w_dc'] - np_gt)))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if count == 1:
                        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                        # Appropriately rescale final axis so that colorbar does not effect formatting
                        pad = 0.01
                        width = 0.02
                        cbar_ax = fig.add_axes([x10 - 2 * pad, y10, width, y11 - y10])
                        cbar = fig.colorbar(im, cax=cbar_ax, format='%.0e',
                                            orientation='vertical')  # Generate colorbar
                        cbar.ax.locator_params(nbins=3)
                        cbar.ax.yaxis.set_ticks_position("left")
                        cbar.ax.tick_params(labelsize=6)
                        cbar.ax.tick_params(rotation=0)
                        tl = cbar.ax.get_yticklabels()

                        # set the alignment for the first and the last
                        tl[0].set_verticalalignment('bottom')
                        tl[-1].set_verticalalignment('top')

                    count += 1

                ax = plt.subplot(gs[4, count])
                ax.imshow(3 * np.abs(langevin_avg - langevin_gt)[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='jet', vmin=0,
                          vmax=np.max(np.abs(np_avgs['rcgan_wo_gr_w_dc'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = plt.subplot(gs[4, count + 1])
                ax.imshow(3 * np.abs(ddrm_avg - ddrm_gt)[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='jet', vmin=0,
                          vmax=np.max(np.abs(np_avgs['rcgan_wo_gr_w_dc'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 1
                for method in keys:
                    if method != 'l1_ssim' and method != 'varnet':
                        ax = plt.subplot(gs[5, count])
                        ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax = plt.subplot(gs[5, count])
                        im = ax.imshow(np.zeros((384, 384)[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length]), cmap='viridis', vmin=0,
                                       vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
                        if method == 'l1_ssim':
                            [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                            # Appropriately rescale final axis so that colorbar does not effect formatting
                            pad = 0.01
                            width = 0.02
                            cbar_ax = fig.add_axes([x10 - 2 * pad, y10, width, y11 - y10])
                            cbar = fig.colorbar(im, cax=cbar_ax, format='%.0e',
                                                orientation='vertical')  # Generate colorbar
                            cbar.ax.locator_params(nbins=3)
                            cbar.ax.yaxis.set_ticks_position("left")
                            cbar.ax.tick_params(labelsize=6)
                            cbar.ax.tick_params(rotation=0)
                            tl = cbar.ax.get_yticklabels()

                            # set the alignment for the first and the last
                            tl[0].set_verticalalignment('bottom')
                            tl[-1].set_verticalalignment('top')

                    count += 1

                ax = plt.subplot(gs[5, count])
                ax.imshow(langevin_std[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = plt.subplot(gs[5, count + 1])
                ax.imshow(ddrm_std[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan_wo_gr_w_dc']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                plt.savefig(f'mr_figs_workshop/workshop_body_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)
                if fig_count == 5:
                    print(fig_count)
                    exit()
                fig_count += 1
