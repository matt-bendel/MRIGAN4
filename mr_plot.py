import torch
import yaml
import os
import types
import json
import pathlib

import numpy as np
import matplotlib.patches as patches

from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.rcGAN import rcGAN
from models.adler import Adler
from models.ohayon import Ohayon
from models.l1_ssim_module import L1SSIMMRI
from utils.math import complex_abs, tensor_to_complex_np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sigpy as sp
from evaluation_scripts.metrics import psnr, ssim
from scipy import ndimage

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
        rcGAN_model = rcGAN.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + '/neurips/rcgan/checkpoint_best.ckpt')
        ohayon_model = Ohayon.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + '/neurips/ohayon_2/checkpoint_best.ckpt')
        adler_model = Adler.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + '/neurips/adler/checkpoint_best.ckpt')
        l1_ssim_model = L1SSIMMRI.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + '/neurips/l1_ssim/checkpoint_best.ckpt')

        rcGAN_model.cuda()
        ohayon_model.cuda()
        adler_model.cuda()
        l1_ssim_model.cuda()

        rcGAN_model.eval()
        ohayon_model.eval()
        adler_model.eval()
        l1_ssim_model.eval()

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std, maps, fname, slice = data
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens_rcgan = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            gens_ohayon = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            gens_adler = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            gens_l1_ssim = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()

            for z in range(cfg.num_z_test):
                gens_rcgan[:, z, :, :, :, :] = rcGAN_model.reformat(rcGAN_model.forward(y, mask))
                gens_ohayon[:, z, :, :, :, :] = ohayon_model.reformat(ohayon_model.forward(y, mask))
                gens_adler[:, z, :, :, :, :] = adler_model.reformat(adler_model.forward(y, mask))
                gens_l1_ssim[:, z, :, :, :, :] = l1_ssim_model.reformat(l1_ssim_model.forward(y, mask))

            avg_rcgan = torch.mean(gens_rcgan, dim=1)
            avg_ohayon = torch.mean(gens_ohayon, dim=1)
            avg_adler = torch.mean(gens_adler, dim=1)
            avg_l1_ssim = torch.mean(gens_l1_ssim, dim=1)

            gt = rcGAN_model.reformat(x)

            # TODO: Add Langevin, L1+SSIM model

            for j in range(y.size(0)):
                np_avgs = {
                    'l1_ssim': None,
                    'rcgan': None,
                    'ohayon': None,
                    'adler': None
                }

                np_samps = {
                    'rcgan': [],
                    'ohayon': [],
                    'adler': []
                }

                np_stds = {
                    'rcgan': None,
                    'ohayon': None,
                    'adler': None
                }

                np_gt = None

                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))

                np_gt = ndimage.rotate(torch.tensor(S.H * tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)

                np_avgs['rcgan'] = ndimage.rotate(torch.tensor(S.H * tensor_to_complex_np((avg_rcgan[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)
                np_avgs['ohayon'] = ndimage.rotate(torch.tensor(S.H * tensor_to_complex_np((avg_ohayon[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)
                np_avgs['adler'] = ndimage.rotate(torch.tensor(S.H * tensor_to_complex_np((avg_adler[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)
                np_avgs['l1_ssim'] = ndimage.rotate(torch.tensor(S.H * tensor_to_complex_np((avg_l1_ssim[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)

                for z in range(cfg.num_z_test):
                    np_samps['rcgan'].append(ndimage.rotate(torch.tensor(S.H * tensor_to_complex_np((gens_rcgan[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))
                    np_samps['ohayon'].append(ndimage.rotate(torch.tensor(
                        S.H * tensor_to_complex_np((gens_ohayon[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))
                    np_samps['adler'].append(ndimage.rotate(torch.tensor(
                        S.H * tensor_to_complex_np((gens_adler[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))

                np_stds['rcgan'] = np.std(np.stack(np_samps['rcgan']), axis=0)
                np_stds['ohayon'] = np.std(np.stack(np_samps['ohayon']), axis=0)
                np_stds['adler'] = np.std(np.stack(np_samps['adler']), axis=0)

                recon_directory = f'/storage/fastMRI_brain/Langevin_Recons_R=8/'
                langevin_recons = np.zeros((32, 384, 384))
                recon_object = None
                exceptions = False

                for l in range(cfg.num_z_test):
                    try:
                        new_filename = recon_directory + fname[j] + f'|langevin|slide_idx_{slice[j]}_R=8_sample={l}_outputs.pt'
                        recon_object = torch.load(new_filename)
                    except Exception as e:
                        print(e)
                        exceptions = True
                        break
                    # temp_recon = unnormalize(recon_object['mvue'], recon_object['zfr'])

                    langevin_recons[l] = ndimage.rotate(complex_abs(recon_object['mvue'][0].permute(1, 2, 0)).cpu().numpy(), 180)

                if exceptions:
                    exceptions = False
                    continue

                langevin_gt = ndimage.rotate(recon_object['gt'][0][0].abs().cpu().numpy(), 180)
                langevin_avg = np.mean(langevin_recons, axis=0)
                langevin_std = np.std(langevin_recons, axis=0)

                keys = ['l1_ssim', 'rcgan', 'ohayon', 'adler']
                zoom_start = 120
                zoom_length = 80

                # TODO: OG fig plot
                # TODO: metrics
                # OG FIG
                nrow = 3
                ncol = 6

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt.clip(0, 0.5*np.max(np_gt)), cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 1
                for method in keys:
                    ax = plt.subplot(gs[0, count])
                    ax.imshow(np_avgs[method].clip(0, 0.5*np.max(np_gt)), cmap='gray', vmin=0, vmax=np.max(np_gt))
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
                ax.imshow(langevin_avg.clip(0, 0.5*np.max(langevin_gt)), cmap='gray', vmin=0, vmax=np.max(langevin_gt))
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
                    im = ax.imshow(2*np.abs(np_avgs[method] - np_gt), cmap='jet', vmin=0, vmax=np.max(np.abs(np_avgs['rcgan'] - np_gt)))
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
                ax.imshow(3*np.abs(langevin_avg - langevin_gt), cmap='jet', vmin=0,
                          vmax=np.max(np.abs(np_avgs['rcgan'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 1
                for method in keys:
                    if method != 'l1_ssim':
                        ax = plt.subplot(gs[2, count])
                        ax.imshow(np_stds[method], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax = plt.subplot(gs[2, count])
                        im = ax.imshow(np.zeros((384, 384)), cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
                        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                        # Appropriately rescale final axis so that colorbar does not effect formatting
                        pad = 0.01
                        width = 0.02
                        cbar_ax = fig.add_axes([x10 - 2*pad, y10, width, y11 - y10])
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
                ax.imshow(langevin_std, cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                plt.savefig(f'mr_figs/body_mri_fig_avg_err_std_{fig_count}.png', bbox_inches='tight', dpi=300)

                # TODO: Samp Grid (my idea): zoomed
                nrow = 1
                ncol = 6
                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                rect = patches.Rectangle((zoom_start, zoom_start), zoom_length, zoom_length, linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax1 = ax
                x_coord = zoom_start + zoom_length
                y_coords = [zoom_start, zoom_start+zoom_length]

                inner = gs[0, 1].subgridspec(2, 2)
                ax = fig.add_subplot(inner[0, 0])
                ax.imshow(np_gt[zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                connection_path_1 = patches.ConnectionPatch([zoom_start, y_coords[1]], [0, 0], coordsA=ax1.transData, coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([x_coord, y_coords[1]], [zoom_length, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                ax = fig.add_subplot(inner[0, 1])
                ax.imshow(np_avgs['l1_ssim'][zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)

                count = 2
                for method in keys:
                    if method != 'l1_ssim':
                        inner = gs[0, count].subgridspec(2, 2)
                        ax = fig.add_subplot(inner[0])
                        ax.imshow(np_avgs[method][zoom_start:zoom_start + zoom_length,
                                  zoom_start:zoom_start + zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.patch.set_edgecolor('red')
                        ax.patch.set_linewidth(2.5)

                        for samp in range(3):
                            ax = fig.add_subplot(inner[samp+1])
                            ax.imshow(np_samps[method][samp][zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])

                        plt.subplots_adjust(wspace=0, hspace=0)
                        count += 1

                inner = gs[0, count].subgridspec(2, 2)
                ax = fig.add_subplot(inner[0])
                ax.imshow(
                    langevin_avg[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                    cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.patch.set_edgecolor('red')
                ax.patch.set_linewidth(2.5)
                for samp in range(3):
                    ax = fig.add_subplot(inner[samp+1])
                    ax.imshow(langevin_recons[samp, zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                              cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)

                plt.savefig('test_my_grid.png', bbox_inches='tight', dpi=300)

                # TODO: top row: zoomed avg, next two rows samps.
                nrow = 3
                ncol = 1

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax1 = ax

                rect = patches.Rectangle((zoom_start, zoom_start), zoom_length, zoom_length, linewidth=1, edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax = plt.subplot(gs[1, 0])
                ax.imshow(np_gt[zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                connection_path_1 = patches.ConnectionPatch([zoom_start, y_coords[1]], [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([x_coord, y_coords[1]], [zoom_length, 0],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                ax = plt.subplot(gs[2, 0])
                ax.imshow(np_avgs['l1_ssim'][zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                          cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                plt.savefig(f'mr_figs/body_mri_fig_left_{fig_count}.png', bbox_inches='tight', dpi=300)

                nrow = 3
                ncol = 4

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                count = 0
                for method in keys:
                    if method != 'l1_ssim':
                        ax = plt.subplot(gs[0, count])
                        ax.imshow(np_avgs[method][zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        count += 1

                ax = plt.subplot(gs[0, count])
                ax.imshow(langevin_avg[zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                for samp in range(2):
                    count = 0
                    for method in keys:
                        if method != 'l1_ssim':
                            ax = plt.subplot(gs[samp+1, count])
                            ax.imshow(np_samps[method][samp][zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            count += 1

                    ax = plt.subplot(gs[samp+1, count])
                    ax.imshow(langevin_recons[samp, zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.savefig(f'mr_figs/body_mri_fig_right_{fig_count}.png', bbox_inches='tight', dpi=300)

                # TODO: Rizwan Idea: zoomed, 1st row avg, 2nd error, 3rd std. dev, 4, 5, 6 samps
                nrow = 6
                ncol = 7

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax1 = ax

                rect = patches.Rectangle((zoom_start, zoom_start), zoom_length, zoom_length, linewidth=1, edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length], cmap='gray',
                          vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                connection_path_1 = patches.ConnectionPatch([x_coord, y_coords[0]], [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([x_coord, y_coords[1]], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                count = 2
                for method in keys:
                    ax = plt.subplot(gs[0, count])
                    ax.imshow(np_avgs[method][zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                              cmap='gray', vmin=0, vmax=np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    count += 1

                ax = plt.subplot(gs[0, count])
                ax.imshow(langevin_avg[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                          cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 2
                for method in keys:
                    ax = plt.subplot(gs[1, count])
                    im = ax.imshow(2*np.abs(np_avgs[method] - np_gt)[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length], cmap='jet', vmin=0, vmax=np.max(np.abs(np_avgs['rcgan'] - np_gt)))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if count == 2:
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
                ax.imshow(3*np.abs(langevin_avg - langevin_gt)[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length], cmap='jet', vmin=0,
                          vmax=np.max(np.abs(np_avgs['rcgan'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 2
                for method in keys:
                    if method != 'l1_ssim':
                        ax = plt.subplot(gs[2, count])
                        ax.imshow(np_stds[method][zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax = plt.subplot(gs[2, count])
                        im = ax.imshow(np.zeros((384, 384))[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
                        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                        # Appropriately rescale final axis so that colorbar does not effect formatting
                        pad = 0.01
                        width = 0.02
                        cbar_ax = fig.add_axes([x10 - 2*pad, y10, width, y11 - y10])
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
                ax.imshow(langevin_std[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                for samp in range(3):
                    count = 3
                    for method in keys:
                        if method != 'l1_ssim':
                            ax = plt.subplot(gs[samp+3, count])
                            ax.imshow(np_samps[method][samp][zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            count += 1

                    ax = plt.subplot(gs[samp+3, count])
                    ax.imshow(langevin_recons[samp, zoom_start:zoom_start+zoom_length, zoom_start:zoom_start+zoom_length], cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.savefig(f'mr_figs/app_mri_fig_{fig_count}.png', bbox_inches='tight', dpi=300)

                # TODO: Rizwan Idea (mine): zoomed, 1st row avg, 2nd error, 3rd std. dev, 4 grid of 4 samps
                nrow = 4
                ncol = 7

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='gray', vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax1 = ax

                rect = patches.Rectangle((zoom_start, zoom_start), zoom_length, zoom_length, linewidth=1, edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length], cmap='gray',
                          vmin=0, vmax=np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                connection_path_1 = patches.ConnectionPatch([x_coord, y_coords[0]], [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([x_coord, y_coords[1]], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                count = 2
                for method in keys:
                    ax = plt.subplot(gs[0, count])
                    ax.imshow(np_avgs[method][zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                              cmap='gray', vmin=0, vmax=np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    count += 1

                ax = plt.subplot(gs[0, count])
                ax.imshow(langevin_avg[zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                          cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 2
                for method in keys:
                    ax = plt.subplot(gs[1, count])
                    im = ax.imshow(2 * np.abs(np_avgs[method] - np_gt)[zoom_start:zoom_start + zoom_length,
                                      zoom_start:zoom_start + zoom_length], cmap='jet', vmin=0,
                                   vmax=np.max(np.abs(np_avgs['rcgan'] - np_gt)))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if count == 2:
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
                ax.imshow(3 * np.abs(langevin_avg - langevin_gt)[zoom_start:zoom_start + zoom_length,
                                      zoom_start:zoom_start + zoom_length], cmap='jet', vmin=0,
                          vmax=np.max(np.abs(np_avgs['rcgan'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 2
                for method in keys:
                    if method != 'l1_ssim':
                        ax = plt.subplot(gs[2, count])
                        ax.imshow(np_stds[method][zoom_start:zoom_start + zoom_length,
                                      zoom_start:zoom_start + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax = plt.subplot(gs[2, count])
                        im = ax.imshow(np.zeros((384, 384))[zoom_start:zoom_start + zoom_length,
                                      zoom_start:zoom_start + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
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
                ax.imshow(langevin_std[zoom_start:zoom_start + zoom_length,
                                      zoom_start:zoom_start + zoom_length], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                count = 3
                for method in keys:
                    if method != 'l1_ssim':
                        inner = gs[3, count].subgridspec(2, 2)
                        for samp in range(4):
                            ax = fig.add_subplot(inner[samp])
                            ax.imshow(np_samps[method][samp][zoom_start:zoom_start + zoom_length,
                                      zoom_start:zoom_start + zoom_length], cmap='gray', vmin=0, vmax=np.max(np_gt))
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_xticks([])
                            ax.set_yticks([])

                        plt.subplots_adjust(wspace=0, hspace=0)
                        count += 1

                inner = gs[3, count].subgridspec(2, 2)
                for samp in range(4):
                    ax = fig.add_subplot(inner[samp])
                    ax.imshow(
                        langevin_recons[samp, zoom_start:zoom_start + zoom_length, zoom_start:zoom_start + zoom_length],
                        cmap='gray', vmin=0, vmax=np.max(langevin_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)

                plt.savefig('test_rizwan_mine.png', bbox_inches='tight', dpi=300)

                if fig_count == 8:
                    exit()
                fig_count += 1

