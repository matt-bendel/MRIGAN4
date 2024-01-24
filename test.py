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
from models.rcGAN import rcGAN, rcGANLatent
from models.rcGAN_no_dc import rcGANNoDC
from models.adler import Adler
from models.ohayon import Ohayon
from models.CoModGAN import InpaintUNet
from models.l1_ssim_module import L1SSIMMRI
from utils.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.metrics import psnr, ssim
from evaluation_scripts.fid.embeddings import VGG16Embedding, AlexNetEmbedding
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
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    if args.mri:
        with open('configs/mri/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        cfg.batch_size = cfg.batch_size * 4

        if args.R != 8:
            cfg.R = args.R

        dm = MRIDataModule(cfg, args.mask_type, big_test=True)

        dm.setup()
        test_loader = dm.test_dataloader()
        if args.rcgan:
            if args.nodc:
                model_alias = rcGANNoDC
            else:
                model_alias = rcGANLatent
        else:
            model_alias = L1SSIMMRI
    elif args.inpaint:
        with open('configs/inpaint/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = CelebAHQDataModule(cfg, args.mask_type)
        dm.setup()
        test_loader = dm.test_dataloader()
        model_alias = InpaintUNet
    elif args.cs:
        args.checkpoint_dir = "/storage/matt_models/cs_baseline/"
        # TODO: LSUN?? data module
        # TODO: CS Lighnting Module
        pass
    else:
        print("No valid application selected. Please include one of the following args: --mri, --inpaint, --cs.")
        exit()

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    lpips_met = lpips.LPIPS(net='alex')
    dists_met = DISTS()

    dists_met = dists_met.cuda()
    lpips_met = lpips_met.cuda()

    print(f"R = {cfg.R}")

    with torch.no_grad():
        model = model_alias.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        # checkpoint_file_gen = pathlib.Path(
        #     f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/trained_models/generator_best_model.pt')
        # checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))
        #
        # g = torch.nn.DataParallel(GeneratorModel(18, 16).cuda())
        # g.load_state_dict(checkpoint_gen['model'])

        # model.generator = torch.nn.DataParallel(model.generator)
        model.cuda()
        model.eval()

        n_samps = [1, 2, 4, 8, 16, 32]

        n_psnrs = []
        n_ssims = []
        n_lpipss = []
        n_distss = []

        print(f"EXPERIMENT: {args.exp_name}")

        for n in n_samps:
            break
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
                y, x, mask, mean, std, maps, _, _ = data
                y = y.cuda()
                x = x.cuda()
                mask = mask.cuda()
                mean = mean.cuda()
                std = std.cuda()

                gens = torch.zeros(size=(y.size(0), n, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
                for z in range(n):
                    gens[:, z, :, :, :, :] = model.reformat(model.forward(y, mask))

                avg = torch.mean(gens, dim=1)

                gt = model.reformat(x)

                batch_psnrs = []
                batchs_ssims = []
                batch_apsds = []

                for j in range(y.size(0)):
                    single_samps = np.zeros((n, cfg.im_size, cfg.im_size))

                    S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))
                    gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                        (avg[j] * std[j] + mean[j]).cpu())

                    avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                    gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                    for z in range(n):
                        np_samp = tensor_to_complex_np((gens[j, z, :, :, :, :] * std[j] + mean[j]).cpu())
                        single_samps[z, :, :] = torch.tensor(S.H * np_samp).abs().numpy()

                    med_np = np.median(single_samps, axis=0)

                    apsds.append(np.mean(np.std(single_samps, axis=0), axis=(0, 1)))
                    psnrs.append(psnr(gt_np, avg_gen_np))
                    ssims.append(ssim(gt_np, avg_gen_np))
                    lpipss.append(lpips_met(rgb(gt_np), rgb(avg_gen_np)).cpu().numpy())
                    distss.append(dists_met(rgb(gt_np, unit_norm=True), rgb(avg_gen_np, unit_norm=True)).cpu().numpy())

            # print('AVG Recon')
            # print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
            # print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
            # print(f'LPIPS: {np.mean(lpipss)} \pm {np.std(lpipss) / np.sqrt(len(lpipss))}')
            # print(f'DISTS: {np.mean(distss)} \pm {np.std(distss) / np.sqrt(len(distss))}')
            #
            # print('Median Recon')
            # print(f'PSNR: {np.mean(med_psnrs)} \pm {np.std(med_psnrs) / np.sqrt(len(med_psnrs))}')
            # print(f'SSIM: {np.mean(med_ssims)} \pm {np.std(med_ssims) / np.sqrt(len(med_ssims))}')
            # print(f'LPIPS: {np.mean(med_lpipss)} \pm {np.std(med_lpipss) / np.sqrt(len(med_lpipss))}')
            # print(f'DISTS: {np.mean(med_distss)} \pm {np.std(med_distss) / np.sqrt(len(med_distss))}')

            # TODO: PSNR HISTOGRAM
            if n == 32 and (args.mask_type == 4 or args.mask_type == 2):
                plt.scatter(psnrs)
                plt.title(f'{args.exp_name}')
                plt.ylabel('PSNR')
                plt.savefig(f'{args.exp_name}_psnr_hist.png')
                plt.close()

            n_psnrs.append(np.mean(psnrs))
            n_ssims.append(np.mean(ssims))
            n_lpipss.append(np.mean(lpipss))
            n_distss.append(np.mean(distss))

        psnr_str = ''
        ssim_str = ''
        lpips_str = ''
        dists_str = ''

        # for i in range(len(n_psnrs)):
        #     psnr_str = f'{psnr_str} {n_psnrs[i]:.2f} &'
        #     ssim_str = f'{ssim_str} {n_ssims[i]:.4f} &'
        #     lpips_str = f'{lpips_str} {n_lpipss[i]:.4f} &'
        #     dists_str = f'{dists_str} {n_distss[i]:.4f} &'
        #
        # print("PSNR and SSIM:")
        # print(f'{psnr_str} {ssim_str}')
        # print("LPIPS and DISTS")
        # print(f'{lpips_str} {dists_str}')
        #
        # print(f'APSD: {np.mean(apsds)}')

    # exit()
    cfids = []
    m_comps = []
    c_comps = []

    inception_embedding = AlexNetEmbedding(parallel=True)
    # CFID_1
    cfid_metric = CFIDMetric(gan=model,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=1)

    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)

    # inception_embedding = VGG16Embedding(parallel=True)
    # # CFID_2
    cfid_metric = CFIDMetric(gan=model,
                             loader=val_dataloader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=1)

    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)

    # inception_embedding = VGG16Embedding(parallel=True)
    # # CFID_3
    cfid_metric = CFIDMetric(gan=model,
                             loader=val_dataloader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=train_dataloader,
                             num_samps=1)

    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)


    inception_embedding = AlexNetEmbedding()
    fid_metric = FIDMetric(gan=model,
                           ref_loader=train_dataloader,
                           loader=test_loader,
                           image_embedding=inception_embedding,
                           condition_embedding=inception_embedding,
                           cuda=True,
                           args=cfg)
    fid, fjd = fid_metric.get_fid()

    # print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
    # print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
    # print(f'APSD: {np.mean(apsds)}')
    for l in range(3):
        print(f'CFID_{l + 1}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')
    #
    print(f'FID: {fid}')
    print(f'FJD: {fjd}')
    print("\n")
