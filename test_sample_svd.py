import torch
import yaml
import os
import types
import json
import pathlib
import lpips

import numpy as np
from matplotlib import gridspec
import sklearn.preprocessing
from data_loaders.MRIDataModule import MRIDataModule
from datasets.fastmri_multicoil_general import FastMRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.mri_unet import MRIUnet
from models.rcGAN import rcGAN
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

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.mask_type = 1

    if args.default_model_descriptor:
        args.num_noise = 1

    if args.mri:
        with open('configs/mri/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        cfg.batch_size = cfg.batch_size * 4
        dm = MRIDataModule(cfg, args.mask_type, big_test=True)

        dm.setup()
        test_loader = dm.test_dataloader()
        if args.rcgan:
            model_alias = rcGAN
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

        n = 128
        count = 5
        current_count = 0

        print(f"{n} SAMPLES")

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
                if current_count >= count:
                    exit()
                single_samps = np.zeros((n, cfg.im_size, cfg.im_size))

                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))
                gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                    (avg[j] * std[j] + mean[j]).cpu())

                avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                for z in range(n):
                    np_samp = tensor_to_complex_np((gens[j, z, :, :, :, :] * std[j] + mean[j]).cpu())
                    single_samps[z, :, :] = torch.tensor(S.H * np_samp).abs().numpy()

                single_samps = single_samps - avg_gen_np[None, :, :]

                cov_mat = np.zeros((n, avg_gen_np.shape[-1] * avg_gen_np.shape[-2]))

                for z in range(n):
                    cov_mat[z, :] = single_samps[z].flatten()

                u, s, vh = np.linalg.svd(cov_mat, full_matrices=False)

                plt.figure()
                plt.scatter(range(len(s)), sklearn.preprocessing.normalize(s.reshape((1,-1))))
                plt.savefig(f'sv_test/test_sv_{current_count}.png')
                plt.close()

                for l in range(5):
                    v_re = vh[l].reshape((384, 384))
                    v_re = (v_re - np.min(v_re)) / (np.max(v_re) - np.min(v_re))
                    plt.figure()
                    plt.imshow(v_re, cmap='viridis')
                    plt.colorbar()
                    plt.savefig(f'sv_test/test_sv_v_{current_count}_{l}.png')
                    plt.close()

                current_count += 1


