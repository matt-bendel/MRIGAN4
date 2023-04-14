import torch
import yaml
import os
import types
import json

import numpy as np

from data_loaders.MRIDataModule import MRIDataModule
from datasets.fastmri_multicoil_general import FastMRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.mri_unet import MRIUnet
from models.rcGAN import rcGAN
from models.CoModGAN import InpaintUNet
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

    if args.mri:
        with open('configs/mri/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = MRIDataModule(cfg, args.mask_type, big_test=True)

        dm.setup()
        test_loader = dm.test_dataloader()
        if args.rcgan:
            model_alias = rcGAN
        else:
            model_alias = MRIUnet
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

    with torch.no_grad():
        model = model_alias.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint-epoch=92.ckpt')
        model = model.cuda()
        model.eval()

        psnrs = []
        ssims = []
        apsds = []

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std, maps, _, _ = data
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
            for z in range(cfg.num_z_test):
                gens[:, z, :, :, :, :] = model.reformat(model.forward(y, mask))

            avg = torch.mean(gens, dim=1)

            gt = model.reformat(x)

            batch_psnrs = []
            batchs_ssims = []
            batch_apsds = []

            for j in range(y.size(0)):
                single_samps = np.zeros((cfg.num_z_test, cfg.im_size, cfg.im_size))
                print(maps[j].shape)

                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))
                gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                    (avg[j] * std[j] + mean[j]).cpu())

                avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                fig = plt.figure()

                generate_image(fig, gt_np, avg_gen_np, f'Recon', 1, 2, 1, disc_num=False)
                im, ax = generate_error_map(fig, gt_np, avg_gen_np, f'Recon', 2, 2, 1)

                plt.savefig(f'test.png')
                plt.close()

                fig = plt.figure()

                generate_image(fig, gt_np, gt_np, f'GT', 1, 1, 1, disc_num=False)

                plt.savefig(f'test_gt.png')
                plt.close()

                new_y_true = fft2c_new(model.reformat(y)[j] * std[j] + mask[j])
                maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y_true.cpu()), calib_width=16,
                                           device=sp.Device(0), crop=0.70,
                                           kernel_width=6).run().get()
                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), maps)
                avg_gen_np = complex_abs(to_tensor(S.H * avg_ksp)).numpy()
                gt_np = complex_abs(to_tensor(S.H * gt_ksp)).numpy()

                fig = plt.figure()

                generate_image(fig, gt_np, avg_gen_np, f'Recon', 1, 2, 1, disc_num=False)
                im, ax = generate_error_map(fig, gt_np, avg_gen_np, f'Recon', 2, 2, 1)

                plt.savefig(f'test_2.png')
                plt.close()
                exit()

                for z in range(cfg.num_z_test):
                    np_samp = tensor_to_complex_np((gens[j, z, :, :, :, :] * std[j] + mean[j]).cpu())
                    single_samps[z, :, :] = torch.tensor(S.H * np_samp).abs().numpy()

                apsds.append(np.mean(np.std(single_samps, axis=0), axis=(0, 1)))
                psnrs.append(psnr(gt_np, avg_gen_np))
                ssims.append(ssim(gt_np, avg_gen_np))

    inception_embedding = VGG16Embedding()

    cfids = []
    m_comps = []
    c_comps = []

    # CFID_1
    cfid_metric = CFIDMetric(gan=model,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=1)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    # CFID_2
    cfid_metric = CFIDMetric(gan=model,
                             loader=val_dataloader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=1)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    # CFID_3
    cfid_metric = CFIDMetric(gan=model,
                             loader=val_dataloader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=train_dataloader,
                             num_samps=1)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    fid_metric = FIDMetric(gan=model,
                            ref_loader=train_dataloader,
                            loader=test_loader,
                            image_embedding=inception_embedding,
                            condition_embedding=inception_embedding,
                            cuda=True,
                            args=cfg)
    #
    fid, _ = fid_metric.get_fid()

    print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
    print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
    print(f'APSD: {np.mean(apsds)}')
    for l in range(3):
        print(f'CFID_{l}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')

    print(f'FID: {fid}')

