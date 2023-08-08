import torch
import yaml
import os
import types
import json
import pathlib
import lpips
import time
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
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric_varnet import CFIDMetric
from evaluation_scripts.fid.fid_metric_varnet import FIDMetric
from evaluation_scripts.metrics import psnr, ssim
import matplotlib.pyplot as plt
from utils.fftc import ifft2c_new, fft2c_new
import sigpy as sp
from DISTS_pytorch import DISTS
import matplotlib.patches as patches


# M_1: 5.59
# C_1: 3.78
# CFID_1: 9.37

# M_2: 6.24
# C_2: 1.69
# CFID_2: 7.93

# M_3: 6.24
# C_3: 1.27
# CFID_3: 7.51

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

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    with open('configs/mri/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    cfg.batch_size = 1
    dm = MRIDataModule(cfg, 1, big_test=True)

    dm.setup()
    test_loader = dm.test_dataloader()
    model_alias = VarNetModule

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    lpips_met = lpips.LPIPS(net='alex')
    lpips_met = lpips_met.cuda()

    dists_met = DISTS()
    dists_met = dists_met.cuda()

    psnr_str = ''
    ssim_str = ''
    lpips_str = ''
    dists_str = ''

    with torch.no_grad():
        model = model_alias.load_from_checkpoint(
            checkpoint_path='/storage/matt_models/mri/e2e_varnet/varnet/varnet_demo/checkpoints/epoch=30-step=284208.ckpt')
        model.cuda()
        model.eval()

        n_samps = [1]

        n_psnrs = []
        n_ssims = []
        n_lpipss = []
        n_distss = []

        for n in n_samps:
            # break
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
            times = []

            for i, data in enumerate(test_loader):
                print(f"Batch: {i}/{len(test_loader)}")
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
                    gt_ksp, avg_ksp = gt[j].cpu().numpy(), recon[j].cpu().numpy()

                    avg_gen_np = avg_ksp
                    gt_np = gt_ksp

                    psnrs.append(psnr(gt_np, avg_gen_np))
                    ssims.append(ssim(gt_np, avg_gen_np))
                    lpipss.append(lpips_met(rgb(gt_np), rgb(avg_gen_np)).cpu().numpy())
                    distss.append(dists_met(rgb(gt_np, unit_norm=True), rgb(avg_gen_np, unit_norm=True)).cpu().numpy())

            print(f'TIME: {np.mean(times)}')
            n_psnrs.append(np.mean(psnrs))
            n_ssims.append(np.mean(ssims))
            n_lpipss.append(np.mean(lpipss))
            n_distss.append(np.mean(distss))



        for i in range(len(n_psnrs)):
            psnr_str = f'{psnr_str} {n_psnrs[i]:.2f} &'
            ssim_str = f'{ssim_str} {n_ssims[i]:.4f} &'
            lpips_str = f'{lpips_str} {n_lpipss[i]:.4f} &'
            dists_str = f'{dists_str} {n_distss[i]:.4f} &'
        #

            # print(f'APSD: {np.mean(apsds)}')
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
                             num_samps=1)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)
    #
    inception_embedding = VGG16Embedding(parallel=True)
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

    inception_embedding = VGG16Embedding(parallel=True)
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

    for l in range(3):
        print(f'CFID_{l+1}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')
    #

    # n_samps = [1, 2, 4, 8, 16, 32]
    # for n in n_samps:
    # #     print(f"{n} SAMPLES")
    inception_embedding = VGG16Embedding()
    fid_metric = FIDMetric(gan=model,
                           ref_loader=train_dataloader,
                           loader=test_loader,
                           image_embedding=inception_embedding,
                           condition_embedding=inception_embedding,
                           cuda=True,
                           args=cfg)
    fid, fjd = fid_metric.get_fid()

    print(f'FID: {fid}; FJD: {fjd}')

    for l in range(3):
        print(f'CFID_{l+1}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')

    print(f'PSNR:\n{psnr_str}')
    print(f'SSIM:\n{ssim_str}')
    print(f'LPIPS:\n{lpips_str}')
    print(f'DISTS:\n{dists_str}')
    # print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
    # print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
    # print(f'APSD: {np.mean(apsds)}')
    # for l in range(3):
    #     print(f'CFID_{l+1}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')
