# import required module
import os
import pathlib
import torch
import numpy as np
# from utils.math import complex_abs
# from utils.parse_args import create_arg_parser
# from evaluation_scripts.fid.embeddings import VGG16Embedding

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.fid.fid_metric_langevin import FIDMetric
from evaluation_scripts.cfid.cfid_metric_langevin import CFIDMetric
import lpips
import pickle
from DISTS_pytorch import DISTS

# M: 2.61
# C: 4.73
# CFID: 7.34

# FID: 20.52

# APSD: 7.6e-6
# PSNR: 33.90 \pm 0.45
# SSIM: 0.8589 \pm 0.0083

def get_cfid():
    print("GETTING INCEPTION EMBEDDING")
    vgg_embedding = VGG16Embedding(parallel=True)

    print("GETTING DATA LOADERS")

    cfid_metric = CFIDMetric(gan=None,
                           ref_loader=None,
                           loader=None,
                           image_embedding=vgg_embedding,
                           condition_embedding=vgg_embedding,
                           cuda=True,
                           args=None)

    cfid = cfid_metric.get_cfid_torch_pinv()
    print(f"CFID: {cfid}")

def get_fid():
    print("GETTING INCEPTION EMBEDDING")
    vgg_embedding = VGG16Embedding(parallel=True)

    print("GETTING DATA LOADERS")

    fid_metric = FIDMetric(gan=None,
                           ref_loader=None,
                           loader=None,
                           image_embedding=vgg_embedding,
                           condition_embedding=vgg_embedding,
                           cuda=True,
                           args=None)

    fid_metric.get_fid()

def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)
    snr = 10 * np.log10(np.mean(gt ** 2) / noise_mse)

    return snr


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
        gt, pred, data_range=maxval
    )

    return ssim


def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling

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

R = 4

# assign directory
ref_directory = '/storage/fastMRI_brain/data/small_T2_test'
recon_directory = f'/storage/matt_models/mri/ddrm_R={R}/'
# recon_directory = f'/storage/matt_models/mri/diff_out/'

# iterate over files in
# that directory

# args = create_arg_parser().parse_args()

# train_loader, _ = create_data_loaders(args, big_test=False)
# get_cfid()
# # n_samps = [1, 2, 4, 8, 16, 32]
# # for n in n_samps:
# #     print(f"{n} SAMPLES")
# get_fid()
# exit()
vals = [1, 2, 4, 8, 16, 32]
# vals = [1, 8]
lpips_met = lpips.LPIPS(net='alex')
dists_met = DISTS()

exceptions = False
count = 0
n_psnrs = []
n_ssims = []
n_distss = []
n_lpipss = []
max_p = 32

psnr_vals = {
    1: [],
    2: [],
    4: [],
    8: [],
    16: [],
    32: []
}

ssim_vals = {
    1: [],
    2: [],
    4: [],
    8: [],
    16: [],
    32: []
}

lpips_vals = {
    1: [],
    2: [],
    4: [],
    8: [],
    16: [],
    32: []
}

dists_vals = {
    1: [],
    2: [],
    4: [],
    8: [],
    16: [],
    32: []
}

for filename in os.listdir(ref_directory):
    for i in range(6):
        recons = np.zeros((max_p, 384, 384))
        recon_object = None

        maps = None
        with open(f'/storage/fastMRI_brain/sense_maps/test_full_res/{filename}_{i}.pkl', 'rb') as inp:
            maps = pickle.load(inp)

        test_mask = maps[0, :, :] > 1e-3

        for j in range(max_p):
            try:
                new_filename = recon_directory + f'{filename}_{i}_sample_{j}.pt'
                recon_object = torch.load(new_filename)
                count += 1
            except Exception as e:
                print(e)
                # print(filename)
                exceptions = True
                continue
            # temp_recon = unnormalize(recon_object['mvue'], recon_object['zfr'])

            recons[j] = recon_object[0, :, :].cpu().numpy()

        if exceptions:
            # print("EXCEPTION")
            exceptions = False
            continue

        new_filename = recon_directory + f'{filename}_{i}_gt.pt'
        gt = torch.load(new_filename)
        gt = torch.load(new_filename)[0, :, :].cpu().numpy() * test_mask

        for P in vals:
            mean = np.mean(recons[:P, :, :], axis=0) * test_mask

            psnr_vals[P].append(psnr(gt, mean))
            ssim_vals[P].append(ssim(gt, mean))
            with torch.no_grad():
                lpips_vals[P].append(lpips_met(rgb(gt), rgb(mean)).numpy())
                dists_vals[P].append(dists_met(rgb(gt, unit_norm=True), rgb(mean, unit_norm=True)))

# for k in vals:
#     print(f"{k} CODE VECTORS")
#     psnr_vals = []
#     ssim_vals = []
#     snr_vals = []
#     apsd_vals = []
#     lpips_vals = []
#     dists_vals = []
#
#     for filename in os.listdir(ref_directory):
#         for i in range(6):
#             recons = np.zeros((k, 384, 384))
#             recon_object = None
#
#             maps = None
#             with open(f'/storage/fastMRI_brain/sense_maps/test_full_res/{filename}_{i}.pkl', 'rb') as inp:
#                 maps = pickle.load(inp)
#
#             test_mask = maps[0, :, :] > 1e-3
#
#             for j in range(k):
#                 try:
#                     new_filename = recon_directory + f'{filename}_{i}_sample_{j}.pt'
#                     recon_object = torch.load(new_filename)
#                     count += 1
#                 except:
#                     # print(filename)
#                     exceptions = True
#                     continue
#                 # temp_recon = unnormalize(recon_object['mvue'], recon_object['zfr'])
#
#                 recons[j] = recon_object[0, :, :].cpu().numpy()
#
#             if exceptions:
#                 # print("EXCEPTION")
#                 exceptions = False
#                 continue
#
#             mean = np.mean(recons, axis=0) * test_mask
#             new_filename = recon_directory + f'{filename}_{i}_gt.pt'
#             gt = torch.load(new_filename)
#             gt = torch.load(new_filename)[0, :, :].cpu().numpy() * test_mask
#             apsd = np.mean(np.std(recons, axis=0), axis=(0, 1))
#
#             apsd_vals.append(apsd)
#             psnr_vals.append(psnr(gt, mean))
#             snr_vals.append(snr(gt, mean))
#             ssim_vals.append(ssim(gt, mean))
#             with torch.no_grad():
#                 lpips_vals.append(lpips_met(rgb(gt), rgb(mean)).numpy())
#                 dists_vals.append(dists_met(rgb(gt, unit_norm=True), rgb(mean, unit_norm=True)))
#
#     # print('AVERAGE')
#     print(psnr_vals)
#     print(np.mean(psnr_vals))
#     print(np.mean(ssim_vals))
#     n_psnrs.append(np.mean(psnr_vals))
#     n_ssims.append(np.mean(ssim_vals))
#     n_lpipss.append(np.mean(lpips_vals))
#     n_distss.append(np.mean(dists_vals))

psnr_str = ''
ssim_str = ''
lpips_str = ''
dists_str = ''

for P in vals:
    psnr_str = f'{psnr_str} {np.mean(psnr_vals[P]):.2f} &'
    ssim_str = f'{ssim_str} {np.mean(ssim_vals[P]):.4f} &'
    lpips_str = f'{lpips_str} {np.mean(lpips_vals[P]):.4f} &'
    dists_str = f'{dists_str} {np.mean(dists_vals[P]):.4f} &'

print("PSNR and SSIM:")
print(f'{psnr_str} {ssim_str}')
print("LPIPS and DISTS")
print(f'{lpips_str} {dists_str}')
