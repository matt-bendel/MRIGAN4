import torch
import yaml
import os
import types
import json

import numpy as np
import sigpy as sp

from data_loaders.MRIDataModule import MRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.mri_unet import MRIUnet
from models.CoModGAN import InpaintUNet
from utils.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.metrics import psnr, ssim
from utils.fftc import ifft2c_new, fft2c_new

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

        dm = MRIDataModule(cfg, args.mask_type)
        dm.setup()
        test_loader = dm.test_dataloader()
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

    with torch.no_grad():
        model = model_alias.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/best_model.ckpt')
        model = model.cuda()
        model.eval()

        psnrs = []
        ssims = []

        losses = {
            'psnr': [],
            'ssim': []
        }

        for i, data in enumerate(test_loader):
            y, x, _, mean, std, mask, maps = data
            y = y.cuda()
            x = x.cuda()
            mean = mean.cuda()
            std = std.cuda()
            mask = mask.cuda()

            num_code = 4
            recons = torch.zeros(y.size(0), num_code, 8, 128, 128, 2).cuda()

            for k in range(num_code):
                z = torch.empty(y.size(0), model.resolution, model.resolution, 2).uniform_(0, 1).cuda()
                z = 2 * torch.bernoulli(z) - 1
                noise_fft = fft2c_new(z)
                # meas_noise = ifft2c_new(mask[:, 0, :, :, :] * noise_fft).permute(0, 3, 1, 2)
                non_noise = ifft2c_new((1 - mask[:, 0, :, :, :]) * noise_fft).permute(0, 3, 1, 2)
                noise = non_noise
                recons[:, k, :, :, :, :] = model.reformat(model.unet(torch.cat([y, noise], dim=1)))

            avg_gen = torch.mean(recons, dim=1)
            gt = model.reformat(x)

            for j in range(y.size(0)):
                S = sp.linop.Multiply((model.args.im_size, model.args.im_size), maps[j].cpu().numpy())
                gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                    (avg_gen[j] * std[j] + mean[j]).cpu())

                avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                losses['ssim'].append(ssim(gt_np, avg_gen_np))
                losses['psnr'].append(psnr(gt_np, avg_gen_np))

    print(f'PSNR: {np.mean(np.mean(losses["psnr"]))}')
    print(f'SSIM: {np.mean(np.mean(losses["ssim"]))}')
