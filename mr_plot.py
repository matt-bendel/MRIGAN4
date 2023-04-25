import torch
import yaml
import os
import types
import json
import pathlib

import numpy as np

from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.rcGAN import rcGAN
from models.adler import Adler
from models.ohayon import Ohayon
from utils.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.metrics import psnr, ssim
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from evaluation_scripts.fid.fid_metric import FIDMetric
import matplotlib.pyplot as plt
import sigpy as sp

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

    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        rcGAN_model = rcGAN.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + '/neurips/rcgan/checkpoint_best.ckpt')
        ohayon_model = Ohayon.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + '/neurips/ohayon_2/checkpoint_best.ckpt')
        adler_model = Adler.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + '/neurips/adler/checkpoint_best.ckpt')

        rcGAN_model.cuda()
        ohayon_model.cuda()
        adler_model.cuda()

        rcGAN_model.eval()
        ohayon_model.eval()
        adler_model.eval()

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

            for z in range(cfg.num_z_test):
                gens_rcgan[:, z, :, :, :, :] = rcGAN_model.reformat(rcGAN_model.forward(y, mask))
                gens_ohayon[:, z, :, :, :, :] = ohayon_model.reformat(ohayon_model.forward(y, mask))
                gens_adler[:, z, :, :, :, :] = adler_model.reformat(adler_model.forward(y, mask))

            avg_rcgan = torch.mean(gens_rcgan, dim=1)
            avg_ohayon = torch.mean(gens_ohayon, dim=1)
            avg_adler = torch.mean(gens_adler, dim=1)

            gt = rcGAN_model.reformat(x)

            np_avgs = {
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

            for j in range(y.size(0)):
                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))

                np_gt = torch.tensor(S.H * tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).abs().numpy()

                np_avgs['rcgan'] = torch.tensor(S.H * tensor_to_complex_np((avg_rcgan[j] * std[j] + mean[j]).cpu())).abs().numpy()
                np_avgs['ohayon'] = torch.tensor(S.H * tensor_to_complex_np((avg_ohayon[j] * std[j] + mean[j]).cpu())).abs().numpy()
                np_avgs['adler'] = torch.tensor(S.H * tensor_to_complex_np((avg_adler[j] * std[j] + mean[j]).cpu())).abs().numpy()

                for z in range(cfg.num_z_test):
                    np_samps['rcgan'].append(torch.tensor(
                        S.H * tensor_to_complex_np((gens_rcgan[j, z] * std[j] + mean[j]).cpu())).abs().numpy())
                    np_samps['ohayon'].append(torch.tensor(
                        S.H * tensor_to_complex_np((gens_ohayon[j, z] * std[j] + mean[j]).cpu())).abs().numpy())
                    np_samps['adler'].append(torch.tensor(
                        S.H * tensor_to_complex_np((gens_adler[j, z] * std[j] + mean[j]).cpu())).abs().numpy())

                np_stds['rcgan'] = np.std(np.stack(np_samps['rcgan']), axis=0)
                np_stds['ohayon'] = np.std(np.stack(np_samps['rcgan']), axis=0)
                np_stds['adler'] = np.std(np.stack(np_samps['rcgan']), axis=0)

                keys = ['rcgan', 'ohayon', 'adler']
                # OG FIG
                fig = plt.figure()
                fig.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout(pad=0.00)

                plt.box(False)
                plt.subplot(3, 4, 1)
                plt.imshow(np_gt, cmap='gray', vmin=0, vmax=np.max(np_gt))
                plt.axis('off')
                plt.grid(visible=None)

                count = 2
                for method in keys:
                    plt.box(False)
                    plt.subplot(3, 4, count)
                    plt.imshow(np_avgs[method], cmap='gray', vmin=0, vmax=np.max(np_gt))
                    plt.axis('off')
                    plt.grid(visible=None)
                    count += 1

                count += 1
                for method in keys:
                    plt.box(False)
                    plt.subplot(3, 4, count)
                    plt.imshow(np.abs(np_avgs[method] - np_gt), cmap='jet', vmin=0, vmax=0.0001)
                    plt.axis('off')
                    plt.grid(visible=None)
                    count += 1

                count += 1
                for method in keys:
                    plt.box(False)
                    plt.subplot(3, 4, count)
                    plt.imshow(np_stds[method], cmap='viridis', vmin=0, vmax=5e-6)
                    plt.axis('off')
                    plt.grid(visible=None)
                    count += 1


                plt.savefig('test_og.png', bbox_inches='tight', dpi=300)
                exit()

