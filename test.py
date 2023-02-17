import torch

import numpy as np
import pytorch_lightning as pl
import sigpy as sp

from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from models.mri_unet import MRIUnet
from pytorch_lightning import seed_everything
from evaluation_scripts.metrics import psnr, ssim
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import tensor_to_complex_np
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.in_chans = 16
    args.out_chans = 16

    args.checkpoint_dir = "/storage/matt_models/genie"
    dm = MRIDataModule(args)
    dm.setup()
    test_loader = dm.test_dataloader()
    best_epoch = 274
    inception_embedding = VGG16Embedding()

    with torch.no_grad():
        print(f"TESTING EPOCH: {best_epoch+1}")
        model = MRIUnet.load_from_checkpoint(checkpoint_path=args.checkpoint_dir + f'/best_model.ckpt')
        model = model.cuda()
        model.eval()

        psnrs = []
        ssims = []

        for i, data in enumerate(test_loader):
            losses = {
                'psnr': [],
                'ssim': []
            }

            y, x, _, mean, std, mask, maps = data
            x_hat = model.forward(y, mask)

            avg_gen = model.reformat(x_hat)
            gt = model.reformat(x)

            for j in range(y.size(0)):
                S = sp.linop.Multiply((args.im_size, args.im_size), maps[j].cpu().numpy())
                gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                    (avg_gen[j] * std[j] + mean[j]).cpu())

                avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                losses['ssim'].append(ssim(gt_np, avg_gen_np))
                losses['psnr'].append(psnr(gt_np, avg_gen_np))

            psnrs.append(np.mean(losses['psnr']))
            ssims.append(np.mean(losses['ssim']))

    print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
    print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
