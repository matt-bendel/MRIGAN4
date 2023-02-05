import torch

import numpy as np
import pytorch_lightning as pl
import sigpy as sp
import sigpy.mri as mr

from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from pytorch_lightning import seed_everything
from evaluation_scripts.metrics import psnr
from data import transforms
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.in_chans = 16
    args.out_chans = 16

    args.checkpoint_dir = "/storage/matt_models"
    dm = MRIDataModule(args)
    dm.setup()
    val_loader = dm.val_dataloader()
    best_epoch = 100
    inception_embedding = VGG16Embedding(parallel=True).cuda()
    best_cfid = 10000000

    for epoch in range(100, 200):
        model = rcGAN.load_from_checkpoint(checkpoint_path=args.checkpoint_dir + f'/checkpoint-{epoch}.ckpt')
        model = model.cuda()
        model.eval()

        psnrs = []
        single_psnrs = []

        for i, data in enumerate(val_loader):
            y, x, y_true, mean, std, mask, maps = data
            y = y.cuda()
            x = x.cuda()
            y_true = y_true.cuda()
            mean = mean.cuda()
            std = std.cuda()
            mask = mask.cuda()

            gens = torch.zeros(size=(y.size(0), 8, args.in_chans, args.im_size, args.im_size)).cuda()
            for z in range(8):
                gens[:, z, :, :, :] = model(y, mask)

            avg = torch.mean(gens, dim=1)

            avg_gen = model.reformat(avg)
            gt = model.reformat(x)

            for j in range(y.size(0)):
                new_y_true = fft2c_new(ifft2c_new(y_true[j]) * std[j] + mean[j])
                S = sp.linop.Multiply((args.im_size, args.im_size), maps)
                gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                    (avg_gen[j] * std[j] + mean[j]).cpu())

                avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                single_gen = torch.zeros(8, args.im_size, args.im_size, 2).cuda()
                single_gen[:, :, :, 0] = gens[j, 0, 0:8, :, :]
                single_gen[:, :, :, 1] = gens[j, 0, 8:16, :, :]

                single_gen_complex_np = tensor_to_complex_np((single_gen * std[j] + mean[j]).cpu())
                single_gen_np = torch.tensor(S.H * single_gen_complex_np).abs().numpy()

                psnrs.append(psnr(gt_np, avg_gen_np))
                single_psnrs.append(psnr(gt_np, single_gen_np))

        psnr_diff = (np.mean(single_psnrs) + 2.5) - np.mean(psnrs)

        if np.abs(psnr_diff) <= 0.25:
            cfid_metric = CFIDMetric(gan=model,
                                     loader=val_loader,
                                     image_embedding=inception_embedding,
                                     condition_embedding=inception_embedding,
                                     cuda=True,
                                     args=args,
                                     ref_loader=False,
                                     num_samps=1)

            cfids = cfid_metric.get_cfid_torch_pinv()

            cfid_val = np.mean(cfids)

            if cfid_val < best_cfid:
                best_epoch = epoch
                best_cfid = cfid_val

    print(f"BEST EPOCH: {best_epoch}")
