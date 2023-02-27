import torch

import pytorch_lightning as pl
import numpy as np
import sigpy as sp

from torch.nn import functional as F
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from models.architectures.mri_unet import Unet
from models.architectures.our_gen_unet_only import UNetModel

from evaluation_scripts.metrics import psnr, ssim
from mail import send_mail


class MRIUnet(pl.LightningModule):
    def __init__(
            self, args, num_realizations, default_model_descriptor, exp_name,
            chans=256,
            num_pool_layers=4,
            drop_prob=0.0,
            lr=0.001,
            lr_step_size=40,
            lr_gamma=0.1,
            weight_decay=0.0, ):
        super().__init__()
        self.args = args
        self.num_realizations = num_realizations
        self.default_model_descriptor = default_model_descriptor
        self.exp_name = exp_name

        self.in_chans = args.in_chans + 2 * self.num_realizations * 2
        self.out_chans = args.out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = UNetModel(self.in_chans, self.out_chans)

        self.resolution = self.args.im_size
        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors, mask):
        noise_vals = []
        for i in range(self.num_realizations):
            # z = torch.randn(num_vectors, self.resolution, self.resolution, 2, device=self.device)
            if self.default_model_descriptor:
                noise_vals.append(ifft2c_new(mask[:, 0, :, :, :]).permute(0, 3, 1, 2))
                break

            z = torch.empty(num_vectors, self.resolution, self.resolution, 2, device=self.device).uniform_(0, 1)
            z = 2 * torch.bernoulli(z) - 1
            noise_fft = fft2c_new(z)
            meas_noise = ifft2c_new(mask[:, 0, :, :, :] * noise_fft).permute(0, 3, 1, 2)
            non_noise = ifft2c_new((1 - mask[:, 0, :, :, :]) * noise_fft).permute(0, 3, 1, 2)
            noise_vals.append(meas_noise)
            # noise_vals.append(z.permute(0, 3, 1, 2))
            noise_vals.append(non_noise)

        return noise_vals

    def reformat(self, samples):
        reformatted_tensor = torch.zeros(size=(samples.size(0), 8, self.resolution, self.resolution, 2),
                                         device=self.device)
        reformatted_tensor[:, :, :, :, 0] = samples[:, 0:8, :, :]
        reformatted_tensor[:, :, :, :, 1] = samples[:, 8:16, :, :]

        return reformatted_tensor

    def readd_measures(self, samples, measures, mask):
        reformatted_tensor = self.reformat(samples)
        measures = fft2c_new(self.reformat(measures))
        reconstructed_kspace = fft2c_new(reformatted_tensor)

        reconstructed_kspace = mask * measures + (1 - mask) * reconstructed_kspace

        image = ifft2c_new(reconstructed_kspace)

        output_im = torch.zeros(size=samples.shape, device=self.device)
        output_im[:, 0:8, :, :] = image[:, :, :, :, 0]
        output_im[:, 8:16, :, :] = image[:, :, :, :, 1]

        return output_im

    def forward(self, y, mask):
        num_vectors = y.size(0)
        if self.num_realizations > 0:
            noise = self.get_noise(num_vectors, mask)
            samples = self.unet(torch.cat([y, torch.cat(noise, dim=1)], dim=1))
        else:
            samples = self.unet(y)

        samples = self.readd_measures(samples, y, mask)
        return samples

    def training_step(self, batch, batch_idx):
        y, x, _, mean, std, mask, _ = batch

        # train generator
        x_hat = self.forward(y, mask)
        loss = F.l1_loss(x_hat, x)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        losses = {
            'psnr': [],
            'single_psnr': [],
            'ssim': []
        }

        y, x, _, mean, std, mask, maps = batch
        x_hat = self.forward(y, mask)

        avg_gen = self.reformat(x_hat)
        gt = self.reformat(x)

        for j in range(y.size(0)):
            S = sp.linop.Multiply((self.args.im_size, self.args.im_size), maps[j].cpu().numpy())
            gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                (avg_gen[j] * std[j] + mean[j]).cpu())

            avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
            gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

            losses['ssim'].append(ssim(gt_np, avg_gen_np))
            losses['psnr'].append(psnr(gt_np, avg_gen_np))

        losses['psnr'] = np.mean(losses['psnr'])
        losses['ssim'] = np.mean(losses['ssim'])

        return losses

    def external_test_step(self, batch, batch_idx):
        y, x, _, mean, std, mask, maps = batch
        y = y.cuda()
        x = x.cuda()
        mean = mean.cuda()
        std = std.cuda()
        mask = mask.cuda()

        new_batch = y, x, _, mean, std, mask, maps

        return self.validation_step(new_batch, batch_idx)

    def validation_step_end(self, batch_parts):
        losses = {
            'psnr': np.mean(batch_parts['psnr']),
            'ssim': np.mean(batch_parts['ssim'])
        }

        return losses

    def validation_epoch_end(self, validation_step_outputs):
        psnrs = []
        ssims = []

        for out in validation_step_outputs:
            psnrs.append(out['psnr'])
            ssims.append(out['ssim'])

        self.log('val_psnr', np.mean(psnrs), sync_dist=True)

        if self.global_rank == 0 and self.current_epoch % 5 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - MRI UNET - {self.exp_name}",
                      f"Metrics:\nPSNR: {np.mean(psnrs):.2f}\nSSIM: {np.mean(ssims):.4f}\n",
                      file_name="variation_gif.gif")

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.unet.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))

        return optim
