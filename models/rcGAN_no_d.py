import os

import torch

import pytorch_lightning as pl
import numpy as np
import sigpy as sp

from torch.nn import functional as F
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from models.architectures.our_gen_unet_only import UNetModel

from evaluation_scripts.metrics import psnr, ssim
from mail import send_mail


class rcGAN(pl.LightningModule):
    def __init__(self, args, num_realizations, default_model_descriptor, exp_name, noise_type):
        super().__init__()
        self.args = args
        self.num_realizations = num_realizations
        self.default_model_descriptor = default_model_descriptor
        self.exp_name = exp_name
        self.noise_type = noise_type

        self.in_chans = args.in_chans + self.num_realizations * 2
        self.out_chans = args.out_chans
        self.generator = UNetModel(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
        )

        self.std_mult = 1
        self.is_good_model = 0
        self.resolution = self.args.im_size
        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors, mask):
        noise_vals = []
        for i in range(self.num_realizations):
            if self.default_model_descriptor:
                noise_vals.append(ifft2c_new(mask[:, 0, :, :, :]).permute(0, 3, 1, 2))
                break

            # if self.noise_type["AWGN"]:
            z = torch.randn(num_vectors, self.resolution, self.resolution, 2, device=self.device)
            # else:
            #     z = torch.empty(num_vectors, self.resolution, self.resolution, 2, device=self.device).uniform_(0, 1)
            #     z = 2 * torch.bernoulli(z) - 1

            # noise_fft = fft2c_new(z)

            # if self.noise_type["structure"] == 1:
            # meas_noise = ifft2c_new(mask[:, 0, :, :, :] * noise_fft).permute(0, 3, 1, 2)
            # noise_vals.append(meas_noise)
            # elif self.noise_type["structure"] == 2:
            # non_noise = ifft2c_new((1 - mask[:, 0, :, :, :]) * noise_fft).permute(0, 3, 1, 2)
            # noise_vals.append(non_noise)
            # else:
            noise_vals.append(z.permute(0, 3, 1, 2))

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
            samples = self.generator(torch.cat([y, torch.cat(noise, dim=1)], dim=1))
        else:
            samples = self.generator(y)

        samples = self.readd_measures(samples, y, mask)
        return samples

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.std_mult * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train + 1))) * torch.std(gens, dim=1).mean()

    def training_step(self, batch, batch_idx):
        y, x, _, mean, std, mask, _ = batch

        gens = torch.zeros(
            size=(y.size(0), self.args.num_z_train, self.args.in_chans, self.args.im_size, self.args.im_size),
            device=self.device)
        for z in range(self.args.num_z_train):
            gens[:, z, :, :, :] = self.forward(y, mask)

        avg_recon = torch.mean(gens, dim=1)

        g_loss = self.l1_std_p(avg_recon, gens, x)

        self.log('g_loss', g_loss, prog_bar=True)

        return g_loss


    def validation_step(self, batch, batch_idx, external_test=False):
        losses = {
            'psnr': [],
            'single_psnr': [],
            'ssim': [],
            'l1': []
        }

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        y, x, y_true, mean, std, mask, maps = batch

        gens = torch.zeros(size=(y.size(0), num_code, self.args.in_chans, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y, mask)

        avg = torch.mean(gens, dim=1)

        avg_gen = self.reformat(avg)
        gt = self.reformat(x)

        for j in range(y.size(0)):
            S = sp.linop.Multiply((self.args.im_size, self.args.im_size), maps[j].cpu().numpy())
            gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                (avg_gen[j] * std[j] + mean[j]).cpu())

            avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
            gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

            single_gen = torch.zeros(8, self.args.im_size, self.args.im_size, 2, device=self.device)
            single_gen[:, :, :, 0] = gens[j, 0, 0:8, :, :]
            single_gen[:, :, :, 1] = gens[j, 0, 8:16, :, :]

            single_gen_complex_np = tensor_to_complex_np((single_gen * std[j] + mean[j]).cpu())
            single_gen_np = torch.tensor(S.H * single_gen_complex_np).abs().numpy()

            if self.global_rank == 0 and batch_idx == 0 and j == 0:
                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[np.expand_dims(avg_gen_np, axis=2), np.expand_dims(np.abs(avg_gen_np - single_gen_np), axis=2)],
                    labels=[f"Recon: PSNR: {psnr(gt_np, avg_gen_np):.2f}", "Error"]
                )

            losses['ssim'].append(ssim(gt_np, avg_gen_np))
            losses['psnr'].append(psnr(gt_np, avg_gen_np))
            losses['single_psnr'].append(psnr(gt_np, single_gen_np))
            losses['l1'].append(np.linalg.norm((gt_np - avg_gen_np), ord=1))

        losses['psnr'] = np.mean(losses['psnr'])
        losses['ssim'] = np.mean(losses['ssim'])
        losses['single_psnr'] = np.mean(losses['single_psnr'])
        losses['l1'] = np.mean(losses['l1'])

        return losses

    def external_test_step(self, batch, batch_idx):
        y, x, _, mean, std, mask, maps = batch
        y = y.cuda()
        x = x.cuda()
        mean = mean.cuda()
        std = std.cuda()
        mask = mask.cuda()

        new_batch = y, x, _, mean, std, mask, maps

        return self.validation_step(new_batch, batch_idx, external_test=True)

    def validation_step_end(self, batch_parts):
        losses = {
            'psnr': np.mean(batch_parts['psnr']),
            'single_psnr': np.mean(batch_parts['single_psnr']),
            'ssim': np.mean(batch_parts['ssim'])
        }

        return losses

    def validation_epoch_end(self, validation_step_outputs):
        psnrs = []
        single_psnrs = []
        ssims = []

        for out in validation_step_outputs:
            psnrs.append(out['psnr'])
            ssims.append(out['ssim'])
            single_psnrs.append(out['single_psnr'])

        avg_psnr = np.mean(psnrs)

        avg_single_psnr = np.mean(single_psnrs)
        psnr_diff = (avg_single_psnr + 2.5) - avg_psnr

        mu_0 = 2e-2
        self.std_mult += mu_0 * psnr_diff

        self.log('val_psnr', avg_psnr, sync_dist=True)
        self.log('val_psnr_diff', psnr_diff, sync_dist=True)

        if np.abs(psnr_diff) <= 0.25:
            self.is_good_model = 1
        else:
            self.is_good_model = 0

        if self.global_rank == 0 and self.current_epoch % 5 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - MRI UNET - {self.exp_name}",
                      f"Metrics:\nPSNR: {np.mean(psnrs):.2f}\nSSIM: {np.mean(ssims):.4f}\n",
                      file_name="variation_gif.gif")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        return opt_g

    def on_save_checkpoint(self, checkpoint):
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model

    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
