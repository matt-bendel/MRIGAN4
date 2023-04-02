import os

import torch
import torchvision

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import sigpy as sp
import sigpy.mri as mr
from matplotlib import cm

from PIL import Image
from torch.nn import functional as F
from data import transforms
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from models.architectures.our_gen_unet_only import UNetModel
from models.architectures.our_disc import DiscriminatorModel
from evaluation_scripts.plotting_scripts import generate_image, generate_error_map
from evaluation_scripts.metrics import psnr, ssim
from evaluation_scripts.plotting_scripts import gif_im, generate_gif
from mail import send_mail


class Adler(pl.LightningModule):
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

        self.discriminator = DiscriminatorModel(
            in_chans=args.in_chans * 3,
            out_chans=args.out_chans
        )

        self.resolution = self.args.im_size
        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors, mask):
        z = torch.randn(num_vectors, 2, self.resolution, self.resolution, device=self.device)
        return z

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

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(input=interpolates, y=y)
        fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(
            self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, y, mask):
        num_vectors = y.size(0)
        noise = self.get_noise(num_vectors, mask)
        samples = self.generator(torch.cat([y, noise], dim=1))
        samples = self.readd_measures(samples, y, mask)
        return samples

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, mask, max_val, _, _, _ = batch

        # train generator
        if optimizer_idx == 0:
            gen1 = self.forward(y, mask)
            gen2 = self.forward(y, mask)

            x_posterior_concat = torch.cat([gen1, gen2], 1)

            fake_pred = self.discriminator(input=x_posterior_concat, y=y)

            # adversarial loss is binary cross-entropy
            g_loss = -fake_pred.mean()

            self.log('g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            gen1 = self.forward(y, mask)
            gen2 = self.forward(y, mask)

            rand_num = np.random.randint(0, 2, 1)
            if rand_num == 0:
                x_expect = torch.cat([x, gen1], 1)
            else:
                x_expect = torch.cat([gen1, x], 1)

            # MAKE PREDICTIONS
            x_posterior_concat = torch.cat([gen1, gen2], 1)
            real_pred = D(input=x_expect, y=y)
            fake_pred = D(input=x_posterior_concat, y=y)

            d_loss = fake_pred.mean() - real_pred.mean()
            d_loss += self.gradient_penalty(x_posterior_concat, x_expect, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        losses = {
            'psnr': [],
            'single_psnr': [],
            'ssim': []
        }

        y, x, mask, max_val, maps, _, _ = batch

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        gens = torch.zeros(size=(y.size(0), 8, self.args.in_chans, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y, mask)

        avg = torch.mean(gens, dim=1)

        avg_gen = self.reformat(avg)
        gt = self.reformat(x)

        for j in range(y.size(0)):
            S = sp.linop.Multiply((self.args.im_size, self.args.im_size), maps[j].cpu().numpy())
            gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * max_val[j]).cpu()), tensor_to_complex_np(
                (avg_gen[j] * max_val[j]).cpu())

            avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
            gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

            single_gen = torch.zeros(8, self.args.im_size, self.args.im_size, 2, device=self.device)
            single_gen[:, :, :, 0] = gens[j, 0, 0:8, :, :]
            single_gen[:, :, :, 1] = gens[j, 0, 8:16, :, :]

            single_gen_complex_np = tensor_to_complex_np((single_gen * max_val[j]).cpu())
            single_gen_np = torch.tensor(S.H * single_gen_complex_np).abs().numpy()

            if self.global_rank == 0 and batch_idx == 0 and j == 0 and self.current_epoch % 5 == 0:
                plot_avg_np = (avg_gen_np - np.min(avg_gen_np)) / (np.max(avg_gen_np) - np.min(avg_gen_np))
                plot_gt_np = (gt_np - np.min(gt_np)) / (np.max(gt_np) - np.min(gt_np))

                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[Image.fromarray(np.uint8(plot_gt_np*255), 'L'), Image.fromarray(np.uint8(plot_avg_np*255), 'L'), Image.fromarray(np.uint8(cm.jet(np.abs(plot_gt_np - plot_avg_np))*255))],
                    caption=["GT", f"Recon: PSNR: {psnr(gt_np, avg_gen_np):.2f}; SINGLE PSNR: {psnr(gt_np, single_gen_np):.2f}", "Error"]
                )

            self.trainer.strategy.barrier()

            losses['ssim'].append(ssim(gt_np, avg_gen_np))
            losses['psnr'].append(psnr(gt_np, avg_gen_np))
            losses['single_psnr'].append(psnr(gt_np, single_gen_np))

        losses['psnr'] = np.mean(losses['psnr'])
        losses['ssim'] = np.mean(losses['ssim'])
        losses['single_psnr'] = np.mean(losses['single_psnr'])

        return losses

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

        if self.global_rank == 0 and self.current_epoch % 2 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - Ohayon",
                      f"Metrics:\nPSNR: {avg_psnr:.2f}\nSINGLE PSNR: {avg_single_psnr:.2f}\nSSIM: {np.mean(ssims):.4f}\nPSNR Diff: {psnr_diff}",
                      file_name="variation_gif.gif")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        return [opt_g, opt_d], []