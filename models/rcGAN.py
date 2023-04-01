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
from models.architectures.patch_disc import PatchDisc
from evaluation_scripts.plotting_scripts import generate_image, generate_error_map
from evaluation_scripts.metrics import psnr, ssim
from evaluation_scripts.plotting_scripts import gif_im, generate_gif
from mail import send_mail
from torchmetrics import PeakSignalNoiseRatio

class rcGAN(pl.LightningModule):
    def __init__(self, args, num_realizations, default_model_descriptor, exp_name, noise_type, num_gpus):
        super().__init__()
        self.args = args
        self.num_realizations = num_realizations
        self.default_model_descriptor = default_model_descriptor
        self.exp_name = exp_name
        self.noise_type = noise_type
        self.num_gpus = num_gpus

        self.in_chans = args.in_chans + self.num_realizations * 2
        self.out_chans = args.out_chans

        self.generator = UNetModel(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
        )

        self.discriminator = PatchDisc(
            input_nc=args.in_chans * 2
        )

        self.std_mult = 1
        self.is_good_model = 0
        self.resolution = self.args.im_size
        self.psnr_1 = PeakSignalNoiseRatio()
        self.psnr_8 = PeakSignalNoiseRatio()

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

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, y, gens):
        patch_out = 94
        fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z_train, patch_out, patch_out), device=self.device)
        for k in range(y.shape[0]):
            cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4], device=self.device)
            cond[0, :, :, :] = y[k, :, :, :]
            cond = cond.repeat(self.args.num_z_train, 1, 1, 1)
            temp = self.discriminator(input=gens[k], y=cond)
            fake_pred[k, :, :, :] = temp[:, 0, :, :]

        gen_pred_loss = torch.mean(fake_pred[0])
        for k in range(y.shape[0] - 1):
            gen_pred_loss += torch.mean(fake_pred[k + 1])

        return - 1e-2 * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.std_mult * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train+ 1))) * torch.std(gens, dim=1).mean()

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, mask, mean, std, _, _, _ = batch

        # train generator
        if optimizer_idx == 1:
            gens = torch.zeros(
                size=(y.size(0), self.args.num_z_train, self.args.in_chans, self.args.im_size, self.args.im_size),
                device=self.device)
            for z in range(self.args.num_z_train):
                gens[:, z, :, :, :] = self.forward(y, mask)

            avg_recon = torch.mean(gens, dim=1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss_generator(y, gens)
            g_loss += self.l1_std_p(avg_recon, gens, x)

            self.log('g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 0:
            x_hat = self.forward(y, mask)

            real_pred = self.discriminator(input=x, y=y)
            fake_pred = self.discriminator(input=x_hat, y=y)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        losses = {
            'psnr': [],
            'single_psnr': [],
            'ssim': []
        }

        y, x, mask, mean, std, maps, _, _ = batch

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        gens = torch.zeros(size=(y.size(0), 8, self.args.in_chans, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y, mask) * std[:, None, None, None] + mean[:, None, None, None] # EXPERIMENTAL UN

        avg = torch.mean(gens, dim=1)

        avg_gen = self.reformat(avg)
        gt = self.reformat(x * std[:, None, None, None] + mean[:, None, None, None])

        mag_avg_list = []
        mag_single_list = []
        mag_gt_list = []

        for j in range(y.size(0)):
            S = sp.linop.Multiply((self.args.im_size, self.args.im_size), sp.from_pytorch(maps))

            ############# EXPERIMENTAL #################
            S_pt = sp.to_pytorch_function(S.H, True, True)
            mag_avg_list.append(S_pt.forward(avg_gen[j]).abs().unsqueeze(0).unsqueeze(0))
            mag_single_list.append(S_pt.forward(self.reformat(gens[j, 0])).abs().unsqueeze(0).unsqueeze(0))
            mag_gt_list.append(S_pt.forward(gt[j]).abs().unsqueeze(0).unsqueeze(0))

        mag_avg_gen = torch.cat(mag_avg_list, dim=0)
        mag_single_gen = torch.cat(mag_single_list, dim=0)
        mag_gt = torch.cat(mag_gt_list, dim=0)

        psnr_8 = self.psnr_8(mag_avg_gen, mag_gt)
        psnr_1 = self.psnr_1(mag_single_gen, mag_gt)

        self.log('psnr_8_step', self.psnr_8, prog_bar=True)
        self.log('psnr_1_step', self.psnr_1, prog_bar=True)
        ############################################

        # TODO: Plot as tensors using torch function
        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 5 == 0:
                avg_gen_np = mag_avg_gen[0, 0, :, :].cpu().numpy()
                gt_np = mag_gt[0, 0, :, :].cpu().numpy()

                plot_avg_np = (avg_gen_np - np.min(avg_gen_np)) / (np.max(avg_gen_np) - np.min(avg_gen_np))
                plot_gt_np = (gt_np - np.min(gt_np)) / (np.max(gt_np) - np.min(gt_np))

                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[Image.fromarray(np.uint8(plot_gt_np*255), 'L'), Image.fromarray(np.uint8(plot_avg_np*255), 'L'), Image.fromarray(np.uint8(cm.jet(5*np.abs(plot_gt_np - plot_avg_np))*255))],
                    caption=["GT", f"Recon: PSNR (NP): {psnr(gt_np, avg_gen_np):.2f}; PSNR (PT): {psnr_8.cpu().numpy():.2f}; SINGLE PSNR: {psnr_1.cpu().numpy():.2f}", "Error"]
                )

            self.trainer.strategy.barrier()

    def validation_epoch_end(self, validation_step_outputs):
        avg_psnr = self.all_gather(self.psnr_8.compute()).mean()
        print(avg_psnr)
        avg_single_psnr = self.all_gather(self.psnr_1.compute()).mean()

        psnr_diff = (avg_single_psnr + 2.5) - avg_psnr

        mu_0 = 2e-2
        self.std_mult += mu_0 * psnr_diff.cpu().numpy()

        if np.abs(psnr_diff) <= 0.25:
            self.is_good_model = 1
        else:
            self.is_good_model = 0

        if self.global_rank == 0 and self.current_epoch % 1 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - rcGAN",
                      f"Std. Dev. Weight: {self.std_mult:.4f}\nMetrics:\nPSNR: {avg_psnr:.2f}\nSINGLE PSNR: {avg_single_psnr:.2f}\nPSNR Diff: {psnr_diff}",
                      file_name="variation_gif.gif")

        self.psnr_8.reset()
        self.psnr_1.reset()

        self.trainer.strategy.barrier()
        print(self.psnr_8.compute())
        exit()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        return [opt_d, opt_g], []

    def on_save_checkpoint(self, checkpoint):
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model

    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
