import os

import torch
import torchvision

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd
import sigpy as sp
from matplotlib import cm

from PIL import Image
from torch.nn import functional as F
from models.architectures.super_res_disc import UNetDiscriminatorSN
from models.architectures.super_res_gen import RRDBNet
from evaluation_scripts.metrics import psnr
from mail import send_mail
from torchmetrics.functional import peak_signal_noise_ratio
from losses.perceptual import PerceptualLoss


class SRrcGAN(pl.LightningModule):
    def __init__(self, args, num_realizations, default_model_descriptor, exp_name, noise_type, num_gpus,
                 upscale_factor=4):
        super().__init__()
        self.args = args
        self.num_realizations = num_realizations
        self.default_model_descriptor = default_model_descriptor
        self.exp_name = exp_name
        self.noise_type = noise_type
        self.num_gpus = num_gpus

        self.in_chans = args.in_chans + self.num_realizations * 2
        self.out_chans = args.out_chans
        self.texture_gain = 0.8

        self.generator = RRDBNet(self.in_chans, self.out_chans, upscale=upscale_factor)

        self.discriminator = UNetDiscriminatorSN(num_in_ch=3)

        self.perceptual_loss = PerceptualLoss()

        self.std_mult = 1
        self.is_good_model = 0

        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors, resolution):
        z = torch.randn(num_vectors, 2, resolution, resolution, device=self.device)
        return z

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = Tensor(real_samples.shape[0], 1, real_samples.shape[-1], real_samples.shape[-1]).fill_(1.0).to(self.device)

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

    def forward(self, y, x_lf):
        num_vectors = y.size(0)
        noise = self.get_noise(num_vectors, y.shape[-1])
        gain_channel = torch.ones([y.size(0), 1, y.shape[-1], y.shape[-1]], device=self.device)

        for i in range(y.size(0)):
            t = gain_channel[i, 0, :, :] * self.texture_gain
            gain_channel[i, 0, :, :] = t

        samples = self.generator(y, noise, x_lf, gain_channel)
        return samples

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, gens):
        gen_pred_loss = self.discriminator(gens.reshape(-1, 3, gens.shape[-1], gens.shape[-1]))

        adv_weight = 1e-4

        return - adv_weight * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.std_mult * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train + 1))) * torch.std(gens, dim=1).mean()

    def gradient_penalty(self, x_hat, x):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, x_lf, _, _ = batch

        # train generator
        if optimizer_idx == 1:
            gens = torch.zeros(
                size=(y.size(0), self.args.num_z_train, self.args.in_chans, x.shape[-1], x.shape[-1]),
                device=self.device)
            for z in range(self.args.num_z_train):
                gens[:, z, :, :, :] = self.forward(y, x_lf)

            avg_recon = torch.mean(gens, dim=1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss_generator(gens)

            # for z in range(self.args.num_z_train):
            #     loss, _ = self.perceptual_loss(gens[:, z, :, :, :], x)
            #     g_loss += 1e-3 * loss

            g_loss += self.l1_std_p(avg_recon, gens, x)

            if torch.isnan(g_loss):
                exit()

            self.log('g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 0:
            x_hat = self.forward(y, x_lf)

            real_pred = self.discriminator(x)
            fake_pred = self.discriminator(x_hat)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x)
            d_loss += self.drift_penalty(real_pred)

            if torch.isnan(d_loss):
                exit()

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        y, x, x_lf, _, _ = batch

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        gens = torch.zeros(size=(y.size(0), 8, self.args.in_chans, x.shape[-1], x.shape[-1]),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y, x_lf)

        avg = torch.mean(gens, dim=1)

        psnr_8s = []
        psnr_1s = []

        psnr_8s.append(peak_signal_noise_ratio(avg, x))
        psnr_1s.append(peak_signal_noise_ratio(gens[:, 0, :, :, :], x))

        psnr_8s = torch.stack(psnr_8s)
        psnr_1s = torch.stack(psnr_1s)

        psnr_8s = torch.nan_to_num(psnr_8s, nan=0.0, posinf=100, neginf=0)
        psnr_1s = torch.nan_to_num(psnr_1s, nan=0.0, posinf=100, neginf=0)

        self.log('psnr_8_step', psnr_8s.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log('psnr_1_step', psnr_1s.mean(), on_step=True, on_epoch=False, prog_bar=True)

        ############################################

        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 50 == 0:
                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[Image.fromarray(np.uint8(x[0].cpu().numpy().transpose(1, 2, 0) * 255), 'RGB'),
                            Image.fromarray(np.uint8(avg[0].cpu().numpy().transpose(1, 2, 0) * 255), 'RGB'),
                            Image.fromarray(np.uint8(gens[0, 0].cpu().numpy().transpose(1, 2, 0) * 255), 'RGB'),
                            Image.fromarray(np.uint8(gens[0, 1].cpu().numpy().transpose(1, 2, 0) * 255), 'RGB')],
                    caption=["GT", f"Recon", "Samp 1", "Samp 2"]
                )

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8s.mean(), 'psnr_1': psnr_1s.mean()}

    def validation_epoch_end(self, validation_step_outputs):
        avg_psnr = self.all_gather(torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()).mean()
        avg_single_psnr = self.all_gather(torch.stack([x['psnr_1'] for x in validation_step_outputs]).mean()).mean()

        avg_psnr = avg_psnr.cpu().numpy()
        avg_single_psnr = avg_single_psnr.cpu().numpy()

        psnr_diff = (avg_single_psnr + 2.5) - avg_psnr
        psnr_diff = psnr_diff

        mu_0 = 2e-2
        if self.current_epoch % 25 == 0:
            self.std_mult += mu_0 * psnr_diff

        if np.abs(psnr_diff) <= 0.25:
            self.is_good_model = 1
        else:
            self.is_good_model = 0

        if self.global_rank == 0 and self.current_epoch % 50 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - rcGAN - SR",
                      f"Std. Dev. Weight: {self.std_mult:.4f}\nMetrics:\nPSNR: {avg_psnr:.2f}\nSINGLE PSNR: {avg_single_psnr:.2f}\nPSNR Diff: {psnr_diff}",
                      file_name="variation_gif.gif")

        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))

        milestones = [50000, 100000, 200000, 300000]
        gamma = 0.5

        schedule_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones, gamma)
        schedule_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones, gamma)

        return [opt_d, opt_g], [schedule_d, schedule_g]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model

    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
