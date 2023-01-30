import os

import torch
import torchvision

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.fftc import ifft2c_new, fft2c_new

from models.architectures.our_gen import GeneratorModel
from models.architectures.patch_disc import PatchDisc

class rcGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.generator = GeneratorModel(
            in_chans=args.in_chans + 4,
            out_chans=args.out_chans,
        )

        self.discriminator = PatchDisc(
            input_nc=args.in_chans * 2
        )

        self.std_mult = 1
        self.resolution = self.args.im_size

    def get_noise(self, num_vectors, mask):
        z_vals = []
        measured_vals = []
        for i in range(2):
            z = torch.randn(num_vectors, self.resolution, self.resolution, 2).cuda()
            noise_fft = fft2c_new(z)
            measured_noise = ifft2c_new(mask[:, 0, :, :, :] * noise_fft).permute(0, 3, 1, 2)
            z_vals.append(z.permute(0, 3, 1, 2))
            measured_vals.append(measured_noise)
        return measured_vals, z_vals

    def reformat(self, samples):
        reformatted_tensor = torch.zeros(size=(samples.size(0), 8, self.resolution, self.resolution, 2)).cuda()
        reformatted_tensor[:, :, :, :, 0] = samples[:, 0:8, :, :]
        reformatted_tensor[:, :, :, :, 1] = samples[:, 8:16, :, :]

        return reformatted_tensor

    def readd_measures(self, samples, measures, mask):
        reformatted_tensor = self.reformat(samples)
        measures = self.reformat(measures)
        reconstructed_kspace = fft2c_new(reformatted_tensor)

        reconstructed_kspace = mask * measures + (1 - mask) * reconstructed_kspace

        image = ifft2c_new(reconstructed_kspace)

        output_im = torch.zeros(size=samples.shape).cuda()
        output_im[:, 0:8, :, :] = image[:, :, :, :, 0]
        output_im[:, 8:16, :, :] = image[:, :, :, :, 1]

        return output_im

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(input=interpolates, y=y)
        fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).cuda()

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
        measured, z = self.get_noise(num_vectors, mask)
        samples = self.generator(y, measured, z)
        samples = self.readd_measures(samples, y, mask)
        return samples

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, y, gens):
        patch_out = 30
        fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z, patch_out, patch_out)).cuda()
        for k in range(y.shape[0]):
            cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4]).cuda()
            cond[0, :, :, :] = y[k, :, :, :]
            cond = cond.repeat(self.args.num_z, 1, 1, 1)
            temp = self.discriminator(input=gens[k], y=cond)
            fake_pred[k, :, :, :] = temp[:, 0, :, :]

        gen_pred_loss = torch.mean(fake_pred[0])
        for k in range(y.shape[0] - 1):
            gen_pred_loss += torch.mean(fake_pred[k + 1])

        return - 1e-2 * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.std_mult * np.sqrt(2 / (np.pi * self.args.num_z * (self.args.num_z + 1))) * torch.std(gens, dim=1).mean()

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, y_true, mean, std, mask = batch

        # train generator
        if optimizer_idx == 0:
            gens = torch.zeros(size=(y.size(0), self.args.num_z, self.args.in_chans, self.args.im_size, self.args.im_size)).cuda()
            for z in range(self.args.num_z):
                gens[:, z, :, :, :] = self.forward(y, mask)

            avg_recon = torch.mean(gens, dim=1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss_generator(y, gens)
            g_loss += self.l1_std_p(avg_recon, gens, x)

            self.log('g_loss', g_loss, sync_dist=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            x_hat = self.forward(y, mask)

            real_pred = self.discriminator(input=x, y=y)
            fake_pred = self.discriminator(input=x_hat, y=y)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, sync_dist=True)

            return d_loss

    def validation_step(self, batch, batch_idx):
        # UPDATE STD DEV REWARD AT END
        pass

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        return [opt_g, opt_d], []

    def on_save_checkpoint(self, checkpoint):
        checkpoint["beta_std"] = self.std_mult

    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
