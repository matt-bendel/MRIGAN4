import torch

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd

from models.comodgan.co_mod_gan import Generator, Discriminator

from evaluation_scripts.metrics import psnr, ssim
from mail import send_mail


class InpaintUNet(pl.LightningModule):
    def __init__(self, args, num_realizations, default_model_descriptor, exp_name):
        super().__init__()
        self.args = args
        self.num_realizations = num_realizations
        self.default_model_descriptor = default_model_descriptor
        self.exp_name = exp_name

        self.generator = Generator(self.args.im_size, self.num_realizations)
        self.discriminator = Discriminator(self.args.im_size)

        self.resolution = self.args.im_size
        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors, mask):
        noise_vals = []
        for i in range(self.num_realizations):
            if self.default_model_descriptor:
                noise_vals.append(mask)
                break

            z = torch.empty(num_vectors, 1, self.resolution, self.resolution, device=self.device).uniform_(0, 1)
            z = 2 * torch.bernoulli(z) - 1
            noise = z * mask[:, None, :, :]
            noise_vals.append(noise)

        return noise_vals

    def readd_measures(self, samples, measures, mask):
        return samples * (1 - mask) + measures * mask

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(input=interpolates, label=y)
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(
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
        if self.num_realizations > 0:
            noise = self.get_noise(num_vectors, mask)
            samples = self.generator(torch.cat([y, torch.cat(noise, dim=1)], dim=1), mask, [torch.randn(y.size(0), 512, device=y.device)])
        else:
            samples = self.generator(y, mask, [torch.randn(y.size(0), 512, device=y.device)])

        samples = self.readd_measures(samples, y, mask)
        return samples

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, y, x_hat):
        gen_pred_loss = self.discriminator(input=x_hat, label=y)

        return - gen_pred_loss.mean()

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, mean, std, mask = batch[0]

        # train generator
        if optimizer_idx == 0:
            x_hat = self.forward(y, mask)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss_generator(y, x_hat)

            self.log('g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            x_hat = self.forward(y, mask)

            real_pred = self.discriminator(input=x, label=y)
            fake_pred = self.discriminator(input=x_hat, label=y)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx):
        losses = {
            'psnr': [],
            'ssim': []
        }

        y, x, mean, std, mask = batch[0]
        x_hat = self.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]
        x = x * std[:, :, None, None] + mean[:, :, None, None]

        for j in range(y.size(0)):
            losses['ssim'].append(ssim(x[j].cpu().numpy().transpose(1, 2, 0), x_hat[j].cpu().numpy().transpose(1, 2, 0), multichannel=True))
            losses['psnr'].append(psnr(x[j].cpu().numpy(), x_hat[j].cpu().numpy()))

        losses['psnr'] = np.mean(losses['psnr'])
        losses['ssim'] = np.mean(losses['ssim'])

        return losses

    def external_test_step(self, batch, batch_idx):
        y, x, mean, std, mask = batch
        y = y.cuda()
        x = x.cuda()
        mean = mean.cuda()
        std = std.cuda()
        mask = mask.cuda()

        new_batch = y, x, mean, std, mask

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

        avg_psnr = np.mean(psnrs)

        self.log('val_ssim', np.mean(ssim), sync_dist=True)

        if self.global_rank == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - CoModGAN - {self.exp_name}",
                      f"Metrics:\nPSNR: {avg_psnr:.2f}\nSSIM: {np.mean(ssims):.4f}",
                      file_name="variation_gif.gif")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        return [opt_g, opt_d], []
