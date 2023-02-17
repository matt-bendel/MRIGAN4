import torch
import math
import pytorch_ssim

import pytorch_lightning as pl
import numpy as np
from models.architectures.mri_unet import Unet

from evaluation_scripts.metrics import psnr, ssim
from mail import send_mail

class InpaintUNet(pl.LightningModule):
    def __init__(
            self, args, num_realizations, default_model_descriptor,
            chans=64,
            num_pool_layers=5,
            drop_prob=0.0,
            lr=0.001,
            lr_step_size=40,
            lr_gamma=0.1,
            weight_decay=0.0,
            loss_weight=0.84):
        super().__init__()
        self.args = args
        self.loss_weight = loss_weight
        self.num_realizations = num_realizations
        self.default_model_descriptor = default_model_descriptor

        self.in_chans = args.in_chans + self.num_realizations * 2
        self.out_chans = args.out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.loss_weight = 0.84
        self.ssim_loss = pytorch_ssim.SSIM()

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

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
        return samples * (1 - mask[:, None, :, :]) + measures * mask[:, None, :, :]

    def log_cosh_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean(self._log_cosh(y_pred - y_true))

    def _log_cosh(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)

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
        y, x, mean, std, mask = batch

        # train generator
        x_hat = self.forward(y, mask)
        loss = (1.0 - self.loss_weight) * (1.0 - self.ssim_loss(x_hat, x))
        loss += self.loss_weight * self.log_cosh_loss(x_hat, x)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        losses = {
            'psnr': [],
            'ssim': []
        }

        y, x, mean, std, mask = batch
        x_hat = self.forward(y, mask)

        for j in range(y.size(0)):
            losses['ssim'].append(ssim(x[j].cpu().numpy(), x_hat[j].cpu().numpy()))
            losses['psnr'].append(psnr(x[j].cpu().numpy(), x_hat[j].cpu().numpy()))

        losses['psnr'] = np.mean(losses['psnr'])
        losses['ssim'] = np.mean(losses['ssim'])

        return losses

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

        self.log('val_ssim', np.mean(ssim), sync_dist=True)

        if self.global_rank == 0 and self.current_epoch % 5 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - INPAINTING",
                      f"Metrics:\nPSNR: {np.mean(psnrs):.2f}\nSSIM: {np.mean(ssims):.4f}\n",
                      file_name="variation_gif.gif")

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]
