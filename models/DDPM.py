import torch

import pytorch_lightning as pl
import numpy as np

from PIL import Image
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from mail import send_mail
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class DDPM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ddpm_net = Unet(
            dim=128,
            dim_mults=(1, 1, 2, 2, 4, 4),
            channels=2,
            resnet_block_groups=2,
            attn_dim_head=24,
            full_attn=(False, False, False, False, True, False)
        )
        self.diffusion = GaussianDiffusion(
            self.ddpm_net,
            image_size=384,
            timesteps=1000
        )

        self.resolution = 384

        self.save_hyperparameters()  # Save passed values

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.float()

        # train generator
        loss = self.diffusion(x)

        self.log('diff_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, external_test=False):
        x = batch
        x = x.float()

        # TODO: Plot as tensors using torch function
        if batch_idx == 0:
            samps = self.diffusion.sample(batch_size=4)
            samps = complex_abs(ifft2c_new(samps.permute(0, 2, 3, 1)))

            for i in range(4):
                samps[i, :, :] = (samps[i, :, :] - samps[i, :, :].min()) / (samps[i, :, :].max() - samps[i, :, :].min())

            if self.global_rank == 0 and self.current_epoch % 5 == 0:
                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[Image.fromarray(np.uint8(samps[0, :, :].cpu().numpy()*255), 'L'), Image.fromarray(np.uint8(samps[1, :, :].cpu().numpy()*255), 'L'), Image.fromarray(np.uint8(samps[2, :, :].cpu().numpy()*255)), Image.fromarray(np.uint8(samps[3, :, :].cpu().numpy()*255))],
                    caption=["Samp 1", "Samp 2", "Samp 3", "Samp 4"]
                )

            self.trainer.strategy.barrier()

        return {}

    def validation_epoch_end(self, validation_step_outputs):
        if self.global_rank == 0 and self.current_epoch % 1 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - DDPM",
                      f"TEMP",
                      file_name="variation_gif.gif")

        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.ddpm_net.parameters(), lr=5e-5, weight_decay=0)

        return opt_g

