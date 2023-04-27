import os

import torch
import torchvision
import pytorch_ssim
import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd
import sigpy as sp
from matplotlib import cm

from PIL import Image
from torch.nn import functional as F
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from models.architectures.our_gen_unet_only import UNetModel
from models.architectures.our_disc import DiscriminatorModel
from models.architectures.patch_disc import PatchDisc
from evaluation_scripts.metrics import psnr
from mail import send_mail
from torchmetrics.functional import peak_signal_noise_ratio
from fastmri.data.transforms import to_tensor

class L1SSIMMRI(pl.LightningModule):
    def __init__(self, args, num_realizations, default_model_descriptor, exp_name, noise_type, num_gpus):
        super().__init__()
        self.args = args
        self.num_realizations = num_realizations
        self.default_model_descriptor = default_model_descriptor
        self.exp_name = exp_name
        self.noise_type = noise_type
        self.num_gpus = num_gpus

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.unet = UNetModel(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
        )

        self.resolution = self.args.im_size
        self.ssim_loss = pytorch_ssim.SSIM()

        self.save_hyperparameters()  # Save passed values

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
        samples = self.unet(y)
        samples = self.readd_measures(samples, y, mask)
        return samples

    def training_step(self, batch, batch_idx):
        y, x, mask, mean, std, _, _, _ = batch

        # train generator
        x_hat = self.forward(y, mask)
        alpha = 0.84

        # adversarial loss is binary cross-entropy
        loss = (1 - alpha) * F.l1_loss(x_hat, x) - alpha * self.ssim_loss(x_hat, x)

        self.log('loss', loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx, external_test=False):
        y, x, mask, mean, std, maps, _, _ = batch

        fig_count = 0

        x_hat = self.reformat(self.forward(y, mask) * std[:, None, None, None] + mean[:, None, None, None])
        gt = self.reformat(x * std[:, None, None, None] + mean[:, None, None, None])

        mag_avg_list = []
        mag_gt_list = []
        psnr_8s = []

        for j in range(y.size(0)):
            S = sp.linop.Multiply((self.args.im_size, self.args.im_size), tensor_to_complex_np(maps[j].cpu()))

            ############# EXPERIMENTAL #################
            # ON CPU
            x_hat_sp_out = torch.tensor(S.H * tensor_to_complex_np(x_hat[j].cpu())).abs().unsqueeze(0).unsqueeze(0).to(self.device)
            gt_sp_out = torch.tensor(S.H * tensor_to_complex_np(gt[j].cpu())).abs().unsqueeze(0).unsqueeze(0).to(self.device)

            # ON GPU
            # avg_sp_out = complex_abs(sp.to_pytorch(S.H * sp.from_pytorch(avg_gen[j], iscomplex=True))).unsqueeze(0).unsqueeze(0)
            # single_sp_out = complex_abs(sp.to_pytorch(S.H * sp.from_pytorch(self.reformat(gens[:, 0])[j], iscomplex=True))).unsqueeze(0).unsqueeze(0)
            # gt_sp_out = complex_abs(sp.to_pytorch(S.H * sp.from_pytorch(gt[j], iscomplex=True))).unsqueeze(0).unsqueeze(0)

            psnr_8s.append(peak_signal_noise_ratio(x_hat_sp_out, gt_sp_out))

            mag_avg_list.append(x_hat_sp_out)
            mag_gt_list.append(gt_sp_out)

        psnr_8s = torch.stack(psnr_8s)
        mag_avg_gen = torch.cat(mag_avg_list, dim=0)
        mag_gt = torch.cat(mag_gt_list, dim=0)

        self.log('psnr_8_step', psnr_8s.mean(), on_step=True, on_epoch=False, prog_bar=True)
        ############################################

        # TODO: Plot as tensors using torch function
        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 5 == 0 and fig_count == 0:
                fig_count += 1
                avg_gen_np = mag_avg_gen[0, 0, :, :].cpu().numpy()
                gt_np = mag_gt[0, 0, :, :].cpu().numpy()

                plot_avg_np = (avg_gen_np - np.min(avg_gen_np)) / (np.max(avg_gen_np) - np.min(avg_gen_np))
                plot_gt_np = (gt_np - np.min(gt_np)) / (np.max(gt_np) - np.min(gt_np))

                np_psnr = psnr(gt_np, avg_gen_np)

                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[Image.fromarray(np.uint8(plot_gt_np*255), 'L'), Image.fromarray(np.uint8(plot_avg_np*255), 'L'), Image.fromarray(np.uint8(cm.jet(5*np.abs(plot_gt_np - plot_avg_np))*255))],
                    caption=["GT", f"Recon: PSNR (NP): {np_psnr:.2f}", "Error"]
                )

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8s.mean()}

    def validation_epoch_end(self, validation_step_outputs):
        # GATHER
        avg_psnr = self.all_gather(torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()).mean()

        # NO GATHER
        # avg_psnr = torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()
        # avg_single_psnr = torch.stack([x['psnr_1'] for x in validation_step_outputs]).mean()

        self.log('val_psnr', avg_psnr, sync_dist=True)

        avg_psnr = avg_psnr.cpu().numpy()

        if self.global_rank == 0 and self.current_epoch % 1 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - L1 + SSIM",
                      f"Metrics:\nPSNR: {avg_psnr:.2f}",
                      file_name="variation_gif.gif")

        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.unet.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        return opt_g
