import pathlib
import shutil
import torch
import numpy as np

from utils.fftc import ifft2c_new, fft2c_new
from utils.get_mask import get_mask

class GANWrapper:
    def __init__(self, gen, args):
        self.args = args
        self.resolution = args.im_size
        self.gen = gen
        self.data_consistency = True

    def get_noise(self, num_vectors, var, mask):
        z_vals = []
        measured_vals = []
        for i in range(2):
            z = torch.randn(num_vectors, self.resolution, self.resolution, 2).cuda()
            noise_fft = fft2c_new(z)
            measured_noise = ifft2c_new(mask[:, 0, :, :, :] * noise_fft).permute(0, 3, 1, 2)
            z_vals.append(z.permute(0, 3, 1, 2))
            measured_vals.append(measured_noise)
        return measured_vals, z_vals


    def update_gen_status(self, val):
        self.gen.eval() if val else self.gen.train()

    def reformat(self, samples):
        reformatted_tensor = torch.zeros(size=(samples.size(0), 8, self.resolution, self.resolution, 2),
                                         device=self.args.device)
        reformatted_tensor[:, :, :, :, 0] = samples[:, 0:8, :, :]
        reformatted_tensor[:, :, :, :, 1] = samples[:, 8:16, :, :]

        return reformatted_tensor

    def readd_measures(self, samples, measures, mask):
        reformatted_tensor = self.reformat(samples)
        reconstructed_kspace = fft2c_new(reformatted_tensor)

        reconstructed_kspace = mask * measures + (1 - mask) * reconstructed_kspace

        image = ifft2c_new(reconstructed_kspace)

        output_im = torch.zeros(size=samples.shape, device=self.args.device)
        output_im[:, 0:8, :, :] = image[:, :, :, :, 0]
        output_im[:, 8:16, :, :] = image[:, :, :, :, 1]

        return output_im

    def __call__(self, y, mask=None):
        num_vectors = y.size(0)
        measured, z = self.get_noise(num_vectors, 1, mask)
        samples = self.gen(y, measured, z, mid_z=None)
        samples = self.readd_measures(samples, y, mask)
        return samples
