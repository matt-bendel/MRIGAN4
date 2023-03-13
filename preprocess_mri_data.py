import os
import torch
import random
import pickle
import h5py

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from pytorch_lightning import seed_everything
import sigpy as sp
import sigpy.mri as mr
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from time import time
import multiprocessing as mp

if __name__ == '__main__':
    for fname in sorted(files):
        kspace = h5py.File(fname, 'r')['kspace']

        if kspace.shape[-1] <= 384 or kspace.shape[1] < 8 or str(
                fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_209_2090296.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_200_2000250.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_201_2010106.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_204_2130024.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_210_2100025.h5':
            continue
        else:
            num_slices = 8  # kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    for i, data in enumerate(val_loader):
        y, x, y_true, mean, std, mask, fname, slice = data

        for j in range(y.size(0)):
            hf = h5py.File(f'/storage/fastMRI_brain/preprocessed_data/train/{fname[j]}_{slice[j]}.h5', 'w')
