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
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    dm = MRIDataModule(args, 2)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = DataLoader(dm.train, shuffle=True, num_workers=num_workers, batch_size=20, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
    # for i, data in enumerate(val_loader):
    #     y, x, y_true, mean, std, mask, fname, slice = data
    #
    #     for j in range(y.size(0)):
    #         hf = h5py.File(f'/storage/fastMRI_brain/preprocessed_data/train/{fname[j]}_{slice[j]}.h5', 'w')
