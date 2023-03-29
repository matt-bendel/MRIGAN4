import os
import torch
import random
import pickle
import yaml

import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.MRIDataModule import MRIDataModule
from datasets.fastmri_multicoil_general import FastMRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from pytorch_lightning import seed_everything
import sigpy as sp
import sigpy.mri as mr
from utils.fftc import ifft2c_new, fft2c_new
from utils.math import complex_abs, tensor_to_complex_np

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    with open('configs/mri/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = FastMRIDataModule(base_path=cfg.data_path, batch_size=cfg.batch_size)
    dm.setup()
    val_loader = dm.val_dataloader()

    for i, data in enumerate(val_loader):
        y, x, mask, max_val, maps, fname, slice = data

        for j in range(y.size(0)):
            new_y_true = fft2c_new(ifft2c_new(y_true[j]) * max_val[j])
            maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y_true.cpu()), calib_width=16,
                                       device=sp.Device(0), crop=0.70,
                                       kernel_width=6).run().get()

            with open(f'/storage/fastMRI_brain/sense_maps/val_full_res/{fname[j]}_{slice[j]}.pkl', 'wb') as outp:
                pickle.dump(maps, outp, pickle.HIGHEST_PROTOCOL)
