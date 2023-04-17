import os
import subprocess
import torch.utils.data as data
import pytorch_lightning as pl

import numpy as np
import time
import torch
from natsort import natsort
import imageio
import glob
import sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

class LRHR_IMGDataset(data.Dataset):
    def __init__(self, cfg, scale, mode):
        super(LRHR_IMGDataset, self).__init__()
        self.scale = scale

        self.hr_file_path = cfg.data_path + f"/{scale}/hr/{mode}"
        self.lr_file_path = cfg.data_path + f"/{scale}/lr/{mode}"
        augment = True

        self.use_flip = True
        self.use_rot = True
        self.use_crop = False
        self.center_crop_hr_size = None

        self.hr_paths = fiFindByWildcard(os.path.join(self.hr_file_path, '*.png'))
        self.lr_paths = fiFindByWildcard(os.path.join(self.lr_file_path, '*.png'))

        self.augment = augment

    def imread(self, img_path):
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img, ] * 3, axis=2)
        return img

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, item):
        hr = self.imread(self.hr_paths[item])
        lr = self.imread(self.lr_paths[item])

        hr = np.transpose(hr, [2, 0, 1])
        lr = np.transpose(lr, [2, 0, 1])

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = hr / 255.0
        lr = lr / 255.0

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        return lr, hr, self.lr_paths[item], self.hr_paths[item]

def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg

class SRDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, args, scale):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.scale = scale

    def prepare_data(self):
        pass

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        train_data = LRHR_IMGDataset(self.args, self.scale, "train")
        dev_data = LRHR_IMGDataset(self.args, self.scale, "val")
        test_data = None#LRHR_IMGDataset(self.args, self.scale, "test")

        self.train, self.validate, self.test = train_data, dev_data, test_data

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            drop_last=False,
            num_workers=5,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            drop_last=False,
            num_workers=5,
            pin_memory=False,
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.test,
    #         batch_size=self.args.batch_size,
    #         num_workers=20,
    #         pin_memory=True,
    #     )