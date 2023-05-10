import os
import subprocess
import torch.utils.data as data
import pytorch_lightning as pl
import cv2
import numpy as np
import time
import torch
from natsort import natsort
import imageio
import glob
import sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from imresize import imresize


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]

def t(array): return torch.Tensor((array.transpose([2, 0, 1])).astype(np.float32)) / 255

def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')

class HRLRDatasetV(data.Dataset):
    def __init__(self, cfg, scale):
        super(HRLRDatasetV, self).__init__()
        self.lr_file_path = cfg.data_path + f"/{scale}/lr/val"
        self.lr_paths = fiFindByWildcard(os.path.join(self.lr_file_path, '*.png'))

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        lr = imread(lr_path)
        # print(lr.shape)
        pad_factor = 2
        # Pad image to be % 2

        h, w, c = lr.shape
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))

        lr_t = t(lr)
        return lr_t, h, w

    def __len__(self):
        return len(self.lr_paths)

class LRHR_IMGDataset(data.Dataset):
    def __init__(self, cfg, scale, mode):
        super(LRHR_IMGDataset, self).__init__()
        self.scale = scale
        self.crop_size = cfg.im_size

        self.hr_file_path = cfg.data_path + f"/{scale}/hr/{mode}"
        augment = True

        self.use_flip = True
        self.use_rot = True
        self.use_crop = False
        self.center_crop_hr_size = None

        self.hr_paths = fiFindByWildcard(os.path.join(self.hr_file_path, '*.png'))

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
        hr = random_crop(hr, self.crop_size)
        lr = imresize(hr, scalar_scale=1 / self.scale)
        lf = imresize(lr, scalar_scale=self.scale)

        hr = np.transpose(hr, [2, 0, 1])
        lr = np.transpose(lr, [2, 0, 1])
        lf = np.transpose(lf, [2, 0, 1])

        if self.use_flip:
            hr, lr, lf = random_flip(hr, lr, lf)

        if self.use_rot:
            hr, lr, lf = random_rotation(hr, lr, lf)

        hr = hr / 255.0
        lr = lr / 255.0
        lf = lf / 255.0

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)
        lf = torch.Tensor(lf)

        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        hr = (hr - mean[:, None, None]) / std[:, None, None]
        lr = (lr - mean[:, None, None]) / std[:, None, None]
        lf = (lf - mean[:, None, None]) / std[:, None, None]
        return lr, hr, mean, std


def random_flip(img, seg, hf):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    hf = hf if random_choice else np.flip(hf, 2).copy()

    return img, seg, hf


def random_rotation(img, seg, hf):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    hf = np.rot90(hf, random_choice, axes=(1, 2)).copy()

    return img, seg, hf


def random_crop(img, size):
    h, w, c = img.shape

    h_start = np.random.randint(0, h - size)
    h_end = h_start + size

    w_start = np.random.randint(0, w - size)
    w_end = w_start + size

    return img[h_start:h_end, w_start:w_end]


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
        dataset = LRHR_IMGDataset(self.args, self.scale, "train")
        train_data, dev_data = torch.utils.data.random_split(
            dataset, [700, 100],
            generator=torch.Generator().manual_seed(0)
        )
        test_data = LRHR_IMGDataset(self.args, self.scale, "val")

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
            batch_size=1,
            drop_last=False,
            num_workers=5,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=1,
            drop_last=False,
            num_workers=5,
            pin_memory=False,
        )
