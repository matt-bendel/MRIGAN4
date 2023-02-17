import pytorch_lightning as pl
from typing import Optional

import torch
import h5py
import numpy as np
import os
import pickle

from torch.utils.data import DataLoader
from utils.cs.get_operator import get_operator

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return self.transform(torch.Tensor(data))

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, mask_type):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create  a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.args = args
        self.mask_type = mask_type
        self.cs_transform = None
        with open(f'cs_ref_operator.pkl', 'rb') as inp:
            self.cs_transform = pickle.load(inp)

    def __call__(self, gt):
        cs_transform = get_operator(self.args.im_size, mask_type=self.mask_type)

        if cs_transform is None:
            cs_transform = self.cs_transform
        masked_im = self.cs_transform.A(gt)

        return masked_im, gt, cs_transform


class BSD400DataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, args, mask_type):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.mask_type = mask_type

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        train_data = Dataset(filename=os.path.join("/storage/bsd400", "train.h5"), transform=DataTransform(self.args, self.mask_type))
        dev_data = Dataset(filename=os.path.join("/storage/bsd400", "valid.h5"), transform=DataTransform(self.args, self.mask_type))
        test_data = None

        self.train, self.validate, self.test = train_data, dev_data, test_data

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            drop_last=True,
            num_workers=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            drop_last=True,
            num_workers=1,
            pin_memory=True,
        )

    def test_dataloader(self):
        return None