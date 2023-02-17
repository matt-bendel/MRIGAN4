import pytorch_lightning as pl
from typing import Optional

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.inpaint.get_mask import get_mask


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

    def __call__(self, gt_im):
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        mask = get_mask(self.args.im_size, mask_type=self.mask_type)
        gt = (gt_im - mean[:, None, None]) / std[:, None, None]
        masked_im = gt * mask

        return gt, masked_im, mean, std, mask


class CelebAHQDataModule(pl.LightningDataModule):
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
        transform = transforms.Compose([transforms.ToTensor(), DataTransform(self.args, self.mask_type)])
        dataset = datasets.ImageFolder(self.args.data_path, transform=transform)
        train_data, dev_data, test_data = torch.utils.data.random_split(
            dataset, [27000, 2000, 1000],
            generator=torch.Generator().manual_seed(0)
        )

        self.train, self.validate, self.test = train_data, dev_data, test_data

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            drop_last=True,
            num_workers=20,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            drop_last=True,
            num_workers=20,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.args.batch_size,
            num_workers=20,
            pin_memory=True,
        )