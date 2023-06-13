import torch.utils.data as data
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from data_loaders.sr.RealESRGANDataset import RealESRGANDataset

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
        train_data = RealESRGANDataset(self.args)
        dev_data = None
        test_data = None

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
