from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from typing import Optional
from data.mri_data import SelectiveSliceData, SelectiveSliceData_Val
from data_loaders.prepare_data import DataTransform

class MRIDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, args):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        if stage == "fit" or stage is None:
            train_data = SelectiveSliceData(
                root=self.args.data_path / 'multicoil_train',
                transform=DataTransform(self.args),
                challenge='multicoil',
                sample_rate=1,
                use_top_slices=True,
                number_of_top_slices=self.args.num_of_top_slices,
                restrict_size=False,
            )

            dev_data = SelectiveSliceData_Val(
                root=self.args.data_path / 'multicoil_val',
                transform=DataTransform(self.args, test=True),
                challenge='multicoil',
                sample_rate=1,
                use_top_slices=True,
                number_of_top_slices=self.args.num_of_top_slices,
                restrict_size=False,
                big_test=False
            )
            self.train, self.validate = train_data, dev_data

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.test = YourCustomDataset(
        #         root_path="/Users/yourusername/path/to/data/test_set/",
        #         ipt="input/",
        #         tgt="target/",
        #         tgt_scale=25,
        #         train_transform=False)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return None