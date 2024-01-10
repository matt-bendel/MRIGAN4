from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from data.mri_data import SelectiveSliceData, SelectiveSliceData_Val, SelectiveSliceData_Test

import pathlib
import cv2
import torch
import numpy as np

from utils.espirit import ifft, fft
from data import transforms
from utils.fftc import ifft2c_new, fft2c_new
from utils.get_mask import get_mask


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, mask_type, use_seed=False, test=False):
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
        self.use_seed = use_seed
        self.args = args
        self.mask = None
        self.test = test
        self.mask_type = mask_type

    def __call__(self, kspace, target, attrs, fname, slice, sense_maps=None, vh=None):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        # GRO Sampling mask:
        mask, nlf = get_mask(self.args.im_size, return_mask=True, R=self.args.R, args=self.args, mask_type=self.mask_type, varnet=True)
        kspace = kspace.transpose(1, 2, 0)
        x = ifft(kspace, (0, 1))  # (768, 396, 16)

        # TODO: Save SVD matrix offline
        coil_compressed_x = ImageCropandKspaceCompression(x, None)  # (384, 384, 8)

        im_tensor = transforms.to_tensor(coil_compressed_x).permute(2, 0, 1, 3)

        # if self.args.im_size == 128:
        #     TODO: Save resized image offline
            # im_tensor = reduce_resolution(im_tensor)

        kspace = fft2c_new(im_tensor)
        masked_kspace = kspace * mask
        input_tensor = ifft2c_new(masked_kspace)

        normalized_input, mean, std = transforms.normalize_instance(input_tensor)
        normalized_gt = transforms.normalize(im_tensor, mean, std)

        final_input = torch.zeros(16, self.args.im_size, self.args.im_size)
        final_input[0:8, :, :] = normalized_input[:, :, :, 0]
        final_input[8:16, :, :] = normalized_input[:, :, :, 1]

        final_gt = torch.zeros(16, self.args.im_size, self.args.im_size)
        final_gt[0:8, :, :] = normalized_gt[:, :, :, 0]
        final_gt[8:16, :, :] = normalized_gt[:, :, :, 1]

        return final_input.float(), final_gt.float(), mask, mean, std, transforms.to_tensor(sense_maps), fname, slice #, nlf



def reduce_resolution(im):
    reduced_im = np.zeros((8, 128, 128, 2))
    for i in range(im.shape[0] // 2):
        reduced_im[i, :, :, 0] = cv2.resize(im[i, :, :, 0].numpy(), dsize=(128, 128),
                                            interpolation=cv2.INTER_LINEAR)
        reduced_im[i, :, :, 1] = cv2.resize(im[i, :, :, 1].numpy(), dsize=(128, 128),
                                            interpolation=cv2.INTER_LINEAR)

    return transforms.to_tensor(reduced_im)


# Helper functions for Transform
def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def unflatten(t, shape_t):
    t = t.reshape(shape_t)
    return t


def ImageCropandKspaceCompression(x, vh):
    w_from = (x.shape[0] - 384) // 2  # crop images into 384x384
    h_from = (x.shape[1] - 384) // 2
    w_to = w_from + 384
    h_to = h_from + 384
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > 8:
        x_tocompression = cropped_x.reshape(384 ** 2, cropped_x.shape[-1])

        if vh is None:
            U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
            coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
            coil_compressed_x = coil_compressed_x[:, 0:8].reshape(384, 384, 8)
        else:
            coil_compressed_x = np.matmul(x_tocompression, vh.conj().T)
            coil_compressed_x = coil_compressed_x[:, 0:8].reshape(384, 384, 8)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x


class MRIDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, args, mask_type, big_test=False):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.mask_type = mask_type
        self.big_test = big_test

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        train_data = SelectiveSliceData(
            root=pathlib.Path(self.args.data_path) / 'multicoil_train',
            transform=DataTransform(self.args, self.mask_type),
            challenge='multicoil',
            sample_rate=1,
            use_top_slices=True,
            number_of_top_slices=self.args.num_of_top_slices,
            restrict_size=False,
        )

        dev_data = SelectiveSliceData_Val(
            root=pathlib.Path(self.args.data_path) / 'multicoil_val',
            transform=DataTransform(self.args, self.mask_type, test=True),
            challenge='multicoil',
            sample_rate=1,
            use_top_slices=True,
            number_of_top_slices=self.args.num_of_top_slices,
            restrict_size=False,
            big_test=self.big_test
        )

        test_data = SelectiveSliceData_Test(
            root=pathlib.Path(self.args.data_path) / 'small_T2_test',
            transform=DataTransform(self.args, self.mask_type, test=True),
            challenge='multicoil',
            sample_rate=1,
            use_top_slices=True,
            number_of_top_slices=6,
            restrict_size=False,
            big_test=True
        )

        self.train, self.validate, self.test = train_data, dev_data, test_data

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            num_workers=4,
            drop_last=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            num_workers=20,
            drop_last=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )