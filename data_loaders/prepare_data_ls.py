import cv2
import torch
import numpy as np
import sigpy as sp

from utils.espirit import ifft, fft
from torch.utils.data import DataLoader
from data import transforms
from data.mri_data_ls import SelectiveSliceData, SelectiveSliceData_Val
from utils.fftc import ifft2c_new, fft2c_new
from utils.get_mask import get_mask
from utils.math import complex_abs


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, use_seed=False, test=False):
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

    def __call__(self, ls_rec, target, sense_maps, fname, slice):
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
        mask = get_mask(384, return_mask=True, R=4)

        gt = transforms.to_tensor(target)
        ls_recon = transforms.to_tensor(ls_rec)

        gt = reduce_resolution(gt)
        ls_recon = reduce_resolution(ls_recon)

        true_image = torch.clone(gt)
        true_measures = fft2c_new(gt) * mask

        normalized_input, mean, std = transforms.normalize_instance(ls_recon)
        normalized_gt = transforms.normalize(true_image, mean, std)

        # For Dynamic Inpainting
        normalized_true_measures = transforms.normalize(ifft2c_new(true_measures), mean, std)
        normalized_true_measures = fft2c_new(normalized_true_measures)

        final_input = torch.zeros(16, 384, 384)
        final_input[0:8, :, :] = normalized_input[:, :, :, 0]
        final_input[8:16, :, :] = normalized_input[:, :, :, 1]

        final_gt = torch.zeros(16, 384, 384)
        final_gt[0:8, :, :] = normalized_gt[:, :, :, 0]
        final_gt[8:16, :, :] = normalized_gt[:, :, :, 1]

        return final_input.float(), final_gt.float(), normalized_true_measures.float(), mean.float(), std.float()


def create_datasets(args, val_only, big_test=False):
    if not val_only:
        train_data = SelectiveSliceData(
            root=args.data_path / f'train_{args.R}',
            transform=DataTransform(args),
            challenge='multicoil',
            sample_rate=1,
            use_top_slices=True,
            number_of_top_slices=args.num_of_top_slices,
            restrict_size=False,
        )

    dev_data = SelectiveSliceData_Val(
        root=args.data_path / f'val_{args.R}',
        transform=DataTransform(args, test=True),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        restrict_size=False,
        big_test=big_test
    )

    return dev_data, train_data if not val_only else dev_data


def create_data_loaders_ls(args, val_only=False, big_test=False):
    dev_data, train_data = create_datasets(args, val_only, big_test=big_test)

    if not val_only:
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
        )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader if not val_only else None, dev_loader


def create_test_dataset(args):
    data = SelectiveSliceData_Val(
        root=args.data_path / 'small_T2_test',
        transform=DataTransform(args, test=True),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        restrict_size=False,
        big_test=True,
        test_set=True
    )
    return data


def create_test_loader(args):
    data = create_test_dataset(args)

    loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    return loader


def reduce_resolution(im):
    reduced_im = np.zeros((8, 384, 384, 2))
    for i in range(im.shape[0] // 2):
        reduced_im[i, :, :, 0] = cv2.resize(im[i, :, :, 0].numpy(), dsize=(384, 384),
                                            interpolation=cv2.INTER_LINEAR)
        reduced_im[i, :, :, 1] = cv2.resize(im[i, :, :, 1].numpy(), dsize=(384, 384),
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


def ImageCropandKspaceCompression(x):
    w_from = (x.shape[0] - 384) // 2  # crop images into 384x384
    h_from = (x.shape[1] - 384) // 2
    w_to = w_from + 384
    h_to = h_from + 384
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > 8:
        x_tocompression = cropped_x.reshape(384 ** 2, cropped_x.shape[-1])
        U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:, 0:8].reshape(384, 384, 8)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x
