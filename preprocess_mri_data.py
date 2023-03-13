import h5py
import pathlib
import cv2
import numpy as np

from utils.math import tensor_to_complex_np
from utils.espirit import ifft, fft
from data import transforms
from utils.fftc import ifft2c_new, fft2c_new

def reduce_resolution(im):
    reduced_im = np.zeros((8, 128, 128, 2))
    for i in range(im.shape[0] // 2):
        reduced_im[i, :, :, 0] = cv2.resize(im[i, :, :, 0].numpy(), dsize=(128, 128),
                                            interpolation=cv2.INTER_LINEAR)
        reduced_im[i, :, :, 1] = cv2.resize(im[i, :, :, 1].numpy(), dsize=(128, 128),
                                            interpolation=cv2.INTER_LINEAR)

    return reduced_im


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

if __name__ == '__main__':
    root = '/storage/fastMRI_brain/data/multicoil_train'
    files = list(pathlib.Path(root).iterdir())

    for fname in sorted(files):
        h5_file = h5py.File(fname, 'r')
        print(fname)
        if h5_file.attrs['acquisition'] != 'AXT2':
            print("CONTINUE")
            continue

        ks = h5_file['kspace']

        if ks.shape[-1] < 384 or ks.shape[1] < 8 or str(
                fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_209_2090296.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_200_2000250.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_201_2010106.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_204_2130024.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_210_2100025.h5':
            continue
        else:
            num_slices = 8  # kspace.shape[0]

        new_ksp = np.zeros((ks.shape[0], 8, 128, 128, 2))
        for j in range(ks.shape[0]):
            kspace = ks[j]
            print(kspace.shape)
            x = tensor_to_complex_np(ifft2c_new(transforms.to_tensor(kspace))).transpose(1, 2, 0)
            print(x.shape)
            # x = ifft(kspace, (0, 1))  # (768, 396, 16)

            print("crop")
            coil_compressed_x = ImageCropandKspaceCompression(x)  # (384, 384, 8)

            print("resize")
            im_tensor = transforms.to_tensor(coil_compressed_x).permute(2, 0, 1, 3)

            new_ksp[j] = reduce_resolution(im_tensor)

        hf = h5py.File(f'/storage/fastMRI_brain/preprocessed_data/multicoil_train/{fname}', 'w')
        hf.attrs = h5_file.attrs
        hf.create_dataset('kspace', new_ksp)
        hf.create_dataset('reconstruction_rss', h5_file['reconstruction_rss'])
        hf.close()


    root = '/storage/fastMRI_brain/data/multicoil_val'
    files = list(pathlib.Path(root).iterdir())

    for fname in sorted(files):
        h5_file = h5py.File(fname, 'r')
        print(fname)
        if h5_file.attrs['acquisition'] != 'AXT2':
            print("CONTINUE")
            continue

        ks = h5_file['kspace']

        if ks.shape[-1] <= 384 or ks.shape[1] < 8 or str(
                fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_209_2090296.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_200_2000250.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_201_2010106.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_204_2130024.h5' or str(
            fname) == '/storage/fastMRI_brain/data/multicoil_val/file_brain_AXT2_210_2100025.h5':
            continue
        else:
            num_slices = 8  # kspace.shape[0]

        new_ksp = np.zeros(ks.shape)
        for j in range(ks.shape[0]):
            kspace = ks[j].transpose(1, 2, 0)
            x = ifft(kspace, (0, 1))  # (768, 396, 16)

            coil_compressed_x = ImageCropandKspaceCompression(x)  # (384, 384, 8)

            im_tensor = transforms.to_tensor(coil_compressed_x).permute(2, 0, 1, 3)

            new_ksp[j] = reduce_resolution(im_tensor)

        hf = h5py.File(f'/storage/fastMRI_brain/preprocessed_data/multicoil_val/{fname}', 'w')
        hf.attrs = h5_file.attrs
        hf.create_dataset('kspace', new_ksp)
        hf.create_dataset('reconstruction_rss', h5_file['reconstruction_rss'])
        hf.close()
