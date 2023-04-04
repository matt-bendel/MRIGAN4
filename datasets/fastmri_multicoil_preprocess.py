"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff
"""


import torch
import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

from typing import Dict, Optional, Sequence, Tuple, Union
import time
import yaml
from warnings import warn
import xml.etree.ElementTree as etree

import cupy as cp

import sys
sys.path.append("..")

import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from utils.get_mask import get_mask

# Coil compression and image cropping
# Needs to be [imgsize, imgsize, num_coils]
# Modified so it uses cupy and is 10x faster
def ImageCropandKspaceCompression(x, size, num_vcoils = 8, vh = None):
    w_from = (x.shape[0] - size) // 2  # crop images into 384x384
    h_from = (x.shape[1] - size) // 2
    w_to = w_from + size
    h_to = h_from + size
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > num_vcoils:
        x_tocompression = cropped_x.reshape(size ** 2, cropped_x.shape[-1])
        
        if vh is None:
            #Convert to a cupy tensor
            with cp.cuda.Device(0):
                U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
                coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
                coil_compressed_x = coil_compressed_x[:, 0:8].reshape(384, 384, 8)
        else:
            coil_compressed_x = np.matmul(x_tocompression, vh.conj().T)
            coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
            Vh = vh
                
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x


def get_compressed(kspace: np.ndarray, img_size, mask_type, accel_rate, attrs: Dict, fname:str, num_vcoils = 8, vh=None):
    #kspace is dimension [num_slices, num_coils, size0, size1]


    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace)
    
    compressed_imgs = []
    Vhs = []

    for i in range(kspace_torch.shape[0]):
        #Compress to 8 virtual coils and crop
        compressed_img, Vh = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img[i]).transpose(1,2,0),
                                                       img_size, num_vcoils, vh)
        compressed_imgs.append(to_tensor(compressed_img))
        Vhs.append(Vh)

    #Combine into one tensor stack
    compressed_imgs = torch.stack(compressed_imgs)
    Vhs = np.stack(Vhs)
    

        
    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(compressed_imgs.permute(0,3,1,2,4))
    
    if vh is None:
        return compressed_k, Vhs
    else:
        return compressed_k

        
# TODO: Change the normalizing expression here or get rid of this step if want to do in dataloader
def get_normalizing_val(kspace: np.ndarray, img_size, mask_type, accel_rate):
    """
    Args:
        kspace: Input k-space of shape (num_coils, rows, cols) for
            multi-coil data or (rows, cols) for single coil data.
        mask: Mask from the test dataset.
        target: Target image.
        attrs: Acquisition related information stored in the HDF5 object.
        fname: File name.
        slice_num: Serial number of the slice.
    Returns:
        A tuple containing, zero-filled input image, the reconstruction
        target, the mean used for normalization, the standard deviations
        used for normalization, the filename, and the slice number.
    """

    #kspace is dimension [num_slices, num_coils, size0, size1]
    #kspace_torch = to_tensor(kspace)
    
    #Apply the mask
    m = np.zeros((384, 384))

    a = np.array(
        [1, 23, 42, 60, 77, 92, 105, 117, 128, 138, 147, 155, 162, 169, 176, 182, 184, 185, 186, 187, 188, 189, 190,
         191, 192, 193, 194, 195,
         196, 197, 198, 199, 200, 204, 210, 217, 224, 231, 239, 248, 258, 269, 281, 294, 309, 326, 344, 363])
    m[:, a] = True

    samp = m
    numcoil = 8
    mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
    mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    mask = get_mask(img_size, R=accel_rate)
    masked_kspace = kspace * mask
    
    #Get the zf imgs
    masked_imgs = fastmri.ifft2c(masked_kspace)
    
    #Get the magnitude imgs for the zf imgs
    zf_mags = fastmri.complex_abs(masked_imgs)


    #Normalized based on the 95th percentile max value of the magnitude
    max_val = np.percentile(zf_mags.cpu(), 95)
    

    return masked_kspace, max_val
    #return zf_imgs, gt_imgs, mask, fname, slice_nums, acquisition
    
    
#%% From fastMRI repository 
def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.
    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.
    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.
    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)
    
def retrieve_metadata(fname):
    with h5py.File(fname, "r") as hf:
        et_root = etree.fromstring(hf["ismrmrd_header"][()])

        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )
        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (
            int(et_query(et_root, rec + ["x"])),
            int(et_query(et_root, rec + ["y"])),
            int(et_query(et_root, rec + ["z"])),
        )

        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max

        num_slices = hf["kspace"].shape[0]

    metadata = {
        "padding_left": padding_left,
        "padding_right": padding_right,
        "encoding_size": enc_size,
        "recon_size": recon_size,
    }

    return metadata, num_slices


if __name__ == '__main__':

    # TODO: Change these parameters for what you need
    #Parameters
    mri_type = 'brain'
    accel_rate = 4
    img_size = 384
    num_vcoils = 8
    dataset_type = 'train' #'train' or 'val
    challenge = 'multicoil' #'multicoil' or 'singlecoil'
    mask_type = 'matt2'
    acquisition = 'AXT2' #'AXT2', 'AXT1POST', 'AXFLAIR', or None

    # Location of the dataset
    if mri_type == 'brain':
        base_dir = "/storage/fastMRI_brain/data/"
    else:
        base_dir = "/storage/fastMRI/data/"

    dataset_dir = os.path.join(base_dir, '{0}_{1}'.format(challenge, dataset_type))
    #dataset_dir = os.path.join(base_dir, 'small_T2_test')
    new_dir = os.path.join(base_dir, '{0}_{1}_{2}coils_general_matt'.format(challenge, dataset_type, num_vcoils))
    #new_dir = os.path.join(base_dir, '{0}_{1}_{2}coils'.format(challenge, 'small_T2_test', num_vcoils))
    
    
    #Create the directory if it does not already exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    # Get the compressed coil images
    files = list(Path(dataset_dir).iterdir())
    
    for fname in tqdm(sorted(files)):
        
        #Skip non-data files
        if fname.name[0] == '.':
            continue
        
        #Recover the metadata
        metadata, num_slices = retrieve_metadata(fname)
    
        with h5py.File(fname, "r") as hf:
            
            #Get the kspace
            kspace = hf['kspace']
            
            #Get the attributes of the volume
            attrs = dict(hf.attrs)
            attrs.update(metadata)

            if acquisition is not None:
                if attrs['acquisition'] != acquisition:
                    continue

            if attrs['encoding_size'][1] < img_size:
                continue
    
            kspace_torch = to_tensor(kspace[:])

            if kspace_torch.shape[1] <= num_vcoils:
                continue


            #Get the virtual coils and the normalization value for the volume
            #start = time.time()
            compressed_k, vhs = get_compressed(kspace_torch, img_size, mask_type, accel_rate, attrs, fname.name, num_vcoils)
            #end = time.time()
            masked_kspace, max_val = get_normalizing_val(compressed_k, img_size, mask_type, accel_rate)
            #end2 = time.time()
            #print(end-start)
            #print(end2-end)

        if num_slices != vhs.shape[0]:
            raise Exception('Problem with {}'.fname.name[0])


        #Save the processed data into the new h5py file
        with h5py.File(os.path.join(new_dir, fname.name), 'w') as nf:
            nf.attrs['max_val'] = max_val
            #Save the Vh from the svd
            vh = nf.create_dataset('vh', vhs.shape, data=vhs)


        
